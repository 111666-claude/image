// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
)

// maxPages is the maximum number of pages (IFDs) that a Reader will walk.
// A TIFF file with more directories than this is rejected, which bounds the
// amount of work that a malicious input can cause when its IFD chain is
// enumerated.
const maxPages = 1 << 16

var (
	// ErrPageOutOfRange is returned by Reader methods when the page number
	// is negative or is not less than NumPages.
	ErrPageOutOfRange = errors.New("tiff: page number out of range")

	// ErrTooManyPages is returned by NewReader when the file's IFD chain
	// contains more than maxPages directories.
	ErrTooManyPages = errors.New("tiff: too many pages")
)

// Reader reads the images, called pages, of a TIFF file. Unlike Decode, which
// only reads the first page, a Reader gives random access to every page in the
// file's main IFD chain. Pages are decoded independently, so decoding one page
// does not decode or disturb any other page.
//
// A Reader does not modify the data it reads. Calling DecodePage repeatedly
// for the same page returns pixel-identical images.
//
// The Reader does not follow SubIFDs. Only the main IFD chain is enumerated,
// and only classic (non-BigTIFF) files are supported.
type Reader struct {
	r         io.ReaderAt
	byteOrder binary.ByteOrder
	ifds      []int64 // IFD offsets, in page order.
}

// NewReader reads the TIFF header from r and enumerates the file's main IFD
// chain, without decoding any pixel data. The returned Reader gives the page
// count and per-page configuration and decoding.
//
// If r does not implement io.ReaderAt, its contents are buffered, as they are
// by Decode. The first IFD offset must be non-zero and point inside the file.
// Malformed, truncated, cyclic or excessively long IFD chains are rejected.
func NewReader(r io.Reader) (*Reader, error) {
	ra := newReaderAt(r)
	byteOrder, ifdOffset, err := readHeader(ra)
	if err != nil {
		return nil, err
	}
	if ifdOffset == 0 {
		return nil, FormatError("missing IFD")
	}

	seen := make(map[int64]struct{})
	var ifds []int64
	for ifdOffset != 0 {
		if ifdOffset < 8 {
			// The 8 byte header occupies [0, 8), so an IFD there would
			// overlap it (and would have to be a cycle or garbage).
			return nil, FormatError("IFD offset out of range")
		}
		if _, ok := seen[ifdOffset]; ok {
			return nil, FormatError("cyclic IFD chain")
		}
		if len(ifds) >= maxPages {
			return nil, fmt.Errorf("%w (limit %d)", ErrTooManyPages, maxPages)
		}
		seen[ifdOffset] = struct{}{}
		ifds = append(ifds, ifdOffset)

		ifdOffset, err = readNextIFDOffset(ra, byteOrder, ifdOffset)
		if err != nil {
			return nil, err
		}
	}

	return &Reader{
		r:         ra,
		byteOrder: byteOrder,
		ifds:      ifds,
	}, nil
}

// readNextIFDOffset returns the offset of the IFD that follows the one at
// ifdOffset, or 0 if it is the last IFD.
func readNextIFDOffset(r io.ReaderAt, byteOrder binary.ByteOrder, ifdOffset int64) (int64, error) {
	// The IFD starts with a 16-bit entry count.
	var p [2]byte
	n, err := r.ReadAt(p[:], ifdOffset)
	if n == 0 {
		// Nothing could be read at ifdOffset: the pointer is out of the
		// file's bounds.
		return 0, FormatError("IFD offset out of range")
	}
	if err != nil {
		// A partial entry count: the file is truncated.
		return 0, io.ErrUnexpectedEOF
	}
	numItems := int64(byteOrder.Uint16(p[:]))
	// numItems is a uint16, so 2+ifdLen*numItems cannot overflow an int64.

	// The next IFD offset immediately follows the IFD's entries.
	nextOffsetPos := ifdOffset + 2 + ifdLen*numItems
	var q [4]byte
	if _, err := r.ReadAt(q[:], nextOffsetPos); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return 0, io.ErrUnexpectedEOF
		}
		return 0, err
	}
	return int64(byteOrder.Uint32(q[:])), nil
}

// NumPages returns the number of pages (images) in the file.
func (r *Reader) NumPages() int {
	return len(r.ifds)
}

// pageDecoder returns a decoder for the page with the given number.
func (r *Reader) pageDecoder(page int) (*decoder, error) {
	if page < 0 || page >= len(r.ifds) {
		return nil, fmt.Errorf("%w: %d (file has %d pages)", ErrPageOutOfRange, page, len(r.ifds))
	}
	d, err := newPageDecoder(r.r, r.byteOrder, r.ifds[page])
	if err != nil {
		return nil, fmt.Errorf("tiff: page %d: %w", page, err)
	}
	// A Reader may be used to decode the same page more than once, and must
	// not modify the caller's input. Copy uncompressed block data rather
	// than decoding in place.
	d.copyBlocks = true
	return d, nil
}

// PageConfig returns the color model and dimensions of the given page without
// decoding its pixel data. Page numbers start at zero. PageConfig(0) is
// equivalent to DecodeConfig, which also reports the first page.
func (r *Reader) PageConfig(page int) (image.Config, error) {
	d, err := r.pageDecoder(page)
	if err != nil {
		return image.Config{}, err
	}
	return d.config, nil
}

// DecodePage decodes the given page and returns it as an image.Image. Page
// numbers start at zero. DecodePage(0) is equivalent to Decode, which also
// decodes the first page.
//
// The image returned for a page does not overlap with the images of other
// pages, and repeated calls for the same page return identical pixels.
func (r *Reader) DecodePage(page int) (image.Image, error) {
	d, err := r.pageDecoder(page)
	if err != nil {
		return nil, err
	}
	return d.decodeImage()
}
