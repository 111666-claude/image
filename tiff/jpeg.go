// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
)

// jpegSOI and jpegEOI are the Start-Of-Image and End-Of-Image markers that
// delimit a complete, self-contained baseline JPEG stream.
var (
	jpegSOI = []byte{0xff, 0xd8}
	jpegEOI = []byte{0xff, 0xd9}
)

// decodeJPEG decodes one new-style (compression 7) JPEG strip and places it at
// its row range in dst. A JPEG strip must hold a complete JPEG stream whose
// pixel dimensions exactly match the strip's width and height. The last strip
// of an image may be shorter than RowsPerStrip.
//
// stripIndex identifies the strip within the image and is used in error
// messages so that a corrupt strip is reported with its position. xmin and
// ymin give the strip's top-left corner in the destination image.
func (d *decoder) decodeJPEG(dst image.Image, stripIndex, xmin, ymin int, offset, n int64, width, height int) error {
	// Read the compressed strip. Reads past the end of the input (for example
	// a StripOffset beyond the file) fail here rather than silently truncating
	// the stream.
	var data []byte
	var err error
	if b, ok := d.r.(*buffer); ok {
		data, err = b.Slice(offset, n)
	} else {
		data, err = safeReadAt(d.r, uint64(n), offset)
	}
	if err != nil {
		return FormatError(fmt.Sprintf("strip %d: cannot read JPEG data: %v", stripIndex, err))
	}

	// A new-style JPEG strip is a full interchange stream, delimited by the
	// SOI and EOI markers.
	if len(data) < 2 || !bytes.Equal(data[:2], jpegSOI) {
		return FormatError(fmt.Sprintf("strip %d: JPEG stream is missing the SOI marker", stripIndex))
	}
	if !bytes.Equal(data[len(data)-2:], jpegEOI) {
		return FormatError(fmt.Sprintf("strip %d: JPEG stream is missing the EOI marker", stripIndex))
	}

	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return FormatError(fmt.Sprintf("strip %d: corrupt JPEG data: %v", stripIndex, err))
	}

	// The decoded strip must cover exactly the rows assigned to it; a
	// dimension mismatch would otherwise silently place pixels on the wrong
	// rows or leave part of the image uninitialized.
	bounds := img.Bounds()
	if bounds.Dx() != width || bounds.Dy() != height {
		return FormatError(fmt.Sprintf("strip %d: JPEG image size %dx%d does not match strip size %dx%d",
			stripIndex, bounds.Dx(), bounds.Dy(), width, height))
	}

	rgba, ok := dst.(*image.RGBA)
	if !ok {
		return FormatError(fmt.Sprintf("strip %d: unsupported destination image type %T", stripIndex, dst))
	}

	// Copy the decoded strip into the destination at its row range. Going
	// through At/RGBA matches the exact YCbCr-to-RGB conversion that the
	// standard library's JPEG decoder exposes, and an RGB JPEG already
	// decodes to *image.RGBA so the common case stays a straight copy.
	if src, isRGBA := img.(*image.RGBA); isRGBA {
		for y := 0; y < height; y++ {
			sRow := src.Pix[src.PixOffset(bounds.Min.X, bounds.Min.Y+y):]
			dRow := rgba.Pix[rgba.PixOffset(xmin, ymin+y):]
			copy(dRow[:width*4], sRow[:width*4])
		}
		return nil
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			i := rgba.PixOffset(xmin+x, ymin+y)
			r, g, b, a := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			rgba.Pix[i+0] = uint8(r >> 8)
			rgba.Pix[i+1] = uint8(g >> 8)
			rgba.Pix[i+2] = uint8(b >> 8)
			rgba.Pix[i+3] = uint8(a >> 8)
		}
	}
	return nil
}
