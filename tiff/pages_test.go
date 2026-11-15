// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"io"
	"os"
	"sort"
	"testing"
)

// pageSpec describes one page of a synthetic multi-page TIFF.
type pageSpec struct {
	width, height int
	compression   uint16
	predictor     bool
	tileWidth     int // Zero means strips.
	tileHeight    int
	seed          int
}

// pagePixel is the value of the pixel at (x, y) in a page with the given seed.
// The seed differs per page, so that pixels read from the wrong page (or the
// wrong offset within a page) are detected.
func pagePixel(seed, x, y int) uint8 {
	return uint8((x*11 + y*29 + seed*53) & 0xff)
}

func expectedPage(spec pageSpec) *image.Gray {
	m := image.NewGray(image.Rect(0, 0, spec.width, spec.height))
	for y := 0; y < spec.height; y++ {
		for x := 0; x < spec.width; x++ {
			m.SetGray(x, y, color.Gray{pagePixel(spec.seed, x, y)})
		}
	}
	return m
}

func deflateData(t *testing.T, p []byte) []byte {
	t.Helper()
	var b bytes.Buffer
	w := zlib.NewWriter(&b)
	if _, err := w.Write(p); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

// packBitsEncode encodes p as a series of literal PackBits runs.
func packBitsEncode(p []byte) []byte {
	var dst []byte
	for len(p) > 0 {
		n := min(len(p), 128)
		dst = append(dst, byte(n-1))
		dst = append(dst, p[:n]...)
		p = p[n:]
	}
	return dst
}

// applyPredictor applies the horizontal differencing predictor to each row of
// an 8 bit per sample, 1 sample per pixel image.
func applyPredictor(p []byte, width, height int) {
	for y := 0; y < height; y++ {
		row := p[y*width : (y+1)*width]
		for x := width - 1; x >= 1; x-- {
			row[x] -= row[x-1]
		}
	}
}

func compressBlock(t *testing.T, compression uint16, p []byte) []byte {
	t.Helper()
	switch compression {
	case cNone, 0:
		return p
	case cDeflate:
		return deflateData(t, p)
	case cPackBits:
		return packBitsEncode(p)
	}
	t.Fatalf("unsupported test compression %d", compression)
	return nil
}

// pageBlocks returns the raw (before compression) blocks of a page.
func pageBlocks(spec pageSpec) [][]byte {
	var blocks [][]byte
	if spec.tileWidth > 0 {
		tw, th := spec.tileWidth, spec.tileHeight
		across := (spec.width + tw - 1) / tw
		down := (spec.height + th - 1) / th
		for ty := 0; ty < down; ty++ {
			for tx := 0; tx < across; tx++ {
				raw := make([]byte, tw*th)
				for ly := 0; ly < th; ly++ {
					for lx := 0; lx < tw; lx++ {
						gx, gy := tx*tw+lx, ty*th+ly
						if gx < spec.width && gy < spec.height {
							raw[ly*tw+lx] = pagePixel(spec.seed, gx, gy)
						}
					}
				}
				blocks = append(blocks, raw)
			}
		}
		return blocks
	}

	raw := make([]byte, spec.width*spec.height)
	for y := 0; y < spec.height; y++ {
		for x := 0; x < spec.width; x++ {
			raw[y*spec.width+x] = pagePixel(spec.seed, x, y)
		}
	}
	if spec.predictor {
		applyPredictor(raw, spec.width, spec.height)
	}
	return [][]byte{raw}
}

// tiffEntry is one IFD entry to be written by appendIFDEntries.
type tiffEntry struct {
	tag   uint16
	typ   uint16
	count uint32
	data  []byte // Encoded in the file's byte order.
}

func shortEntry(bo byteOrder, tag uint16, v uint16) tiffEntry {
	data := make([]byte, 2)
	bo.PutUint16(data, v)
	return tiffEntry{tag: tag, typ: dtShort, count: 1, data: data}
}

func longEntry(bo byteOrder, tag uint16, v uint32) tiffEntry {
	data := make([]byte, 4)
	bo.PutUint32(data, v)
	return tiffEntry{tag: tag, typ: dtLong, count: 1, data: data}
}

func longsEntry(bo byteOrder, tag uint16, vs []uint32) tiffEntry {
	data := make([]byte, 4*len(vs))
	for i, v := range vs {
		bo.PutUint32(data[4*i:], v)
	}
	return tiffEntry{tag: tag, typ: dtLong, count: uint32(len(vs)), data: data}
}

// appendIFDEntries appends one IFD, initialized with the offset of the next
// IFD set to zero, and returns b and the offset of that next-IFD field.
func appendIFDEntries(b []byte, bo byteOrder, entries []tiffEntry) ([]byte, int) {
	sort.Slice(entries, func(i, j int) bool { return entries[i].tag < entries[j].tag })

	start := len(b)
	b = append(b, make([]byte, 2+ifdLen*len(entries)+4)...)
	bo.PutUint16(b[start:], uint16(len(entries)))
	for i, e := range entries {
		p := start + 2 + i*ifdLen
		bo.PutUint16(b[p+0:], e.tag)
		bo.PutUint16(b[p+2:], e.typ)
		bo.PutUint32(b[p+4:], e.count)
		if datalen := int(e.count) * int(lengths[e.typ]); datalen <= 4 {
			copy(b[p+8:p+12], e.data)
		} else {
			bo.PutUint32(b[p+8:], uint32(len(b)))
			b = append(b, e.data...)
		}
	}
	return b, start + 2 + ifdLen*len(entries)
}

// newPageTIFFHeader returns an 8 byte TIFF header. The package's newTIFF
// helper writes a header that is only correct for big-endian files, so page
// tests need their own.
func newPageTIFFHeader(bo byteOrder) []byte {
	b := make([]byte, 8)
	if bo.Uint16([]byte{1, 0}) == 0x1 {
		copy(b, leHeader)
	} else {
		copy(b, beHeader)
	}
	return b
}

// buildMultiPageTIFF builds a classic TIFF whose main IFD chain has one IFD
// per page. Every page is 8 bit grayscale with sample values derived from its
// seed, so that pages are distinguishable.
func buildMultiPageTIFF(t *testing.T, bo byteOrder, specs []pageSpec) []byte {
	t.Helper()

	b := newPageTIFFHeader(bo)
	type layout struct {
		spec    pageSpec
		offsets []uint32
		counts  []uint32
	}
	layouts := make([]layout, len(specs))
	for i, spec := range specs {
		for _, raw := range pageBlocks(spec) {
			blk := compressBlock(t, spec.compression, raw)
			layouts[i].offsets = append(layouts[i].offsets, uint32(len(b)))
			layouts[i].counts = append(layouts[i].counts, uint32(len(blk)))
			b = append(b, blk...)
		}
		layouts[i].spec = spec
	}

	var ifdOffsets, nextFields []int
	for _, l := range layouts {
		entries := []tiffEntry{
			longEntry(bo, tImageWidth, uint32(l.spec.width)),
			longEntry(bo, tImageLength, uint32(l.spec.height)),
			shortEntry(bo, tBitsPerSample, 8),
			shortEntry(bo, tCompression, l.spec.compression),
			shortEntry(bo, tPhotometricInterpretation, pBlackIsZero),
			shortEntry(bo, tSamplesPerPixel, 1),
		}
		if l.spec.tileWidth > 0 {
			entries = append(entries,
				longEntry(bo, tTileWidth, uint32(l.spec.tileWidth)),
				longEntry(bo, tTileLength, uint32(l.spec.tileHeight)),
				longsEntry(bo, tTileOffsets, l.offsets),
				longsEntry(bo, tTileByteCounts, l.counts),
			)
		} else {
			entries = append(entries,
				longsEntry(bo, tStripOffsets, l.offsets),
				longEntry(bo, tRowsPerStrip, uint32(l.spec.height)),
				longsEntry(bo, tStripByteCounts, l.counts),
			)
			if l.spec.predictor {
				entries = append(entries, shortEntry(bo, tPredictor, prHorizontal))
			}
		}
		ifdOffsets = append(ifdOffsets, len(b))
		var nextField int
		b, nextField = appendIFDEntries(b, bo, entries)
		nextFields = append(nextFields, nextField)
	}

	for i := range ifdOffsets {
		var next uint32
		if i+1 < len(ifdOffsets) {
			next = uint32(ifdOffsets[i+1])
		}
		bo.PutUint32(b[nextFields[i]:], next)
	}
	bo.PutUint32(b[4:8], uint32(ifdOffsets[0]))
	return b
}

// nextIFDField returns the offset of the next-IFD field of the IFD at
// ifdOffset.
func nextIFDField(bo binary.ByteOrder, b []byte, ifdOffset int) int {
	n := int(bo.Uint16(b[ifdOffset:]))
	return ifdOffset + 2 + ifdLen*n
}

var multiPageSpecs = []pageSpec{
	// Page 0: uncompressed strips with a horizontal predictor. The predictor
	// is applied in place while decoding, so this page catches decoders that
	// modify (or alias) data shared between page decodes.
	{width: 40, height: 30, compression: cNone, predictor: true, seed: 1},
	// Page 1: Deflate compressed strips, a different size.
	{width: 17, height: 23, compression: cDeflate, seed: 2},
	// Page 2: PackBits compressed tiles, again a different size, with tiles
	// clipped at the right and bottom edges.
	{width: 50, height: 9, compression: cPackBits, tileWidth: 16, tileHeight: 16, seed: 3},
}

func TestReaderPages(t *testing.T) {
	byteOrders := []struct {
		name string
		bo   byteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	}
	inputs := []struct {
		name string
		newR func([]byte) io.Reader
	}{
		{"ReaderAt", func(b []byte) io.Reader { return bytes.NewReader(b) }},
		// Hide io.ReaderAt so that the Reader buffers its input.
		{"buffered", func(b []byte) io.Reader { return ioReader{bytes.NewReader(b)} }},
	}

	for _, endian := range byteOrders {
		for _, input := range inputs {
			t.Run(endian.name+"/"+input.name, func(t *testing.T) {
				data := buildMultiPageTIFF(t, endian.bo, multiPageSpecs)
				original := bytes.Clone(data)

				r, err := NewReader(input.newR(data))
				if err != nil {
					t.Fatalf("NewReader: %v", err)
				}
				if got, want := r.NumPages(), len(multiPageSpecs); got != want {
					t.Fatalf("NumPages: got %d, want %d", got, want)
				}

				for i, spec := range multiPageSpecs {
					cfg, err := r.PageConfig(i)
					if err != nil {
						t.Fatalf("PageConfig(%d): %v", i, err)
					}
					if cfg.Width != spec.width || cfg.Height != spec.height {
						t.Errorf("PageConfig(%d): got %dx%d, want %dx%d", i, cfg.Width, cfg.Height, spec.width, spec.height)
					}
					if cfg.ColorModel != color.GrayModel {
						t.Errorf("PageConfig(%d): got color model %v, want GrayModel", i, cfg.ColorModel)
					}
				}

				// Read the pages out of order, and some of them twice. Every
				// read must return that page's pixels, independently of the
				// pages decoded before it.
				for _, page := range []int{2, 0, 1, 1, 2, 0, 2} {
					img, err := r.DecodePage(page)
					if err != nil {
						t.Fatalf("DecodePage(%d): %v", page, err)
					}
					compare(t, expectedPage(multiPageSpecs[page]), img)
				}

				// Decode and DecodeConfig still describe the first page.
				img, err := Decode(input.newR(data))
				if err != nil {
					t.Fatalf("Decode: %v", err)
				}
				compare(t, expectedPage(multiPageSpecs[0]), img)
				cfg, err := DecodeConfig(input.newR(data))
				if err != nil {
					t.Fatalf("DecodeConfig: %v", err)
				}
				if got, want := cfg, (image.Config{
					ColorModel: color.GrayModel,
					Width:      multiPageSpecs[0].width,
					Height:     multiPageSpecs[0].height,
				}); got != want {
					t.Errorf("DecodeConfig: got %v, want %v", got, want)
				}

				for _, page := range []int{-1, len(multiPageSpecs), len(multiPageSpecs) + 1} {
					if _, err := r.DecodePage(page); !errors.Is(err, ErrPageOutOfRange) {
						t.Errorf("DecodePage(%d): got error %v, want ErrPageOutOfRange", page, err)
					}
					if _, err := r.PageConfig(page); !errors.Is(err, ErrPageOutOfRange) {
						t.Errorf("PageConfig(%d): got error %v, want ErrPageOutOfRange", page, err)
					}
				}

				if !bytes.Equal(data, original) {
					t.Error("the input was modified")
				}
			})
		}
	}
}

// TestReaderPageErrors tests the malformed IFD chains that NewReader rejects.
func TestReaderPageErrors(t *testing.T) {
	bo := binary.LittleEndian
	makeTwoPages := func(t *testing.T) []byte {
		return buildMultiPageTIFF(t, bo, []pageSpec{
			{width: 4, height: 4, compression: cNone, seed: 1},
			{width: 3, height: 2, compression: cPackBits, seed: 2},
		})
	}

	t.Run("no IFD", func(t *testing.T) {
		data := makeTwoPages(t)
		bo.PutUint32(data[4:8], 0)
		_, err := NewReader(bytes.NewReader(data))
		if err == nil || !contains(err.Error(), "missing IFD") {
			t.Fatalf("got error %v, want missing IFD", err)
		}
	})

	t.Run("IFD offset out of range", func(t *testing.T) {
		for _, offset := range []uint32{4, 8 << 20, 0xfffffff0} {
			data := makeTwoPages(t)
			bo.PutUint32(data[4:8], offset)
			_, err := NewReader(ioReader{bytes.NewReader(data)})
			if err == nil || !contains(err.Error(), "IFD offset out of range") {
				t.Errorf("offset %#x: got error %v, want IFD offset out of range", offset, err)
			}
		}
	})

	t.Run("truncated IFD", func(t *testing.T) {
		data := makeTwoPages(t)
		first := int(bo.Uint32(data[4:8]))
		// Cut the file in the middle of the first IFD's entries.
		truncated := data[:first+2+ifdLen]
		_, err := NewReader(bytes.NewReader(truncated))
		if err != io.ErrUnexpectedEOF {
			t.Fatalf("got error %v, want io.ErrUnexpectedEOF", err)
		}
	})

	t.Run("truncated entry count", func(t *testing.T) {
		data := makeTwoPages(t)
		first := int(bo.Uint32(data[4:8]))
		_, err := NewReader(bytes.NewReader(data[:first+1]))
		if err != io.ErrUnexpectedEOF {
			t.Fatalf("got error %v, want io.ErrUnexpectedEOF", err)
		}
	})

	t.Run("cyclic IFD chain", func(t *testing.T) {
		for _, target := range []string{"first", "self"} {
			data := makeTwoPages(t)
			first := int(bo.Uint32(data[4:8]))
			second := int(bo.Uint32(data[nextIFDField(bo, data, first):]))
			if second == 0 {
				t.Fatal("test file does not have a second IFD")
			}
			next := first
			if target == "self" {
				next = second
			}
			bo.PutUint32(data[nextIFDField(bo, data, second):], uint32(next))
			_, err := NewReader(bytes.NewReader(data))
			if err == nil || !contains(err.Error(), "cyclic IFD chain") {
				t.Errorf("%s cycle: got error %v, want cyclic IFD chain", target, err)
			}
		}
	})

	t.Run("too many pages", func(t *testing.T) {
		// Each IFD here has no entries, so the chain is only 6 bytes per
		// page: the entry count, then the next IFD offset.
		const n = maxPages + 1
		data := make([]byte, 8, 8+6*n)
		copy(data, newPageTIFFHeader(bo))
		bo.PutUint32(data[4:8], 8)
		for i := 0; i < n; i++ {
			off := 8 + 6*i
			next := uint32(off + 6)
			if i == n-1 {
				next = 0
			}
			var rec [6]byte
			bo.PutUint16(rec[0:], 0)
			bo.PutUint32(rec[2:], next)
			data = append(data, rec[:]...)
		}
		_, err := NewReader(bytes.NewReader(data))
		if !errors.Is(err, ErrTooManyPages) {
			t.Fatalf("got error %v, want ErrTooManyPages", err)
		}
	})

	t.Run("bad header", func(t *testing.T) {
		if _, err := NewReader(bytes.NewReader([]byte("not a tiff"))); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
		if _, err := NewReader(bytes.NewReader(nil)); err != io.ErrUnexpectedEOF {
			t.Fatalf("empty input: got %v, want io.ErrUnexpectedEOF", err)
		}
	})
}

// TestReaderSinglePageFiles checks that the pages of the existing single-page
// test files decode exactly as Decode does, for every supported compression
// and pixel layout.
func TestReaderSinglePageFiles(t *testing.T) {
	names := []string{
		"bw-uncompressed.tiff",
		"bw-deflate.tiff",
		"bw-packbits.tiff",
		"video-001.tiff",
		"video-001-16bit.tiff",
		"video-001-gray.tiff",
		"video-001-gray-16bit.tiff",
		"video-001-paletted.tiff",
		"video-001-strip-64.tiff",
		"video-001-tile-64x64.tiff",
		"blue-purple-pink.lzwcompressed.tiff",
		"bw-gopher_ccittGroup3.tiff",
		"bw-gopher_ccittGroup4.tiff",
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(testdataDir + name)
			if err != nil {
				t.Fatal(err)
			}
			want, err := Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			wantCfg, err := DecodeConfig(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}

			// Use ioReader to exercise the buffered path too.
			r, err := NewReader(ioReader{bytes.NewReader(data)})
			if err != nil {
				t.Fatal(err)
			}
			if got := r.NumPages(); got != 1 {
				t.Fatalf("NumPages: got %d, want 1", got)
			}
			if got, err := r.PageConfig(0); err != nil {
				t.Fatalf("PageConfig(0): %v", err)
			} else if got.Width != wantCfg.Width || got.Height != wantCfg.Height ||
				!palettesEqual(got.ColorModel, wantCfg.ColorModel) {
				t.Errorf("PageConfig(0): got %v, want %v", got, wantCfg)
			}
			got, err := r.DecodePage(0)
			if err != nil {
				t.Fatalf("DecodePage(0): %v", err)
			}
			compare(t, want, got)

			// A second read of the same page must agree with the first.
			got2, err := r.DecodePage(0)
			if err != nil {
				t.Fatalf("DecodePage(0) (second read): %v", err)
			}
			compare(t, got, got2)
		})
	}
}

func contains(s, substr string) bool {
	return bytes.Contains([]byte(s), []byte(substr))
}

// palettesEqual reports whether two color models are equal, including the
// common case of a color.Palette, which is a slice and so not comparable with
// ==.
func palettesEqual(a, b color.Model) bool {
	pa, aok := a.(color.Palette)
	pb, bok := b.(color.Palette)
	if !aok || !bok {
		return a == b
	}
	if len(pa) != len(pb) {
		return false
	}
	for i := range pa {
		if pa[i] != pb[i] {
			return false
		}
	}
	return true
}
