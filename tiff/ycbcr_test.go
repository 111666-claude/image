// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"strings"
	"testing"
)

// Deterministic sample patterns for the hand-built YCbCr fixtures, keyed on
// absolute pixel (or chroma) coordinates so that strips and tiles agree on
// shared values.
func testY(x, y int) uint8    { return uint8(3*x + 5*y + 1) }
func testCb(cx, cy int) uint8 { return uint8(7*cx + 2*cy + 40) }
func testCr(cx, cy int) uint8 { return uint8(11*cx + 13*cy + 80) }

// ycbcrChunk returns the uncompressed data for the region
// [x0,x0+w)×[y0,y0+h) of a YCbCr image, laid out as subsampling blocks of
// hsub*vsub Y samples followed by one Cb and one Cr sample. The region is
// padded to a whole number of blocks, as required by the spec.
func ycbcrChunk(x0, y0, w, h, hsub, vsub int) []byte {
	var b []byte
	for by := 0; by < (h+vsub-1)/vsub; by++ {
		for bx := 0; bx < (w+hsub-1)/hsub; bx++ {
			for j := 0; j < vsub; j++ {
				for i := 0; i < hsub; i++ {
					b = append(b, testY(x0+bx*hsub+i, y0+by*vsub+j))
				}
			}
			b = append(b, testCb((x0+bx*hsub)/hsub, (y0+by*vsub)/vsub))
			b = append(b, testCr((x0+bx*hsub)/hsub, (y0+by*vsub)/vsub))
		}
	}
	return b
}

// buildTIFFWithChunks appends each chunk of pixel data to a fresh TIFF and
// records the offsets and byte counts under the given tags.
func buildTIFFWithChunks(enc byteOrder, entries map[uint16]interface{}, tagOffsets, tagCounts uint16, chunks ...[]byte) []byte {
	b := newTIFF(enc)
	var offs, counts []uint32
	for _, c := range chunks {
		offs = append(offs, uint32(len(b)))
		b = append(b, c...)
		counts = append(counts, uint32(len(c)))
	}
	entries[tagOffsets] = offs
	entries[tagCounts] = counts
	return appendIFD(b, enc, entries)
}

// checkYCbCr verifies that img holds exactly the pattern produced by
// ycbcrChunk for a w×h image with the given subsampling.
func checkYCbCr(t *testing.T, img image.Image, ratio image.YCbCrSubsampleRatio, w, h, hsub, vsub int) {
	t.Helper()
	yimg, ok := img.(*image.YCbCr)
	if !ok {
		t.Fatalf("got %T, want *image.YCbCr", img)
	}
	if yimg.SubsampleRatio != ratio {
		t.Fatalf("subsample ratio: got %v, want %v", yimg.SubsampleRatio, ratio)
	}
	if b := yimg.Bounds(); b.Dx() != w || b.Dy() != h {
		t.Fatalf("bounds: got %v, want %dx%d", b, w, h)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if got, want := yimg.Y[yimg.YOffset(x, y)], testY(x, y); got != want {
				t.Fatalf("Y at (%d, %d): got %d, want %d", x, y, got, want)
			}
		}
	}
	for cy := 0; cy < (h+vsub-1)/vsub; cy++ {
		for cx := 0; cx < (w+hsub-1)/hsub; cx++ {
			ci := cy*yimg.CStride + cx
			if got, want := yimg.Cb[ci], testCb(cx, cy); got != want {
				t.Fatalf("Cb at (%d, %d): got %d, want %d", cx, cy, got, want)
			}
			if got, want := yimg.Cr[ci], testCr(cx, cy); got != want {
				t.Fatalf("Cr at (%d, %d): got %d, want %d", cx, cy, got, want)
			}
		}
	}
}

func ycbcrEntries(w, h int, subsampling *[2]uint16) map[uint16]interface{} {
	entries := map[uint16]interface{}{
		tImageWidth:                uint32(w),
		tImageLength:               uint32(h),
		tBitsPerSample:             []uint16{8, 8, 8},
		tCompression:               uint16(cNone),
		tPhotometricInterpretation: uint16(pYCbCr),
	}
	if subsampling != nil {
		entries[tYCbCrSubSampling] = []uint16{subsampling[0], subsampling[1]}
	}
	return entries
}

// TestDecodeYCbCr444 tests an uncompressed 4:4:4 image in a single strip.
func TestDecodeYCbCr444(t *testing.T) {
	const w, h = 5, 3
	entries := ycbcrEntries(w, h, &[2]uint16{1, 1})
	entries[tRowsPerStrip] = uint32(h)
	data := buildTIFFWithChunks(binary.LittleEndian, entries, tStripOffsets, tStripByteCounts,
		ycbcrChunk(0, 0, w, h, 1, 1))

	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Width != w || cfg.Height != h {
		t.Fatalf("config size: got %dx%d, want %dx%d", cfg.Width, cfg.Height, w, h)
	}
	if cfg.ColorModel != color.YCbCrModel {
		t.Fatalf("color model: got %v, want YCbCrModel", cfg.ColorModel)
	}

	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	checkYCbCr(t, img, image.YCbCrSubsampleRatio444, w, h, 1, 1)
}

// TestDecodeYCbCr422 tests a 4:2:2 image with odd width and two strips.
func TestDecodeYCbCr422(t *testing.T) {
	const w, h = 5, 3
	entries := ycbcrEntries(w, h, &[2]uint16{2, 1})
	entries[tRowsPerStrip] = uint32(2)
	data := buildTIFFWithChunks(binary.BigEndian, entries, tStripOffsets, tStripByteCounts,
		ycbcrChunk(0, 0, w, 2, 2, 1),
		ycbcrChunk(0, 2, w, 1, 2, 1))

	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	checkYCbCr(t, img, image.YCbCrSubsampleRatio422, w, h, 2, 1)
}

// TestDecodeYCbCr420 tests a 4:2:0 image whose height is not a multiple of
// the vertical subsampling factor, split into two strips. The subsampling
// tag is omitted to exercise the 2,2 default.
func TestDecodeYCbCr420(t *testing.T) {
	const w, h = 6, 5
	entries := ycbcrEntries(w, h, nil)
	entries[tRowsPerStrip] = uint32(4)
	data := buildTIFFWithChunks(binary.LittleEndian, entries, tStripOffsets, tStripByteCounts,
		ycbcrChunk(0, 0, w, 4, 2, 2),
		ycbcrChunk(0, 4, w, 1, 2, 2))

	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	checkYCbCr(t, img, image.YCbCrSubsampleRatio420, w, h, 2, 2)
}

// TestDecodeYCbCr420Tiled tests a 4:2:0 image stored as tiles that extend
// past the right and bottom edges of the image.
func TestDecodeYCbCr420Tiled(t *testing.T) {
	const w, h = 20, 12
	entries := ycbcrEntries(w, h, &[2]uint16{2, 2})
	entries[tTileWidth] = uint32(16)
	entries[tTileLength] = uint32(16)
	data := buildTIFFWithChunks(binary.LittleEndian, entries, tTileOffsets, tTileByteCounts,
		ycbcrChunk(0, 0, 16, 16, 2, 2),
		ycbcrChunk(16, 0, 16, 16, 2, 2))

	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	checkYCbCr(t, img, image.YCbCrSubsampleRatio420, w, h, 2, 2)
}

// TestDecodeYCbCrReferenceBlackWhite tests that explicit ReferenceBlackWhite
// values (here the limited "video" range) are used to normalize the samples.
func TestDecodeYCbCrReferenceBlackWhite(t *testing.T) {
	const w, h = 4, 1
	entries := ycbcrEntries(w, h, &[2]uint16{1, 1})
	entries[tRowsPerStrip] = uint32(h)
	entries[tReferenceBlackWhite] = [][2]uint32{
		{16, 1}, {235, 1}, // Y: 16..235
		{128, 1}, {240, 1}, // Cb: neutral 128, max 240
		{128, 1}, {240, 1}, // Cr: neutral 128, max 240
	}
	chunk := []byte{
		16, 128, 128, // limited-range black -> YCbCr(0, 128, 128)
		235, 128, 128, // limited-range white -> YCbCr(255, 128, 128)
		128, 240, 16, // -> YCbCr(130, 255, 1)
		100, 100, 200, // -> YCbCr(98, 96, 210)
	}
	data := buildTIFFWithChunks(binary.LittleEndian, entries, tStripOffsets, tStripByteCounts, chunk)

	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	yimg := img.(*image.YCbCr)
	want := []color.YCbCr{
		{Y: 0, Cb: 128, Cr: 128},
		{Y: 255, Cb: 128, Cr: 128},
		{Y: 130, Cb: 255, Cr: 1},
		{Y: 98, Cb: 96, Cr: 210},
	}
	for x, w := range want {
		if got := yimg.YCbCrAt(x, 0); got != w {
			t.Errorf("pixel (%d, 0): got %v, want %v", x, got, w)
		}
	}
	if r, g, b, _ := img.At(0, 0).RGBA(); r != 0 || g != 0 || b != 0 {
		t.Errorf("RGBA at (0, 0): got (%d, %d, %d), want black", r, g, b)
	}
	if r, g, b, _ := img.At(1, 0).RGBA(); r != 0xffff || g != 0xffff || b != 0xffff {
		t.Errorf("RGBA at (1, 0): got (%d, %d, %d), want white", r, g, b)
	}
}

// TestDecodeYCbCrErrors tests that malformed YCbCr images are rejected with
// an error.
func TestDecodeYCbCrErrors(t *testing.T) {
	base := func() map[uint16]interface{} {
		entries := ycbcrEntries(6, 5, &[2]uint16{2, 2})
		entries[tRowsPerStrip] = uint32(4)
		return entries
	}
	chunk := func() []byte { return ycbcrChunk(0, 0, 6, 4, 2, 2) }

	tests := []struct {
		name    string
		mutate  func(entries map[uint16]interface{})
		chunks  [][]byte
		wantErr string
	}{{
		name: "invalid horizontal subsampling factor",
		mutate: func(e map[uint16]interface{}) {
			e[tYCbCrSubSampling] = []uint16{3, 2}
		},
		wantErr: "invalid YCbCr subsampling",
	}, {
		name: "zero subsampling factor",
		mutate: func(e map[uint16]interface{}) {
			e[tYCbCrSubSampling] = []uint16{0, 0}
		},
		wantErr: "invalid YCbCr subsampling",
	}, {
		name: "unsupported but valid subsampling",
		mutate: func(e map[uint16]interface{}) {
			e[tYCbCrSubSampling] = []uint16{2, 4}
		},
		wantErr: "unsupported feature",
	}, {
		name: "wrong number of samples",
		mutate: func(e map[uint16]interface{}) {
			e[tBitsPerSample] = []uint16{8, 8}
		},
		wantErr: "wrong number of samples for YCbCr",
	}, {
		name: "16 bits per sample",
		mutate: func(e map[uint16]interface{}) {
			e[tBitsPerSample] = []uint16{16, 16, 16}
		},
		wantErr: "unsupported feature",
	}, {
		name:    "not enough pixel data",
		chunks:  [][]byte{chunk()[:len(chunk())/2], ycbcrChunk(0, 4, 6, 1, 2, 2)},
		wantErr: "not enough pixel data",
	}, {
		name: "strip height not aligned to subsampling",
		mutate: func(e map[uint16]interface{}) {
			e[tRowsPerStrip] = uint32(3)
		},
		chunks:  [][]byte{chunk(), chunk()},
		wantErr: "strip height is not a multiple",
	}, {
		name: "reference black not below white",
		mutate: func(e map[uint16]interface{}) {
			e[tReferenceBlackWhite] = [][2]uint32{
				{0, 1}, {255, 1},
				{200, 1}, {100, 1},
				{128, 1}, {255, 1},
			}
		},
		wantErr: "invalid ReferenceBlackWhite",
	}, {
		name: "reference black white zero denominator",
		mutate: func(e map[uint16]interface{}) {
			e[tReferenceBlackWhite] = [][2]uint32{
				{0, 1}, {255, 1},
				{128, 0}, {255, 1},
				{128, 1}, {255, 1},
			}
		},
		wantErr: "invalid ReferenceBlackWhite",
	}, {
		name: "reference black white wrong count",
		mutate: func(e map[uint16]interface{}) {
			e[tReferenceBlackWhite] = [][2]uint32{{0, 1}, {255, 1}}
		},
		wantErr: "invalid ReferenceBlackWhite",
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			entries := base()
			if tc.mutate != nil {
				tc.mutate(entries)
			}
			chunks := tc.chunks
			if chunks == nil {
				chunks = [][]byte{chunk(), ycbcrChunk(0, 4, 6, 1, 2, 2)}
			}
			data := buildTIFFWithChunks(binary.LittleEndian, entries, tStripOffsets, tStripByteCounts, chunks...)
			_, err := Decode(bytes.NewReader(data))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Decode: got %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}

// TestDecodeYCbCrStripOffsetOutOfRange tests that a strip offset pointing
// past the end of the file is reported as an error.
func TestDecodeYCbCrStripOffsetOutOfRange(t *testing.T) {
	entries := ycbcrEntries(6, 5, &[2]uint16{2, 2})
	entries[tRowsPerStrip] = uint32(5)
	entries[tStripOffsets] = []uint32{0x7fffffff}
	entries[tStripByteCounts] = []uint32{64}
	data := appendIFD(newTIFF(binary.LittleEndian), binary.LittleEndian, entries)
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("Decode: got nil error, want non-nil")
	}
}

// TestDecodeYCbCrTileNotAligned tests that tiles whose dimensions are not
// multiples of the subsampling factors are rejected.
func TestDecodeYCbCrTileNotAligned(t *testing.T) {
	entries := ycbcrEntries(16, 16, &[2]uint16{2, 2})
	entries[tTileWidth] = uint32(9)
	entries[tTileLength] = uint32(9)
	entries[tTileOffsets] = []uint32{8, 8, 8, 8}
	entries[tTileByteCounts] = []uint32{0, 0, 0, 0}
	data := appendIFD(newTIFF(binary.LittleEndian), binary.LittleEndian, entries)
	_, err := Decode(bytes.NewReader(data))
	if want := "tile dimensions are not a multiple"; err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("Decode: got %v, want error containing %q", err, want)
	}
}
