// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bmp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"testing"

	_ "image/png"
)

const testdataDir = "../testdata/"

func compare(img0, img1 image.Image) error {
	b := img1.Bounds()
	if !b.Eq(img0.Bounds()) {
		return fmt.Errorf("wrong image size: want %s, got %s", img0.Bounds(), b)
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c0 := img0.At(x, y)
			c1 := img1.At(x, y)
			r0, g0, b0, a0 := c0.RGBA()
			r1, g1, b1, a1 := c1.RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return fmt.Errorf("pixel at (%d, %d) has wrong color: want %v, got %v", x, y, c0, c1)
			}
		}
	}
	return nil
}

// TestDecode tests that decoding a PNG image and a BMP image result in the
// same pixel data.
func TestDecode(t *testing.T) {
	testCases := []string{
		"colormap",
		"colormap-0",
		"colormap-251",
		"video-001",
		"yellow_rose-small",
		"yellow_rose-small-v5",
		"bmp_1bpp",
		"bmp_4bpp",
		"bmp_8bpp",
	}

	for _, tc := range testCases {
		f0, err := os.Open(testdataDir + tc + ".png")
		if err != nil {
			t.Errorf("%s: Open PNG: %v", tc, err)
			continue
		}
		defer f0.Close()
		img0, _, err := image.Decode(f0)
		if err != nil {
			t.Errorf("%s: Decode PNG: %v", tc, err)
			continue
		}

		f1, err := os.Open(testdataDir + tc + ".bmp")
		if err != nil {
			t.Errorf("%s: Open BMP: %v", tc, err)
			continue
		}
		defer f1.Close()
		img1, _, err := image.Decode(f1)
		if err != nil {
			t.Errorf("%s: Decode BMP: %v", tc, err)
			continue
		}

		if err := compare(img0, img1); err != nil {
			t.Errorf("%s: %v", tc, err)
			continue
		}
	}
}

func TestDecodeConstructed(t *testing.T) {
	for _, tc := range []struct {
		name    string
		b       []byte
		wantErr error
	}{{
		name: "1x1 paletted",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 1,
			colorsUsed:   2,
			colorTable: []colorTableEntry{
				{0, 0, 0},
				{0xff, 0xff, 0xff},
			},
			data: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: nil, // successful base case
	}, {
		name: "1x1 rgb",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 24,
			data: []byte{
				0, 0, 0, 0,
			},
		}.Bytes(),
		wantErr: nil, // successful base case
	}, {
		name: "invalid palette index",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 8,
			colorsUsed:   2,
			colorTable: []colorTableEntry{
				{0, 0, 0},
				{0xff, 0xff, 0xff},
			},
			data: []byte{
				2, 0, 0, 0, // index 2
			},
		}.Bytes(),
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "0x1",
		b: bmpBuilder{
			width:        1,
			height:       0,
			planes:       1,
			bitsPerPixel: 24,
			data: []byte{
				0, 0, 0, 0,
			},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "excessive width and height",
		b: bmpBuilder{
			width:        2147483647,
			height:       2147483647,
			planes:       1,
			bitsPerPixel: 24,
		}.Bytes(),
		wantErr: ErrUnsupported,
	}} {
		img, _, err := image.Decode(bytes.NewReader(tc.b))
		if err != tc.wantErr {
			t.Errorf("%v: Decode error %v; want %v", tc.name, err, tc.wantErr)
		}
		if err != nil {
			continue
		}
		_ = img.At(0, 0) // try accessing a pixel
	}
}

// TestEOF tests that decoding a BMP image returns io.ErrUnexpectedEOF
// when there are no headers or data is empty
func TestEOF(t *testing.T) {
	_, err := Decode(bytes.NewReader(nil))
	if err != io.ErrUnexpectedEOF {
		t.Errorf("Error should be io.ErrUnexpectedEOF on nil but got %v", err)
	}
}

// checkPalettedPix verifies that img is an *image.Paletted whose palette
// indices match want, row by row from the top.
func checkPalettedPix(t *testing.T, img image.Image, want [][]uint8) {
	t.Helper()
	p, ok := img.(*image.Paletted)
	if !ok {
		t.Fatalf("decoded image is %T, want *image.Paletted", img)
	}
	if got := p.Bounds().Dx(); got != len(want[0]) {
		t.Fatalf("width: got %d, want %d", got, len(want[0]))
	}
	if got := p.Bounds().Dy(); got != len(want) {
		t.Fatalf("height: got %d, want %d", got, len(want))
	}
	for y, row := range want {
		for x, w := range row {
			if got := p.ColorIndexAt(x, y); got != w {
				t.Fatalf("pixel at (%d, %d): got palette index %d, want %d", x, y, got, w)
			}
		}
	}
}

var rleTestPalette = []colorTableEntry{
	{0x00, 0x00, 0x00},
	{0x11, 0x11, 0x11},
	{0x22, 0x22, 0x22},
	{0x33, 0x33, 0x33},
	{0x44, 0x44, 0x44},
	{0x55, 0x55, 0x55},
	{0x66, 0x66, 0x66},
	{0x77, 0x77, 0x77},
}

// TestDecodeRLE8 decodes a hand-built 6x3 BI_RLE8 image that exercises
// encoded runs, absolute mode, end-of-line and end-of-bitmap commands.
func TestDecodeRLE8(t *testing.T) {
	b := bmpBuilder{
		width:        6,
		height:       3,
		planes:       1,
		bitsPerPixel: 8,
		compression:  1, // BI_RLE8
		colorsUsed:   4,
		colorTable:   rleTestPalette[:4],
		data: []byte{
			// Bottom row: two encoded runs.
			0x03, 0x02, // 3 pixels of color 2
			0x03, 0x00, // 3 pixels of color 0
			0x00, 0x00, // end of line
			// Middle row: absolute mode, 6 literal pixels (even, no padding).
			0x00, 0x06, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01,
			0x00, 0x00, // end of line
			// Top row: three encoded runs.
			0x02, 0x01, // 2 pixels of color 1
			0x02, 0x02, // 2 pixels of color 2
			0x02, 0x03, // 2 pixels of color 3
			0x00, 0x01, // end of bitmap
		},
	}.Bytes()

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	checkPalettedPix(t, img, [][]uint8{
		{1, 1, 2, 2, 3, 3},
		{0, 1, 0, 1, 0, 1},
		{2, 2, 2, 0, 0, 0},
	})
}

// TestDecodeRLE4 decodes a hand-built 5x3 BI_RLE4 image that exercises
// encoded runs with alternating nibbles, absolute mode with an odd pixel
// count (and hence a padding byte), end-of-line and end-of-bitmap commands.
func TestDecodeRLE4(t *testing.T) {
	b := bmpBuilder{
		width:        5,
		height:       3,
		planes:       1,
		bitsPerPixel: 4,
		compression:  2, // BI_RLE4
		colorsUsed:   4,
		colorTable:   rleTestPalette[:4],
		data: []byte{
			// Bottom row: absolute mode, 5 pixels in 3 bytes plus 1 pad byte.
			0x00, 0x05, 0x00, 0x11, 0x20, 0x00,
			0x00, 0x00, // end of line
			// Middle row: encoded run, alternating nibbles 3 and 3.
			0x05, 0x33,
			0x00, 0x00, // end of line
			// Top row: encoded run, alternating nibbles 1 and 2.
			0x05, 0x12,
			0x00, 0x01, // end of bitmap
		},
	}.Bytes()

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	checkPalettedPix(t, img, [][]uint8{
		{1, 2, 1, 2, 1},
		{3, 3, 3, 3, 3},
		{0, 0, 1, 1, 2},
	})
}

// TestDecodeRLEDelta decodes a hand-built 4x4 BI_RLE8 image whose only
// pixels are written after a delta command.
func TestDecodeRLEDelta(t *testing.T) {
	b := bmpBuilder{
		width:        4,
		height:       4,
		planes:       1,
		bitsPerPixel: 8,
		compression:  1, // BI_RLE8
		colorsUsed:   8,
		colorTable:   rleTestPalette,
		data: []byte{
			0x00, 0x02, 0x01, 0x02, // delta: right 1, up 2
			0x02, 0x07, // 2 pixels of color 7
			0x00, 0x01, // end of bitmap
		},
	}.Bytes()

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	checkPalettedPix(t, img, [][]uint8{
		{0, 0, 0, 0},
		{0, 7, 7, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	})
}

// TestDecodeRLEConfig checks that DecodeConfig works for RLE-compressed
// images.
func TestDecodeRLEConfig(t *testing.T) {
	b := bmpBuilder{
		width:        6,
		height:       3,
		planes:       1,
		bitsPerPixel: 8,
		compression:  1, // BI_RLE8
		colorsUsed:   4,
		colorTable:   rleTestPalette[:4],
		data:         []byte{0x00, 0x01},
	}.Bytes()

	c, err := DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	}
	if c.Width != 6 || c.Height != 3 {
		t.Errorf("dimensions: got %dx%d, want 6x3", c.Width, c.Height)
	}
	if _, ok := c.ColorModel.(color.Palette); !ok {
		t.Errorf("color model: got %T, want color.Palette", c.ColorModel)
	}
}

// TestDecodeRLEErrors checks that malformed RLE data is rejected.
func TestDecodeRLEErrors(t *testing.T) {
	rle8 := func(data []byte) []byte {
		return bmpBuilder{
			width:        4,
			height:       2,
			planes:       1,
			bitsPerPixel: 8,
			compression:  1,
			colorsUsed:   2,
			colorTable:   rleTestPalette[:2],
			data:         data,
		}.Bytes()
	}
	for _, tc := range []struct {
		name    string
		b       []byte
		wantErr error
	}{{
		name:    "run overflows row",
		b:       rle8([]byte{0x05, 0x01, 0x00, 0x01}),
		wantErr: errInvalidRLE,
	}, {
		name:    "run crosses row boundary",
		b:       rle8([]byte{0x03, 0x01, 0x03, 0x01, 0x00, 0x01}),
		wantErr: errInvalidRLE,
	}, {
		name:    "delta overflows row",
		b:       rle8([]byte{0x00, 0x02, 0x05, 0x00, 0x00, 0x01}),
		wantErr: errInvalidRLE,
	}, {
		name:    "delta overflows image",
		b:       rle8([]byte{0x00, 0x02, 0x00, 0x05, 0x00, 0x01}),
		wantErr: errInvalidRLE,
	}, {
		name:    "absolute mode overflows row",
		b:       rle8([]byte{0x00, 0x05, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01}),
		wantErr: errInvalidRLE,
	}, {
		name:    "missing end of bitmap",
		b:       rle8([]byte{0x04, 0x01, 0x00, 0x00, 0x04, 0x01}),
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name:    "truncated encoded run",
		b:       rle8([]byte{0x04}),
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name:    "truncated absolute mode",
		b:       rle8([]byte{0x00, 0x04, 0x00, 0x01}),
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name:    "truncated delta",
		b:       rle8([]byte{0x00, 0x02, 0x01}),
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name:    "invalid palette index in run",
		b:       rle8([]byte{0x02, 0x05, 0x00, 0x01}),
		wantErr: errInvalidPaletteIndex,
	}, {
		name:    "invalid palette index in absolute mode",
		b:       rle8([]byte{0x00, 0x03, 0x00, 0x05, 0x01, 0x00, 0x00, 0x01}),
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "invalid palette index in RLE4 run",
		b: bmpBuilder{
			width:        4,
			height:       2,
			planes:       1,
			bitsPerPixel: 4,
			compression:  2,
			colorsUsed:   2,
			colorTable:   rleTestPalette[:2],
			data:         []byte{0x02, 0x12, 0x00, 0x01},
		}.Bytes(),
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "RLE8 compression with 4bpp",
		b: bmpBuilder{
			width:        4,
			height:       2,
			planes:       1,
			bitsPerPixel: 4,
			compression:  1,
			colorsUsed:   2,
			colorTable:   rleTestPalette[:2],
			data:         []byte{0x00, 0x01},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "RLE4 compression with 8bpp",
		b: bmpBuilder{
			width:        4,
			height:       2,
			planes:       1,
			bitsPerPixel: 8,
			compression:  2,
			colorsUsed:   2,
			colorTable:   rleTestPalette[:2],
			data:         []byte{0x00, 0x01},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}} {
		_, err := Decode(bytes.NewReader(tc.b))
		if err != tc.wantErr {
			t.Errorf("%v: Decode error %v; want %v", tc.name, err, tc.wantErr)
		}
	}
}

type bmpBuilder struct {
	width           int32
	height          int32
	planes          uint16
	bitsPerPixel    uint16
	compression     uint32
	imageSize       uint32
	xppm            uint32
	yppm            uint32
	colorsUsed      uint32
	colorsImportant uint32
	colorTable      []colorTableEntry
	data            []byte
}

type colorTableEntry struct {
	r, g, b byte
}

func (b bmpBuilder) Bytes() []byte {
	buf := []byte{
		0x42, 0x4d, // 'BM'
		0x00, 0x00, 0x00, 0x00, // file size
		0x00, 0x00, 0x00, 0x00, // reserved
		0x00, 0x00, 0x00, 0x00, // data offset
		0x28, 0x00, 0x00, 0x00, // header size (40)
	}
	buf = binary.LittleEndian.AppendUint32(buf, uint32(b.width))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(b.height))
	buf = binary.LittleEndian.AppendUint16(buf, b.planes)
	buf = binary.LittleEndian.AppendUint16(buf, b.bitsPerPixel)
	buf = binary.LittleEndian.AppendUint32(buf, b.compression)
	buf = binary.LittleEndian.AppendUint32(buf, b.imageSize)
	buf = binary.LittleEndian.AppendUint32(buf, b.xppm)
	buf = binary.LittleEndian.AppendUint32(buf, b.yppm)
	buf = binary.LittleEndian.AppendUint32(buf, b.colorsUsed)
	buf = binary.LittleEndian.AppendUint32(buf, b.colorsImportant)
	for _, e := range b.colorTable {
		buf = append(buf, e.r, e.g, e.b, 0)
	}
	binary.LittleEndian.PutUint32(buf[10:], uint32(len(buf))) // data offset
	buf = append(buf, b.data...)
	binary.LittleEndian.PutUint32(buf[2:], uint32(len(buf))) // file size
	return buf
}
