// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bmp

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"testing"
)

// bitfieldsBMP constructs a BI_BITFIELDS BMP file. Masks for the
// BITMAPINFOHEADER layout (infoLen == 40) are written between the header and
// the pixel data; larger headers carry the masks in the header itself.
type bitfieldsBMP struct {
	width, height int32 // A negative height means top-down.
	infoLen       uint32
	bpp           uint16
	masks         []uint32
	pixels        []byte
	// pixelOffset overrides the computed pixel data offset when non-zero.
	pixelOffset uint32
}

func (f bitfieldsBMP) Bytes() []byte {
	if f.infoLen == 0 {
		f.infoLen = infoHeaderLen
	}
	const (
		fileHeaderLen = 14
		maskOffset    = fileHeaderLen + infoHeaderLen
	)
	// The masks always start at offset 54 (after the file header and the first
	// 40 bytes of DIB header). A BITMAPINFOHEADER does not include them, so
	// they extend the header area by 4*len(masks) bytes.
	size := fileHeaderLen + int(f.infoLen)
	if f.infoLen == infoHeaderLen {
		size += 4 * len(f.masks)
	}
	b := make([]byte, size)
	copy(b, "BM")
	binary.LittleEndian.PutUint32(b[14:], f.infoLen)
	binary.LittleEndian.PutUint32(b[18:], uint32(f.width))
	binary.LittleEndian.PutUint32(b[22:], uint32(f.height))
	binary.LittleEndian.PutUint16(b[26:], 1) // planes
	binary.LittleEndian.PutUint16(b[28:], f.bpp)
	binary.LittleEndian.PutUint32(b[30:], biBitFields)
	binary.LittleEndian.PutUint32(b[34:], uint32(len(f.pixels))) // image size
	for i, m := range f.masks {
		binary.LittleEndian.PutUint32(b[maskOffset+4*i:], m)
	}
	offset := size
	if f.pixelOffset != 0 {
		offset = int(f.pixelOffset)
	}
	binary.LittleEndian.PutUint32(b[10:], uint32(offset))
	b = append(b, f.pixels...)
	binary.LittleEndian.PutUint32(b[2:], uint32(len(b)))
	return b
}

// rgb565Row returns little-endian RGB565 pixels padded to a 4-byte boundary.
// The padding is filled with a pattern that would decode to a conspicuous
// color if it were mistaken for a pixel.
func rgb565Row(pixels ...uint16) []byte {
	b := make([]byte, (2*len(pixels)+3)&^3)
	for i := range b {
		b[i] = 0xff
	}
	for i, p := range pixels {
		binary.LittleEndian.PutUint16(b[2*i:], p)
	}
	return b
}

// rgb565Rows returns the rows in BMP file order: bottom-up unless topDown.
func rgb565Rows(topDown bool, rows ...[3]uint16) []byte {
	var b []byte
	for i := range rows {
		j := len(rows) - 1 - i
		if topDown {
			j = i
		}
		b = append(b, rgb565Row(rows[j][0], rows[j][1], rows[j][2])...)
	}
	return b
}

func TestDecodeBitFieldsRGB565(t *testing.T) {
	// Two 3-pixel rows. The last pixel of the second row exercises channel
	// scaling: 1/2/1 in 5/6/5 bits becomes 8/8/8 in 8 bits.
	top := [3]uint16{0xf800, 0x07e0, 0x001f}
	bottom := [3]uint16{0xffff, 0x0000, 0x0841}
	want := []color.RGBA{
		{0xff, 0x00, 0x00, 0xff}, // top row
		{0x00, 0xff, 0x00, 0xff},
		{0x00, 0x00, 0xff, 0xff},
		{0xff, 0xff, 0xff, 0xff}, // bottom row
		{0x00, 0x00, 0x00, 0xff},
		{0x08, 0x08, 0x08, 0xff},
	}

	masks := []uint32{0xf800, 0x07e0, 0x001f}
	for _, tc := range []struct {
		name    string
		infoLen uint32
		topDown bool
		masks   []uint32
	}{{
		name:    "BITMAPINFOHEADER bottom-up",
		infoLen: infoHeaderLen,
		masks:   masks,
	}, {
		name:    "BITMAPINFOHEADER top-down",
		infoLen: infoHeaderLen,
		topDown: true,
		masks:   masks,
	}, {
		name:    "BITMAPV2INFOHEADER",
		infoLen: v2InfoHeaderLen,
		masks:   masks,
	}, {
		name:    "BITMAPV4HEADER",
		infoLen: v4InfoHeaderLen,
		masks:   masks,
	}, {
		name:    "BITMAPV5HEADER",
		infoLen: v5InfoHeaderLen,
		masks:   masks,
	}} {
		height := int32(2)
		if tc.topDown {
			height = -height
		}
		file := bitfieldsBMP{
			width:   3,
			height:  height,
			infoLen: tc.infoLen,
			bpp:     16,
			masks:   tc.masks,
			pixels:  rgb565Rows(tc.topDown, top, bottom),
		}.Bytes()

		img, err := Decode(bytes.NewReader(file))
		if err != nil {
			t.Errorf("%s: Decode: %v", tc.name, err)
			continue
		}
		rgba, ok := img.(*image.RGBA)
		if !ok {
			t.Errorf("%s: got %T, want *image.RGBA", tc.name, img)
			continue
		}
		if got := rgba.Bounds(); got != image.Rect(0, 0, 3, 2) {
			t.Errorf("%s: bounds %v, want (0,0)-(3,2)", tc.name, got)
			continue
		}
		for i, want := range want {
			x, y := i%3, i/3
			if got := rgba.RGBAAt(x, y); got != want {
				t.Errorf("%s: pixel (%d,%d) = %v, want %v", tc.name, x, y, got, want)
			}
		}

		cfg, err := DecodeConfig(bytes.NewReader(file))
		if err != nil {
			t.Errorf("%s: DecodeConfig: %v", tc.name, err)
			continue
		}
		if cfg.Width != 3 || cfg.Height != 2 || cfg.ColorModel != color.RGBAModel {
			t.Errorf("%s: config = %v", tc.name, cfg)
		}
	}
}

func TestDecodeBitFieldsAlpha(t *testing.T) {
	// Masks are ARGB (alpha in the high byte) and the pixel bytes are
	// little-endian, so the first pixel is stored as 40 80 FF 80.
	pixels := []byte{
		0x40, 0x80, 0xff, 0x80,
		0x00, 0x00, 0x00, 0x00,
	}
	want := []color.NRGBA{
		{0xff, 0x80, 0x40, 0x80},
		{0x00, 0x00, 0x00, 0x00},
	}
	masks := []uint32{0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000}
	for _, tc := range []struct {
		name    string
		infoLen uint32
	}{{
		name:    "BITMAPV3INFOHEADER",
		infoLen: v3InfoHeaderLen,
	}, {
		name:    "BITMAPV4HEADER",
		infoLen: v4InfoHeaderLen,
	}, {
		name:    "BITMAPV5HEADER",
		infoLen: v5InfoHeaderLen,
	}} {
		file := bitfieldsBMP{
			width:   2,
			height:  1,
			infoLen: tc.infoLen,
			bpp:     32,
			masks:   masks,
			pixels:  pixels,
		}.Bytes()
		img, err := Decode(bytes.NewReader(file))
		if err != nil {
			t.Errorf("%s: Decode: %v", tc.name, err)
			continue
		}
		nrgba, ok := img.(*image.NRGBA)
		if !ok {
			t.Errorf("%s: got %T, want *image.NRGBA", tc.name, img)
			continue
		}
		for x, want := range want {
			if got := nrgba.NRGBAAt(x, 0); got != want {
				t.Errorf("%s: pixel (%d,0) = %v, want %v", tc.name, x, got, want)
			}
		}
		cfg, err := DecodeConfig(bytes.NewReader(file))
		if err != nil {
			t.Errorf("%s: DecodeConfig: %v", tc.name, err)
			continue
		}
		if cfg.ColorModel != color.NRGBAModel {
			t.Errorf("%s: ColorModel = %v, want NRGBAModel", tc.name, cfg.ColorModel)
		}
	}
}

func TestDecodeBitFieldsAlpha1555(t *testing.T) {
	// ARGB1555 (A1R5G5B5): 0xFC00 is opaque red, 0x001F is transparent blue
	// and 0x0000 is transparent black.
	file := bitfieldsBMP{
		width:   3,
		height:  1,
		infoLen: v4InfoHeaderLen,
		bpp:     16,
		masks:   []uint32{0x7c00, 0x03e0, 0x001f, 0x8000},
		pixels:  rgb565Row(0xfc00, 0x001f, 0x0000),
	}.Bytes()
	img, err := Decode(bytes.NewReader(file))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	nrgba, ok := img.(*image.NRGBA)
	if !ok {
		t.Fatalf("got %T, want *image.NRGBA", img)
	}
	for x, want := range []color.NRGBA{
		{0xff, 0x00, 0x00, 0xff},
		{0x00, 0x00, 0xff, 0x00},
		{0x00, 0x00, 0x00, 0x00},
	} {
		if got := nrgba.NRGBAAt(x, 0); got != want {
			t.Errorf("pixel (%d,0) = %v, want %v", x, got, want)
		}
	}
}

// TestDecodeBitFieldsAlphaAfterBitmapInfoHeader checks the Windows CE
// ALPHABITFIELDS layout: a BITMAPINFOHEADER followed by four masks. The fourth
// mask is honoured because it is written explicitly.
func TestDecodeBitFieldsAlphaAfterBitmapInfoHeader(t *testing.T) {
	file := bitfieldsBMP{
		width:   2,
		height:  1,
		infoLen: infoHeaderLen,
		bpp:     32,
		masks:   []uint32{0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000},
		pixels: []byte{
			0x40, 0x80, 0xff, 0x80,
			0x00, 0x00, 0x00, 0x00,
		},
	}.Bytes()
	img, err := Decode(bytes.NewReader(file))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	nrgba, ok := img.(*image.NRGBA)
	if !ok {
		t.Fatalf("got %T, want *image.NRGBA", img)
	}
	for x, want := range []color.NRGBA{
		{0xff, 0x80, 0x40, 0x80},
		{0x00, 0x00, 0x00, 0x00},
	} {
		if got := nrgba.NRGBAAt(x, 0); got != want {
			t.Errorf("pixel (%d,0) = %v, want %v", x, got, want)
		}
	}
}

func TestDecodeBitFields32BitNoAlpha(t *testing.T) {
	// Masks need not be byte aligned: 2, 12 and 10 bits for red, green and
	// blue leave many zero bits. The most significant 8 bits of each channel
	// are used, as Chromium's BMP decoder does.
	file := bitfieldsBMP{
		width:   1,
		height:  1,
		infoLen: v4InfoHeaderLen,
		bpp:     32,
		masks:   []uint32{0xc0000000, 0x3ff00000, 0x000ffc00},
		// red = 0b10, green = all 1s, blue = all 1s.
		pixels: []byte{0x00, 0xfc, 0xff, 0xbf},
	}.Bytes()
	img, err := Decode(bytes.NewReader(file))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	rgba, ok := img.(*image.RGBA)
	if !ok {
		t.Fatalf("got %T, want *image.RGBA", img)
	}
	if got, want := rgba.RGBAAt(0, 0), (color.RGBA{0xaa, 0xff, 0xff, 0xff}); got != want {
		t.Errorf("pixel (0,0) = %v, want %v", got, want)
	}
}

func TestDecodeBitFieldsErrors(t *testing.T) {
	validPixels := []byte{0, 0, 0, 0}
	for _, tc := range []struct {
		name    string
		b       []byte
		wantErr error
	}{{
		name: "overlapping masks",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0x7c00, 0x7c00, 0x001f}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "zero red mask",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0x0000, 0x07e0, 0x001f}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "zero green mask",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0xf800, 0x0000, 0x001f}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "zero blue mask",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0xf800, 0x07e0, 0x0000}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "mask wider than 16-bit pixel",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0x1f0000, 0x07e0, 0x001f}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "non-contiguous mask",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0xf820, 0x07c0, 0x001f}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "overlapping alpha mask",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 32, infoLen: v4InfoHeaderLen,
			masks:  []uint32{0x00ff0000, 0x0000ff00, 0x000000ff, 0x800000ff},
			pixels: validPixels,
		}.Bytes(),
		wantErr: errInvalidBitFields,
	}, {
		name: "unsupported bit depth",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 24,
			masks: []uint32{0xff0000, 0xff00, 0xff}, pixels: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "pixel offset before masks",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0xf800, 0x07e0, 0x001f}, pixels: []byte{0, 0, 0, 0},
			pixelOffset: fileHeaderLen + infoHeaderLen + 8,
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "pixel offset after masks",
		b: bitfieldsBMP{
			width: 1, height: 1, bpp: 16,
			masks: []uint32{0xf800, 0x07e0, 0x001f}, pixels: []byte{0, 0, 0, 0},
			pixelOffset: fileHeaderLen + infoHeaderLen + 20,
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "Nx0 dimensions",
		b: bitfieldsBMP{
			width: 1, height: 0, bpp: 16,
			masks: []uint32{0xf800, 0x07e0, 0x001f},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "excessive dimensions",
		b: bitfieldsBMP{
			width: 2147483647, height: 2147483647, bpp: 16,
			masks: []uint32{0xf800, 0x07e0, 0x001f},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}} {
		if _, err := Decode(bytes.NewReader(tc.b)); err != tc.wantErr {
			t.Errorf("%s: Decode error %v; want %v", tc.name, err, tc.wantErr)
		}
		if _, err := DecodeConfig(bytes.NewReader(tc.b)); err != tc.wantErr {
			t.Errorf("%s: DecodeConfig error %v; want %v", tc.name, err, tc.wantErr)
		}
	}
}

func TestDecodeBitFieldsTruncated(t *testing.T) {
	// Masks follow a BITMAPINFOHEADER, so a file truncated in the middle of
	// them must fail without allocating an image.
	file := bitfieldsBMP{
		width: 1, height: 1, bpp: 16,
		masks:  []uint32{0xf800, 0x07e0, 0x001f},
		pixels: []byte{0, 0, 0, 0},
	}.Bytes()
	file = file[:fileHeaderLen+infoHeaderLen+6]
	if _, err := Decode(bytes.NewReader(file)); err != io.ErrUnexpectedEOF {
		t.Errorf("truncated masks: Decode error %v; want %v", err, io.ErrUnexpectedEOF)
	}

	// A truncated BITMAPV4HEADER must be reported before any size checks are
	// made against header fields that were never read.
	file = bitfieldsBMP{
		width: 1, height: 1, bpp: 32, infoLen: v4InfoHeaderLen,
		masks:  []uint32{0xff0000, 0xff00, 0xff, 0xff000000},
		pixels: []byte{0, 0, 0, 0},
	}.Bytes()
	file = file[:fileHeaderLen+infoHeaderLen]
	if _, err := Decode(bytes.NewReader(file)); err != io.ErrUnexpectedEOF {
		t.Errorf("truncated header: Decode error %v; want %v", err, io.ErrUnexpectedEOF)
	}
}
