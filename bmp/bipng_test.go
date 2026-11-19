// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bmp

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// makeIconPNG returns the PNG encoding of a w*h NRGBA image whose pixels have
// distinct, deterministic colors, including fully and partially transparent
// ones.
func makeIconPNG(t *testing.T, w, h int) ([]byte, *image.NRGBA) {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{
				R: uint8(17*x + 5),
				G: uint8(29*y + 7),
				B: uint8(11*(x+y) + 3),
				A: uint8((x + y*w) * 255 / (w*h - 1)),
			})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return buf.Bytes(), img
}

// wrapPNG hand-crafts a BMP file (BITMAPINFOHEADER, BI_PNG compression) whose
// pixel data is the given PNG payload.
func wrapPNG(width, height int32, imageSize uint32, payload []byte) []byte {
	return bmpBuilder{
		width:        width,
		height:       height,
		planes:       1,
		bitsPerPixel: 32,
		compression:  biPNG,
		imageSize:    imageSize,
		data:         payload,
	}.Bytes()
}

// TestDecodeBIPNG decodes a hand-crafted BMP file that embeds a PNG icon and
// checks every pixel, including the alpha channel.
func TestDecodeBIPNG(t *testing.T) {
	pngData, want := makeIconPNG(t, 5, 3)
	b := wrapPNG(5, 3, uint32(len(pngData)), pngData)

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(want, img); err != nil {
		t.Errorf("pixel mismatch: %v", err)
	}
	// The decoded image must preserve transparency.
	if _, ok := img.(*image.NRGBA); !ok {
		t.Errorf("decoded image has type %T, want *image.NRGBA", img)
	}
	opaque := true
	for y := 0; y < 3; y++ {
		for x := 0; x < 5; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			if a != 0xffff {
				opaque = false
			}
		}
	}
	if opaque {
		t.Error("decoded image is fully opaque, want transparency preserved")
	}

	cfg, err := DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	}
	if cfg.Width != 5 || cfg.Height != 3 {
		t.Errorf("DecodeConfig: got %dx%d, want 5x3", cfg.Width, cfg.Height)
	}
	if cfg.ColorModel != color.NRGBAModel {
		t.Errorf("DecodeConfig: ColorModel is %T, want color.NRGBAModel", cfg.ColorModel)
	}
}

// TestDecodeBIPNGTrailingData checks that extra bytes after the PNG payload
// are ignored.
func TestDecodeBIPNGTrailingData(t *testing.T) {
	pngData, want := makeIconPNG(t, 4, 2)
	payload := append(append([]byte{}, pngData...), []byte("trailing garbage not part of the PNG")...)
	b := wrapPNG(4, 2, uint32(len(payload)), payload)

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(want, img); err != nil {
		t.Errorf("pixel mismatch: %v", err)
	}
}

// TestDecodeBIPNGZeroSizeHeader checks that a BMP header declaring 0x0 adopts
// the PNG dimensions.
func TestDecodeBIPNGZeroSizeHeader(t *testing.T) {
	pngData, want := makeIconPNG(t, 3, 4)
	b := wrapPNG(0, 0, uint32(len(pngData)), pngData)

	cfg, err := DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	}
	if cfg.Width != 3 || cfg.Height != 4 {
		t.Errorf("DecodeConfig: got %dx%d, want 3x4", cfg.Width, cfg.Height)
	}
	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(want, img); err != nil {
		t.Errorf("pixel mismatch: %v", err)
	}
}

// TestDecodeBIPNGConfigDoesNotDecodePixels checks that DecodeConfig only
// needs the PNG header: a payload truncated right after the IHDR chunk still
// yields a valid Config, while Decode of the same data fails.
func TestDecodeBIPNGConfigDoesNotDecodePixels(t *testing.T) {
	pngData, _ := makeIconPNG(t, 6, 6)
	// PNG signature (8 bytes) plus one full IHDR chunk (4+4+13+4 bytes).
	const ihdrEnd = 8 + 25
	truncated := pngData[:ihdrEnd]
	b := wrapPNG(6, 6, uint32(len(truncated)), truncated)

	cfg, err := DecodeConfig(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	}
	if cfg.Width != 6 || cfg.Height != 6 {
		t.Errorf("DecodeConfig: got %dx%d, want 6x6", cfg.Width, cfg.Height)
	}
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Error("Decode of truncated payload: got nil error, want non-nil")
	}
}

// pngIHDROnly builds a minimal PNG data stream containing only the signature
// and an IHDR chunk with the given dimensions.
func pngIHDROnly(w, h uint32) []byte {
	data := make([]byte, 13)
	binary.BigEndian.PutUint32(data[0:4], w)
	binary.BigEndian.PutUint32(data[4:8], h)
	data[8] = 8 // bit depth
	data[9] = 6 // color type: RGBA
	chunk := append([]byte("IHDR"), data...)
	out := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}
	out = binary.BigEndian.AppendUint32(out, uint32(len(data)))
	out = append(out, chunk...)
	out = binary.BigEndian.AppendUint32(out, crc32.ChecksumIEEE(chunk))
	return out
}

func TestDecodeBIPNGErrors(t *testing.T) {
	pngData, _ := makeIconPNG(t, 4, 4)

	// offsetOutOfRange patches the pixel-data offset to point past the end
	// of the file.
	offsetOutOfRange := func(b []byte) []byte {
		binary.LittleEndian.PutUint32(b[10:14], uint32(len(b))+100)
		return b
	}
	// offsetIntoHeader patches the pixel-data offset to point inside the
	// BMP headers.
	offsetIntoHeader := func(b []byte) []byte {
		binary.LittleEndian.PutUint32(b[10:14], 20)
		return b
	}

	testCases := []struct {
		name    string
		b       []byte
		wantErr error // nil means any non-nil error is acceptable
	}{{
		name: "bad PNG payload",
		b:    wrapPNG(4, 4, 24, []byte("this is not a PNG file!!")),
	}, {
		name: "empty payload",
		b:    wrapPNG(4, 4, 0, nil),
	}, {
		name:    "size mismatch",
		b:       wrapPNG(9, 2, uint32(len(pngData)), pngData),
		wantErr: errPNGSizeMismatch,
	}, {
		name: "truncated payload",
		// Truncated before the end of the PNG IHDR chunk, so that even
		// DecodeConfig cannot proceed. A payload truncated after IHDR is
		// covered by TestDecodeBIPNGConfigDoesNotDecodePixels.
		b: wrapPNG(4, 4, 20, pngData[:20]),
	}, {
		name: "offset out of range",
		b:    offsetOutOfRange(wrapPNG(4, 4, uint32(len(pngData)), pngData)),
	}, {
		name: "offset overlaps headers",
		b:    offsetIntoHeader(wrapPNG(4, 4, uint32(len(pngData)), pngData)),
	}, {
		name: "oversized PNG dimensions",
		// The PNG decoder itself rejects dimensions that would overflow;
		// any non-nil error is the expected, clear result.
		b: wrapPNG(0, 0, 0, pngIHDROnly(1<<31-1, 1<<31-1)),
	}, {
		name:    "top-down not allowed",
		b:       wrapPNG(4, -4, uint32(len(pngData)), pngData),
		wantErr: ErrUnsupported,
	}}

	for _, tc := range testCases {
		_, err := Decode(bytes.NewReader(tc.b))
		if err == nil {
			t.Errorf("%s: Decode: got nil error, want non-nil", tc.name)
		} else if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
			t.Errorf("%s: Decode: got error %v, want %v", tc.name, err, tc.wantErr)
		}
		_, err = DecodeConfig(bytes.NewReader(tc.b))
		if err == nil {
			t.Errorf("%s: DecodeConfig: got nil error, want non-nil", tc.name)
		} else if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
			t.Errorf("%s: DecodeConfig: got error %v, want %v", tc.name, err, tc.wantErr)
		}
	}
}

// TestDecodeBIPNGGap checks that a gap between the BMP headers and the PNG
// payload (data offset beyond the header end) is skipped correctly.
func TestDecodeBIPNGGap(t *testing.T) {
	pngData, want := makeIconPNG(t, 2, 2)
	b := wrapPNG(2, 2, uint32(len(pngData)), pngData)
	// Insert a 4-byte gap between the BMP headers and the PNG payload, and
	// move the data offset past it.
	offset := binary.LittleEndian.Uint32(b[10:14])
	gap := []byte{0xde, 0xad, 0xbe, 0xef}
	b = append(b[:offset], append(gap, b[offset:]...)...)
	binary.LittleEndian.PutUint32(b[10:14], offset+uint32(len(gap)))
	binary.LittleEndian.PutUint32(b[2:6], uint32(len(b)))

	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(want, img); err != nil {
		t.Errorf("pixel mismatch: %v", err)
	}
}
