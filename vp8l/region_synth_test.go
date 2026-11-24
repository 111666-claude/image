// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"testing"
)

// bitWriter packs a VP8L bit-stream, least significant bit first.
type bitWriter struct {
	buf  []byte
	bits uint32
	n    uint32
}

func (w *bitWriter) write(v, n uint32) {
	w.bits |= (v & (1<<n - 1)) << w.n
	w.n += n
	for w.n >= 8 {
		w.buf = append(w.buf, byte(w.bits))
		w.bits >>= 8
		w.n -= 8
	}
}

func (w *bitWriter) bytes() []byte {
	if w.n != 0 {
		w.buf = append(w.buf, byte(w.bits))
		w.bits, w.n = 0, 0
	}
	return w.buf
}

// writeSimpleHuffman writes a one-symbol Huffman tree. The single symbol has
// a one-bit code, so decoding consumes exactly one bit per symbol.
func (w *bitWriter) writeSimpleHuffman(symbol uint32) {
	w.write(1, 1) // useSimple
	w.write(0, 1) // nSymbols = 1
	// firstSymbolLengthCode is one bit: 0 means a 1-bit symbol, 1 means an
	// 8-bit symbol. Use 8 bits so any literal, including 0xff, is
	// representable.
	w.write(1, 1)
	w.write(symbol, 8)
}

// encodeFlatVP8L builds a no-transform, no-color-cache, single-group VP8L
// stream of a uniform image.
func encodeFlatVP8L(width, height int, c color.NRGBA) []byte {
	var w bitWriter
	w.write(0x2f, 8)             // Magic.
	w.write(uint32(width-1), 14) // Width minus one.
	w.write(uint32(height-1), 14)
	w.write(0, 1) // hasAlpha hint.
	w.write(0, 3) // Version.
	w.write(0, 1) // No transforms.
	w.write(0, 1) // No color cache.
	w.write(0, 1) // Single Huffman group (no meta-image).
	// Five one-symbol trees: green, red, blue, alpha, distance.
	w.writeSimpleHuffman(uint32(c.G))
	w.writeSimpleHuffman(uint32(c.R))
	w.writeSimpleHuffman(uint32(c.B))
	w.writeSimpleHuffman(uint32(c.A))
	w.writeSimpleHuffman(0)
	// Every pixel emits the green symbol (which is < nLiteralCodes).
	for i := 0; i < width*height; i++ {
		w.write(0, 1)
	}
	return w.bytes()
}

// TestSyntheticFlatRegion checks small images of odd dimensions and various
// alpha values against full Decode.
func TestSyntheticFlatRegion(t *testing.T) {
	sizes := []struct{ w, h int }{
		{1, 1}, {3, 1}, {1, 7}, {7, 3}, {13, 11}, {31, 17}, {64, 64}, {100, 33},
	}
	colors := []color.NRGBA{
		{R: 0x11, G: 0x22, B: 0x33, A: 0xff},
		{R: 0xaa, G: 0xbb, B: 0xcc, A: 0x00}, // Fully transparent.
		{R: 0x00, G: 0x00, B: 0x00, A: 0x80}, // Semi-transparent black.
	}
	for _, sz := range sizes {
		for _, c := range colors {
			stream := encodeFlatVP8L(sz.w, sz.h, c)
			fullImg, err := Decode(bytes.NewReader(stream))
			if err != nil {
				t.Fatalf("w=%d h=%d: full decode: %v", sz.w, sz.h, err)
			}
			full := fullImg.(*image.NRGBA)
			for _, r := range []image.Rectangle{
				image.Rect(0, 0, sz.w, sz.h),
				image.Rect(0, 0, 1, 1),
				image.Rect(sz.w/2, sz.h/2, sz.w, sz.h),
				image.Rect(1, 1, 2, 2),
			} {
				r = r.Intersect(full.Rect)
				if r.Empty() {
					continue
				}
				got, err := DecodeRegion(context.Background(), bytes.NewReader(stream), r, nil)
				if err != nil {
					t.Fatalf("w=%d h=%d r=%v: %v", sz.w, sz.h, r, err)
				}
				for y := r.Min.Y; y < r.Max.Y; y++ {
					for x := r.Min.X; x < r.Max.X; x++ {
						if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
							t.Fatalf("w=%d h=%d r=%v at (%d,%d): got %v want %v",
								sz.w, sz.h, r, x, y, got.NRGBAAt(x, y), full.NRGBAAt(x, y))
						}
					}
				}
			}
		}
	}
}

// TestSyntheticRingWrap decodes a region of an image larger than the ring
// buffer, forcing the ring to wrap, and requires the pixels to match full
// Decode. The stream has no LZ77 references, so every distance check passes
// and the test isolates ring addressing.
func TestSyntheticRingWrap(t *testing.T) {
	const width, height = 1200, 1000 // 1.2M pixels, above the 1<<20 ring.
	c := color.NRGBA{R: 0x10, G: 0x40, B: 0x80, A: 0xc0}
	stream := encodeFlatVP8L(width, height, c)

	fullImg, err := Decode(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("full decode: %v", err)
	}
	full := fullImg.(*image.NRGBA)

	for _, r := range []image.Rectangle{
		image.Rect(900, 900, width, height),
		image.Rect(0, 0, width, 1),
		image.Rect(width/2, height/2, width/2+64, height/2+64),
	} {
		got, err := DecodeRegion(context.Background(), bytes.NewReader(stream), r, nil)
		if err != nil {
			t.Fatalf("%v: %v", r, err)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
					t.Fatalf("%v: mismatch at (%d,%d)", r, x, y)
				}
			}
		}
	}
}

// TestSyntheticHostileDimensions claims the largest 14-bit dimensions with no
// data and a small memory cap; the decode must fail with a bounded error.
func TestSyntheticHostileDimensions(t *testing.T) {
	const wm1, hm1 = 16383, 16383
	bits := uint64(0x2f) | uint64(wm1)<<8 | uint64(hm1)<<22
	header := []byte{
		byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24), byte(bits >> 32),
	}
	_, err := DecodeRegion(context.Background(), bytes.NewReader(header),
		image.Rect(0, 0, 8, 8), &Options{MaxMemory: 4 << 20})
	if err == nil {
		t.Fatal("hostile dimensions: got nil error")
	}
}
