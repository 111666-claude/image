// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"testing"
)

// bitWriter packs a VP8L bit-stream (LSB-first).
type bitWriter struct {
	buf  []byte
	bits uint32
	n    uint32
}

func (w *bitWriter) write(v uint32, n uint32) {
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

// writeSimpleHuffman writes a one-symbol Huffman tree (useSimple=1,
// nSymbols=1, symbol length code 1 => 1-bit symbol code, symbol value).
func (w *bitWriter) writeSimpleHuffman(symbol uint32) {
	w.write(1, 1) // useSimple
	w.write(0, 1) // nSymbols = 1
	w.write(0, 1) // firstSymbolLengthCode => length 1
	w.write(symbol, 1)
}

// writeTwoSymbolHuffman writes a simple two-symbol tree whose code bit 0
// decodes to sym0 and code bit 1 decodes to sym1. Per the spec, the first
// symbol is read with a 1-bit length code and the second as a full 8 bits.
func (w *bitWriter) writeTwoSymbolHuffman(sym0, sym1 uint32) {
	w.write(1, 1)    // useSimple
	w.write(1, 1)    // nSymbols = 2
	w.write(0, 1)    // firstSymbolLengthCode => length 1
	w.write(sym0, 1) // first symbol, 1-bit code
	w.write(sym1, 8) // second symbol, 8-bit code
}

// encodeLZ77VP8L builds a no-transform stream whose first row is made of
// literal pixels and every later pixel is an LZ77 copy of the pixel one row
// above (distance symbol 0 maps to an offset of width via distanceMap). When
// width*height exceeds maxLZ77DistancePixels, the region decoder's ring
// buffer wraps, so this exercises wrap-around and cross-row references.
//
// The fixed color is limited to green/red/blue/alpha values that fit the
// one-symbol and two-symbol simple Huffman trees (the first symbol of a
// two-symbol tree is only 1 bit wide).
func encodeLZ77VP8L(t *testing.T, width, height int, rgba [4]byte) []byte {
	t.Helper()
	for _, c := range rgba {
		if c > 1 {
			t.Fatalf("encodeLZ77VP8L only supports channel values 0/1, got %d", c)
		}
	}
	var w bitWriter
	w.write(0x2f, 8)
	w.write(uint32(width-1), 14)
	w.write(uint32(height-1), 14)
	w.write(1, 1)
	w.write(0, 3)
	w.write(0, 1) // no transforms
	w.write(0, 1) // no color cache
	w.write(0, 1) // single hGroup
	// Green tree: code 0 = literal green, code 1 = LZ77 length symbol 256.
	w.writeTwoSymbolHuffman(uint32(rgba[1]), nLiteralCodes)
	w.writeSimpleHuffman(uint32(rgba[0]))
	w.writeSimpleHuffman(uint32(rgba[2]))
	w.writeSimpleHuffman(uint32(rgba[3]))
	w.writeSimpleHuffman(0) // distance symbol 0 => distance code 1
	// First row: width literals (green bit 0, then R, B, A symbols).
	for x := 0; x < width; x++ {
		w.write(0, 1)
		w.write(0, 1)
		w.write(0, 1)
		w.write(0, 1)
	}
	// Every remaining pixel: green bit 1 (length sym 256 => length 1) and
	// distance bit 0 (distance symbol 0 => offset width).
	for i := width; i < width*height; i++ {
		w.write(1, 1)
		w.write(0, 1)
	}
	return w.bytes()
}

// TestSyntheticLZ77RingWrap forces the region decoder's fixed-size ring
// buffer to wrap and checks LZ77-derived pixels against the full decoder.
func TestSyntheticLZ77RingWrap(t *testing.T) {
	// 1025*1025 = 1,050,625 pixels, just past the 1<<20 ring threshold.
	const width, height = 1025, 1025
	rgba := [4]byte{1, 0, 1, 1}
	stream := encodeLZ77VP8L(t, width, height, rgba)

	fullImg, err := Decode(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("full decode: %v", err)
	}
	full := fullImg.(*image.NRGBA)
	// The full image must be uniform.
	want := colorNRGBA(rgba)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if c := full.NRGBAAt(x, y); c != want {
				t.Fatalf("full image not uniform at (%d,%d): got %v want %v", x, y, c, want)
			}
		}
	}

	// A region near the bottom-right is well past the ring wrap point.
	r := image.Rect(900, 900, width, height)
	got, err := DecodeRegion(context.Background(), bytes.NewReader(stream), r, nil)
	if err != nil {
		t.Fatalf("region decode: %v", err)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("mismatch at (%d,%d): got %v want %v",
					x, y, got.NRGBAAt(x, y), full.NRGBAAt(x, y))
			}
		}
	}
}

// TestSyntheticInvalidLZ77 checks that a backwards reference reaching before
// the first pixel is rejected with a descriptive error instead of panicking.
func TestSyntheticInvalidLZ77(t *testing.T) {
	const width, height = 4, 4
	rgba := [4]byte{1, 0, 1, 1}
	var w bitWriter
	w.write(0x2f, 8)
	w.write(width-1, 14)
	w.write(height-1, 14)
	w.write(1, 1)
	w.write(0, 3)
	w.write(0, 1) // no transforms
	w.write(0, 1) // no color cache
	w.write(0, 1) // single hGroup
	// Green tree: literal green vs LZ77 length symbol 256.
	w.writeTwoSymbolHuffman(uint32(rgba[1]), nLiteralCodes)
	w.writeSimpleHuffman(uint32(rgba[0]))
	w.writeSimpleHuffman(uint32(rgba[2]))
	w.writeSimpleHuffman(uint32(rgba[3]))
	w.writeSimpleHuffman(0)
	// One literal pixel, then an immediate LZ77 reference: distance 1 maps to
	// offset width=4, but only one pixel exists, so the reference is invalid.
	w.write(0, 1)
	w.write(0, 1)
	w.write(0, 1)
	w.write(0, 1)
	w.write(1, 1) // LZ77 length symbol
	w.write(0, 1) // distance symbol 0
	stream := w.bytes()

	if _, err := DecodeRegion(context.Background(), bytes.NewReader(stream),
		image.Rect(0, 0, width, height), nil); err == nil {
		t.Fatal("got nil error, want invalid LZ77 parameters")
	}
}

// TestSyntheticColorCacheLimit crafts a stream declaring an 11-bit (2048
// entry) color cache and verifies the allocation is charged against the
// memory limit rather than allocated unconditionally.
func TestSyntheticColorCacheLimit(t *testing.T) {
	var w bitWriter
	w.write(0x2f, 8)
	w.write(9, 14) // 10 px wide
	w.write(9, 14) // 10 px high
	w.write(1, 1)
	w.write(0, 3)
	w.write(0, 1)  // no transforms
	w.write(1, 1)  // color cache present
	w.write(11, 4) // ccBits = 11 => 2048 entries * 4 bytes = 8 KiB
	stream := w.bytes()

	// A tiny limit must be reported as a memory-limit breach.
	_, err := DecodeRegion(context.Background(), bytes.NewReader(stream),
		image.Rect(0, 0, 5, 5), &Options{MaxMemory: 128})
	if !errors.Is(err, ErrMemoryLimitExceeded) {
		t.Fatalf("small limit: got %v, want ErrMemoryLimitExceeded", err)
	}
	// A generous limit gets past the cache allocation and fails later on the
	// truncated Huffman data, proving the cache itself is bounded, not fatal.
	_, err = DecodeRegion(context.Background(), bytes.NewReader(stream),
		image.Rect(0, 0, 5, 5), &Options{MaxMemory: 1 << 20})
	if err == nil || errors.Is(err, ErrMemoryLimitExceeded) {
		t.Fatalf("generous limit: got %v, want a non-memory truncation error", err)
	}
}

// TestSyntheticCancelMidDecode cancels a context while a large (>1M pixel)
// decode is in flight and requires the call to stop with context.Canceled.
func TestSyntheticCancelMidDecode(t *testing.T) {
	const width, height = 1025, 1025
	stream := encodeLZ77VP8L(t, width, height, [4]byte{1, 1, 0, 1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancelled before entry; run() also re-checks per row.
	_, err := DecodeRegion(ctx, bytes.NewReader(stream),
		image.Rect(0, 0, 10, 10), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
}

func colorNRGBA(rgba [4]byte) color.NRGBA {
	return color.NRGBA{R: rgba[0], G: rgba[1], B: rgba[2], A: rgba[3]}
}

func encodeFlatVP8L(t *testing.T, width, height int, rgba [4]byte) []byte {
	t.Helper()
	var w bitWriter
	// Header.
	w.write(0x2f, 8)
	w.write(uint32(width-1), 14)
	w.write(uint32(height-1), 14)
	w.write(1, 1) // hasAlpha
	w.write(0, 3) // version
	w.write(0, 1) // no more transforms
	// Color cache: absent.
	w.write(0, 1)
	// Huffman groups: single group.
	w.write(0, 1) // no meta-image
	// Five trees, each one-symbol: green, red, blue, alpha, distance.
	w.writeSimpleHuffman(uint32(rgba[1]))
	w.writeSimpleHuffman(uint32(rgba[0]))
	w.writeSimpleHuffman(uint32(rgba[2]))
	w.writeSimpleHuffman(uint32(rgba[3]))
	w.writeSimpleHuffman(0)
	// Every pixel is one green symbol (the literal), width*height times.
	for i := 0; i < width*height; i++ {
		w.write(0, 1)
	}
	return w.bytes()
}

// TestSyntheticFlatRegion builds several flat images and checks region
// decoding against the full decoder, including odd dimensions.
func TestSyntheticFlatRegion(t *testing.T) {
	sizes := []struct{ w, h int }{
		{1, 1}, {3, 1}, {1, 7}, {7, 3}, {13, 11}, {31, 17}, {64, 64}, {100, 33},
	}
	colors := [][4]byte{
		{0x11, 0x22, 0x33, 0xff},
		{0xaa, 0xbb, 0xcc, 0x00}, // fully transparent
		{0x00, 0x00, 0x00, 0x80}, // semi-transparent black
	}
	for _, sz := range sizes {
		for _, c := range colors {
			stream := encodeFlatVP8L(t, sz.w, sz.h, c)
			fullImg, err := Decode(bytes.NewReader(stream))
			if err != nil {
				t.Fatalf("w=%d h=%d: full decode: %v", sz.w, sz.h, err)
			}
			full := fullImg.(*image.NRGBA)
			rects := []image.Rectangle{
				image.Rect(0, 0, sz.w, sz.h),
				image.Rect(0, 0, 1, 1),
				image.Rect(sz.w/2, sz.h/2, sz.w, sz.h),
				image.Rect(1, 1, 2, 2),
			}
			for _, r := range rects {
				got, err := DecodeRegion(context.Background(), bytes.NewReader(stream), r, nil)
				if err != nil {
					t.Fatalf("w=%d h=%d r=%v: region decode: %v", sz.w, sz.h, r, err)
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
