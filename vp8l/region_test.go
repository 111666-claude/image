// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"image"
	"os"
	"testing"
)

var vp8lTestFiles = []string{
	"blue-purple-pink.lossless.webp",
	"blue-purple-pink-large.lossless.webp",
	"gopher-doc.1bpp.lossless.webp",
	"gopher-doc.2bpp.lossless.webp",
	"gopher-doc.4bpp.lossless.webp",
	"gopher-doc.8bpp.lossless.webp",
	"tux.lossless.webp",
	"yellow_rose.lossless.webp",
}

// extractVP8L returns the VP8L chunk payload of a RIFF/WEBP file.
func extractVP8L(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("../testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) < 12 || string(raw[8:12]) != "WEBP" {
		t.Fatalf("%s: not a WEBP file", name)
	}
	for p := 12; p+8 <= len(raw); {
		fourcc := string(raw[p : p+4])
		n := int(binary.LittleEndian.Uint32(raw[p+4 : p+8]))
		q := p + 8
		if q+n > len(raw) {
			t.Fatalf("%s: chunk %q overruns file", name, fourcc)
		}
		if fourcc == "VP8L" {
			return raw[q : q+n]
		}
		p = q + n + (n & 1)
	}
	t.Fatalf("%s: no VP8L chunk", name)
	return nil
}

func decodeFullVP8L(t *testing.T, payload []byte) *image.NRGBA {
	t.Helper()
	m, err := Decode(bytes.NewReader(payload))
	if err != nil {
		t.Fatal(err)
	}
	return m.(*image.NRGBA)
}

func decodeRegionVP8L(t *testing.T, payload []byte, r image.Rectangle) *image.NRGBA {
	t.Helper()
	m, err := DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
	if err != nil {
		t.Fatalf("%v: %v", r, err)
	}
	return m
}

// TestDecodeRegionMatchesDecode checks that DecodeRegion matches Decode on the
// whole image, on tile-straddling rectangles (predictor tiles are up to 32 px,
// Huffman-group tiles 8 or 16 px, color-index packing 2-8 px), on single
// pixels and on the four edges.
func TestDecodeRegionMatchesDecode(t *testing.T) {
	for _, name := range vp8lTestFiles {
		payload := extractVP8L(t, name)
		full := decodeFullVP8L(t, payload)
		b := full.Bounds()
		rects := []image.Rectangle{
			b,
			image.Rect(0, 0, 1, 1),
			image.Rect(b.Max.X-1, b.Max.Y-1, b.Max.X, b.Max.Y),
			image.Rect(0, 0, b.Max.X, 1),
			image.Rect(0, b.Max.Y-1, b.Max.X, b.Max.Y),
			image.Rect(0, 0, 1, b.Max.Y),
			image.Rect(b.Max.X-1, 0, b.Max.X, b.Max.Y),
			image.Rect(7, 7, 9, 9),
			image.Rect(15, 15, 17, 17),
			image.Rect(31, 31, 33, 33),
			image.Rect(3, 5, 47, 29),
			image.Rect(b.Max.X/2-13, b.Max.Y/2-7, b.Max.X/2+19, b.Max.Y/2+11),
		}
		for _, r := range rects {
			r = r.Intersect(b)
			if r.Empty() {
				continue
			}
			got := decodeRegionVP8L(t, payload, r)
			if got.Rect != r {
				t.Errorf("%s %v: bounds got %v", name, r, got.Rect)
				continue
			}
			if got.Stride != r.Dx()*4 {
				t.Errorf("%s %v: stride %d, want %d", name, r, got.Stride, r.Dx()*4)
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
						t.Fatalf("%s %v: mismatch at (%d,%d): got %v want %v",
							name, r, x, y, got.NRGBAAt(x, y), full.NRGBAAt(x, y))
					}
				}
			}
		}
	}
}

// TestDecodeRegionBounds checks clipping, empty and out-of-bounds rectangles.
func TestDecodeRegionBounds(t *testing.T) {
	payload := extractVP8L(t, "tux.lossless.webp")
	full := decodeFullVP8L(t, payload)
	b := full.Bounds()

	// Fully out-of-bounds: empty image, no error.
	empty, err := DecodeRegion(context.Background(), bytes.NewReader(payload),
		image.Rect(b.Max.X+10, b.Max.Y+10, b.Max.X+20, b.Max.Y+20), nil)
	if err != nil {
		t.Fatalf("out-of-bounds: %v", err)
	}
	if !empty.Rect.Empty() || len(empty.Pix) != 0 {
		t.Fatalf("out-of-bounds: rect=%v pixLen=%d, want empty", empty.Rect, len(empty.Pix))
	}

	// Partially overlapping: clipped.
	r := image.Rect(b.Max.X-5, b.Max.Y-5, b.Max.X+50, b.Max.Y+50)
	got := decodeRegionVP8L(t, payload, r)
	want := r.Intersect(b)
	if got.Rect != want {
		t.Fatalf("partial: got %v, want %v", got.Rect, want)
	}
	for y := want.Min.Y; y < want.Max.Y; y++ {
		for x := want.Min.X; x < want.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("partial: mismatch at (%d,%d)", x, y)
			}
		}
	}

	// Negative origin: clipped.
	r = image.Rect(-50, -50, 12, 13)
	got = decodeRegionVP8L(t, payload, r)
	if got.Rect != r.Intersect(b) {
		t.Fatalf("negative: got %v, want %v", got.Rect, r.Intersect(b))
	}

	// Degenerate rectangle is canonical and error-free.
	empty, err = DecodeRegion(context.Background(), bytes.NewReader(payload),
		image.Rect(10, 10, 10, 20), nil)
	if err != nil || !empty.Rect.Empty() {
		t.Fatalf("degenerate: rect=%v err=%v", empty.Rect, err)
	}

	// Extreme coordinates cannot overflow the clipping arithmetic.
	huge := image.Rect(-1<<30, -1<<30, 1<<30, 1<<30)
	got = decodeRegionVP8L(t, payload, huge)
	if got.Rect != b {
		t.Fatalf("huge: got %v, want %v", got.Rect, b)
	}
}

// TestDecodeRegionRepeated checks that repeated calls on independent readers
// return identical results.
func TestDecodeRegionRepeated(t *testing.T) {
	payload := extractVP8L(t, "blue-purple-pink.lossless.webp")
	full := decodeFullVP8L(t, payload)
	r := image.Rect(11, 9, 73, 41)
	var first *image.NRGBA
	for i := 0; i < 3; i++ {
		got := decodeRegionVP8L(t, payload, r)
		if first == nil {
			first = got
			continue
		}
		if !bytes.Equal(got.Pix, first.Pix) || got.Rect != first.Rect || got.Stride != first.Stride {
			t.Fatalf("call %d differs from call 0", i)
		}
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if first.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("mismatch at (%d,%d)", x, y)
			}
		}
	}
}

// TestDecodeRegionTransparent confirms alpha semantics on images with
// transparent pixels, including a rectangle that contains only transparent
// pixels.
func TestDecodeRegionTransparent(t *testing.T) {
	for _, name := range []string{"tux.lossless.webp", "yellow_rose.lossless.webp"} {
		payload := extractVP8L(t, name)
		full := decodeFullVP8L(t, payload)
		b := full.Bounds()

		// Find a fully transparent pixel to build a transparent rectangle.
		var tr image.Rectangle
		foundTransparent := false
		for y := b.Min.Y; y < b.Max.Y && !foundTransparent; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if full.NRGBAAt(x, y).A == 0 {
					tr = image.Rect(x, y, x+1, y+1)
					foundTransparent = true
					break
				}
			}
		}
		if !foundTransparent {
			t.Logf("%s: no fully transparent pixel", name)
		} else {
			got := decodeRegionVP8L(t, payload, tr)
			if got.NRGBAAt(tr.Min.X, tr.Min.Y) != full.NRGBAAt(tr.Min.X, tr.Min.Y) {
				t.Errorf("%s: transparent pixel mismatch", name)
			}
		}

		// The whole image with transparency included matches byte for byte.
		got := decodeRegionVP8L(t, payload, b)
		if !bytes.Equal(got.Pix, full.Pix) {
			t.Fatalf("%s: full-region decode differs from Decode", name)
		}
	}
}

// TestDecodeRegionMemoryLimit exercises the cap with limits below any pixel
// allocation.
func TestDecodeRegionMemoryLimit(t *testing.T) {
	payload := extractVP8L(t, "blue-purple-pink.lossless.webp")
	full := decodeFullVP8L(t, payload)
	r := image.Rect(0, 0, 10, 10)
	for _, limit := range []int64{1, 64, 1024, 4096, 1 << 16} {
		_, err := DecodeRegion(context.Background(), bytes.NewReader(payload), r,
			&Options{MaxMemory: limit})
		if !errors.Is(err, ErrMemoryLimitExceeded) {
			t.Errorf("limit %d: got %v, want ErrMemoryLimitExceeded", limit, err)
		}
	}
	got, err := DecodeRegion(context.Background(), bytes.NewReader(payload), r,
		&Options{MaxMemory: 64 << 20})
	if err != nil {
		t.Fatalf("generous limit: %v", err)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("generous limit: mismatch at (%d,%d)", x, y)
			}
		}
	}
}

// TestDecodeRegionCancelled checks context cancellation.
func TestDecodeRegionCancelled(t *testing.T) {
	payload := extractVP8L(t, "blue-purple-pink-large.lossless.webp")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := DecodeRegion(ctx, bytes.NewReader(payload), image.Rect(0, 0, 50, 50), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled: got %v, want context.Canceled", err)
	}
}

// TestDecodeRegionTruncated feeds progressively truncated bit-streams and
// requires an error (never a panic) on every prefix that still parses the
// header.
func TestDecodeRegionTruncated(t *testing.T) {
	payload := extractVP8L(t, "tux.lossless.webp")
	r := image.Rect(5, 5, 60, 60)
	for n := 0; n < len(payload); n += 37 {
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("prefix %d: panic: %v", n, rec)
				}
			}()
			_, err := DecodeRegion(context.Background(), bytes.NewReader(payload[:n]), r, nil)
			if err == nil {
				t.Errorf("prefix %d: got nil error, want non-nil", n)
			}
		}()
	}
}

// TestDecodeRegionHostileDimensions crafts a VP8L header claiming the largest
// representable dimensions followed by no data: decoding must fail with a
// bounded error, never a huge allocation.
func TestDecodeRegionHostileDimensions(t *testing.T) {
	const wm1, hm1 = 16383, 16383
	bits := uint64(0x2f) |
		uint64(wm1)<<8 |
		uint64(hm1)<<22
	header := []byte{
		byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24), byte(bits >> 32),
	}
	_, err := DecodeRegion(context.Background(), bytes.NewReader(header),
		image.Rect(0, 0, 8, 8), &Options{MaxMemory: 8 << 20})
	if err == nil {
		t.Fatal("hostile dimensions: got nil error")
	}
}

// TestDecodeRegionColorIndexing checks the color-index transform across a
// rectangle boundary, using the 1/2/4 bpp fixtures whose palette packing makes
// the coded row narrower than the output row.
func TestDecodeRegionColorIndexing(t *testing.T) {
	for _, name := range []string{
		"gopher-doc.1bpp.lossless.webp",
		"gopher-doc.2bpp.lossless.webp",
		"gopher-doc.4bpp.lossless.webp",
		"gopher-doc.8bpp.lossless.webp",
	} {
		payload := extractVP8L(t, name)
		full := decodeFullVP8L(t, payload)
		b := full.Bounds()
		// Odd column bounds exercise the half-byte packing of the last coded
		// pixel.
		for _, r := range []image.Rectangle{
			image.Rect(1, 0, b.Max.X-1, b.Max.Y),
			image.Rect(3, 2, b.Max.X-3, 9),
			image.Rect(b.Max.X/2+1, 0, b.Max.X/2+2, b.Max.Y),
		} {
			r = r.Intersect(b)
			if r.Empty() {
				continue
			}
			got := decodeRegionVP8L(t, payload, r)
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
						t.Fatalf("%s %v: mismatch at (%d,%d)", name, r, x, y)
					}
				}
			}
		}
	}
}
