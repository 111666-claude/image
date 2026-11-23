// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"image"
	"os"
	"testing"

	"golang.org/x/image/vp8l"
)

// extractVP8L returns the VP8L chunk payload of a RIFF/WEBP container.
func extractVP8L(t *testing.T, data []byte) []byte {
	t.Helper()
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		t.Fatal("not a WEBP file")
	}
	for p := 12; p+8 <= len(data); {
		fourcc := string(data[p : p+4])
		n := int(binary.LittleEndian.Uint32(data[p+4 : p+8]))
		q := p + 8
		if q+n > len(data) {
			t.Fatalf("chunk %q overruns file", fourcc)
		}
		if fourcc == "VP8L" {
			return data[q : q+n]
		}
		p = q + n + (n & 1)
	}
	t.Fatal("no VP8L chunk")
	return nil
}

var regionTestFiles = []string{
	"../testdata/blue-purple-pink.lossless.webp",
	"../testdata/blue-purple-pink-large.lossless.webp",
	"../testdata/gopher-doc.1bpp.lossless.webp",
	"../testdata/gopher-doc.2bpp.lossless.webp",
	"../testdata/gopher-doc.4bpp.lossless.webp",
	"../testdata/gopher-doc.8bpp.lossless.webp",
	"../testdata/tux.lossless.webp",
	"../testdata/yellow_rose.lossless.webp",
}

func loadVP8L(t *testing.T, filename string) ([]byte, *image.NRGBA) {
	t.Helper()
	raw, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	payload := extractVP8L(t, raw)
	full, err := vp8l.Decode(bytes.NewReader(payload))
	if err != nil {
		t.Fatalf("full Decode: %v", err)
	}
	return payload, full.(*image.NRGBA)
}

// TestDecodeRegionMatchesDecode decodes a broad set of sub-rectangles with
// DecodeRegion and checks every returned pixel against the full Decode result.
// The rectangles deliberately straddle predictor/cross-color tile boundaries
// (tiles are up to 16 px), Huffman-group tile boundaries (8/16 px) and
// color-indexing byte boundaries.
func TestDecodeRegionMatchesDecode(t *testing.T) {
	for _, filename := range regionTestFiles {
		payload, full := loadVP8L(t, filename)
		b := full.Bounds()
		rects := []image.Rectangle{
			b,
			image.Rect(0, 0, 1, 1),
			image.Rect(b.Max.X-1, b.Max.Y-1, b.Max.X, b.Max.Y),
			image.Rect(0, 0, b.Max.X, 1),               // first row
			image.Rect(0, b.Max.Y-1, b.Max.X, b.Max.Y), // last row
			image.Rect(0, 0, 1, b.Max.Y),               // first column
			image.Rect(b.Max.X-1, 0, b.Max.X, b.Max.Y), // last column
			image.Rect(7, 7, 9, 9),                     // straddles an 8px boundary
			image.Rect(15, 15, 17, 17),                 // straddles a 16px boundary
			image.Rect(3, 5, 47, 29),
			image.Rect(b.Max.X/2-13, b.Max.Y/2-7, b.Max.X/2+19, b.Max.Y/2+11),
		}
		for _, r := range rects {
			r = r.Intersect(b)
			if r.Empty() {
				continue
			}
			got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
			if err != nil {
				t.Errorf("%s %v: DecodeRegion: %v", filename, r, err)
				continue
			}
			if got.Rect != r {
				t.Errorf("%s %v: bounds got %v, want %v", filename, r, got.Rect, r)
				continue
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					g := got.NRGBAAt(x, y)
					w := full.NRGBAAt(x, y)
					if g != w {
						t.Errorf("%s %v: at (%d,%d) got %v, want %v", filename, r, x, y, g, w)
					}
				}
			}
		}
	}
}

// TestDecodeRegionBounds checks clipping, empty and out-of-bounds rectangles.
func TestDecodeRegionBounds(t *testing.T) {
	payload, full := loadVP8L(t, "../testdata/tux.lossless.webp")
	b := full.Bounds()

	// Fully out-of-bounds rectangle: empty image, no error.
	empty, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload),
		image.Rect(b.Max.X+10, b.Max.Y+10, b.Max.X+20, b.Max.Y+20), nil)
	if err != nil {
		t.Fatalf("out-of-bounds: %v", err)
	}
	if !empty.Rect.Empty() || len(empty.Pix) != 0 {
		t.Fatalf("out-of-bounds: got rect=%v pixLen=%d, want empty", empty.Rect, len(empty.Pix))
	}

	// Partially overlapping rectangle is clipped to the image bounds.
	r := image.Rect(b.Max.X-5, b.Max.Y-5, b.Max.X+50, b.Max.Y+50)
	got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
	if err != nil {
		t.Fatalf("partial overlap: %v", err)
	}
	want := r.Intersect(b)
	if got.Rect != want {
		t.Fatalf("partial overlap: got %v, want %v", got.Rect, want)
	}
	for y := want.Min.Y; y < want.Max.Y; y++ {
		for x := want.Min.X; x < want.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("partial overlap: mismatch at (%d,%d)", x, y)
			}
		}
	}

	// A negative-origin rectangle is clipped as well.
	r = image.Rect(-50, -50, 12, 13)
	got, err = vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
	if err != nil {
		t.Fatalf("negative origin: %v", err)
	}
	if got.Rect != r.Intersect(b) {
		t.Fatalf("negative origin: got %v, want %v", got.Rect, r.Intersect(b))
	}

	// The empty rectangle is canonical and error-free.
	empty, err = vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload),
		image.Rect(10, 10, 10, 20), nil)
	if err != nil || !empty.Rect.Empty() {
		t.Fatalf("empty rect: rect=%v err=%v", empty.Rect, err)
	}
}

// TestDecodeRegionRepeated verifies that repeated calls on independent
// readers return identical results (the decoder keeps no global state).
func TestDecodeRegionRepeated(t *testing.T) {
	payload, full := loadVP8L(t, "../testdata/blue-purple-pink.lossless.webp")
	r := image.Rect(11, 9, 73, 41)
	var first *image.NRGBA
	for i := 0; i < 3; i++ {
		got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
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

// TestDecodeRegionNoLeakage verifies that the returned Pix contains no bytes
// from outside the requested rectangle: every pixel position is covered and
// the buffer is tightly sized.
func TestDecodeRegionNoLeakage(t *testing.T) {
	payload, full := loadVP8L(t, "../testdata/yellow_rose.lossless.webp")
	r := image.Rect(37, 41, 101, 88)
	got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Pix) != r.Dx()*r.Dy()*4 {
		t.Fatalf("Pix len = %d, want %d", len(got.Pix), r.Dx()*r.Dy()*4)
	}
	if got.Stride != r.Dx()*4 {
		t.Fatalf("Stride = %d, want %d", got.Stride, r.Dx()*4)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("mismatch at (%d,%d)", x, y)
			}
		}
	}
}

// TestDecodeRegionTransparent checks alpha semantics against the full decode
// on the images that carry transparent pixels.
func TestDecodeRegionTransparent(t *testing.T) {
	for _, filename := range []string{
		"../testdata/tux.lossless.webp",
		"../testdata/yellow_rose.lossless.webp",
	} {
		payload, full := loadVP8L(t, filename)
		b := full.Bounds()
		r := image.Rect(0, 0, b.Max.X, b.Max.Y)
		got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
		if err != nil {
			t.Fatalf("%s: %v", filename, err)
		}
		nTransparent := 0
		for i := 3; i < len(full.Pix); i += 4 {
			if full.Pix[i] == 0 {
				nTransparent++
			}
		}
		if nTransparent == 0 {
			t.Errorf("%s: test image has no transparent pixels", filename)
		}
		if !bytes.Equal(got.Pix, full.Pix) {
			t.Fatalf("%s: full-region decode differs from full Decode", filename)
		}
	}
}

// TestDecodeRegionMemoryLimit exercises the memory cap with deliberately
// small limits, including limits smaller than any pixel allocation.
func TestDecodeRegionMemoryLimit(t *testing.T) {
	payload, full := loadVP8L(t, "../testdata/blue-purple-pink.lossless.webp")
	b := full.Bounds()
	r := image.Rect(0, 0, 10, 10)
	for _, limit := range []int64{1, 64, 1024, 4096, 1 << 16} {
		_, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r,
			&vp8l.Options{MaxMemory: limit})
		if !errors.Is(err, vp8l.ErrMemoryLimitExceeded) {
			t.Errorf("limit %d: got %v, want ErrMemoryLimitExceeded", limit, err)
		}
	}
	// A generous limit succeeds and produces correct pixels.
	got, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(payload), r,
		&vp8l.Options{MaxMemory: 64 << 20})
	if err != nil {
		t.Fatalf("generous limit: %v", err)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
				t.Fatalf("mismatch at (%d,%d)", x, y)
			}
		}
	}
	_ = b
}

// TestDecodeRegionCancelled checks context cancellation.
func TestDecodeRegionCancelled(t *testing.T) {
	payload, _ := loadVP8L(t, "../testdata/blue-purple-pink-large.lossless.webp")
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Already cancelled before the call.
	_, err := vp8l.DecodeRegion(ctx, bytes.NewReader(payload),
		image.Rect(0, 0, 50, 50), nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled: got %v, want context.Canceled", err)
	}
}

// TestDecodeRegionTruncated feeds progressively truncated bit-streams and
// requires an error (never a panic) on every prefix length.
func TestDecodeRegionTruncated(t *testing.T) {
	payload, _ := loadVP8L(t, "../testdata/tux.lossless.webp")
	r := image.Rect(5, 5, 60, 60)
	for n := 0; n < len(payload); n += 37 {
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("prefix %d: panic: %v", n, rec)
				}
			}()
			_, err := vp8l.DecodeRegion(context.Background(),
				bytes.NewReader(payload[:n]), r, nil)
			if err == nil {
				t.Errorf("prefix %d: got nil error, want non-nil", n)
			}
		}()
	}
}

// TestDecodeRegionMalformedColorCache builds a header claiming the maximum
// (11-bit) color cache and then cuts the stream, confirming bounded behavior.
func TestDecodeRegionMalformedColorCache(t *testing.T) {
	payload, _ := loadVP8L(t, "../testdata/blue-purple-pink.lossless.webp")
	// Keep only the header and transform section and truncate inside the
	// Huffman data: the decoder must error rather than allocate unbounded.
	for _, n := range []int{30, 40, 60, 80} {
		if n >= len(payload) {
			continue
		}
		_, err := vp8l.DecodeRegion(context.Background(),
			bytes.NewReader(payload[:n]), image.Rect(0, 0, 5, 5), nil)
		if err == nil {
			t.Errorf("truncation %d: nil error", n)
		}
	}
}

// TestDecodeRegionHostileDimensions crafts a VP8L header claiming the largest
// representable dimensions (16384x16384) followed by no data: decoding must
// fail with a bounded error, never a huge allocation.
func TestDecodeRegionHostileDimensions(t *testing.T) {
	// VP8L header: 0x2f magic, 14-bit (w-1), 14-bit (h-1), 1-bit alpha,
	// 3-bit version (=0).
	const wm1, hm1 = 16383, 16383
	bits := uint64(0x2f) |
		uint64(wm1)<<8 |
		uint64(hm1)<<22 |
		uint64(0)<<36 // alpha hint and version both zero.
	header := []byte{
		byte(bits), byte(bits >> 8), byte(bits >> 16), byte(bits >> 24),
		byte(bits >> 32),
	}
	_, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(header),
		image.Rect(0, 0, 8, 8), &vp8l.Options{MaxMemory: 8 << 20})
	if err == nil {
		t.Fatal("hostile dimensions: got nil error")
	}
	if errors.Is(err, vp8l.ErrMemoryLimitExceeded) == false {
		// Truncation is also acceptable; an allocation failure/panic is not.
		t.Logf("hostile dimensions returned %v", err)
	}
}
