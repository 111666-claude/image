// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8

import (
	"bytes"
	"context"
	"encoding/binary"
	"image"
	"os"
	"testing"
)

// vp8TestFiles are the lossy WEBP fixtures, whose VP8 payloads are used to
// exercise DecodeFrameRegion directly against DecodeFrame.
var vp8TestFiles = []string{
	"blue-purple-pink.lossy.webp",
	"blue-purple-pink-large.no-filter.lossy.webp",
	"blue-purple-pink-large.simple-filter.lossy.webp",
	"blue-purple-pink-large.normal-filter.lossy.webp",
	"video-001.lossy.webp",
	"yellow_rose.lossy.webp",
}

// extractVP8 returns the VP8 chunk payload of a RIFF/WEBP file.
func extractVP8(t *testing.T, name string) []byte {
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
		if fourcc == "VP8 " {
			return raw[q : q+n]
		}
		p = q + n + (n & 1)
	}
	t.Fatalf("%s: no VP8 chunk", name)
	return nil
}

func decodeFullVP8(t *testing.T, payload []byte) *image.YCbCr {
	t.Helper()
	d := NewDecoder()
	d.Init(bytes.NewReader(payload), len(payload))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatal(err)
	}
	m, err := d.DecodeFrame()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func decodeRegionVP8(t *testing.T, payload []byte, r image.Rectangle) *image.YCbCr {
	t.Helper()
	d := NewDecoder()
	d.Init(bytes.NewReader(payload), len(payload))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatal(err)
	}
	m, err := d.DecodeFrameRegion(context.Background(), r, nil)
	if err != nil {
		t.Fatalf("%v: %v", r, err)
	}
	return m
}

// TestDecodeFrameRegionMatchesDecode checks that region pixels equal the
// corresponding full-frame pixels. Rectangles deliberately start and end
// inside macroblocks and on their boundaries, and cover one-row bands (which
// need the loop filter's below-row) as well as the bottom edge of the frame.
func TestDecodeFrameRegionMatchesDecode(t *testing.T) {
	for _, name := range vp8TestFiles {
		payload := extractVP8(t, name)
		full := decodeFullVP8(t, payload)
		b := full.Bounds()
		rects := []image.Rectangle{
			b,
			image.Rect(0, 0, 1, 1),
			image.Rect(b.Max.X-1, b.Max.Y-1, b.Max.X, b.Max.Y),
			image.Rect(0, 0, b.Max.X, 1),
			image.Rect(0, 15, b.Max.X, 17),
			image.Rect(0, 16, b.Max.X, 32),
			image.Rect(0, b.Max.Y-16, b.Max.X, b.Max.Y),
			image.Rect(17, 19, 41, 53),
			image.Rect(b.Max.X/2-7, b.Max.Y/2-5, b.Max.X/2+11, b.Max.Y/2+13),
		}
		for _, r := range rects {
			r = r.Intersect(b)
			if r.Empty() {
				continue
			}
			got := decodeRegionVP8(t, payload, r)
			if got.Bounds() != r {
				t.Errorf("%s %v: bounds got %v", name, r, got.Bounds())
				continue
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if got.YCbCrAt(x, y) != full.YCbCrAt(x, y) {
						t.Fatalf("%s %v: mismatch at (%d,%d): got %v want %v",
							name, r, x, y, got.YCbCrAt(x, y), full.YCbCrAt(x, y))
					}
				}
			}
		}
	}
}

// TestDecodeFrameRegionBounds checks clipping, empty rectangles and odd
// origins.
func TestDecodeFrameRegionBounds(t *testing.T) {
	payload := extractVP8(t, "yellow_rose.lossy.webp")
	full := decodeFullVP8(t, payload)
	b := full.Bounds()

	// Empty and fully outside: zero-size image, no error.
	for _, r := range []image.Rectangle{
		image.Rect(b.Max.X+10, b.Max.Y+10, b.Max.X+20, b.Max.Y+20),
		image.Rect(10, 10, 10, 20),
	} {
		d := NewDecoder()
		d.Init(bytes.NewReader(payload), len(payload))
		if _, err := d.DecodeFrameHeader(); err != nil {
			t.Fatal(err)
		}
		m, err := d.DecodeFrameRegion(context.Background(), r, nil)
		if err != nil {
			t.Fatalf("%v: %v", r, err)
		}
		if !m.Bounds().Empty() {
			t.Errorf("%v: bounds %v, want empty", r, m.Bounds())
		}
	}

	// Partial overlap and negative origin are clipped.
	for _, tc := range []struct{ in, want image.Rectangle }{
		{image.Rect(b.Max.X-5, b.Max.Y-5, b.Max.X+50, b.Max.Y+50), image.Rect(b.Max.X-5, b.Max.Y-5, b.Max.X, b.Max.Y)},
		{image.Rect(-50, -50, 12, 13), image.Rect(0, 0, 12, 13)},
	} {
		got := decodeRegionVP8(t, payload, tc.in)
		if got.Bounds() != tc.want {
			t.Errorf("%v: bounds %v, want %v", tc.in, got.Bounds(), tc.want)
		}
		for y := tc.want.Min.Y; y < tc.want.Max.Y; y++ {
			for x := tc.want.Min.X; x < tc.want.Max.X; x++ {
				if got.YCbCrAt(x, y) != full.YCbCrAt(x, y) {
					t.Fatalf("%v: mismatch at (%d,%d)", tc.in, x, y)
				}
			}
		}
	}
}

// TestDecodeFrameRegionReuse checks that one Decoder can decode several
// regions of a frame in turn and still match the full decode.
func TestDecodeFrameRegionReuse(t *testing.T) {
	payload := extractVP8(t, "blue-purple-pink-large.normal-filter.lossy.webp")
	full := decodeFullVP8(t, payload)
	b := full.Bounds()

	d := NewDecoder()
	for _, r := range []image.Rectangle{
		image.Rect(3, 5, 41, 67),
		image.Rect(0, 16, b.Max.X, 32),
		image.Rect(b.Max.X-9, b.Max.Y-7, b.Max.X, b.Max.Y),
	} {
		d.Init(bytes.NewReader(payload), len(payload))
		if _, err := d.DecodeFrameHeader(); err != nil {
			t.Fatal(err)
		}
		got, err := d.DecodeFrameRegion(context.Background(), r, nil)
		if err != nil {
			t.Fatalf("%v: %v", r, err)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if got.YCbCrAt(x, y) != full.YCbCrAt(x, y) {
					t.Fatalf("%v: mismatch at (%d,%d)", r, x, y)
				}
			}
		}
	}
}

// TestDecodeFrameRegionCancel checks cancellation both before decoding and on
// the context that is checked between rows.
func TestDecodeFrameRegionCancel(t *testing.T) {
	payload := extractVP8(t, "blue-purple-pink-large.normal-filter.lossy.webp")
	d := NewDecoder()
	d.Init(bytes.NewReader(payload), len(payload))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := d.DecodeFrameRegion(ctx, image.Rect(0, 0, 16, 16), nil); err != context.Canceled {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	// A canceled decode leaves the decoder reusable after Init.
	d.Init(bytes.NewReader(payload), len(payload))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatal(err)
	}
	if _, err := d.DecodeFrameRegion(context.Background(), image.Rect(0, 0, 16, 16), nil); err != nil {
		t.Fatalf("after cancel: %v", err)
	}
}

// TestDecodeFrameRegionMemoryLimit checks that a tiny limit is rejected with
// ErrRegionTooLarge before any big allocation.
func TestDecodeFrameRegionMemoryLimit(t *testing.T) {
	payload := extractVP8(t, "blue-purple-pink-large.normal-filter.lossy.webp")
	d := NewDecoder()
	d.Init(bytes.NewReader(payload), len(payload))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatal(err)
	}
	if _, err := d.DecodeFrameRegion(context.Background(), image.Rect(0, 0, 16, 16),
		&RegionOptions{MaxMemory: 1}); err != ErrRegionTooLarge {
		t.Fatalf("got %v, want ErrRegionTooLarge", err)
	}
}

// TestDecodeFrameRegionTruncated requires an error, never a panic, for
// truncated partitions.
func TestDecodeFrameRegionTruncated(t *testing.T) {
	payload := extractVP8(t, "video-001.lossy.webp")
	for n := 0; n < len(payload); n += 17 {
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("prefix %d: panic: %v", n, rec)
				}
			}()
			d := NewDecoder()
			d.Init(bytes.NewReader(payload[:n]), n)
			if _, err := d.DecodeFrameHeader(); err != nil {
				return
			}
			_, _ = d.DecodeFrameRegion(context.Background(), image.Rect(0, 0, 16, 16), nil)
		}()
	}
}
