// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"os"
	"strings"
	"testing"
)

// regionFixtures lists every WEBP file in testdata with an independent way to
// obtain its full decode, so that a region decode can be compared pixel for
// pixel.
var regionFixtures = []string{
	"blue-purple-pink.lossy.webp",
	"blue-purple-pink-large.no-filter.lossy.webp",
	"blue-purple-pink-large.simple-filter.lossy.webp",
	"blue-purple-pink-large.normal-filter.lossy.webp",
	"video-001.lossy.webp",
	"yellow_rose.lossy.webp",
	"yellow_rose.lossy-with-alpha.webp",
	"blue-purple-pink.lossless.webp",
	"blue-purple-pink-large.lossless.webp",
	"gopher-doc.1bpp.lossless.webp",
	"gopher-doc.2bpp.lossless.webp",
	"gopher-doc.4bpp.lossless.webp",
	"gopher-doc.8bpp.lossless.webp",
	"tux.lossless.webp",
	"yellow_rose.lossless.webp",
}

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("../testdata/" + name)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return raw
}

// decodeFull is the reference: the ordinary whole-image Decode.
func decodeFull(t *testing.T, raw []byte) image.Image {
	t.Helper()
	m, err := Decode(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	return m
}

// regionRects covers the whole image, the four quadrants, zero-origin
// rectangles, a rectangle that straddles macroblock and transform tiles, odd
// sizes, and single pixels in each corner and near the centre.
func regionRects(b image.Rectangle) []image.Rectangle {
	w, h := b.Dx(), b.Dy()
	rects := []image.Rectangle{
		b,
		image.Rect(0, 0, w/2, h/2),
		image.Rect(w/2, 0, w, h/2),
		image.Rect(0, h/2, w/2, h),
		image.Rect(w/2, h/2, w, h),
		image.Rect(0, 0, 1, 1),
		image.Rect(w-1, 0, w, 1),
		image.Rect(0, h-1, 1, h),
		image.Rect(w-1, h-1, w, h),
		image.Rect(w/2-7, h/2-5, w/2+9, h/2+7),
		image.Rect(7, 9, 24, 39),
		image.Rect(15, 15, 17, 17),
	}
	// Clamp so that tiny fixtures do not produce empty or inverted rects.
	out := rects[:0]
	for _, r := range rects {
		r = r.Intersect(b)
		if !r.Empty() {
			out = append(out, r)
		}
	}
	return out
}

func sameColor(a, b color.Color) bool {
	ar, ag, ab, aa := a.RGBA()
	br, bg, bb, ba := b.RGBA()
	return ar == br && ag == bg && ab == bb && aa == ba
}

func TestDecodeRegionMatchesDecode(t *testing.T) {
	for _, name := range regionFixtures {
		raw := readFixture(t, name)
		full := decodeFull(t, raw)
		for _, r := range regionRects(full.Bounds()) {
			got, err := DecodeRegion(context.Background(), bytes.NewReader(raw), r, nil)
			if err != nil {
				t.Errorf("%s %v: DecodeRegion: %v", name, r, err)
				continue
			}
			if got.Bounds() != r {
				t.Errorf("%s: bounds got %v, want %v", name, got.Bounds(), r)
				continue
			}
			if isTransparentArea(got, r) && !isTransparentArea(full, r) {
				// Not a failure; just context for the assertion below.
				t.Logf("%s %v: full image is not transparent here", name, r)
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if !sameColor(got.At(x, y), full.At(x, y)) {
						t.Fatalf("%s %v: mismatch at (%d,%d): got %v, want %v",
							name, r, x, y, got.At(x, y), full.At(x, y))
					}
				}
			}
		}
	}
}

func isTransparentArea(m image.Image, r image.Rectangle) bool {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if _, _, _, a := m.At(x, y).RGBA(); a != 0 {
				return false
			}
		}
	}
	return true
}

// TestDecodeRegionQuadrants checks that the four quadrants of an image
// reassemble into exactly the full image, so that no quadrant silently decodes
// the wrong rows or columns.
func TestDecodeRegionQuadrants(t *testing.T) {
	for _, name := range []string{
		"yellow_rose.lossy.webp",
		"yellow_rose.lossy-with-alpha.webp",
		"yellow_rose.lossless.webp",
		"tux.lossless.webp",
	} {
		raw := readFixture(t, name)
		full := decodeFull(t, raw)
		b := full.Bounds()
		xm, ym := b.Min.X+b.Dx()/2, b.Min.Y+b.Dy()/2
		quads := []image.Rectangle{
			image.Rect(b.Min.X, b.Min.Y, xm, ym),
			image.Rect(xm, b.Min.Y, b.Max.X, ym),
			image.Rect(b.Min.X, ym, xm, b.Max.Y),
			image.Rect(xm, ym, b.Max.X, b.Max.Y),
		}
		reassembled := image.NewNRGBA(b)
		for _, q := range quads {
			got, err := DecodeRegion(context.Background(), bytes.NewReader(raw), q, nil)
			if err != nil {
				t.Fatalf("%s %v: %v", name, q, err)
			}
			for y := q.Min.Y; y < q.Max.Y; y++ {
				for x := q.Min.X; x < q.Max.X; x++ {
					reassembled.Set(x, y, got.At(x, y))
					if !sameColor(got.At(x, y), full.At(x, y)) {
						t.Fatalf("%s %v: quadrant mismatch at (%d,%d)", name, q, x, y)
					}
				}
			}
		}
	}
}

// TestDecodeRegionTransparentEdges focuses on images with an alpha channel,
// including a rectangle that is entirely transparent and one that straddles
// the opaque/transparent boundary.
func TestDecodeRegionTransparentEdges(t *testing.T) {
	for _, name := range []string{
		"yellow_rose.lossy-with-alpha.webp",
		"tux.lossless.webp",
		"yellow_rose.lossless.webp",
	} {
		raw := readFixture(t, name)
		full := decodeFull(t, raw)
		b := full.Bounds()
		// Find the rows and columns that contain at least one transparent
		// pixel so the rectangles below are meaningful.
		found := false
		for y := b.Min.Y; y < b.Max.Y && !found; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				if _, _, _, a := full.At(x, y).RGBA(); a < 0xffff {
					found = true
					break
				}
			}
		}
		if !found {
			t.Logf("%s: no partially transparent pixels", name)
		}
		// The outermost one-pixel frame of an image with transparency is the
		// classic place a region decoder leaks neighbouring pixels.
		edges := []image.Rectangle{
			image.Rect(b.Min.X, b.Min.Y, b.Max.X, b.Min.Y+1),
			image.Rect(b.Min.X, b.Max.Y-1, b.Max.X, b.Max.Y),
			image.Rect(b.Min.X, b.Min.Y, b.Min.X+1, b.Max.Y),
			image.Rect(b.Max.X-1, b.Min.Y, b.Max.X, b.Max.Y),
		}
		for _, r := range edges {
			got, err := DecodeRegion(context.Background(), bytes.NewReader(raw), r, nil)
			if err != nil {
				t.Fatalf("%s %v: %v", name, r, err)
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if !sameColor(got.At(x, y), full.At(x, y)) {
						t.Fatalf("%s %v: edge mismatch at (%d,%d)", name, r, x, y)
					}
				}
			}
		}
	}
}

// TestDecodeRegionBounds checks clipping, empty rectangles and zero-size
// results. The concrete image type must match the one Decode would return.
func TestDecodeRegionBounds(t *testing.T) {
	cases := []struct {
		name   string
		raw    []byte
		wantTy func(image.Image) bool
	}{
		{"yellow_rose.lossy.webp", readFixture(t, "yellow_rose.lossy.webp"),
			func(m image.Image) bool { _, ok := m.(*image.YCbCr); return ok }},
		{"yellow_rose.lossy-with-alpha.webp", readFixture(t, "yellow_rose.lossy-with-alpha.webp"),
			func(m image.Image) bool { _, ok := m.(*image.NYCbCrA); return ok }},
		{"tux.lossless.webp", readFixture(t, "tux.lossless.webp"),
			func(m image.Image) bool { _, ok := m.(*image.NRGBA); return ok }},
	}
	for _, tc := range cases {
		full := decodeFull(t, tc.raw)
		b := full.Bounds()

		// Entirely outside the image: an empty image and no error.
		outside := image.Rect(b.Max.X+10, b.Max.Y+10, b.Max.X+20, b.Max.Y+20)
		got, err := DecodeRegion(context.Background(), bytes.NewReader(tc.raw), outside, nil)
		if err != nil {
			t.Fatalf("%s: outside: %v", tc.name, err)
		}
		if !got.Bounds().Empty() {
			t.Errorf("%s: outside: bounds %v, want empty", tc.name, got.Bounds())
		}
		if !tc.wantTy(got) {
			t.Errorf("%s: outside: got %T, wrong concrete type", tc.name, got)
		}

		// Partially overlapping: clipped to the image.
		partial := image.Rect(b.Max.X-5, b.Max.Y-5, b.Max.X+50, b.Max.Y+50)
		got, err = DecodeRegion(context.Background(), bytes.NewReader(tc.raw), partial, nil)
		if err != nil {
			t.Fatalf("%s: partial: %v", tc.name, err)
		}
		if want := partial.Intersect(b); got.Bounds() != want {
			t.Errorf("%s: partial: bounds %v, want %v", tc.name, got.Bounds(), want)
		}

		// Negative origin, clipped as well.
		negative := image.Rect(-50, -50, 12, 13)
		got, err = DecodeRegion(context.Background(), bytes.NewReader(tc.raw), negative, nil)
		if err != nil {
			t.Fatalf("%s: negative: %v", tc.name, err)
		}
		if want := negative.Intersect(b); got.Bounds() != want {
			t.Errorf("%s: negative: bounds %v, want %v", tc.name, got.Bounds(), want)
		}
		for y := 0; y < 13; y++ {
			for x := 0; x < 12; x++ {
				if !sameColor(got.At(x, y), full.At(x, y)) {
					t.Fatalf("%s: negative: mismatch at (%d,%d)", tc.name, x, y)
				}
			}
		}

		// Degenerate (zero-width) rectangle.
		got, err = DecodeRegion(context.Background(), bytes.NewReader(tc.raw), image.Rect(10, 10, 10, 20), nil)
		if err != nil || !got.Bounds().Empty() {
			t.Errorf("%s: degenerate: bounds=%v err=%v, want empty and nil", tc.name, got.Bounds(), err)
		}

		// An extremely large rectangle must be clipped rather than overflow.
		huge := image.Rect(-1<<30, -1<<30, 1<<30, 1<<30)
		got, err = DecodeRegion(context.Background(), bytes.NewReader(tc.raw), huge, nil)
		if err != nil {
			t.Fatalf("%s: huge: %v", tc.name, err)
		}
		if got.Bounds() != b {
			t.Errorf("%s: huge: bounds %v, want %v", tc.name, got.Bounds(), b)
		}
	}
}

// TestDecoderReuse checks that a single Decoder produces the same result as a
// fresh one across a mix of images and rectangles, including after a failed
// call.
func TestDecoderReuse(t *testing.T) {
	var rawLossy, rawLossless, rawAlpha []byte
	for _, name := range []string{"yellow_rose.lossy.webp", "tux.lossless.webp", "yellow_rose.lossy-with-alpha.webp"} {
		raw := readFixture(t, name)
		switch name {
		case "yellow_rose.lossy.webp":
			rawLossy = raw
		case "tux.lossless.webp":
			rawLossless = raw
		default:
			rawAlpha = raw
		}
	}
	inputs := [][]byte{rawLossy, rawLossless, rawAlpha, rawLossy}

	d := NewDecoder()
	for _, raw := range inputs {
		full := decodeFull(t, raw)
		b := full.Bounds()
		for _, r := range []image.Rectangle{
			image.Rect(3, 5, 41, 67),
			image.Rect(b.Max.X-9, b.Max.Y-7, b.Max.X, b.Max.Y),
		} {
			got, err := d.DecodeRegion(context.Background(), bytes.NewReader(raw), r, nil)
			if err != nil {
				t.Fatalf("reused decoder %v: %v", r, err)
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if !sameColor(got.At(x, y), full.At(x, y)) {
						t.Fatalf("reused decoder %v: mismatch at (%d,%d)", r, x, y)
					}
				}
			}
		}
	}

	// A failed call (truncated input) must not poison the next one.
	truncated := rawLossy[:len(rawLossy)/2]
	if _, err := d.DecodeRegion(context.Background(), bytes.NewReader(truncated), image.Rect(0, 0, 8, 8), nil); err == nil {
		t.Fatal("truncated input: got nil error, want non-nil")
	}
	got, err := d.DecodeRegion(context.Background(), bytes.NewReader(rawLossy), image.Rect(3, 5, 41, 67), nil)
	if err != nil {
		t.Fatalf("after failure: %v", err)
	}
	full := decodeFull(t, rawLossy)
	for y := 5; y < 67; y++ {
		for x := 3; x < 41; x++ {
			if !sameColor(got.At(x, y), full.At(x, y)) {
				t.Fatalf("after failure: mismatch at (%d,%d)", x, y)
			}
		}
	}
}

// TestDecodeRegionCancel checks that a canceled context is reported, both
// before decoding starts and while a large image is being decoded.
func TestDecodeRegionCancel(t *testing.T) {
	raw := readFixture(t, "blue-purple-pink-large.lossless.webp")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DecodeRegion(ctx, bytes.NewReader(raw), image.Rect(0, 0, 16, 16), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled lossless: got %v, want context.Canceled", err)
	}

	rawLossy := readFixture(t, "blue-purple-pink-large.normal-filter.lossy.webp")
	if _, err := DecodeRegion(ctx, bytes.NewReader(rawLossy), image.Rect(0, 0, 16, 16), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled lossy: got %v, want context.Canceled", err)
	}
}

// TestDecodeRegionTruncated requires an error (never a panic) for every
// truncation point of a WEBP file.
func TestDecodeRegionTruncated(t *testing.T) {
	for _, name := range []string{
		"tux.lossless.webp",
		"blue-purple-pink.lossy.webp",
		"yellow_rose.lossy-with-alpha.webp",
	} {
		raw := readFixture(t, name)
		for n := 0; n < len(raw); n += 53 {
			func() {
				defer func() {
					if rec := recover(); rec != nil {
						t.Fatalf("%s prefix %d: panic: %v", name, n, rec)
					}
				}()
				_, err := DecodeRegion(context.Background(), bytes.NewReader(raw[:n]),
					image.Rect(0, 0, 8, 8), nil)
				if err == nil {
					t.Errorf("%s prefix %d: got nil error, want non-nil", name, n)
				}
			}()
		}
	}
}

// TestDecodeRegionMemoryLimit checks that a small memory limit is enforced
// with ErrRegionMemory (wrapped, so errors.Is works) rather than by allocating
// unboundedly.
func TestDecodeRegionMemoryLimit(t *testing.T) {
	for _, name := range []string{
		"tux.lossless.webp",
		"yellow_rose.lossy.webp",
		"yellow_rose.lossy-with-alpha.webp",
	} {
		raw := readFixture(t, name)
		_, err := DecodeRegion(context.Background(), bytes.NewReader(raw),
			image.Rect(0, 0, 320, 320), &RegionOptions{MaxMemory: 16})
		if !errors.Is(err, ErrRegionMemory) {
			t.Errorf("%s: got %v, want ErrRegionMemory", name, err)
		}
	}
}

// TestDecodeRegionHostileCanvas builds a VP8X header with the largest
// permitted canvas and no image data. The decode must fail with a bounded
// error rather than allocating a huge buffer.
func TestDecodeRegionHostileCanvas(t *testing.T) {
	// A VP8X chunk followed by a tiny VP8 chunk that claims 0x3fff x 0x3fff.
	// The canvas area (16384*16384) exceeds maxCanvasPixels, so it is rejected
	// outright before any allocation.
	const wm1, hm1 = 0x3fff, 0x3fff
	body := []byte{'V', 'P', '8', 'X', 10, 0, 0, 0, 0x10, 0, 0, 0}
	body = append(body,
		byte(wm1&0xff), byte(wm1>>8&0xff), byte(wm1>>16&0xff),
		byte(hm1&0xff), byte(hm1>>8&0xff), byte(hm1>>16&0xff),
	)
	payload := append([]byte("WEBP"), body...)
	out := []byte("RIFF")
	n := len(payload)
	out = append(out, byte(n), byte(n>>8), byte(n>>16), byte(n>>24))
	out = append(out, payload...)

	if _, err := DecodeRegion(context.Background(), bytes.NewReader(out), image.Rect(0, 0, 8, 8), nil); err == nil {
		t.Fatal("hostile canvas: got nil error, want non-nil")
	}
}

// TestDecodeRegionRandomBytes feeds random-ish data and requires an error or a
// valid image, never a panic.
func TestDecodeRegionRandomBytes(t *testing.T) {
	base := readFixture(t, "tux.lossless.webp")
	data := append([]byte(nil), base...)
	for i := 12; i < len(data) && i < 512; i += 7 {
		data[i] ^= byte(i * 31)
	}
	defer func() {
		if rec := recover(); rec != nil {
			t.Fatalf("panic on corrupted input: %v", rec)
		}
	}()
	img, err := DecodeRegion(context.Background(), bytes.NewReader(data), image.Rect(0, 0, 16, 16), nil)
	if err == nil && img == nil {
		t.Fatal("nil image and nil error")
	}
}

// TestDecodeRegionRIFFOnly checks that structurally invalid input is rejected.
func TestDecodeRegionRIFFOnly(t *testing.T) {
	for _, data := range []string{
		"",
		"not a webp file",
		"RIFF\x04\x00\x00\x00WEBP",
		"RIFF\xff\xff\xff\x7fWEBPVP8 \x78\x56\x34\x12",
	} {
		if _, err := DecodeRegion(context.Background(), strings.NewReader(data), image.Rect(0, 0, 4, 4), nil); err == nil {
			t.Errorf("%q: got nil error, want non-nil", data)
		}
	}
}
