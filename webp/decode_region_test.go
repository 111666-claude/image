// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp

import (
	"bytes"
	"context"
	"image"
	"os"
	"testing"
)

func TestDecodeRegionLossless(t *testing.T) {
	testCases := []string{
		"blue-purple-pink",
		"blue-purple-pink-large",
		"gopher-doc.1bpp",
		"gopher-doc.2bpp",
		"gopher-doc.4bpp",
		"gopher-doc.8bpp",
		"tux",
		"yellow_rose",
	}
	for _, tc := range testCases {
		f, err := os.Open("../testdata/" + tc + ".lossless.webp")
		if err != nil {
			t.Fatal(err)
		}
		full, err := Decode(f)
		f.Close()
		if err != nil {
			t.Fatalf("%s: full Decode: %v", tc, err)
		}
		fullNRGBA := full.(*image.NRGBA)
		b := fullNRGBA.Bounds()

		// Re-open for region decode: Decode consumes the reader.
		f2, err := os.Open("../testdata/" + tc + ".lossless.webp")
		if err != nil {
			t.Fatal(err)
		}
		r := image.Rect(b.Max.X/4, b.Max.Y/4, (b.Max.X*3)/4, (b.Max.Y*3)/4)
		got, err := DecodeRegion(context.Background(), f2, r, nil)
		f2.Close()
		if err != nil {
			t.Fatalf("%s: DecodeRegion: %v", tc, err)
		}
		if got.Rect != r {
			t.Errorf("%s: bounds got %v, want %v", tc, got.Rect, r)
			continue
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if got.NRGBAAt(x, y) != fullNRGBA.NRGBAAt(x, y) {
					t.Fatalf("%s: mismatch at (%d,%d): got %v want %v",
						tc, x, y, got.NRGBAAt(x, y), fullNRGBA.NRGBAAt(x, y))
				}
			}
		}
	}
}

func TestDecodeRegionLossyRejected(t *testing.T) {
	f, err := os.Open("../testdata/blue-purple-pink.lossy.webp")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := DecodeRegion(context.Background(), f, image.Rect(0, 0, 4, 4), nil); err == nil {
		t.Fatal("lossy image: got nil error, want non-nil")
	}
}

func TestDecodeRegionEmptyAndClip(t *testing.T) {
	f, err := os.Open("../testdata/tux.lossless.webp")
	if err != nil {
		t.Fatal(err)
	}
	empty, err := DecodeRegion(context.Background(), f, image.Rect(9999, 9999, 10000, 10000), nil)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if !empty.Rect.Empty() || len(empty.Pix) != 0 {
		t.Fatalf("got rect=%v pixLen=%d, want empty", empty.Rect, len(empty.Pix))
	}
}

// TestDecodeRegionTruncatedContainer verifies that a truncated RIFF container
// surfaces an error rather than a panic, at every truncation point that still
// leaves the headers parseable.
func TestDecodeRegionTruncatedContainer(t *testing.T) {
	raw, err := os.ReadFile("../testdata/tux.lossless.webp")
	if err != nil {
		t.Fatal(err)
	}
	for _, cut := range []int{40, 100, 500, 1000, len(raw) - 1} {
		if cut >= len(raw) {
			continue
		}
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("cut=%d: panic: %v", cut, rec)
				}
			}()
			if _, err := DecodeRegion(context.Background(), bytes.NewReader(raw[:cut]),
				image.Rect(0, 0, 10, 10), nil); err == nil {
				t.Errorf("cut=%d: got nil error, want non-nil", cut)
			}
		}()
	}
}
