// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"bytes"
	"context"
	"errors"
	"image"
	"runtime"
	"testing"
)

// TestDecodeRegionRingWrap decodes rectangles of an image larger than the
// region decoder's rolling ring buffer. The image is 1.43 megapixels, above
// the 1<<20-pixel ring, so pixel addressing wraps while decoding; every pixel
// must still equal the corresponding pixel of a full Decode. It also carries
// fully transparent regions, so alpha is exercised on the wrapped path.
func TestDecodeRegionRingWrap(t *testing.T) {
	payload := extractVP8L(t, "region-large.lossless.webp")
	full := decodeFullVP8L(t, payload)
	if int64(full.Rect.Dx())*int64(full.Rect.Dy()) <= maxRegionRingPixels {
		t.Fatalf("fixture has %d pixels, not above the %d-pixel ring",
			full.Rect.Dx()*full.Rect.Dy(), maxRegionRingPixels)
	}
	b := full.Rect
	for _, r := range []image.Rectangle{
		b,
		image.Rect(b.Max.X-200, b.Max.Y-150, b.Max.X, b.Max.Y),
		image.Rect(0, 0, b.Max.X, 1),
		image.Rect(0, b.Max.Y-1, b.Max.X, b.Max.Y),
		image.Rect(b.Max.X-1, 0, b.Max.X, b.Max.Y),
		image.Rect(b.Max.X/2, b.Max.Y/2, b.Max.X/2+64, b.Max.Y/2+64),
		image.Rect(1, 1, 2, 2),
	} {
		got, err := DecodeRegion(context.Background(), bytes.NewReader(payload), r, nil)
		if err != nil {
			t.Fatalf("%v: %v", r, err)
		}
		if got.Rect != r {
			t.Fatalf("bounds got %v, want %v", got.Rect, r)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if got.NRGBAAt(x, y) != full.NRGBAAt(x, y) {
					t.Fatalf("%v: mismatch at (%d,%d): got %v, want %v",
						r, x, y, got.NRGBAAt(x, y), full.NRGBAAt(x, y))
				}
			}
		}
	}
}

// encodeManyGroupsVP8L builds a stream whose Huffman-group meta-image
// references group index 65535, the largest possible. Decoding it must be
// rejected by the memory limit before the 65536-entry group slice is
// allocated; without the up-front charge the decoder would first allocate
// roughly 180 MiB.
func encodeManyGroupsVP8L() []byte {
	var w bitWriter
	w.write(0x2f, 8) // Magic.
	w.write(3, 14)   // Width minus one: 4 pixels.
	w.write(3, 14)   // Height minus one: 4 pixels.
	w.write(0, 1)    // hasAlpha hint.
	w.write(0, 3)    // Version.
	w.write(0, 1)    // No transforms.
	w.write(0, 1)    // No top-level color cache.
	// The top-level Huffman-group count is derived from the maximum index
	// referenced by the huffman meta-image, which is itself entropy coded.
	// hBits=6 makes the 4x4 image a single tile, and the meta-image pixel's
	// (red, green) bytes are 0xffff, i.e. group 65535.
	w.write(1, 1) // Use a huffman meta-image.
	w.write(4, 3) // hBits = 6, so the 4x4 image is one tile.
	w.write(0, 1) // No meta-image color cache. (decodePix with
	// topLevel=false does not read a useMeta bit, so the five trees follow.)
	w.writeSimpleHuffman(255) // Green: a literal 0xff.
	w.writeSimpleHuffman(255) // Red: high byte of the index.
	w.writeSimpleHuffman(0)   // Blue.
	w.writeSimpleHuffman(0)   // Alpha.
	w.writeSimpleHuffman(0)   // Distance (unused).
	// Emit the single meta-image pixel: each 8-bit tree emits one 8-bit code.
	// The decoded pixel is (R, G, B, A) = (0xff, 0xff, 0, 0); the group index
	// is R<<8 | G = 65535.
	w.write(255, 8) // Green symbol 0xff.
	w.write(255, 8) // Red symbol 0xff.
	w.write(0, 8)   // Blue symbol 0.
	w.write(0, 8)   // Alpha symbol 0.
	// Real image pixels would follow, but the group-table charge rejects the
	// stream first.
	return w.bytes()
}

// TestDecodeRegionManyGroupsMemory checks that a stream declaring the maximum
// number of Huffman groups is rejected by the memory limit rather than
// allocating the group table first. The group table for 65536 groups would be
// roughly 180 MiB, so the test measures bytes allocated during the call and
// requires it to stay well below that.
func TestDecodeRegionManyGroupsMemory(t *testing.T) {
	stream := encodeManyGroupsVP8L()
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, err := DecodeRegion(context.Background(), bytes.NewReader(stream),
		image.Rect(0, 0, 4, 4), &Options{MaxMemory: 8 << 20})
	runtime.ReadMemStats(&after)
	if !errors.Is(err, ErrMemoryLimitExceeded) {
		t.Fatalf("got %v, want ErrMemoryLimitExceeded", err)
	}
	// The up-front charge must reject the stream before the 65536-entry group
	// slice is allocated. Allow a few MiB of unrelated decoder state, but
	// nothing near the ~180 MiB group table.
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 32<<20 {
		t.Fatalf("allocated %d bytes before rejecting the stream, want < 32 MiB", alloc)
	}
}
