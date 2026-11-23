// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l_test

import (
	"bytes"
	"context"
	"image"
	"testing"

	"golang.org/x/image/vp8l"
)

// FuzzDecodeRegion feeds arbitrary bytes to DecodeRegion. The decoder must
// always return either an image or an error, never panic, and must respect
// a tight memory limit even for hostile dimension claims.
func FuzzDecodeRegion(f *testing.F) {
	// Seed with a valid VP8L header prefix and a few truncated payloads.
	f.Add([]byte{0x2f, 0x00, 0x00, 0x00, 0x00})
	f.Add([]byte{0x2f, 0xff, 0xff, 0xff, 0xff})
	f.Add(make([]byte, 64))

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if rec := recover(); rec != nil {
				t.Fatalf("panic on input % x: %v", data, rec)
			}
		}()
		// A 1 MiB cap is far below what a 16384x16384 header would demand.
		img, err := vp8l.DecodeRegion(context.Background(), bytes.NewReader(data),
			image.Rect(0, 0, 8, 8), &vp8l.Options{MaxMemory: 1 << 20})
		if err != nil {
			return
		}
		if img == nil {
			t.Fatal("nil image and nil error")
		}
		// Every in-bounds pixel access must be safe and the Pix tightly
		// covers the rectangle.
		b := img.Bounds()
		if !b.Empty() && len(img.Pix) < b.Dx()*b.Dy()*4 {
			t.Fatalf("Pix too small: %d bytes for %v", len(img.Pix), b)
		}
	})
}
