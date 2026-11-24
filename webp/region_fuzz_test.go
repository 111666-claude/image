// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp_test

import (
	"bytes"
	"context"
	"image"
	"testing"

	"golang.org/x/image/webp"
)

// FuzzDecodeRegion feeds arbitrary bytes to webp.DecodeRegion. The decoder
// must return either an image or an error, never panic, and must stay within
// the memory limit even for hostile canvas or dimension claims.
func FuzzDecodeRegion(f *testing.F) {
	f.Add([]byte("RIFF\x04\x00\x00\x00WEBP"))
	f.Add([]byte("RIFF\x00\x00\x00\x00WEBPVP8L\x05\x00\x00\x00\x2f\x00\x00\x00\x00"))
	f.Add(bytes.Repeat([]byte{0}, 64))
	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if rec := recover(); rec != nil {
				t.Fatalf("panic on input %x: %v", data, rec)
			}
		}()
		img, err := webp.DecodeRegion(context.Background(), bytes.NewReader(data),
			image.Rect(0, 0, 16, 16), &webp.RegionOptions{MaxMemory: 1 << 20})
		if err != nil {
			return
		}
		if img == nil {
			t.Fatal("nil image and nil error")
		}
		b := img.Bounds()
		if !b.Empty() && b.Dx()*b.Dy() > 1<<20 {
			t.Fatalf("decoded region %v exceeds the request", b)
		}
	})
}
