// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp

import (
	"bytes"
	"context"
	"image"
	"testing"
)

// buildVP8XAlphaFile assembles a minimal VP8X + ALPH + VP8X-less WEBP file from
// the given uncompressed alpha plane and a lossy VP8 payload.
func buildVP8XAlphaFile(t *testing.T, vp8Payload, alphaPlane []byte, w, h int, filter byte) []byte {
	t.Helper()
	var chunks []byte
	appendChunk := func(fourcc string, data []byte) {
		chunks = append(chunks, fourcc...)
		n := len(data)
		chunks = append(chunks, byte(n), byte(n>>8), byte(n>>16), byte(n>>24))
		chunks = append(chunks, data...)
		if n&1 != 0 {
			chunks = append(chunks, 0)
		}
	}

	vp8x := make([]byte, 10)
	vp8x[0] = 1 << 4 // Alpha present.
	wp, hp := w-1, h-1
	vp8x[4], vp8x[5], vp8x[6] = byte(wp), byte(wp>>8), byte(wp>>16)
	vp8x[7], vp8x[8], vp8x[9] = byte(hp), byte(hp>>8), byte(hp>>16)
	appendChunk("VP8X", vp8x)

	alph := append([]byte{(filter << 2) & 0x0c}, alphaPlane...) // Compression 0, filter in bits 2-3.
	appendChunk("ALPH", alph)
	appendChunk("VP8 ", vp8Payload)

	body := append([]byte("WEBP"), chunks...)
	out := append([]byte("RIFF"), byte(len(body)), byte(len(body)>>8), byte(len(body)>>16), byte(len(body)>>24))
	return append(out, body...)
}

// extractVP8Chunk returns the VP8 payload of a WEBP file.
func extractVP8Chunk(t *testing.T, raw []byte) []byte {
	t.Helper()
	for p := 12; p+8 <= len(raw); {
		fourcc := string(raw[p : p+4])
		n := int(raw[p+4]) | int(raw[p+5])<<8 | int(raw[p+6])<<16 | int(raw[p+7])<<24
		q := p + 8
		if fourcc == "VP8 " {
			return raw[q : q+n]
		}
		p = q + n + (n & 1)
	}
	t.Fatal("no VP8 chunk")
	return nil
}

// TestDecodeRegionUncompressedAlpha checks the uncompressed (filter 0) ALPH
// path, which the testdata fixtures do not cover, for every filter mode. It
// builds the file from the lossy yellow_rose payload and a synthetic alpha
// plane, then requires the region decode to match a full Decode.
func TestDecodeRegionUncompressedAlpha(t *testing.T) {
	raw := readFixture(t, "yellow_rose.lossy.webp")
	vp8Payload := extractVP8Chunk(t, raw)
	cfg := fullConfig(t, raw)

	w, h := cfg.Width, cfg.Height
	// A deterministic, non-repeating alpha plane so each filter mode produces
	// a different plane but a consistent one.
	plane := make([]byte, w*h)
	for i := range plane {
		plane[i] = byte(i*37 + i/7)
	}

	for filter := byte(0); filter <= 3; filter++ {
		// Use arbitrary bytes as the coded plane and take the reference alpha
		// values from the package's own whole-plane unfilterAlpha, which is
		// what full Decode applies.
		encoded := make([]byte, w*h)
		for i := range encoded {
			encoded[i] = byte(i*37 + i/7)
		}
		want := append([]byte(nil), encoded...)
		unfilterAlpha(want, w, filter)

		file := buildVP8XAlphaFile(t, vp8Payload, encoded, w, h, filter)

		full, err := Decode(bytes.NewReader(file))
		if err != nil {
			t.Fatalf("filter %d: full Decode: %v", filter, err)
		}
		fa, ok := full.(*image.NYCbCrA)
		if !ok {
			t.Fatalf("filter %d: got %T, want *image.NYCbCrA", filter, full)
		}
		// Sanity check: the reference matches full Decode's alpha plane.
		for i := range want {
			if fa.A[i] != want[i] {
				t.Fatalf("filter %d: full Decode alpha differs at %d: got %d want %d", filter, i, fa.A[i], want[i])
			}
		}

		for _, r := range regionRects(full.Bounds()) {
			got, err := DecodeRegion(context.Background(), bytes.NewReader(file), r, nil)
			if err != nil {
				t.Fatalf("filter %d %v: %v", filter, r, err)
			}
			ga, ok := got.(*image.NYCbCrA)
			if !ok {
				t.Fatalf("filter %d %v: got %T, want *image.NYCbCrA", filter, r, got)
			}
			if ga.Bounds() != r {
				t.Fatalf("filter %d: bounds %v, want %v", filter, ga.Bounds(), r)
			}
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if ga.NYCbCrAAt(x, y) != fa.NYCbCrAAt(x, y) {
						t.Fatalf("filter %d %v: mismatch at (%d,%d): got %v, want %v",
							filter, r, x, y, ga.NYCbCrAAt(x, y), fa.NYCbCrAAt(x, y))
					}
				}
			}
		}
	}
}

func fullConfig(t *testing.T, raw []byte) image.Config {
	t.Helper()
	cfg, err := DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

// forwardFilterAlpha applies the forward ALPH filter, the inverse of the
// decoder's reconstruction. Row 0 is always coded horizontally.
func forwardFilterAlpha(plane []byte, w, h int, filter byte) []byte {
	out := append([]byte(nil), plane...)
	// Filter row by row in reverse order so that the predictor reads
	// unfiltered rows.
	for y := h - 1; y >= 0; y-- {
		row := out[y*w : (y+1)*w]
		switch {
		case y == 0:
			if filter != 0 {
				for i := w - 1; i >= 1; i-- {
					row[i] -= row[i-1]
				}
			}
		case filter == 1:
			for i := w - 1; i >= 1; i-- {
				row[i] -= row[i-1]
			}
		case filter == 2:
			prev := out[(y-1)*w : y*w]
			for i := 0; i < w; i++ {
				row[i] -= prev[i]
			}
		case filter == 3:
			prev := out[(y-1)*w : y*w]
			// Reconstruct left-to-right prediction values from the already
			// reconstructed row: the decoder computes cur[i-1] after adding,
			// i.e. the original pixel. So the forward filter subtracts
			// plane[i-1] + prev[i] - prev[i-1] clamped.
			orig := append([]byte(nil), row...)
			for i := w - 1; i >= 1; i-- {
				a := int(orig[i-1])
				b := int(prev[i])
				c := int(prev[i-1])
				x := a + b - c
				if x < 0 {
					x = 0
				} else if x > 255 {
					x = 255
				}
				row[i] = orig[i] - uint8(x)
			}
			row[0] = orig[0] - prev[0]
		}
	}
	return out
}

// TestDecodeRegionFuzzCorpus feeds arbitrary bytes and requires an error or a
// valid image, never a panic, under a tight memory limit.
func TestDecodeRegionFuzzCorpus(t *testing.T) {
	base := readFixture(t, "yellow_rose.lossy-with-alpha.webp")
	ctx := context.Background()
	for i := 0; i < 64; i++ {
		data := append([]byte(nil), base...)
		if len(data) > 0 {
			for j := 12 + i; j < len(data); j += 97 {
				data[j] ^= byte(i + j)
			}
		}
		func() {
			defer func() {
				if rec := recover(); rec != nil {
					t.Fatalf("case %d: panic: %v", i, rec)
				}
			}()
			img, err := DecodeRegion(ctx, bytes.NewReader(data), image.Rect(0, 0, 16, 16),
				&RegionOptions{MaxMemory: 1 << 20})
			if err == nil && img == nil {
				t.Fatalf("case %d: nil image and nil error", i)
			}
		}()
	}
}
