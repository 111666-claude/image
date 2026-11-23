// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"math/rand"
	"testing"
)

// TestPredictorRowEquivalentToFull builds random predictor-transform tile
// parameters and residual images, reconstructs them both with the original
// whole-array inversePredictor and with the streaming per-row
// inversePredictorRow, and requires identical output. Random tile modes make
// the right-edge handling of TR-based modes (3, 5, 9, 10) part of the check.
func TestPredictorRowEquivalentToFull(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, bits := range []uint32{0, 1, 2, 3, 4} {
		for _, dims := range [][2]int32{
			{1, 1}, {3, 2}, {7, 5}, {13, 11}, {17, 9}, {32, 16}, {33, 17},
		} {
			w, h := dims[0], dims[1]
			tileBytes := nTiles(w, bits) * nTiles(h, bits) * 4
			tilePix := make([]byte, tileBytes)
			for i := range tilePix {
				tilePix[i] = byte(rng.Intn(256))
			}
			// A valid VP8L stream never places a TR-based predictor mode
			// (3, 5, 9, 10) in a tile touching the image's right edge, since
			// the top-right neighbor does not exist there. Sanitize those
			// tiles to mode 1 (L), which the streaming and whole-array
			// implementations handle identically without a border read.
			// Other tiles keep all 16 mode values, including the otherwise
			// unused modes 14/15, whose residual-pass-through behavior must
			// also match.
			tile := int32(1) << bits
			tilesPerRow := nTiles(w, bits)
			for ty := int32(0); ty < nTiles(h, bits); ty++ {
				for tx := int32(0); tx < tilesPerRow; tx++ {
					if (tx+1)*tile >= w {
						idx := 4 * (ty*tilesPerRow + tx)
						tilePix[idx+1] = (tilePix[idx+1] & 0xf0) | 0x01
					}
				}
			}
			tr := transform{
				transformType: transformTypePredictor,
				oldWidth:      w,
				bits:          bits,
				pix:           tilePix,
			}

			residual := make([]byte, 4*w*h)
			for i := range residual {
				residual[i] = byte(rng.Intn(256))
			}

			// Reference: the original whole-array implementation.
			want := make([]byte, len(residual))
			copy(want, residual)
			inversePredictor(&tr, want, h)

			// Streaming: feed one row at a time.
			p := &regionPipeline{
				w:   w,
				h:   h,
				acc: &memAccount{limit: 1 << 30},
			}
			got := make([]byte, 0, len(residual))
			for y := int32(0); y < h; y++ {
				row := make([]byte, 4*w)
				copy(row, residual[y*4*w:(y+1)*4*w])
				out, err := p.inversePredictorRow(&tr, row, y, int(w))
				if err != nil {
					t.Fatalf("bits=%d w=%d h=%d y=%d: %v", bits, w, h, y, err)
				}
				got = append(got, out...)
			}
			if !bytesEqual(got, want) {
				t.Fatalf("bits=%d w=%d h=%d: streaming predictor differs from full inverse", bits, w, h)
			}
		}
	}
}

// TestCrossColorRowEquivalentToFull does the same equivalence check for the
// cross-color transform, whose tile coefficients are indexed per tile row.
func TestCrossColorRowEquivalentToFull(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, bits := range []uint32{0, 2, 4} {
		w, h := int32(19), int32(13)
		tilePix := make([]byte, nTiles(w, bits)*nTiles(h, bits)*4)
		for i := range tilePix {
			tilePix[i] = byte(rng.Intn(256))
		}
		tr := transform{
			transformType: transformTypeCrossColor,
			oldWidth:      w,
			bits:          bits,
			pix:           tilePix,
		}
		residual := make([]byte, 4*w*h)
		for i := range residual {
			residual[i] = byte(rng.Intn(256))
		}
		want := make([]byte, len(residual))
		copy(want, residual)
		inverseCrossColor(&tr, want, h)

		got := make([]byte, 0, len(residual))
		for y := int32(0); y < h; y++ {
			row := make([]byte, 4*w)
			copy(row, residual[y*4*w:(y+1)*4*w])
			inverseCrossColorRow(&tr, row, y, int(w))
			got = append(got, row...)
		}
		if !bytesEqual(got, want) {
			t.Fatalf("bits=%d: streaming cross-color differs from full inverse", bits)
		}
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
