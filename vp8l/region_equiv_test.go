// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestPredictorRowEquivalentToFull builds random predictor tiles and residual
// images, reconstructs them with the whole-array inversePredictor and with the
// row-streaming inversePredictorRow, and requires identical bytes.
//
// It deliberately includes TR-based predictor modes (3, 5, 9 and 10) in the
// rightmost tile. A conforming encoder never emits those, because the top-right
// neighbour does not exist there, but the whole-array decoder still evaluates
// them: its contiguous row-major buffer makes the last column's top-right read
// alias the first pixel of the current row. A region decoder that must match
// Decode pixel for pixel has to reproduce that aliasing rather than substitute
// the top neighbour or the edge pixel.
func TestPredictorRowEquivalentToFull(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, bits := range []uint32{0, 1, 2, 3, 4} {
		for _, dims := range [][2]int32{
			{1, 1}, {3, 2}, {7, 5}, {13, 11}, {17, 9}, {32, 16}, {33, 17},
		} {
			w, h := dims[0], dims[1]
			tilePix := make([]byte, nTiles(w, bits)*nTiles(h, bits)*4)
			for i := range tilePix {
				tilePix[i] = byte(rng.Intn(256))
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

			// Reference: the whole-array implementation used by Decode.
			want := make([]byte, len(residual))
			copy(want, residual)
			inversePredictor(&tr, want, h)

			// Streaming: feed one row at a time.
			p := &regionPipeline{
				w:   w,
				h:   h,
				acc: &memAccount{limit: 1 << 30},
			}
			var got []byte
			for y := int32(0); y < h; y++ {
				row := make([]byte, 4*w)
				copy(row, residual[y*4*w:(y+1)*4*w])
				out, err := p.inversePredictorRow(&tr, row, y, int(w))
				if err != nil {
					t.Fatalf("bits=%d w=%d h=%d y=%d: %v", bits, w, h, y, err)
				}
				got = append(got, out...)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("bits=%d w=%d h=%d: streaming predictor differs from full inverse", bits, w, h)
			}
		}
	}
}

// TestCrossColorRowEquivalentToFull does the same for the cross-color
// transform, whose tile coefficients are selected per tile row.
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

		var got []byte
		for y := int32(0); y < h; y++ {
			row := make([]byte, 4*w)
			copy(row, residual[y*4*w:(y+1)*4*w])
			inverseCrossColorRow(&tr, row, y, int(w))
			got = append(got, row...)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("bits=%d: streaming cross-color differs from full inverse", bits)
		}
	}
}

// TestColorIndexingRowEquivalentToFull checks the row expansion of packed
// color indices, including widths that are not a multiple of the packing.
func TestColorIndexingRowEquivalentToFull(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	palette := make([]byte, 4*256)
	for i := range palette {
		palette[i] = byte(rng.Intn(256))
	}
	for _, bits := range []uint32{0, 1, 2, 3} {
		for _, w := range []int32{1, 3, 5, 8, 9, 17} {
			h := int32(3)
			tr := transform{
				transformType: transformTypeColorIndexing,
				oldWidth:      w,
				bits:          bits,
				pix:           palette,
			}
			inW := w
			if bits != 0 {
				inW = nTiles(w, bits)
			}
			residual := make([]byte, 4*inW*h)
			for i := range residual {
				residual[i] = byte(rng.Intn(256))
			}
			want := inverseColorIndexing(&tr, append([]byte(nil), residual...), h)

			var got []byte
			for y := int32(0); y < h; y++ {
				in := make([]byte, 4*inW)
				copy(in, residual[y*4*inW:(y+1)*4*inW])
				out := make([]byte, 4*w)
				inverseColorIndexingRow(&tr, in, out, int(inW))
				got = append(got, out...)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("bits=%d w=%d: streaming color indexing differs from full inverse", bits, w)
			}
		}
	}
}

// TestSubtractGreenRowEquivalent checks the row-local subtract-green inverse
// against the whole-array version.
func TestSubtractGreenRowEquivalent(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	w, h := int32(11), int32(5)
	residual := make([]byte, 4*w*h)
	for i := range residual {
		residual[i] = byte(rng.Intn(256))
	}
	want := inverseSubtractGreen(&transform{}, append([]byte(nil), residual...), h)
	var got []byte
	for y := int32(0); y < h; y++ {
		row := append([]byte(nil), residual[y*4*w:(y+1)*4*w]...)
		inverseSubtractGreenRow(row)
		got = append(got, row...)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("streaming subtract-green differs from full inverse")
	}
}
