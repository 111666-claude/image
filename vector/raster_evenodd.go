// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

// This file contains the even-odd winding rule variants of the accumulate
// functions in raster_fixed.go and raster_floating.go.
//
// The xxxLineTo methods store, per pixel column, the change in signed
// winding area contributed by each edge segment. Those values do not depend
// on the fill rule: the running sum along a scanline is the (signed) winding
// number W of the ideal path, scaled by the pixel's partial coverage.
//
// Non-zero fill coverage is clamp(|W|) to [0, 1].
//
// Even-odd fill coverage follows the triangle wave f(W) = 1 - |mod(W, 2) - 1|
// (Skia and other 2-D libraries use the same identity): an even integer
// winding number maps to 0 and an odd integer maps to 1, while fractional
// values give the anti-aliased transition across an edge. Running sums are
// reset to zero at the start of every row (scanline), since winding numbers
// do not carry across rows.

// fixedFoldEvenOdd folds the signed, scaled winding number w (an int2ϕ value
// in units of 1<<(2*ϕ)) onto the even-odd triangle wave, returned in the
// same units. The result is always in the range [0, 1<<(2*ϕ)].
func fixedFoldEvenOdd(w int2ϕ) int2ϕ {
	// m is w modulo 2 (i.e. modulo 1<<(2*ϕ+1) in int2ϕ units). The
	// unsigned remainder is always in [0, period), and since period is at
	// most 1<<31 it is non-negative when viewed as an int2ϕ (int32).
	const period = 1 << (2*ϕ + 1)
	m := int2ϕ(uint32(w) % period)
	// f(w) = 1 - |m - 1| in winding units: m == 0 (even winding) maps to
	// 0 and m == 1 (odd winding) maps to 1, with linear ramps in between.
	one := int2ϕ(fxOne * fxOne) // 1 in int2ϕ units.
	a := m - one
	if a < 0 {
		a = -a
	}
	return one - a
}

// fixedAccumulateMaskEvenOdd is fixedAccumulateMask with the even-odd rule.
// width is the rasterizer width: buf is laid out as rows of width samples,
// and the running sum restarts on each row.
func fixedAccumulateMaskEvenOdd(buf []uint32, width int32) {
	if width <= 0 {
		return
	}
	height := int32(len(buf)) / width
	for y := int32(0); y < height; y++ {
		row := buf[y*width : (y+1)*width]
		acc := int2ϕ(0)
		for i, v := range row {
			acc += int2ϕ(v)
			a := fixedFoldEvenOdd(acc) >> (2*ϕ - 16)
			if a > 0xffff {
				a = 0xffff
			}
			row[i] = uint32(a)
		}
	}
}

// fixedAccumulateOpSrcEvenOdd is fixedAccumulateOpSrc with the even-odd rule.
func fixedAccumulateOpSrcEvenOdd(dst []uint8, src []uint32, width int32) {
	// Sanity check that len(dst) >= len(src).
	if len(dst) < len(src) || width <= 0 {
		return
	}

	height := int32(len(src)) / width
	for y := int32(0); y < height; y++ {
		row := src[y*width : (y+1)*width]
		acc := int2ϕ(0)
		for i, v := range row {
			acc += int2ϕ(v)
			a := fixedFoldEvenOdd(acc) >> (2*ϕ - 8)
			if a > 0xff {
				a = 0xff
			}
			dst[y*width+int32(i)] = uint8(a)
		}
	}
}

// fixedAccumulateOpOverEvenOdd is fixedAccumulateOpOver with the even-odd
// rule.
func fixedAccumulateOpOverEvenOdd(dst []uint8, src []uint32, width int32) {
	// Sanity check that len(dst) >= len(src).
	if len(dst) < len(src) || width <= 0 {
		return
	}

	height := int32(len(src)) / width
	for y := int32(0); y < height; y++ {
		row := src[y*width : (y+1)*width]
		acc := int2ϕ(0)
		for i, v := range row {
			acc += int2ϕ(v)
			a := fixedFoldEvenOdd(acc) >> (2*ϕ - 16)
			if a > 0xffff {
				a = 0xffff
			}
			// This algorithm comes from the standard library's image/draw
			// package.
			idx := y*width + int32(i)
			dstA := uint32(dst[idx]) * 0x101
			maskA := uint32(a)
			outA := dstA*(0xffff-maskA)/0xffff + maskA
			dst[idx] = uint8(outA >> 8)
		}
	}
}
