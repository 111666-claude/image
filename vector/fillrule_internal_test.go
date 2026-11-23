// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import (
	"image"
	"image/draw"
	"math"
	"testing"
)

// renderInternal runs one path through both the fixed and floating point
// rasterizer math implementations and returns the two alpha masks under the
// given fill rule and DrawOp. The geometry and size are identical.
func renderInternal(w, h int, build func(z *Rasterizer), rule FillRule, op draw.Op) (fixed, floating *image.Alpha) {
	mask := func(useFloat bool) *image.Alpha {
		z := NewRasterizer(w, h)
		z.setUseFloatingPointMath(useFloat)
		z.FillRule = rule
		z.DrawOp = op
		build(z)
		m := image.NewAlpha(z.Bounds())
		z.Draw(m, m.Bounds(), image.Opaque, image.Point{})
		return m
	}
	return mask(false), mask(true)
}

func starBuild(cx, cy, outer float32) func(z *Rasterizer) {
	pts := [5][2]float32{}
	for i := range pts {
		a := -math.Pi/2 + 2*math.Pi*float64(i)/5
		pts[i] = [2]float32{
			cx + outer*float32(math.Cos(a)),
			cy + outer*float32(math.Sin(a)),
		}
	}
	return func(z *Rasterizer) {
		for k := 0; k < 5; k++ {
			x, y := pts[(k*2)%5][0], pts[(k*2)%5][1]
			if k == 0 {
				z.MoveTo(x, y)
			} else {
				z.LineTo(x, y)
			}
		}
		z.ClosePath()
	}
}

func ringsBuild(cx, cy, rOuter, rInner float32) func(z *Rasterizer) {
	const n = 24
	ring := func(z *Rasterizer, rad float32, reverse bool) {
		pt := func(i int) (float32, float32) {
			a := 2 * math.Pi * float64(i) / n
			return cx + rad*float32(math.Cos(a)), cy + rad*float32(math.Sin(a))
		}
		if reverse {
			x, y := pt(0)
			z.MoveTo(x, y)
			for i := n - 1; i >= 1; i-- {
				x, y = pt(i)
				z.LineTo(x, y)
			}
		} else {
			x, y := pt(0)
			z.MoveTo(x, y)
			for i := 1; i < n; i++ {
				x, y = pt(i)
				z.LineTo(x, y)
			}
		}
		z.ClosePath()
	}
	return func(z *Rasterizer) {
		ring(z, rOuter, false)
		ring(z, rInner, false)
	}
}

// TestEvenOddFixedFloatingConsistency requires the fixed and floating point
// math implementations to produce the same even-odd coverage for a
// self-intersecting star and a same-winding ring pair, under both DrawOps.
func TestEvenOddFixedFloatingConsistency(t *testing.T) {
	cases := []struct {
		name  string
		w, h  int
		build func(z *Rasterizer)
	}{
		{"star", 48, 48, starBuild(24, 24, 22)},
		{"rings", 48, 48, ringsBuild(24, 24, 22, 11)},
	}
	for _, tc := range cases {
		for _, op := range []draw.Op{draw.Over, draw.Src} {
			fixed, floating := renderInternal(tc.w, tc.h, tc.build, EvenOdd, op)
			bad, worst := 0, 0
			for i := range fixed.Pix {
				d := int(fixed.Pix[i]) - int(floating.Pix[i])
				if d < 0 {
					d = -d
				}
				if d > worst {
					worst = d
				}
				// The two math implementations carry different rounding
				// error (see TestFixedFloatingCloseness for the non-zero
				// case); a few levels of alpha difference at edges is
				// expected, but wholesale disagreement is not.
				if d > 8 {
					bad++
				}
			}
			if maxBad := tc.w * tc.h / 25; bad > maxBad {
				t.Errorf("%s/op%v: %d pixels differ (max %d)", tc.name, op, bad, worst)
			}
		}
	}
}

// TestEvenOddRowReset checks the per-scanline running-sum reset directly:
// every row of an even-odd mask that lies entirely outside the path must be
// zero even when earlier rows have non-trivial parity.
func TestEvenOddRowReset(t *testing.T) {
	const w, h = 16, 32
	build := func(z *Rasterizer) {
		// A rectangle covering only rows 8..24.
		z.MoveTo(3, 8)
		z.LineTo(13, 8)
		z.LineTo(13, 24)
		z.LineTo(3, 24)
		z.ClosePath()
	}
	for _, useFloat := range []bool{false, true} {
		z := NewRasterizer(w, h)
		z.setUseFloatingPointMath(useFloat)
		z.FillRule = EvenOdd
		build(z)
		m := image.NewAlpha(z.Bounds())
		z.Draw(m, m.Bounds(), image.Opaque, image.Point{})
		for x := 0; x < w; x++ {
			if m.AlphaAt(x, 0).A != 0 || m.AlphaAt(x, h-1).A != 0 {
				t.Errorf("useFloat=%v: running sum leaked across rows at x=%d", useFloat, x)
			}
		}
		if m.AlphaAt(8, 16).A < 0xfe {
			t.Errorf("useFloat=%v: interior pixel got %#02x, want ~0xff", useFloat, m.AlphaAt(8, 16).A)
		}
	}
}

// TestEvenOddFoldIdentity unit-tests the fold helpers for the canonical
// winding-number values, in both math variants.
func TestEvenOddFoldIdentity(t *testing.T) {
	// Fixed point: winding k in int2ϕ units is k << (2*ϕ).
	for k := -6; k <= 6; k++ {
		w := int2ϕ(k) << (2 * ϕ)
		got := fixedFoldEvenOdd(w)
		want := int2ϕ(0)
		if k%2 != 0 {
			want = int2ϕ(fxOne * fxOne)
		}
		// Allow one unit of rounding slack.
		d := got - want
		if d < 0 {
			d = -d
		}
		if d > 1 {
			t.Errorf("fixedFoldEvenOdd(%d): got %v, want %v", k, got, want)
		}
	}

	for k := -6; k <= 6; k++ {
		got := floatingFoldEvenOdd(float32(k))
		want := float32(0)
		if k%2 != 0 {
			want = 1
		}
		if d := got - want; d < -1e-6 || d > 1e-6 {
			t.Errorf("floatingFoldEvenOdd(%d): got %v, want %v", k, got, want)
		}
	}

	// Fractional winding ramps linearly between the integer anchors.
	if got := floatingFoldEvenOdd(0.5); math.Abs(float64(got)-0.5) > 1e-6 {
		t.Errorf("floatingFoldEvenOdd(0.5): got %v, want 0.5", got)
	}
	if got := floatingFoldEvenOdd(1.5); math.Abs(float64(got)-0.5) > 1e-6 {
		t.Errorf("floatingFoldEvenOdd(1.5): got %v, want 0.5", got)
	}
	if got := fixedFoldEvenOdd(int2ϕ(1) << (2*ϕ - 1)); got != int2ϕ(fxOne*fxOne/2) {
		t.Errorf("fixedFoldEvenOdd(0.5): got %v, want %v", got, int2ϕ(fxOne*fxOne/2))
	}
}

// BenchmarkEvenOddStar measures the even-odd accumulation cost against the
// default non-zero rule on identical geometry.
func BenchmarkEvenOddStar(b *testing.B) {
	for _, rule := range []FillRule{NonZero, EvenOdd} {
		name := "nonzero"
		if rule == EvenOdd {
			name = "evenodd"
		}
		b.Run(name, func(b *testing.B) {
			build := starBuild(128, 128, 120)
			dst := image.NewAlpha(image.Rect(0, 0, 256, 256))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				z := NewRasterizer(256, 256)
				z.FillRule = rule
				build(z)
				z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
			}
		})
	}
}
