// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import (
	"image"
	"math"
	"testing"
)

// quadPoint evaluates the quadratic Bézier (a, b, c) at parameter t, in
// float64, as an analytic reference for the float32 flattening code.
func quadPoint(t, ax, ay, bx, by, cx, cy float64) (x, y float64) {
	u := 1 - t
	return u*u*ax + 2*u*t*bx + t*t*cx, u*u*ay + 2*u*t*by + t*t*cy
}

// cubePoint evaluates the cubic Bézier (a, b, c, d) at parameter t.
func cubePoint(t, ax, ay, bx, by, cx, cy, dx, dy float64) (x, y float64) {
	u := 1 - t
	return u*u*u*ax + 3*u*u*t*bx + 3*u*t*t*cx + t*t*t*dx,
		u*u*u*ay + 3*u*u*t*by + 3*u*t*t*cy + t*t*t*dy
}

// pointSegmentDistance returns the distance from p to the segment ab.
func pointSegmentDistance(px, py, ax, ay, bx, by float64) float64 {
	dx, dy := bx-ax, by-ay
	t := 0.0
	if l2 := dx*dx + dy*dy; l2 > 0 {
		t = ((px-ax)*dx + (py-ay)*dy) / l2
		t = math.Max(0, math.Min(1, t))
	}
	return math.Hypot(px-(ax+t*dx), py-(ay+t*dy))
}

// hausdorff approximates the Hausdorff distance between a densely sampled
// curve and a polyline, in both directions.
func hausdorff(samples, poly [][2]float64) float64 {
	maxD := 0.0
	// Curve samples to polyline segments.
	for _, s := range samples {
		d := math.Inf(+1)
		for i := 0; i+1 < len(poly); i++ {
			d = math.Min(d, pointSegmentDistance(s[0], s[1],
				poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1]))
		}
		maxD = math.Max(maxD, d)
	}
	// Polyline vertices to (densely sampled) curve.
	for _, p := range poly {
		d := math.Inf(+1)
		for i := 0; i+1 < len(samples); i++ {
			d = math.Min(d, pointSegmentDistance(p[0], p[1],
				samples[i][0], samples[i][1], samples[i+1][0], samples[i+1][1]))
		}
		maxD = math.Max(maxD, d)
	}
	return maxD
}

// flattenQuadPolyline reproduces, point for point, the polyline that
// Rasterizer.QuadTo emits for the given curve and tolerance.
func flattenQuadPolyline(z *Rasterizer, ax, ay, bx, by, cx, cy float32) [][2]float64 {
	devsq := devSquared(ax, ay, bx, by, cx, cy)
	n := z.flattenSegments(devsq)
	poly := [][2]float64{{float64(ax), float64(ay)}}
	if n > 1 {
		t, nInv := float32(0), 1/float32(n)
		for i := 0; i < n-1; i++ {
			t += nInv
			abx, aby := lerp(t, ax, ay, bx, by)
			bcx, bcy := lerp(t, bx, by, cx, cy)
			x, y := lerp(t, abx, aby, bcx, bcy)
			poly = append(poly, [2]float64{float64(x), float64(y)})
		}
	}
	return append(poly, [2]float64{float64(cx), float64(cy)})
}

// flattenCubePolyline reproduces, point for point, the polyline that
// Rasterizer.CubeTo emits for the given curve and tolerance.
func flattenCubePolyline(z *Rasterizer, ax, ay, bx, by, cx, cy, dx, dy float32) [][2]float64 {
	devsq := devSquared(ax, ay, bx, by, dx, dy)
	if devsqAlt := devSquared(ax, ay, cx, cy, dx, dy); devsq < devsqAlt {
		devsq = devsqAlt
	}
	n := z.flattenSegments(devsq)
	poly := [][2]float64{{float64(ax), float64(ay)}}
	if n > 1 {
		t, nInv := float32(0), 1/float32(n)
		for i := 0; i < n-1; i++ {
			t += nInv
			abx, aby := lerp(t, ax, ay, bx, by)
			bcx, bcy := lerp(t, bx, by, cx, cy)
			cdx, cdy := lerp(t, cx, cy, dx, dy)
			abcx, abcy := lerp(t, abx, aby, bcx, bcy)
			bcdx, bcdy := lerp(t, bcx, bcy, cdx, cdy)
			x, y := lerp(t, abcx, abcy, bcdx, bcdy)
			poly = append(poly, [2]float64{float64(x), float64(y)})
		}
	}
	return append(poly, [2]float64{float64(dx), float64(dy)})
}

func denseQuad(ax, ay, bx, by, cx, cy float32, n int) [][2]float64 {
	s := make([][2]float64, n+1)
	fax, fay, fbx, fby, fcx, fcy := float64(ax), float64(ay), float64(bx), float64(by), float64(cx), float64(cy)
	for i := range s {
		x, y := quadPoint(float64(i)/float64(n), fax, fay, fbx, fby, fcx, fcy)
		s[i] = [2]float64{x, y}
	}
	return s
}

func denseCube(ax, ay, bx, by, cx, cy, dx, dy float32, n int) [][2]float64 {
	s := make([][2]float64, n+1)
	fax, fay := float64(ax), float64(ay)
	fbx, fby := float64(bx), float64(by)
	fcx, fcy := float64(cx), float64(cy)
	fdx, fdy := float64(dx), float64(dy)
	for i := range s {
		x, y := cubePoint(float64(i)/float64(n), fax, fay, fbx, fby, fcx, fcy, fdx, fdy)
		s[i] = [2]float64{x, y}
	}
	return s
}

// TestFlattenQuadHausdorff checks, against an analytic Bézier reference,
// that flattening a quadratic Bézier stays within the tolerance-derived
// error bound |D|/(4n²) ≤ τ.
func TestFlattenQuadHausdorff(t *testing.T) {
	quads := [][6]float32{
		{10, 10, 50, 90, 90, 10},    // gentle arc
		{0, 0, 200, 0, 0, 20},       // sharp turn
		{0, 0, 0.5, 0.4, 1, 0},      // tiny, below threshold at large τ
		{100, 0, 100, 100, 0, 100},  // quarter-circle-like
		{0, 0, 1000, 1000, 2000, 0}, // large, high curvature
	}
	tolerances := []float32{4, 1, 0.5, 0.25, DefaultFlattenTolerance, 0.05}
	for _, q := range quads {
		ax, ay, bx, by, cx, cy := q[0], q[1], q[2], q[3], q[4], q[5]
		devsq := devSquared(ax, ay, bx, by, cx, cy)
		samples := denseQuad(ax, ay, bx, by, cx, cy, 4096)
		for _, tau := range tolerances {
			z := &Rasterizer{FlattenTolerance: tau}
			n := z.flattenSegments(devsq)
			poly := flattenQuadPolyline(z, ax, ay, bx, by, cx, cy)
			if got := len(poly) - 1; got != n {
				t.Errorf("q=%v τ=%v: polyline has %d segments, flattenSegments says %d", q, tau, got, n)
			}
			h := hausdorff(samples, poly)
			// The analytic bound for a quadratic split into n
			// equal-parameter segments is |D|/(4n²), which the segment
			// count formula keeps at or below τ.
			bound := math.Sqrt(float64(devsq)) / (4 * float64(n) * float64(n))
			if h > bound*1.02+0.01 {
				t.Errorf("q=%v τ=%v n=%d: Hausdorff %v exceeds analytic bound %v", q, tau, n, h, bound)
			}
			if devsq >= 1 && h > float64(tau)*1.02+0.01 {
				t.Errorf("q=%v τ=%v n=%d: Hausdorff %v exceeds tolerance %v", q, tau, n, h, tau)
			}
		}
	}
}

// TestFlattenCubeHausdorff checks cubic flattening against the analytic
// reference. The cubic flatness measure is heuristic (each interior control
// point is treated as the control of a quadratic over the same chord), so
// the bound is |D|/(2n) rather than a strict τ guarantee.
func TestFlattenCubeHausdorff(t *testing.T) {
	cubes := [][8]float32{
		{0, 50, 80, 150, 20, -50, 100, 50}, // S-shape
		{0, 0, 30, 60, 70, 60, 100, 0},     // arc-like
		{14, 14, 8, 2, 5, 20, 2, 8},        // the CubeTo of TestBasicPath
		{0, 0, 0, 100, 100, 100, 100, 0},   // symmetric bulge
	}
	tolerances := []float32{4, 1, 0.5, 0.25, DefaultFlattenTolerance, 0.05}
	for _, c := range cubes {
		ax, ay, bx, by := c[0], c[1], c[2], c[3]
		cx, cy, dx, dy := c[4], c[5], c[6], c[7]
		devsq := devSquared(ax, ay, bx, by, dx, dy)
		if alt := devSquared(ax, ay, cx, cy, dx, dy); devsq < alt {
			devsq = alt
		}
		samples := denseCube(ax, ay, bx, by, cx, cy, dx, dy, 4096)
		for _, tau := range tolerances {
			z := &Rasterizer{FlattenTolerance: tau}
			n := z.flattenSegments(devsq)
			poly := flattenCubePolyline(z, ax, ay, bx, by, cx, cy, dx, dy)
			h := hausdorff(samples, poly)
			bound := math.Sqrt(float64(devsq)) / (2 * float64(n))
			if h > bound*1.02+0.01 {
				t.Errorf("c=%v τ=%v n=%d: Hausdorff %v exceeds analytic bound %v", c, tau, n, h, bound)
			}
		}
	}
}

// TestFlattenSegmentsPointBudget checks that the segment count is
// deterministic and bounded for every degenerate input: zero, negative and
// non-finite tolerances, extreme curvature, and out-of-range coordinates.
func TestFlattenSegmentsPointBudget(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(+1))

	// Degenerate devsq values always yield a single segment.
	z := &Rasterizer{}
	for _, devsq := range []float32{0, -1, nan, inf, -inf, 0.1, 0.333 - 1e-6} {
		if n := z.flattenSegments(devsq); n != 1 {
			t.Errorf("devsq=%v: got %d segments, want 1", devsq, n)
		}
	}

	// Non-positive or non-finite tolerances fall back to the default, which
	// matches the historical formula for this devsq (180 → 5 segments).
	const devsq = 180
	const want = 5
	for _, tau := range []float32{0, -1, -inf, nan, inf} {
		z := &Rasterizer{FlattenTolerance: tau}
		if n := z.flattenSegments(devsq); n != want {
			t.Errorf("τ=%v: got %d segments, want default %d", tau, n, want)
		}
	}

	// Smaller tolerances subdivide more; larger ones subdivide less.
	prev := 1
	for _, tau := range []float32{8, 2, 1, 0.5, 0.25, 0.1, 0.05} {
		z := &Rasterizer{FlattenTolerance: tau}
		n := z.flattenSegments(1e4)
		if n < prev {
			t.Errorf("τ=%v: n=%d decreased below previous %d", tau, n, prev)
		}
		prev = n
	}

	// The point budget caps extreme curvature and huge coordinates.
	for _, maxN := range []int{2, 3, 17, 256, DefaultMaxFlattenPoints} {
		z := &Rasterizer{FlattenTolerance: 1e-20, MaxFlattenPoints: maxN}
		for _, devsq := range []float32{1, 1e6, 1e18, 3e38} {
			if n := z.flattenSegments(devsq); n < 1 || n > maxN {
				t.Errorf("maxN=%d devsq=%v: n=%d out of [1, %d]", maxN, devsq, n, maxN)
			}
		}
	}
	// Budget values below 2 mean the default budget.
	for _, maxN := range []int{0, 1, -100} {
		z := &Rasterizer{FlattenTolerance: 1e-20, MaxFlattenPoints: maxN}
		if n := z.flattenSegments(3e38); n != DefaultMaxFlattenPoints {
			t.Errorf("maxN=%d: got %d, want default cap %d", maxN, n, DefaultMaxFlattenPoints)
		}
	}
}

// TestFlattenSegmentsDefaultMatchesLegacy pins the zero-value Rasterizer to
// the historical subdivision formula, keeping existing golden output
// bit-for-bit compatible.
func TestFlattenSegmentsDefaultMatchesLegacy(t *testing.T) {
	z := &Rasterizer{}
	for devsq := float32(0); devsq < 1e6; devsq = devsq*1.7 + 0.013 {
		want := 1
		if devsq >= 0.333 {
			want = 1 + int(math.Sqrt(math.Sqrt(3*float64(devsq))))
		}
		if got := z.flattenSegments(devsq); got != want {
			t.Fatalf("devsq=%v: got %d, want legacy %d", devsq, got, want)
		}
	}
}

// TestFlattenDegenerateGeometry checks that pathological coordinates
// terminate promptly, move the pen to the segment end point, and do not
// panic during rasterization.
func TestFlattenDegenerateGeometry(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(+1))

	testCases := []struct {
		bx, by, cx, cy, dx, dy float32
	}{
		{nan, 0, 1, 1, 2, 2},
		{inf, 0, 1, 1, 2, 2},
		{0, 0, -inf, inf, 2, 2},
		{1e30, 1e30, -1e30, 1e30, 2, 2},
		{1e19, 0, 0, 1e19, 2, 2},
		{math.MaxFloat32, 0, 0, math.MaxFloat32, 2, 2},
	}
	for _, tc := range testCases {
		z := NewRasterizer(8, 8)
		// A small budget keeps the huge-but-finite cases fast; the count
		// itself is checked in TestFlattenSegmentsPointBudget.
		z.MaxFlattenPoints = 4
		z.MoveTo(1, 1)
		z.QuadTo(tc.bx, tc.by, tc.dx, tc.dy)
		z.CubeTo(tc.bx, tc.by, tc.cx, tc.cy, tc.dx, tc.dy)
		px, py := z.Pen()
		z.ClosePath()
		dst := image.NewAlpha(z.Bounds())
		z.Draw(dst, z.Bounds(), image.Opaque, image.Point{})
		if isnan32(tc.dx) {
			if !isnan32(px) || !isnan32(py) {
				t.Errorf("%v: pen=(%v,%v), want NaN", tc, px, py)
			}
		} else if px != tc.dx || py != tc.dy {
			t.Errorf("%v: pen=(%v,%v), want (%v,%v)", tc, px, py, tc.dx, tc.dy)
		}
	}
}

func isnan32(f float32) bool { return f != f }

// TestFlattenDeterminismAndReset checks that rasterizing the same input
// twice gives identical output, and that Reset discards any custom
// flattening configuration along with the path state.
func TestFlattenDeterminismAndReset(t *testing.T) {
	draw := func(z *Rasterizer) *image.Alpha {
		z.MoveTo(8, 8)
		z.QuadTo(60, 4, 40, 50)
		z.CubeTo(30, 70, 10, 60, 8, 8)
		z.ClosePath()
		dst := image.NewAlpha(z.Bounds())
		z.Draw(dst, z.Bounds(), image.Opaque, image.Point{})
		return dst
	}
	same := func(a, b *image.Alpha) bool {
		if len(a.Pix) != len(b.Pix) {
			return false
		}
		for i := range a.Pix {
			if a.Pix[i] != b.Pix[i] {
				return false
			}
		}
		return true
	}

	fresh := draw(NewRasterizer(64, 64))
	if again := draw(NewRasterizer(64, 64)); !same(fresh, again) {
		t.Error("two fresh Rasterizers gave different output")
	}

	z := NewRasterizer(64, 64)
	z.FlattenTolerance = 2
	z.MaxFlattenPoints = 3
	coarse := draw(z)
	if same(fresh, coarse) {
		t.Error("coarse flattening unexpectedly matched the default output")
	}

	// After Reset, the custom tolerance and budget are gone and the output
	// matches a fresh Rasterizer again.
	z.Reset(64, 64)
	if z.FlattenTolerance != 0 || z.MaxFlattenPoints != 0 {
		t.Error("Reset did not clear the flattening configuration")
	}
	if reset := draw(z); !same(fresh, reset) {
		t.Error("output after Reset differs from a fresh Rasterizer")
	}
}

// TestFlattenLinesAndClosedPathDirection checks that LineTo and ClosePath
// are independent of the flattening settings and that reversing a closed
// path's direction does not change its coverage.
func TestFlattenLinesAndClosedPathDirection(t *testing.T) {
	points := [][2]float32{{4, 4}, {60, 10}, {30, 55}, {10, 40}}
	render := func(tau float32, reverse bool) *image.Alpha {
		z := NewRasterizer(64, 64)
		z.FlattenTolerance = tau
		n := len(points)
		for i := 0; i <= n; i++ {
			p := points[i%n]
			if reverse {
				p = points[(n-i)%n]
			}
			if i == 0 {
				z.MoveTo(p[0], p[1])
			} else if i == n {
				z.ClosePath()
			} else {
				z.LineTo(p[0], p[1])
			}
		}
		dst := image.NewAlpha(z.Bounds())
		z.Draw(dst, z.Bounds(), image.Opaque, image.Point{})
		return dst
	}
	base := render(0, false)
	for _, m := range []*image.Alpha{render(0.01, false), render(100, false), render(0, true)} {
		for i := range base.Pix {
			if base.Pix[i] != m.Pix[i] {
				t.Fatalf("line-only masks differ at pixel %d", i)
			}
		}
	}
}

// TestFlattenEdgePixels compares rendered edge coverage against the analytic
// position of a quadratic Bézier edge.
//
// The path is the closed sliver between the chord x=32 and the curve from
// (32, 4) via (56, 32) to (32, 60). For this curve y(t) = 4 + 56t is linear
// and x(t) = 32 + 48t(1-t), so the expected coverage of each pixel row is
// known in closed form.
func TestFlattenEdgePixels(t *testing.T) {
	for _, tau := range []float32{DefaultFlattenTolerance, 0.5, 0.05} {
		z := NewRasterizer(64, 64)
		z.FlattenTolerance = tau
		z.MoveTo(32, 4)
		z.QuadTo(56, 32, 32, 60)
		z.ClosePath()
		dst := image.NewAlpha(z.Bounds())
		z.Draw(dst, z.Bounds(), image.Opaque, image.Point{})

		for y := 6; y < 58; y++ {
			sum := 0
			for x := 0; x < 64; x++ {
				sum += int(dst.Pix[y*dst.Stride+x])
			}
			// t at the row's vertical center; the curve is the right
			// boundary of the filled sliver.
			tt := (float64(y) + 0.5 - 4) / 56
			want := 48 * tt * (1 - tt)
			got := float64(sum) / 0xff
			// The edge is within τ of the true curve by construction;
			// allow one more pixel for rasterization rounding.
			if math.Abs(got-want) > float64(tau)+1 {
				t.Errorf("τ=%v y=%d: row coverage %.2f, want %.2f ± %.2f",
					tau, y, got, want, float64(tau)+1)
			}
		}
	}
}
