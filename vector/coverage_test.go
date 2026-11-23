// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import (
	"context"
	"image"
	"image/color"
	"image/draw"
	"math"
	"sync"
	"testing"
	"time"
)

// This file tests the coverage / alpha-mask export (CoverageBuffer and
// Rasterizer.Coverage). The geometric reference is an independent, brute-force
// supersampled non-zero-winding rasterizer (see referenceCoverage), rather
// than anything sharing the package's own accumulation code.

// pointInPolygon reports whether (px, py) is inside the simple polygon given
// by verts, using a non-zero winding number. Coordinates are float64 on
// purpose, so this does not reuse the Rasterizer's fixed/float32 arithmetic.
func pointInPolygon(verts []image.Point, px, py float64) bool {
	wn := 0
	for i := range verts {
		a := verts[i]
		b := verts[(i+1)%len(verts)]
		ax, ay := float64(a.X), float64(a.Y)
		bx, by := float64(b.X), float64(b.Y)
		if ay <= py {
			if by > py && crossSide(ax, ay, bx, by, px, py) > 0 {
				wn++
			}
		} else {
			if by <= py && crossSide(ax, ay, bx, by, px, py) < 0 {
				wn--
			}
		}
	}
	return wn != 0
}

// crossSide is the sign of the cross product of (b-a) and (p-a).
func crossSide(ax, ay, bx, by, px, py float64) float64 {
	return (bx-ax)*(py-ay) - (by-ay)*(px-ax)
}

// referenceCoverage returns the independent supersampled 8-bit coverage of the
// polygon verts over a w*h cell grid, laid out as w*h row-major bytes matching
// a Rasterizer whose origin is (0, 0). J is the number of sub-samples per
// axis. The Rasterizer and this reference both implement non-zero winding,
// saturated to [0, 1].
func referenceCoverage(w, h, J int, verts []image.Point) []byte {
	out := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			inside := 0
			for j := 0; j < J; j++ {
				for i := 0; i < J; i++ {
					px := float64(x) + (float64(i)+0.5)/float64(J)
					py := float64(y) + (float64(j)+0.5)/float64(J)
					if pointInPolygon(verts, px, py) {
						inside++
					}
				}
			}
			// Round to nearest to mirror an 8-bit alpha.
			out[y*w+x] = byte((2*inside*0xff + J*J) / (2 * J * J))
		}
	}
	return out
}

// polygonArea returns the unsigned shoelace area of verts.
func polygonArea(verts []image.Point) float64 {
	area := 0.0
	for i := range verts {
		a, b := verts[i], verts[(i+1)%len(verts)]
		area += float64(a.X*b.Y - b.X*a.Y)
	}
	if area < 0 {
		area = -area
	}
	return area / 2
}

// assertCoverageClose compares the Rasterizer's coverage against the
// independent supersampled non-zero-winding reference.
//
// clip is the rectangle (in cell coordinates) to which both renderings are
// confined; pass the Rasterizer's bounds when the polygon may extend outside
// it, since the visible polygon area is the clipped area, not the full
// polygon area.
//
// The two renderers use different anti-aliasing conventions (analytic
// sub-pixel area vs. point sampling), so individual edge pixels can differ
// even when both are correct. Instead of requiring byte-identical edges, this
// checks the invariants every correct non-zero-winding rasterizer must
// satisfy:
//
//   - the total covered area matches the polygon's area inside clip;
//   - pixels well inside the polygon are fully covered (0xff);
//   - pixels well outside are empty (0x00);
//   - every value is a legal 8-bit alpha and edge coverage is conserved
//     between the two renderings (their sums over the edge band agree).
//
// Byte-identical agreement between the package's own fixed and floating
// backends is checked in TestCoverageFloatingVsFixed.
func assertCoverageClose(t *testing.T, label string, got []byte, w, h, J int, verts []image.Point, clip image.Rectangle) {
	t.Helper()
	ref := referenceCoverage(w, h, J, verts)
	if len(got) != len(ref) {
		t.Fatalf("%s: len(got)=%d, len(ref)=%d", label, len(got), len(ref))
	}

	// 1. Total covered area vs. the polygon's area, estimated independently by
	// the point-sampling reference over the same clip rectangle. The reference
	// already ignores samples outside [0,w)x[0,h), so its sum is the visible
	// (clipped) area; comparing both sums to each other (and to the exact
	// shoelace area when nothing is clipped) avoids any shared code path.
	var gotSum, refSum float64
	for i := range got {
		gotSum += float64(got[i]) / 255.0
		refSum += float64(ref[i]) / 255.0
	}
	trueArea := polygonArea(verts)
	// When the polygon fits inside the clip, both renderings must match its
	// exact shoelace area. Otherwise, require them to agree with each other
	// and not to exceed the full polygon area.
	fullyInside := true
	for _, p := range verts {
		if !p.In(clip) {
			fullyInside = false
		}
	}
	const areaTol = 0.01 // 1% relative area error.
	if fullyInside {
		if e := math.Abs(gotSum-trueArea) / trueArea; e > areaTol {
			t.Errorf("%s: rasterizer area %.4f vs true area %.4f, error %.3f%%",
				label, gotSum, trueArea, 100*e)
		}
		if e := math.Abs(refSum-trueArea) / trueArea; e > areaTol {
			t.Errorf("%s: reference area %.4f vs true area %.4f, error %.3f%%",
				label, refSum, trueArea, 100*e)
		}
	} else {
		denom := refSum
		if denom == 0 {
			denom = 1
		}
		if e := math.Abs(gotSum-refSum) / denom; e > areaTol {
			t.Errorf("%s: clipped area: rasterizer %.4f vs reference %.4f differ by %.3f%%",
				label, gotSum, refSum, 100*e)
		}
		if gotSum > trueArea+1 {
			t.Errorf("%s: rasterizer area %.4f exceeds full polygon area %.4f", label, gotSum, trueArea)
		}
	}

	// 2. Classify each pixel from the reference's sub-samples: a pixel whose
	// center is comfortably inside (all sub-samples covered) or comfortably
	// outside (none covered) has a convention-independent expected value. We
	// leave a one-pixel-wide classification margin around the reference's own
	// edge band, since an edge passing through a pixel center makes even the
	// point sampler partial.
	interiorMismatch, exteriorMismatch := 0, 0
	edgeConservation := 0.0
	nEdge := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			// A pixel is "safely interior/exterior" only if its whole 3x3
			// neighborhood agrees, which keeps true edge pixels out of the
			// exact-match set.
			safeIn, safeOut := true, true
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= w || yy >= h {
						safeIn, safeOut = false, false
						continue
					}
					if ref[yy*w+xx] != 0xff {
						safeIn = false
					}
					if ref[yy*w+xx] != 0x00 {
						safeOut = false
					}
				}
			}
			switch {
			case safeIn:
				if got[i] != 0xff {
					interiorMismatch++
				}
			case safeOut:
				if got[i] != 0x00 {
					exteriorMismatch++
				}
			default:
				edgeConservation += float64(got[i]) - float64(ref[i])
				nEdge++
				if got[i] > 0xff {
					t.Errorf("%s: i=%d illegal alpha %#02x", label, i, got[i])
				}
			}
		}
	}
	if interiorMismatch != 0 {
		t.Errorf("%s: %d interior pixels are not fully covered", label, interiorMismatch)
	}
	if exteriorMismatch != 0 {
		t.Errorf("%s: %d exterior pixels are not empty", label, exteriorMismatch)
	}
	// 3. Coverage over the edge band is conserved (same area, redistributed
	// among neighboring pixels by the different sampling phases).
	if nEdge > 0 {
		meanBias := math.Abs(edgeConservation) / float64(nEdge)
		if meanBias > 4.0 {
			t.Errorf("%s: edge coverage not conserved: mean bias %.2f across %d edge pixels",
				label, meanBias, nEdge)
		}
	}
}

// extractCoverage runs a Coverage export over the Rasterizer's full bounds and
// returns the row-major bytes.
func extractCoverage(t *testing.T, z *Rasterizer) []byte {
	t.Helper()
	buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z.Coverage(nil, buf); err != nil {
		t.Fatalf("Coverage: %v", err)
	}
	w, h := z.size.X, z.size.Y
	out := make([]byte, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out[y*w+x] = buf.AlphaAt(x, y).A
		}
	}
	return out
}

// TestCoverageHalfCoveredPolygon is the half-coverage / slanted-edge fixture:
// a triangle gives a spread of partial coverage along its edges.
func TestCoverageHalfCoveredPolygon(t *testing.T) {
	const w, h = 24, 24
	verts := []image.Point{
		{2, 2},
		{20, 4},
		{6, 20},
	}

	z := NewRasterizer(w, h)
	z.MoveTo(2, 2)
	z.LineTo(20, 4)
	z.LineTo(6, 20)
	z.ClosePath()

	got := extractCoverage(t, z)
	// J=12 supersampling keeps the reference's own quantization well under
	// the Rasterizer's sub-pixel approximation error.
	assertCoverageClose(t, "triangle", got, w, h, 12, verts, image.Rect(0, 0, w, h))

	// Fully-insured invariants: a corner outside the shape is empty and the
	// interior is solid.
	if got[0] != 0x00 {
		t.Errorf("corner coverage: got %#02x, want 0x00", got[0])
	}
	// Some edge cell must be partially covered (the point of anti-aliasing).
	sawPartial := false
	for _, a := range got {
		if a != 0x00 && a != 0xff {
			sawPartial = true
		}
	}
	if !sawPartial {
		t.Error("expected partial-coverage edge pixels, found only 0x00/0xff")
	}
}

// TestCoverageOverlappingPaths checks overlapping and nested sub-paths,
// including opposite winding, against the non-zero-winding reference.
func TestCoverageOverlappingPaths(t *testing.T) {
	const w, h = 32, 32

	type sub struct {
		verts []image.Point
		neg   bool // traverse in reverse (opposite winding)
	}
	outer := []image.Point{{4, 4}, {28, 4}, {28, 28}, {4, 28}}
	inner := []image.Point{{10, 10}, {22, 10}, {22, 22}, {10, 22}}

	z := NewRasterizer(w, h)
	feed := func(vs []image.Point, neg bool) {
		order := make([]image.Point, len(vs))
		copy(order, vs)
		if neg {
			for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
				order[i], order[j] = order[j], order[i]
			}
		}
		z.MoveTo(float32(order[0].X), float32(order[0].Y))
		for _, p := range order[1:] {
			z.LineTo(float32(p.X), float32(p.Y))
		}
		z.ClosePath()
	}
	feed(outer, false)
	feed(inner, true) // hole with opposite winding

	got := extractCoverage(t, z)

	// Reference: outer polygon minus inner polygon (non-zero winding nets the
	// reversed inner ring to zero). Build a combined vertex check.
	refOuter := referenceCoverage(w, h, 8, outer)
	refInner := referenceCoverage(w, h, 8, inner)
	want := make([]byte, len(refOuter))
	for i := range want {
		if refInner[i] == 0 {
			want[i] = refOuter[i]
		}
	}
	const tol = 3
	for i := range want {
		d := int(got[i]) - int(want[i])
		if d < -tol || d > tol {
			t.Errorf("i=%d (x=%d,y=%d): got %#02x, want ~%#02x", i, i%w, i/w, got[i], want[i])
		}
	}

	// The hole's center must be empty and the ring's interior solid.
	if c := got[16*w+16]; c != 0x00 {
		t.Errorf("hole center: got %#02x, want 0x00", c)
	}
	if c := got[6*w+6]; c != 0xff {
		t.Errorf("ring interior: got %#02x, want 0xff", c)
	}
}

// TestCoverageNegativeAndOutOfBounds uses non-zero (negative) coordinates and
// a shape extending beyond the Rasterizer's bounds. Coverage must be clipped
// and mapped by shared origin.
func TestCoverageNegativeAndOutOfBounds(t *testing.T) {
	const w, h = 16, 16
	// Square from (-4,-4) to (12,12): negative corner, extends out of bounds
	// on the top/left while fitting inside on bottom/right.
	verts := []image.Point{
		{-4, -4},
		{12, -4},
		{12, 12},
		{-4, 12},
	}

	z := NewRasterizer(w, h)
	z.MoveTo(-4, -4)
	z.LineTo(12, -4)
	z.LineTo(12, 12)
	z.LineTo(-4, 12)
	z.ClosePath()

	got := extractCoverage(t, z)
	assertCoverageClose(t, "negative coords", got, w, h, 8, verts, image.Rect(0, 0, w, h))

	// Cell (0,0) is interior despite the shape's negative origin.
	if got[0] != 0xff {
		t.Errorf("cell (0,0): got %#02x, want 0xff", got[0])
	}
	// The far corner (15,15) is outside the square.
	if got[15*w+15] != 0x00 {
		t.Errorf("cell (15,15): got %#02x, want 0x00", got[15*w+15])
	}
}

// TestCoverageNonZeroOrigin maps a Rasterizer into a CoverageBuffer whose
// rectangle does not start at the origin, verifying the shared-origin mapping
// and clipping to the intersection.
func TestCoverageNonZeroOrigin(t *testing.T) {
	const w, h = 20, 20
	z := NewRasterizer(w, h)
	z.MoveTo(2, 2)
	z.LineTo(18, 2)
	z.LineTo(18, 18)
	z.LineTo(2, 18)
	z.ClosePath()

	// Buffer covers cells (5,6)..(15,16), with padding stride.
	rect := image.Rect(5, 6, 15, 16)
	buf, err := NewCoverageBuffer(rect, rect.Dx()+3, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z.Coverage(nil, buf); err != nil {
		t.Fatalf("Coverage: %v", err)
	}

	full := extractCoverage(t, z)
	for y := rect.Min.Y; y < rect.Max.Y; y++ {
		for x := rect.Min.X; x < rect.Max.X; x++ {
			if g, w := buf.AlphaAt(x, y).A, full[y*w+x]; g != w {
				t.Errorf("(%d,%d): got %#02x, want %#02x", x, y, g, w)
			}
		}
	}
	// Out-of-bounds points read as transparent.
	if a := buf.AlphaAt(0, 0).A; a != 0x00 {
		t.Errorf("AlphaAt outside bounds: got %#02x, want 0x00", a)
	}
}

// TestCoverageEmptyPathAndEmptyRasterizer checks degenerate inputs yield zero
// coverage and never panic.
func TestCoverageEmptyPathAndEmptyRasterizer(t *testing.T) {
	// Rasterizer with no paths.
	z := NewRasterizer(8, 8)
	got := extractCoverage(t, z)
	for i, a := range got {
		if a != 0 {
			t.Fatalf("empty path: i=%d got %#02x, want 0x00", i, a)
		}
	}

	// Only degenerate (zero-length / horizontal) segments.
	z2 := NewRasterizer(8, 8)
	z2.MoveTo(1, 1)
	z2.LineTo(1, 1)
	z2.LineTo(7, 1)
	z2.ClosePath()
	got2 := extractCoverage(t, z2)
	for i, a := range got2 {
		if a != 0 {
			t.Fatalf("degenerate path: i=%d got %#02x, want 0x00", i, a)
		}
	}

	// Zero-value / zero-size Rasterizer: intersection is empty.
	var z0 Rasterizer
	buf, err := NewCoverageBuffer(image.Rect(0, 0, 4, 4), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z0.Coverage(nil, buf); err != nil {
		t.Fatalf("zero-value Rasterizer Coverage: %v", err)
	}
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if buf.AlphaAt(x, y).A != 0 {
				t.Fatalf("zero-value Rasterizer wrote coverage at (%d,%d)", x, y)
			}
		}
	}

	// Empty intersection writes nothing and returns no error.
	big := NewRasterizer(4, 4)
	disjoint, err := NewCoverageBuffer(image.Rect(100, 100, 104, 104), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := big.Coverage(nil, disjoint); err != nil {
		t.Fatalf("disjoint Coverage: %v", err)
	}
	if a := disjoint.AlphaAt(100, 100).A; a != 0 {
		t.Errorf("disjoint buffer modified: got %#02x, want 0x00", a)
	}
}

// TestCoverageIndependentOfOpAndSource documents that coverage is purely
// geometric: DrawOp and a possibly-transparent source cannot change it.
func TestCoverageIndependentOfOpAndSource(t *testing.T) {
	mk := func() *Rasterizer {
		z := NewRasterizer(12, 12)
		z.MoveTo(2, 2)
		z.LineTo(10, 2)
		z.LineTo(10, 10)
		z.ClosePath()
		return z
	}

	base := extractCoverage(t, mk())

	zSrc := mk()
	zSrc.DrawOp = draw.Src
	if got := extractCoverage(t, zSrc); !bytesEqual(got, base) {
		t.Error("coverage changed with draw.Src")
	}

	// A transparent source is irrelevant to Coverage, but also confirm Draw
	// with it leaves a destination untouched while Coverage still reports the
	// shape.
	zTransparent := mk()
	dst := image.NewAlpha(zTransparent.Bounds())
	for i := range dst.Pix {
		dst.Pix[i] = 0x40
	}
	transparent := image.NewUniform(color.Alpha{A: 0x00})
	zTransparent.Draw(dst, dst.Bounds(), transparent, image.Point{})
	for i, a := range dst.Pix {
		if a != 0x40 {
			t.Fatalf("transparent source modified dst at %d: got %#02x", i, a)
		}
	}
	if got := extractCoverage(t, zTransparent); !bytesEqual(got, base) {
		t.Error("coverage changed after drawing a transparent source")
	}
}

// TestCoverageMatchesDrawSrc proves the exported bytes equal an opaque-source,
// draw.Src render into image.Alpha, for both the full-bounds fast path and a
// sub-rectangle slow path, and across both math implementations.
func TestCoverageMatchesDrawSrc(t *testing.T) {
	for _, size := range []int{16, 600} { // 16: fixed math, 600: floating math.
		mk := func() *Rasterizer {
			s := float32(size)
			z := NewRasterizer(size, size)
			z.MoveTo(1, 1)
			z.QuadTo(s*0.9, 4, s-2, s-2)
			z.CubeTo(4, s*0.7, s*0.6, 3, 3, s*0.5)
			z.ClosePath()
			return z
		}

		// Full bounds: exercises the accumulate OpSrc fast path.
		z1 := mk()
		alpha := image.NewAlpha(z1.Bounds())
		z1.DrawOp = draw.Src
		z1.Draw(alpha, alpha.Bounds(), image.Opaque, image.Point{})

		z2 := mk()
		buf, err := NewCoverageBuffer(z2.Bounds(), 0, nil)
		if err != nil {
			t.Fatalf("size=%d: NewCoverageBuffer: %v", size, err)
		}
		if _, err := z2.Coverage(nil, buf); err != nil {
			t.Fatalf("size=%d: Coverage: %v", size, err)
		}
		for i := range alpha.Pix {
			if alpha.Pix[i] != buf.pix[i] {
				t.Fatalf("size=%d full-bounds: i=%d got %#02x, want %#02x", size, i, buf.pix[i], alpha.Pix[i])
			}
		}

		// Sub-rectangle originating at (0,0): this exercises the mask-based
		// slow path in Draw while staying inside that path's origin
		// assumption. Coverage's own non-origin mapping is covered separately
		// by TestCoverageNonZeroOrigin.
		z3 := mk()
		loose := image.NewAlpha(image.Rect(0, 0, size+5, size+5))
		z3.DrawOp = draw.Src
		rect := image.Rect(0, 0, size-2, size-1)
		z3.Draw(loose, rect, image.Opaque, image.Point{})

		z4 := mk()
		buf4, err := NewCoverageBuffer(z4.Bounds(), 0, nil)
		if err != nil {
			t.Fatalf("size=%d: NewCoverageBuffer: %v", size, err)
		}
		if _, err := z4.Coverage(nil, buf4); err != nil {
			t.Fatalf("size=%d: Coverage: %v", size, err)
		}
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			for x := rect.Min.X; x < rect.Max.X; x++ {
				if g, w := loose.AlphaAt(x, y).A, buf4.AlphaAt(x, y).A; g != w {
					t.Fatalf("size=%d sub-rect (%d,%d): got %#02x, want %#02x", size, x, y, g, w)
				}
			}
		}
	}
}

// TestCoverageRecomposition is the end-to-end "re-synthesis" fixture: use the
// exported coverage as an image.Alpha mask with draw.DrawMask and confirm it
// reproduces Rasterizer.Draw for both ops and several sources.
func TestCoverageRecomposition(t *testing.T) {
	mk := func() *Rasterizer {
		z := NewRasterizer(20, 20)
		z.MoveTo(3, 3)
		z.QuadTo(18, 5, 16, 17)
		z.CubeTo(6, 18, 4, 9, 3, 3)
		z.ClosePath()
		return z
	}
	sources := []image.Image{
		image.Opaque,
		image.NewUniform(color.RGBA{0x40, 0x80, 0xc0, 0xff}),
		image.NewUniform(color.RGBA{0x40, 0x80, 0xc0, 0x80}),
	}

	for _, op := range []draw.Op{draw.Over, draw.Src} {
		for si, src := range sources {
			// Reference: direct Draw onto an RGBA destination.
			z1 := mk()
			direct := image.NewRGBA(z1.Bounds())
			for i := range direct.Pix {
				direct.Pix[i] = byte(0x11 * (i + 1))
			}
			z1.DrawOp = op
			z1.Draw(direct, direct.Bounds(), src, image.Point{})

			// Recomposed: export coverage, then DrawMask with the same src.
			z2 := mk()
			buf, err := NewCoverageBuffer(z2.Bounds(), 0, nil)
			if err != nil {
				t.Fatalf("NewCoverageBuffer: %v", err)
			}
			if _, err := z2.Coverage(nil, buf); err != nil {
				t.Fatalf("Coverage: %v", err)
			}
			recomposed := image.NewRGBA(z2.Bounds())
			for i := range recomposed.Pix {
				recomposed.Pix[i] = byte(0x11 * (i + 1))
			}
			draw.DrawMask(recomposed, recomposed.Bounds(), src, image.Point{},
				buf.Alpha(), image.Point{}, op)

			// The package's Draw and the stdlib's draw.DrawMask quantize the
			// same Porter-Duff math in slightly different orders, so they can
			// differ by one unit on intermediate pixels; the mask itself is
			// byte-identical (see TestCoverageMatchesDrawSrc).
			const tol = 1
			for i := range direct.Pix {
				d := int(direct.Pix[i]) - int(recomposed.Pix[i])
				if d < -tol || d > tol {
					t.Errorf("op=%v src=%d i=%d: direct %#02x, recomposed %#02x",
						op, si, i, direct.Pix[i], recomposed.Pix[i])
				}
			}
		}
	}
}

// TestCoverageReusableAndNonDestructive verifies Clear, repeated Coverage and
// interleaved Draw/Coverage calls: the path's area values survive, so every
// export is identical and Draw results don't drift.
func TestCoverageReusableAndNonDestructive(t *testing.T) {
	z := NewRasterizer(18, 18)
	z.MoveTo(2, 2)
	z.LineTo(15, 3)
	z.QuadTo(16, 12, 4, 15)
	z.ClosePath()

	first := extractCoverage(t, z)

	// Repeated Coverage without Reset must give identical results.
	for k := 0; k < 3; k++ {
		if got := extractCoverage(t, z); !bytesEqual(got, first) {
			t.Fatalf("Coverage call %d differs", k+2)
		}
	}

	// Interleave Draw and Coverage and confirm neither perturbs the other.
	dstA := image.NewAlpha(z.Bounds())
	z.DrawOp = draw.Src
	z.Draw(dstA, dstA.Bounds(), image.Opaque, image.Point{})
	if got := extractCoverage(t, z); !bytesEqual(got, first) {
		t.Fatal("Coverage changed after a Draw")
	}
	dstB := image.NewAlpha(z.Bounds())
	z.Draw(dstB, dstB.Bounds(), image.Opaque, image.Point{})
	if !bytesEqual(dstA.Pix, dstB.Pix) {
		t.Fatal("Draw results differ across an intervening Coverage")
	}

	// Clear zeros the buffer, then a fresh export repopulates it.
	buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z.Coverage(nil, buf); err != nil {
		t.Fatalf("Coverage: %v", err)
	}
	buf.Clear()
	for i, a := range buf.pix {
		if a != 0 {
			t.Fatalf("Clear left non-zero at %d", i)
		}
	}
	if _, err := z.Coverage(nil, buf); err != nil {
		t.Fatalf("Coverage after Clear: %v", err)
	}
	for i := range first {
		if buf.pix[i] != first[i] {
			t.Fatalf("re-export after Clear at %d: got %#02x, want %#02x", i, buf.pix[i], first[i])
		}
	}
}

// TestCoverageRepeatedReset stresses Reset at varying sizes (including zero),
// above and below the floating-point threshold, and repeated many times.
func TestCoverageRepeatedReset(t *testing.T) {
	var z Rasterizer
	sizes := [][2]int{
		{0, 0}, {1, 1}, {16, 16}, {512, 512}, {513, 8}, {8, 600}, {32, 32}, {0, 4},
	}
	for repeat := 0; repeat < 3; repeat++ {
		for _, sh := range sizes {
			z.Reset(sh[0], sh[1])
			// A zero-dimension Rasterizer has empty Bounds; there is nowhere
			// to export. Verify Coverage itself handles that without a buffer.
			if sh[0] <= 0 || sh[1] <= 0 {
				if _, err := z.Coverage(nil, nil); err != ErrNilCoverageBuffer {
					t.Fatalf("Reset(%d,%d): nil-buffer error = %v", sh[0], sh[1], err)
				}
				continue
			}
			z.MoveTo(0, 0)
			z.LineTo(float32(sh[0]), 0)
			z.LineTo(float32(sh[0]), float32(sh[1]))
			z.ClosePath()
			buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
			if err != nil {
				t.Fatalf("Reset(%d,%d): NewCoverageBuffer: %v", sh[0], sh[1], err)
			}
			if _, err := z.Coverage(nil, buf); err != nil {
				t.Fatalf("Reset(%d,%d): Coverage: %v", sh[0], sh[1], err)
			}
		}
	}
}

// TestCoverageBasicMaskGolden cross-checks against the package's existing
// basicMask golden fixture used by the Draw tests, so coverage stays
// consistent with the original tests.
func TestCoverageBasicMaskGolden(t *testing.T) {
	z := NewRasterizer(16, 16)
	z.MoveTo(2, 2)
	z.LineTo(8, 2)
	z.QuadTo(14, 2, 14, 14)
	z.CubeTo(8, 2, 5, 20, 2, 8)
	z.ClosePath()

	got := extractCoverage(t, z)
	const tol = 2 // matches testBasicPath's rounding allowance.
	for i := range basicMask {
		d := int(got[i]) - int(basicMask[i])
		if d < -tol || d > tol {
			t.Errorf("i=%d: got %#02x, golden %#02x", i, got[i], basicMask[i])
		}
	}
}

// TestNewCoverageBufferErrors checks the stable error / resource-limit
// contract, including caller-supplied buffers, stride and the size cap.
func TestNewCoverageBufferErrors(t *testing.T) {
	pix := make([]byte, 64)
	cases := []struct {
		name   string
		rect   image.Rectangle
		stride int
		pix    []byte
		want   error
	}{
		{"empty rect", image.Rect(0, 0, 0, 4), 0, nil, ErrInvalidCoverageRect},
		// image.Rect canonicalizes, so build a genuinely non-canonical
		// rectangle (Min greater than Max) by hand.
		{"inverted rect", image.Rectangle{Min: image.Point{4, 4}, Max: image.Point{1, 1}}, 0, nil, ErrInvalidCoverageRect},
		{"negative stride", image.Rect(0, 0, 4, 4), -1, nil, ErrInvalidCoverageStride},
		{"stride below width", image.Rect(0, 0, 4, 4), 3, nil, ErrInvalidCoverageStride},
		{"pix too small", image.Rect(0, 0, 8, 8), 10, pix, ErrCoverageBufferTooSmall},
		{"excessive size", image.Rect(0, 0, 1<<20, 1<<20), 0, nil, ErrCoverageBufferTooSmall},
		{"excessive stride", image.Rect(0, 0, 4, 4), maxCoverageBufferBytes, nil, ErrCoverageBufferTooSmall},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := NewCoverageBuffer(tc.rect, tc.stride, tc.pix)
			if err != tc.want {
				t.Fatalf("got %v, want %v", err, tc.want)
			}
			if b != nil {
				t.Fatal("expected nil buffer on error")
			}
		})
	}

	// A caller-supplied buffer that is exactly large enough is accepted and
	// cleared, and the returned Alpha aliases the same storage.
	rect := image.Rect(0, 0, 4, 2)
	backing := make([]byte, 4*2)
	for i := range backing {
		backing[i] = 0xff
	}
	buf, err := NewCoverageBuffer(rect, 4, backing)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	for i, a := range backing {
		if a != 0 {
			t.Fatalf("caller buffer not cleared at %d", i)
		}
	}
	// Aliasing check: mutating the buffer is observed through Alpha.
	buf.pix[0] = 0x7f
	if buf.Alpha().Pix[0] != 0x7f {
		t.Fatal("Alpha() does not alias the CoverageBuffer storage")
	}
	if buf.Stride() != 4 || buf.Bounds() != rect {
		t.Fatal("Stride/Bounds mismatch")
	}
	if cm := buf.ColorModel(); cm != color.AlphaModel {
		t.Fatalf("ColorModel = %v, want AlphaModel", cm)
	}
}

// TestCoverageNilBuffer documents the nil-destination error.
func TestCoverageNilBuffer(t *testing.T) {
	z := NewRasterizer(4, 4)
	if _, err := z.Coverage(nil, nil); err != ErrNilCoverageBuffer {
		t.Fatalf("got %v, want ErrNilCoverageBuffer", err)
	}
}

// TestCoverageCancellation checks both pre-canceled and mid-run cancellation.
func TestCoverageCancellation(t *testing.T) {
	// Pre-canceled: nothing is written and the context error is returned.
	z := NewRasterizer(16, 16)
	z.MoveTo(1, 1)
	z.LineTo(15, 1)
	z.LineTo(15, 15)
	z.ClosePath()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z.Coverage(ctx, buf); err != context.Canceled {
		t.Fatalf("pre-canceled: got %v, want context.Canceled", err)
	}
	for i, a := range buf.pix {
		if a != 0 {
			t.Fatalf("pre-canceled Coverage wrote at %d", i)
		}
	}

	// Mid-run cancellation on a large rasterizer returns the context error.
	// We drive cancellation from a custom context whose Err flips after many
	// checks, so the outcome is deterministic and does not depend on timing.
	const big = 2048
	zb := NewRasterizer(big, big)
	zb.MoveTo(0, 0)
	zb.LineTo(big, 0)
	zb.LineTo(big, big)
	zb.ClosePath()
	bbuf, err := NewCoverageBuffer(zb.Bounds(), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer big: %v", err)
	}
	cctx := &cancelAfterNContext{after: 2}
	if _, err := zb.Coverage(cctx, bbuf); err != errCancelAfterN {
		t.Fatalf("mid-run: got %v, want errCancelAfterN", err)
	}
}

// TestCoverageConcurrentReads confirms many goroutines can read a stable
// CoverageBuffer concurrently. Run under -race to validate the contract.
func TestCoverageConcurrentReads(t *testing.T) {
	z := NewRasterizer(32, 32)
	z.MoveTo(4, 4)
	z.LineTo(28, 4)
	z.LineTo(28, 28)
	z.ClosePath()
	buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
	if err != nil {
		t.Fatalf("NewCoverageBuffer: %v", err)
	}
	if _, err := z.Coverage(nil, buf); err != nil {
		t.Fatalf("Coverage: %v", err)
	}

	const goroutines = 16
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			a := buf.Alpha()
			for k := 0; k < 100; k++ {
				for y := 0; y < 32; y++ {
					for x := 0; x < 32; x++ {
						if buf.AlphaAt(x, y).A != a.AlphaAt(x, y).A {
							t.Error("concurrent read mismatch")
							return
						}
					}
				}
			}
		}()
	}
	wg.Wait()
}

// TestCoverageFloatingVsFixed ensures both math backends produce coverage
// consistent with the independent reference, and that the rasterizer selects
// the floating backend once a dimension exceeds floatingPointMathThreshold.
func TestCoverageFloatingVsFixed(t *testing.T) {
	// Below the threshold the fixed-point backend is exact enough that both
	// implementations agree with the reference and with each other.
	const w1, h1 = 256, 256
	verts1 := []image.Point{{12, 12}, {244, 38}, {128, 230}}

	zf := NewRasterizer(w1, h1)
	if zf.useFloatingPointMath {
		t.Fatal("expected fixed point math for size <= 512")
	}
	zf.MoveTo(12, 12)
	zf.LineTo(244, 38)
	zf.LineTo(128, 230)
	zf.ClosePath()
	gotFixed := extractCoverage(t, zf)
	assertCoverageClose(t, "fixed triangle", gotFixed, w1, h1, 10, verts1, image.Rect(0, 0, w1, h1))

	zfloat := NewRasterizer(w1, h1)
	zfloat.setUseFloatingPointMath(true)
	zfloat.MoveTo(12, 12)
	zfloat.LineTo(244, 38)
	zfloat.LineTo(128, 230)
	zfloat.ClosePath()
	gotFloat := extractCoverage(t, zfloat)
	assertCoverageClose(t, "floating triangle", gotFloat, w1, h1, 10, verts1, image.Rect(0, 0, w1, h1))

	// The package's two backends implement the same analytic algorithm with
	// different arithmetic, so the vast majority of pixels agree exactly and
	// the mean absolute difference is small. This mirrors the package's
	// existing TestFixedFloatingCloseness (which allows a 1% mean error).
	var total int
	for i := range gotFixed {
		d := int(gotFixed[i]) - int(gotFloat[i])
		if d < 0 {
			d = -d
		}
		total += d
	}
	meanErr := float64(total) / float64(len(gotFixed)*255)
	const backendMeanTol = 0.01
	if meanErr > backendMeanTol {
		t.Errorf("fixed/floating mean error %.4f%% > %.4f%%", 100*meanErr, 100*backendMeanTol)
	}

	// Above the threshold the floating backend is selected automatically and
	// still matches the reference; the fixed backend is intentionally not used
	// at this scale because its fixed-point arithmetic overflows (that is the
	// reason floatingPointMathThreshold exists).
	const w2, h2 = 600, 600
	verts2 := []image.Point{{20, 20}, {580, 60}, {300, 560}}
	z := NewRasterizer(w2, h2)
	if !z.useFloatingPointMath {
		t.Fatal("expected floating point math for size > 512")
	}
	z.MoveTo(20, 20)
	z.LineTo(580, 60)
	z.LineTo(300, 560)
	z.ClosePath()
	assertCoverageClose(t, "large floating triangle", extractCoverage(t, z), w2, h2, 10, verts2, image.Rect(0, 0, w2, h2))
}

// TestCoverageHalfCoverageExactSquare checks a shape that covers exactly half
// of a column of pixels, validating the 0x80-ish partial coverage against the
// supersampled reference.
func TestCoverageHalfCoverageExactSquare(t *testing.T) {
	const w, h = 8, 8
	// Square occupying x in [0,4], so the first four columns are fully covered
	// and the rest empty; add a slanted edge to force partial pixels.
	verts := []image.Point{{0, 0}, {4, 0}, {4, 8}, {0, 8}}
	z := NewRasterizer(w, h)
	z.MoveTo(0, 0)
	z.LineTo(4, 0)
	z.LineTo(4, 8)
	z.LineTo(0, 8)
	z.ClosePath()
	got := extractCoverage(t, z)
	assertCoverageClose(t, "half square", got, w, h, 8, verts, image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		if got[y*w+0] != 0xff || got[y*w+3] != 0xff {
			t.Errorf("row %d: expected fully-covered left columns", y)
		}
		if got[y*w+7] != 0x00 {
			t.Errorf("row %d: expected empty right column", y)
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

// errCancelAfterN and cancelAfterNContext provide a deterministic, timing-free
// cancellable context whose Err flips after a fixed number of checks.
var errCancelAfterN = contextError("coverage-test: canceled after N checks")

type contextError string

func (e contextError) Error() string { return string(e) }

type cancelAfterNContext struct {
	after  int
	checks int
}

func (c *cancelAfterNContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterNContext) Done() <-chan struct{}       { return nil }
func (c *cancelAfterNContext) Err() error {
	c.checks++
	if c.checks > c.after {
		return errCancelAfterN
	}
	return nil
}
func (c *cancelAfterNContext) Value(any) any { return nil }

// BenchmarkCoverageGlyph measures exporting the coverage mask for the same
// glyph used by the Draw benchmarks, at a size below and above the floating
// point threshold.
func BenchmarkCoverageGlyph64(b *testing.B)  { benchmarkCoverageGlyph(b, 64) }
func BenchmarkCoverageGlyph1024(b *testing.B) { benchmarkCoverageGlyph(b, 1024) }

func benchmarkCoverageGlyph(b *testing.B, height int) {
	width, data := scaledBenchmarkGlyphData(height)
	z := NewRasterizer(width, height)
	buf, err := NewCoverageBuffer(z.Bounds(), 0, nil)
	if err != nil {
		b.Fatalf("NewCoverageBuffer: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		z.Reset(width, height)
		for _, d := range data {
			switch d.n {
			case 0:
				z.MoveTo(d.px, d.py)
			case 1:
				z.LineTo(d.px, d.py)
			case 2:
				z.QuadTo(d.px, d.py, d.qx, d.qy)
			}
		}
		buf.Clear()
		if _, err := z.Coverage(nil, buf); err != nil {
			b.Fatalf("Coverage: %v", err)
		}
	}
}
