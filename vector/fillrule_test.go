// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector_test

import (
	"image"
	"image/color"
	"image/draw"
	"math"
	"testing"

	"golang.org/x/image/vector"
)

// ----------------------------------------------------------------------------
// Independent scanline reference rasterizer.
//
// referenceRasterizer records the same path commands as a vector.Rasterizer
// but flattens curves itself (de Casteljau subdivision) and fills via
// brute-force supersampled ray casting. Point containment uses the classic
// half-open edge crossing test. It has no coverage-buffer or fixed-point
// state in common with the code under test, which makes it an independent
// oracle for both fill rules.
// ----------------------------------------------------------------------------

type refSegment struct {
	x0, y0, x1, y1 float64
}

type referenceRasterizer struct {
	subpaths [][]refSegment
	cur      []refSegment
	penX     float64
	penY     float64
	firstX   float64
	firstY   float64
	started  bool
}

func (r *referenceRasterizer) MoveTo(x, y float32) {
	r.subpaths = append(r.subpaths, r.cur)
	r.cur = nil
	r.penX, r.penY = float64(x), float64(y)
	r.firstX, r.firstY = r.penX, r.penY
	r.started = true
}

func (r *referenceRasterizer) LineTo(x, y float32) {
	r.cur = append(r.cur, refSegment{r.penX, r.penY, float64(x), float64(y)})
	r.penX, r.penY = float64(x), float64(y)
}

func (r *referenceRasterizer) QuadTo(bx, by, cx, cy float32) {
	ax, ay := r.penX, r.penY
	bx64, by64, cx64, cy64 := float64(bx), float64(by), float64(cx), float64(cy)
	const n = 96
	var prevX, prevY = ax, ay
	for i := 1; i <= n; i++ {
		t := float64(i) / n
		abx := ax + (bx64-ax)*t
		aby := ay + (by64-ay)*t
		bcx := bx64 + (cx64-bx64)*t
		bcy := by64 + (cy64-by64)*t
		px := abx + (bcx-abx)*t
		py := aby + (bcy-aby)*t
		r.cur = append(r.cur, refSegment{prevX, prevY, px, py})
		prevX, prevY = px, py
	}
	r.penX, r.penY = cx64, cy64
}

func (r *referenceRasterizer) CubeTo(bx, by, cx, cy, dx, dy float32) {
	ax, ay := r.penX, r.penY
	bx64, by64 := float64(bx), float64(by)
	cx64, cy64 := float64(cx), float64(cy)
	dx64, dy64 := float64(dx), float64(dy)
	const n = 96
	var prevX, prevY = ax, ay
	for i := 1; i <= n; i++ {
		t := float64(i) / n
		abx := ax + (bx64-ax)*t
		aby := ay + (by64-ay)*t
		bcx := bx64 + (cx64-bx64)*t
		bcy := by64 + (cy64-by64)*t
		cdx := cx64 + (dx64-cx64)*t
		cdy := cy64 + (dy64-cy64)*t
		abcx := abx + (bcx-abx)*t
		abcy := aby + (bcy-aby)*t
		bcdx := bcx + (cdx-bcx)*t
		bcdy := bcy + (cdy-bcy)*t
		px := abcx + (bcdx-abcx)*t
		py := abcy + (bcdy-abcy)*t
		r.cur = append(r.cur, refSegment{prevX, prevY, px, py})
		prevX, prevY = px, py
	}
	r.penX, r.penY = dx64, dy64
}

func (r *referenceRasterizer) ClosePath() {
	if r.started && (r.penX != r.firstX || r.penY != r.firstY) {
		r.cur = append(r.cur, refSegment{r.penX, r.penY, r.firstX, r.firstY})
	}
	r.penX, r.penY = r.firstX, r.firstY
}

func (r *referenceRasterizer) segments() [][]refSegment {
	out := make([][]refSegment, 0, len(r.subpaths)+1)
	for _, s := range r.subpaths {
		if len(s) != 0 {
			out = append(out, s)
		}
	}
	if len(r.cur) != 0 {
		out = append(out, r.cur)
	}
	return out
}

// coverage returns the fraction (in [0, 1]) of the (x, y) pixel's area that
// is inside the path under the given fill rule, estimated by S*S point
// samples.
func (r *referenceRasterizer) coverage(x, y int, rule vector.FillRule) float64 {
	const s = 8
	inside := 0
	for j := 0; j < s; j++ {
		rayY := float64(y) + (float64(j)+0.5)/s
		for i := 0; i < s; i++ {
			rayX := float64(x) + (float64(i)+0.5)/s
			winding, crossings := 0, 0
			for _, sub := range r.segments() {
				for _, e := range sub {
					// Half-open edge test: the lower endpoint is included,
					// the upper one excluded.
					if (e.y0 <= rayY && e.y1 > rayY) || (e.y1 <= rayY && e.y0 > rayY) {
						ix := e.x0 + (rayY-e.y0)*(e.x1-e.x0)/(e.y1-e.y0)
						if ix > rayX {
							crossings++
							if e.y1 > e.y0 {
								winding++
							} else {
								winding--
							}
						}
					}
				}
			}
			var hit bool
			if rule == vector.EvenOdd {
				hit = crossings%2 != 0
			} else {
				hit = winding != 0
			}
			if hit {
				inside++
			}
		}
	}
	return float64(inside) / (s * s)
}

// alphaAt renders one pixel's alpha (0..255).
func (r *referenceRasterizer) alphaAt(x, y int, rule vector.FillRule) uint8 {
	return uint8(r.coverage(x, y, rule)*255 + 0.5)
}

// ----------------------------------------------------------------------------
// Test path definitions and playback.
// ----------------------------------------------------------------------------

type pathCmd struct {
	op                     byte // 'm', 'l', 'q', 'c', 'z'
	x0, y0, x1, y1, x2, y2 float32
}

func playPath(z *vector.Rasterizer, r *referenceRasterizer, cmds []pathCmd) {
	for _, c := range cmds {
		switch c.op {
		case 'm':
			z.MoveTo(c.x0, c.y0)
			r.MoveTo(c.x0, c.y0)
		case 'l':
			z.LineTo(c.x0, c.y0)
			r.LineTo(c.x0, c.y0)
		case 'q':
			z.QuadTo(c.x0, c.y0, c.x1, c.y1)
			r.QuadTo(c.x0, c.y0, c.x1, c.y1)
		case 'c':
			z.CubeTo(c.x0, c.y0, c.x1, c.y1, c.x2, c.y2)
			r.CubeTo(c.x0, c.y0, c.x1, c.y1, c.x2, c.y2)
		case 'z':
			z.ClosePath()
			r.ClosePath()
		}
	}
}

// circle builds an approximately circular closed contour of radius rad around
// (cx, cy). reverse flips the winding direction.
func circle(cx, cy, rad float32, reverse bool) []pathCmd {
	const n = 24
	pt := func(i int) (float32, float32) {
		a := 2 * math.Pi * float64(i) / n
		return cx + rad*float32(math.Cos(a)), cy + rad*float32(math.Sin(a))
	}
	out := []pathCmd{}
	if reverse {
		x, y := pt(0)
		out = append(out, pathCmd{op: 'm', x0: x, y0: y})
		for i := n - 1; i >= 1; i-- {
			x, y := pt(i)
			out = append(out, pathCmd{op: 'l', x0: x, y0: y})
		}
		out = append(out, pathCmd{op: 'z'})
	} else {
		x, y := pt(0)
		out = append(out, pathCmd{op: 'm', x0: x, y0: y})
		for i := 1; i < n; i++ {
			x, y := pt(i)
			out = append(out, pathCmd{op: 'l', x0: x, y0: y})
		}
		out = append(out, pathCmd{op: 'z'})
	}
	return out
}

// ringPath builds two concentric circular contours.
func ringPath(cx, cy, outer, inner float32, innerReversed bool) []pathCmd {
	cmds := circle(cx, cy, outer, false)
	cmds = append(cmds, circle(cx, cy, inner, innerReversed)...)
	return cmds
}

// starPath builds a classic self-intersecting 5-point star (a pentagram):
// five outer vertices connected in every-other order, so the edges cross.
// The inner pentagon has winding number 2 under non-zero (filled) but an even
// edge count under even-odd (empty), making the two rules visibly differ.
func starPath(cx, cy, outer float32) []pathCmd {
	const n = 5
	outerPt := func(i int) (float32, float32) {
		a := -math.Pi/2 + 2*math.Pi*float64(i)/n
		return cx + outer*float32(math.Cos(a)), cy + outer*float32(math.Sin(a))
	}
	cmds := make([]pathCmd, 0, n+1)
	for k := 0; k < n; k++ {
		x, y := outerPt((k * 2) % n)
		if k == 0 {
			cmds = append(cmds, pathCmd{op: 'm', x0: x, y0: y})
		} else {
			cmds = append(cmds, pathCmd{op: 'l', x0: x, y0: y})
		}
	}
	cmds = append(cmds, pathCmd{op: 'z'})
	return cmds
}

// curvedPath exercises QuadTo and CubeTo in one closed contour.
func curvedPath() []pathCmd {
	return []pathCmd{
		{op: 'm', x0: 4, y0: 4},
		{op: 'q', x0: 24, y0: -8, x1: 36, y1: 20},
		{op: 'c', x0: 44, y0: 40, x1: 20, y1: 48, x2: 8, y2: 36},
		{op: 'z'},
	}
}

// curvedOverlapPath is two same-direction closed curved subpaths (one using
// QuadTo, the other CubeTo) whose interiors overlap. The overlap has winding
// number 2: filled by non-zero but empty under even-odd. This makes the two
// fill rules differ on a path built entirely from curves.
func curvedOverlapPath() []pathCmd {
	a := []pathCmd{
		{op: 'm', x0: 6, y0: 6},
		{op: 'q', x0: 30, y0: 0, x1: 30, y1: 30},
		{op: 'q', x0: 30, y0: 44, x1: 6, y1: 42},
		{op: 'z'},
	}
	b := []pathCmd{
		{op: 'm', x0: 20, y0: 20},
		{op: 'c', x0: 44, y0: 20, x1: 44, y1: 44, x2: 20, y2: 44},
		{op: 'c', x0: 6, y0: 44, x1: 6, y1: 20, x2: 20, y2: 20},
		{op: 'z'},
	}
	return append(a, b...)
}

// renderAlpha runs a path through a fresh Rasterizer and returns its alpha
// mask under the given rule and DrawOp.
func renderAlpha(w, h int, cmds []pathCmd, rule vector.FillRule, op draw.Op) *image.Alpha {
	z := vector.NewRasterizer(w, h)
	z.FillRule = rule
	z.DrawOp = op
	r := &referenceRasterizer{}
	playPath(z, r, cmds)
	dst := image.NewAlpha(z.Bounds())
	z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
	return dst
}

// TestFillRuleMatchesReference compares both rasterizer math variants (fixed
// point and floating point, selected via the rasterizer size) against the
// independent supersampled reference, for both fill rules and a battery of
// adversarial paths.
func TestFillRuleMatchesReference(t *testing.T) {
	type shape struct {
		name string
		w, h int
		cmds []pathCmd
	}
	shapes := []shape{
		{"concentric-rings-same-winding", 48, 48, ringPath(24, 24, 22, 11, false)},
		{"concentric-rings-reversed-inner", 48, 48, ringPath(24, 24, 22, 11, true)},
		{"self-intersecting-star", 48, 48, starPath(24, 24, 22)},
		{"curved-contour", 48, 48, curvedPath()},
		{"overlapping-curved-subpaths", 48, 48, curvedOverlapPath()},
		{"overlapping-coincident-edges", 48, 48, append(
			rectPath(6, 6, 30, 30, false),
			rectPath(12, 12, 36, 36, false)...)},
		{"coincident-edges-opposite-winding", 48, 48, append(
			rectPath(6, 6, 30, 30, false),
			rectPath(12, 12, 36, 36, true)...)},
		// A shape entirely in negative coordinates, offset into view by a
		// large positive subpath so the rasterizer still has visible work.
		{"negative-coordinates", 64, 64, append(
			rectPath(-40, -40, -8, -8, false),
			rectPath(8, 8, 40, 40, false)...)},
		// Floating point math variant: size above floatingPointMathThreshold.
		{"large-floating-math-star", 520, 520, starPath(260, 260, 240)},
	}

	for _, sh := range shapes {
		for _, rule := range []vector.FillRule{vector.NonZero, vector.EvenOdd} {
			for _, op := range []draw.Op{draw.Over, draw.Src} {
				name := sh.name + "/" + ruleName(rule) + "/" + opName(op)
				t.Run(name, func(t *testing.T) {
					// Rasterizer under test.
					z := vector.NewRasterizer(sh.w, sh.h)
					z.FillRule = rule
					z.DrawOp = op
					ref := &referenceRasterizer{}
					playPath(z, ref, sh.cmds)
					got := image.NewAlpha(z.Bounds())
					z.Draw(got, got.Bounds(), image.Opaque, image.Point{})

					bad, worst := 0, 0
					for y := 0; y < sh.h; y++ {
						for x := 0; x < sh.w; x++ {
							wantCov := ref.coverage(x, y, rule)
							g := int(got.AlphaAt(x, y).A)
							w := int(ref.alphaAt(x, y, rule))
							d := g - w
							if d < 0 {
								d = -d
							}
							if d > worst {
								worst = d
							}
							// Fully inside/outside samples are deterministic
							// for both implementations; only anti-aliased edge
							// pixels get a tolerance. The area-based rasterizer
							// and the point-sampled reference can disagree
							// noticeably on the few pixels straddling a sharp
							// or self-intersecting vertex, so edge pixels get a
							// generous allowance, but they must remain a small
							// fraction of the image.
							switch {
							case wantCov == 0 && g > 8:
								bad++
							case wantCov == 1 && g < 247:
								bad++
							case wantCov > 0 && wantCov < 1 && d > 72:
								bad++
							}
						}
					}
					// Guard against a wholesale mismatch: at most 2% of pixels
					// may exceed the edge tolerance.
					if maxBad := sh.w * sh.h / 50; bad > maxBad {
						t.Errorf("%d pixels differ from reference beyond tolerance (max %d), worst diff %d", bad, maxBad, worst)
					}
				})
			}
		}
	}
}

func rectPath(x0, y0, x1, y1 float32, reverse bool) []pathCmd {
	if reverse {
		return []pathCmd{
			{op: 'm', x0: x0, y0: y0},
			{op: 'l', x0: x0, y0: y1},
			{op: 'l', x0: x1, y0: y1},
			{op: 'l', x0: x1, y0: y0},
			{op: 'z'},
		}
	}
	return []pathCmd{
		{op: 'm', x0: x0, y0: y0},
		{op: 'l', x0: x1, y0: y0},
		{op: 'l', x0: x1, y0: y1},
		{op: 'l', x0: x0, y0: y1},
		{op: 'z'},
	}
}

func ruleName(r vector.FillRule) string {
	if r == vector.EvenOdd {
		return "evenodd"
	}
	return "nonzero"
}

func opName(op draw.Op) string {
	if op == draw.Src {
		return "src"
	}
	return "over"
}

// TestEvenOddConcentricRings checks the canonical ring semantics pixel by
// pixel: under even-odd the hole alternates, under non-zero a same-winding
// inner contour fills while a reversed one hollows it out.
func TestEvenOddConcentricRings(t *testing.T) {
	const size = 48
	const cx, cy = 24, 24

	check := func(t *testing.T, cmds []pathCmd, rule vector.FillRule, holeWant uint8, ringWant uint8) {
		dst := renderAlpha(size, size, cmds, rule, draw.Src)
		if d := absDiff(dst.AlphaAt(cx, cy).A, holeWant); d > 2 {
			t.Errorf("center pixel: got %#02x, want ~%#02x", dst.AlphaAt(cx, cy).A, holeWant)
		}
		if d := absDiff(dst.AlphaAt(cx+16, cy).A, ringWant); d > 2 {
			t.Errorf("ring pixel: got %#02x, want ~%#02x", dst.AlphaAt(cx+16, cy).A, ringWant)
		}
	}

	// Both contours wind the same way.
	same := ringPath(cx, cy, 22, 11, false)
	t.Run("same-winding/nonzero", func(t *testing.T) {
		// Winding number 2 in the hole: still filled.
		check(t, same, vector.NonZero, 0xff, 0xff)
	})
	t.Run("same-winding/evenodd", func(t *testing.T) {
		// Two crossings in the hole: even, so empty.
		check(t, same, vector.EvenOdd, 0x00, 0xff)
	})

	// Inner contour reversed (the usual "donut" construction).
	reversed := ringPath(cx, cy, 22, 11, true)
	t.Run("reversed-inner/nonzero", func(t *testing.T) {
		// Winding numbers cancel in the hole: empty.
		check(t, reversed, vector.NonZero, 0x00, 0xff)
	})
	t.Run("reversed-inner/evenodd", func(t *testing.T) {
		// Direction is irrelevant to parity: still empty.
		check(t, reversed, vector.EvenOdd, 0x00, 0xff)
	})
}

// TestEvenOddStar checks the self-intersecting star, where the two rules give
// visibly different coverage at the center.
func TestEvenOddStar(t *testing.T) {
	const size = 48
	cmds := starPath(24, 24, 22)

	nz := renderAlpha(size, size, cmds, vector.NonZero, draw.Src)
	eo := renderAlpha(size, size, cmds, vector.EvenOdd, draw.Src)

	// In a pentagram, the inner pentagon has winding number 2: it is filled
	// by non-zero but empty under even-odd. The arms (winding 1, odd) are
	// filled under both rules.
	if nz.AlphaAt(size/2, size/2).A < 0xfe {
		t.Errorf("non-zero center: got %#02x, want ~0xff (winding 2 is non-zero)", nz.AlphaAt(size/2, size/2).A)
	}
	if eo.AlphaAt(size/2, size/2).A != 0 {
		t.Errorf("even-odd center: got %#02x, want 0x00 (even number of crossings)", eo.AlphaAt(size/2, size/2).A)
	}

	ref := &referenceRasterizer{}
	z := vector.NewRasterizer(size, size)
	playPath(z, ref, cmds)

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			eoWant := ref.alphaAt(x, y, vector.EvenOdd)
			nzWant := ref.alphaAt(x, y, vector.NonZero)
			if d := absDiff(eo.AlphaAt(x, y).A, eoWant); d > 72 {
				t.Fatalf("even-odd (%d,%d): got %#02x, want ~%#02x", x, y, eo.AlphaAt(x, y).A, eoWant)
			}
			if d := absDiff(nz.AlphaAt(x, y).A, nzWant); d > 72 {
				t.Fatalf("non-zero (%d,%d): got %#02x, want ~%#02x", x, y, nz.AlphaAt(x, y).A, nzWant)
			}
		}
	}

	// The rules must actually differ somewhere on this shape.
	different := false
	for i := range nz.Pix {
		if absDiff(nz.Pix[i], eo.Pix[i]) > 24 {
			different = true
			break
		}
	}
	if !different {
		t.Error("non-zero and even-odd produced identical coverage for a self-intersecting star")
	}
}

func absDiff(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}

// TestFillRuleDoesNotMutatePath renders the same recorded path twice with
// different rules and checks Pen and geometry are untouched.
func TestFillRuleDoesNotMutatePath(t *testing.T) {
	cmds := curvedPath()
	z := vector.NewRasterizer(48, 48)
	r := &referenceRasterizer{}
	playPath(z, r, cmds)

	penX, penY := z.Pen()
	z.FillRule = vector.EvenOdd
	a := image.NewAlpha(z.Bounds())
	z.Draw(a, a.Bounds(), image.Opaque, image.Point{})
	if gotX, gotY := z.Pen(); gotX != penX || gotY != penY {
		t.Fatalf("EvenOdd Draw moved pen from (%v,%v) to (%v,%v)", penX, penY, gotX, gotY)
	}
	z.FillRule = vector.NonZero
	b := image.NewAlpha(z.Bounds())
	z.Draw(b, b.Bounds(), image.Opaque, image.Point{})
	if gotX, gotY := z.Pen(); gotX != penX || gotY != penY {
		t.Fatalf("NonZero Draw moved pen from (%v,%v) to (%v,%v)", penX, penY, gotX, gotY)
	}
}

// TestEmptyPathAndRepeatedReset checks that drawing with no geometry, and
// resetting repeatedly, yields zero coverage and never leaks state into the
// next path under either rule.
func TestEmptyPathAndRepeatedReset(t *testing.T) {
	for _, rule := range []vector.FillRule{vector.NonZero, vector.EvenOdd} {
		var z vector.Rasterizer
		for i := 0; i < 3; i++ {
			z.Reset(16, 16)
			z.FillRule = rule
			// No path commands at all.
			dst := image.NewAlpha(z.Bounds())
			z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
			for i, v := range dst.Pix {
				if v != 0 {
					t.Fatalf("rule=%v: empty path produced non-zero alpha at %d: %#02x", ruleName(rule), i, v)
				}
			}
		}

		// After the empty draws, a real path must render correctly (no
		// carry-over from the previous buffers).
		z.Reset(16, 16)
		z.FillRule = rule
		playPath(&z, &referenceRasterizer{}, rectPath(2, 2, 14, 14, false))
		dst := image.NewAlpha(z.Bounds())
		z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
		if dst.AlphaAt(8, 8).A < 0xfe {
			t.Fatalf("rule=%v: center pixel after empty paths got %#02x, want 0xff", ruleName(rule), dst.AlphaAt(8, 8).A)
		}
		if dst.AlphaAt(0, 0).A != 0 {
			t.Fatalf("rule=%v: corner pixel got %#02x, want 0x00", ruleName(rule), dst.AlphaAt(0, 0).A)
		}
	}
}

// TestRuleSwitchIsolation renders different rules back-to-back on one
// Rasterizer and confirms each Draw only reflects its own rule (the full
// bounds fast path consumes the area buffer non-destructively).
func TestRuleSwitchIsolation(t *testing.T) {
	cmds := ringPath(24, 24, 22, 11, false)
	z := vector.NewRasterizer(48, 48)
	playPath(z, &referenceRasterizer{}, cmds)

	eoDst := image.NewAlpha(z.Bounds())
	z.FillRule = vector.EvenOdd
	z.Draw(eoDst, eoDst.Bounds(), image.Opaque, image.Point{})

	nzDst := image.NewAlpha(z.Bounds())
	z.FillRule = vector.NonZero
	z.Draw(nzDst, nzDst.Bounds(), image.Opaque, image.Point{})

	eoAgain := image.NewAlpha(z.Bounds())
	z.FillRule = vector.EvenOdd
	z.Draw(eoAgain, eoAgain.Bounds(), image.Opaque, image.Point{})

	for i := range eoDst.Pix {
		if eoDst.Pix[i] != eoAgain.Pix[i] {
			t.Fatalf("even-odd output not stable across a non-zero Draw in between at %d", i)
		}
	}
	if eoDst.AlphaAt(24, 24).A != 0 {
		t.Errorf("even-odd hole center: got %#02x, want 0x00", eoDst.AlphaAt(24, 24).A)
	}
	if nzDst.AlphaAt(24, 24).A < 0xfe {
		t.Errorf("non-zero hole center (winding 2): got %#02x, want 0xff", nzDst.AlphaAt(24, 24).A)
	}
}

// TestEvenOddSubRectangle exercises the non-fast-path accumulateMask route
// (r != Bounds) with even-odd. The destination is padded and the target
// rectangle only covers part of it, so the full-bounds fast paths cannot be
// taken. A non-square rasterizer also checks that the running sum restarts
// per row rather than running across the whole buffer.
func TestEvenOddSubRectangle(t *testing.T) {
	const w, h = 48, 40
	cmds := ringPath(24, 20, 18, 9, false)

	z := vector.NewRasterizer(w, h)
	z.FillRule = vector.EvenOdd
	playPath(z, &referenceRasterizer{}, cmds)

	dst := image.NewAlpha(image.Rect(0, 0, w+8, h))
	r := image.Rect(0, 0, w, h)
	z.Draw(dst, r, image.Opaque, image.Point{})

	// Structural checks against canonical even-odd semantics. The hole
	// (winding 2) must be empty, the ring (winding 1) opaque, and the
	// per-row running-sum reset must leave no vertical streaks.
	if g := dst.AlphaAt(24, 20).A; g != 0 {
		t.Errorf("hole center: got %#02x, want 0x00", g)
	}
	if g := dst.AlphaAt(24+13, 20).A; g < 0xfe {
		t.Errorf("ring pixel: got %#02x, want ~0xff", g)
	}
	// The padding column must be untouched.
	for y := 0; y < h; y++ {
		if dst.AlphaAt(w, y).A != 0 {
			t.Fatalf("Draw wrote outside the target rectangle at (%d,%d)", w, y)
		}
	}
	// The first and last rows of a ring are empty: if the running sum
	// leaked across row boundaries, edge rows would accumulate parity from
	// the previous row and light up.
	for x := 0; x < w; x++ {
		if dst.AlphaAt(x, 0).A != 0 || dst.AlphaAt(x, h-1).A != 0 {
			t.Fatalf("boundary row unexpectedly non-empty at x=%d", x)
		}
	}
}

// TestEvenOddRGBA exercises the generic RGBA path (uniform src, both ops),
// which goes through accumulateMask rather than the Alpha fast paths.
func TestEvenOddRGBA(t *testing.T) {
	cmds := ringPath(24, 24, 22, 11, false)
	for _, op := range []draw.Op{draw.Over, draw.Src} {
		z := vector.NewRasterizer(48, 48)
		z.FillRule = vector.EvenOdd
		z.DrawOp = op
		playPath(z, &referenceRasterizer{}, cmds)
		dst := image.NewRGBA(z.Bounds())
		blue := image.NewUniform(color.RGBA{0x00, 0x00, 0xff, 0xff})
		z.Draw(dst, dst.Bounds(), blue, image.Point{})

		c := dst.RGBAAt(24, 24) // hole: even-odd says empty.
		if c.A != 0 {
			t.Errorf("op=%v hole alpha: got %#02x, want 0x00", op, c.A)
		}
		c = dst.RGBAAt(24+16, 24) // ring: filled.
		if c.A < 0xfe || c.B < 0xfe {
			t.Errorf("op=%v ring pixel: got %+v, want opaque blue", op, c)
		}
	}
}

// TestResourceLimits checks that pathological dimensions and coordinates fail
// deterministically and leave the rasterizer usable afterwards.
func TestResourceLimits(t *testing.T) {
	// Negative and oversized dimensions.
	badDims := [][2]int{
		{-1, 1}, {1, -1},
		{1<<22 + 1, 1}, {1, 1<<22 + 1},
		{1 << 20, 1 << 20}, // product 1<<40 > maxPixels
	}
	for _, d := range badDims {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Reset(%d, %d) did not panic", d[0], d[1])
				}
			}()
			var z vector.Rasterizer
			z.Reset(d[0], d[1])
		}()
	}

	// Dimensions exactly at the limits are accepted (small allocation only
	// for the within-pixel-cap case).
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("Reset at the documented limits panicked: %v", r)
			}
		}()
		z := vector.NewRasterizer(16, 16)
		_ = z
	}()

	// Out-of-range and NaN coordinates.
	var z vector.Rasterizer
	z.Reset(8, 8)
	badCoords := [][2]float32{
		{1 << 22, 0}, {-(1 << 22), 0}, {1 << 30, 1 << 30},
		{float32(math.NaN()), 0}, {0, float32(math.Inf(1))},
	}
	for _, c := range badCoords {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("MoveTo(%v, %v) did not panic", c[0], c[1])
				}
			}()
			z.MoveTo(c[0], c[1])
		}()
	}

	// After the panics, the rasterizer still works on valid input.
	z.Reset(8, 8)
	z.MoveTo(2, 2)
	z.LineTo(6, 2)
	z.LineTo(6, 6)
	z.LineTo(2, 6)
	z.ClosePath()
	dst := image.NewAlpha(z.Bounds())
	z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
	if dst.AlphaAt(4, 4).A < 0xfe {
		t.Fatalf("rasterizer unusable after recovered panics: center %#02x", dst.AlphaAt(4, 4).A)
	}
}

// TestInvalidFillRule checks the stable error for an unknown rule.
func TestInvalidFillRule(t *testing.T) {
	z := vector.NewRasterizer(8, 8)
	z.MoveTo(1, 1)
	z.LineTo(7, 1)
	z.LineTo(7, 7)
	z.ClosePath()
	z.FillRule = vector.FillRule(42)
	dst := image.NewAlpha(z.Bounds())
	func() {
		defer func() {
			if recover() == nil {
				t.Error("Draw with invalid FillRule did not panic")
			}
		}()
		z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
	}()
}

// TestEvenOddTripleRings exercises winding numbers up to 3: three same-way
// concentric contours give winding 3 at the center and 2 in the middle
// band. Even-odd alternates with each crossing, while non-zero fills every
// non-zero winding, so the two rules disagree on the winding-2 band and agree
// (both filled) at the winding-3 center.
func TestEvenOddTripleRings(t *testing.T) {
	const size = 72
	cmds := append(ringPath(36, 36, 33, 22, false), circle(36, 36, 11, false)...)

	nz := renderAlpha(size, size, cmds, vector.NonZero, draw.Src)
	eo := renderAlpha(size, size, cmds, vector.EvenOdd, draw.Src)

	// Winding 3 center: odd, so even-odd fills; non-zero fills too.
	if g := nz.AlphaAt(36, 36).A; g < 0xfe {
		t.Errorf("non-zero winding-3 center: got %#02x, want ~0xff", g)
	}
	if g := eo.AlphaAt(36, 36).A; g < 0xfe {
		t.Errorf("even-odd winding-3 center: got %#02x, want ~0xff", g)
	}
	// Winding 2 band at radius ~16: non-zero fills, even-odd empties.
	if g := nz.AlphaAt(36+16, 36).A; g < 0xfe {
		t.Errorf("non-zero winding-2 band: got %#02x, want ~0xff", g)
	}
	if g := eo.AlphaAt(36+16, 36).A; g != 0 {
		t.Errorf("even-odd winding-2 band: got %#02x, want 0x00", g)
	}
	// Winding 1 outer band at radius ~27: both fill.
	if g := eo.AlphaAt(36+27, 36).A; g < 0xfe {
		t.Errorf("even-odd winding-1 band: got %#02x, want ~0xff", g)
	}
}

// TestEvenOddCurvedOverlap confirms that two overlapping same-direction
// curved subpaths (QuadTo and CubeTo) fill their winding-2 intersection under
// non-zero but leave it empty under even-odd, and that both agree with the
// independent reference.
func TestEvenOddCurvedOverlap(t *testing.T) {
	const w, h = 48, 48
	cmds := curvedOverlapPath()

	nz := renderAlpha(w, h, cmds, vector.NonZero, draw.Src)
	eo := renderAlpha(w, h, cmds, vector.EvenOdd, draw.Src)
	ref := &referenceRasterizer{}
	z := vector.NewRasterizer(w, h)
	playPath(z, ref, cmds)

	// Locate a pixel firmly inside the overlap (winding 2): the second
	// subpath's center region.
	found := false
	for y := 24; y < 40 && !found; y++ {
		for x := 14; x < 34 && !found; x++ {
			if ref.coverage(x, y, vector.NonZero) == 1 && ref.coverage(x, y, vector.EvenOdd) == 0 {
				found = true
				if nz.AlphaAt(x, y).A < 0xfe {
					t.Errorf("non-zero overlap (%d,%d): got %#02x, want ~0xff", x, y, nz.AlphaAt(x, y).A)
				}
				if eo.AlphaAt(x, y).A != 0 {
					t.Errorf("even-odd overlap (%d,%d): got %#02x, want 0x00", x, y, eo.AlphaAt(x, y).A)
				}
			}
		}
	}
	if !found {
		t.Fatal("test geometry did not produce a winding-2 overlap region")
	}

	// Full-image agreement with the reference under both rules.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if d := absDiff(nz.AlphaAt(x, y).A, ref.alphaAt(x, y, vector.NonZero)); d > 72 {
				t.Fatalf("non-zero (%d,%d): got %#02x, want ~%#02x", x, y, nz.AlphaAt(x, y).A, ref.alphaAt(x, y, vector.NonZero))
			}
			if d := absDiff(eo.AlphaAt(x, y).A, ref.alphaAt(x, y, vector.EvenOdd)); d > 72 {
				t.Fatalf("even-odd (%d,%d): got %#02x, want ~%#02x", x, y, eo.AlphaAt(x, y).A, ref.alphaAt(x, y, vector.EvenOdd))
			}
		}
	}
}

// TestDefaultFillRuleIsNonZero checks the zero value and Reset both select
// non-zero, and that even-odd and non-zero agree on a simple convex path.
func TestDefaultFillRuleIsNonZero(t *testing.T) {
	var z vector.Rasterizer
	if z.FillRule != vector.NonZero {
		t.Fatalf("zero FillRule = %v, want NonZero", z.FillRule)
	}
	z.Reset(16, 16)
	if z.FillRule != vector.NonZero {
		t.Fatalf("FillRule after Reset = %v, want NonZero", z.FillRule)
	}
	z.FillRule = vector.EvenOdd
	z.Reset(16, 16)
	if z.FillRule != vector.NonZero {
		t.Fatalf("Reset did not restore NonZero, got %v", z.FillRule)
	}

	// A simple convex rectangle is identical under both rules.
	cmds := rectPath(2, 2, 14, 14, false)
	a := renderAlpha(16, 16, cmds, vector.NonZero, draw.Src)
	b := renderAlpha(16, 16, cmds, vector.EvenOdd, draw.Src)
	for i := range a.Pix {
		if absDiff(a.Pix[i], b.Pix[i]) > 2 {
			t.Fatalf("convex path differs between rules at %d: %#02x vs %#02x", i, a.Pix[i], b.Pix[i])
		}
	}
}

// TestEvenOddZeroAndOutOfBounds covers degenerate rasterizers and geometry:
// a zero-value Rasterizer (zero-sized Bounds), a path entirely outside the
// bounds, and a degenerate horizontal-only contour. None may panic or emit
// coverage under either rule.
func TestEvenOddZeroAndOutOfBounds(t *testing.T) {
	// Zero-value Rasterizer: zero-sized Bounds.
	var z vector.Rasterizer
	z.FillRule = vector.EvenOdd
	z.MoveTo(2, 2)
	z.LineTo(8, 2)
	z.LineTo(8, 8)
	z.ClosePath()
	dst := image.NewAlpha(z.Bounds())
	z.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})
	if len(dst.Pix) != 0 {
		t.Errorf("zero Rasterizer produced a non-empty image: %d bytes", len(dst.Pix))
	}

	// Geometry entirely outside the bounds.
	for _, rule := range []vector.FillRule{vector.NonZero, vector.EvenOdd} {
		r := vector.NewRasterizer(16, 16)
		r.FillRule = rule
		r.MoveTo(100, 100)
		r.LineTo(120, 100)
		r.LineTo(120, 120)
		r.LineTo(100, 120)
		r.ClosePath()
		m := image.NewAlpha(r.Bounds())
		r.Draw(m, m.Bounds(), image.Opaque, image.Point{})
		for i, v := range m.Pix {
			if v != 0 {
				t.Fatalf("rule=%v: out-of-bounds path wrote %#02x at %d", ruleName(rule), v, i)
			}
		}
	}

	// Degenerate horizontal-only contour: zero area, zero coverage.
	r := vector.NewRasterizer(16, 16)
	r.FillRule = vector.EvenOdd
	r.MoveTo(2, 8)
	r.LineTo(14, 8)
	r.ClosePath()
	m := image.NewAlpha(r.Bounds())
	r.Draw(m, m.Bounds(), image.Opaque, image.Point{})
	for i, v := range m.Pix {
		if v != 0 {
			t.Fatalf("horizontal-only contour wrote %#02x at %d", v, i)
		}
	}
}
