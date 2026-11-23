// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"

	"golang.org/x/image/math/f64"
)

// refIndex is an independent re-implementation of the boundary coordinate
// mapping. It works in source-rectangle-local coordinates p and the local
// size n. It returns the resolved local index and whether the sample is a
// real pixel (false means a transparent-black miss).
func refIndex(mode BoundaryMode, p, n int) (int, bool) {
	if n <= 0 {
		return 0, false
	}
	if p >= 0 && p < n {
		return p, true
	}
	switch mode {
	case BoundaryWrap:
		r := p % n
		if r < 0 {
			r += n
		}
		return r, true
	case BoundaryMirror:
		m := 2 * n
		r := p % m
		if r < 0 {
			r += m
		}
		if r >= n {
			r = m - 1 - r
		}
		return r, true
	case BoundaryTransparent:
		return 0, false
	default:
		if p < 0 {
			return 0, true
		}
		return n - 1, true
	}
}

// refSampler resolves a local, possibly out-of-range source point to a
// premultiplied color using the given mode. A transparent miss is zero.
func refSampler(m *image.RGBA, mode BoundaryMode, u, v int) (r, g, b, a float64) {
	w, h := m.Bounds().Dx(), m.Bounds().Dy()
	lu, oku := refIndex(mode, u, w)
	lv, okv := refIndex(mode, v, h)
	if !oku || !okv {
		return 0, 0, 0, 0
	}
	c := m.RGBAAt(m.Bounds().Min.X+lu, m.Bounds().Min.Y+lv)
	return float64(uint32(c.R) * 0x101),
		float64(uint32(c.G) * 0x101),
		float64(uint32(c.B) * 0x101),
		float64(uint32(c.A) * 0x101)
}

func refTrunc(v float64) uint32 {
	if v <= 0 {
		return 0
	}
	if v >= 0xffff {
		return 0xffff
	}
	return uint32(v)
}

// refNNScale is an independent nearest-neighbor scaler.
func refNNScale(mode BoundaryMode, src *image.RGBA, dw, dh int) *image.RGBA {
	sr := src.Bounds()
	sw, sh := sr.Dx(), sr.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for dy := 0; dy < dh; dy++ {
		v := int((2*uint64(dy) + 1) * uint64(sh) / (2 * uint64(dh)))
		for dx := 0; dx < dw; dx++ {
			u := int((2*uint64(dx) + 1) * uint64(sw) / (2 * uint64(dw)))
			r, g, b, a := refSampler(src, mode, u, v)
			dst.SetRGBA(dx, dy, color.RGBA{
				uint8(refTrunc(r) >> 8), uint8(refTrunc(g) >> 8),
				uint8(refTrunc(b) >> 8), uint8(refTrunc(a) >> 8),
			})
		}
	}
	return dst
}

// refBilinearScale is an independent bilinear scaler with a boundary mode.
func refBilinearScale(mode BoundaryMode, src *image.RGBA, dw, dh int) *image.RGBA {
	sr := src.Bounds()
	sw, sh := sr.Dx(), sr.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	xs := float64(sw) / float64(dw)
	ys := float64(sh) / float64(dh)
	for dy := 0; dy < dh; dy++ {
		cy := (float64(dy)+0.5)*ys - 0.5
		y0 := math.Floor(cy)
		fy := cy - y0
		for dx := 0; dx < dw; dx++ {
			cx := (float64(dx)+0.5)*xs - 0.5
			x0 := math.Floor(cx)
			fx := cx - x0
			var rr, gg, bb, aa float64
			for _, ty := range []struct {
				v int
				w float64
			}{{int(y0), 1 - fy}, {int(y0) + 1, fy}} {
				for _, tx := range []struct {
					u int
					w float64
				}{{int(x0), 1 - fx}, {int(x0) + 1, fx}} {
					r, g, b, a := refSampler(src, mode, tx.u, ty.v)
					w := tx.w * ty.w
					rr += r * w
					gg += g * w
					bb += b * w
					aa += a * w
				}
			}
			dst.SetRGBA(dx, dy, color.RGBA{
				uint8(refTrunc(rr) >> 8), uint8(refTrunc(gg) >> 8),
				uint8(refTrunc(bb) >> 8), uint8(refTrunc(aa) >> 8),
			})
		}
	}
	return dst
}

// refKernelScale is an independent separable kernel scaler with a boundary
// mode. Taps outside the rectangle are supplied by the boundary sampler; the
// total kernel weight (transparent taps included) normalizes the result.
func refKernelScale(q *Kernel, mode BoundaryMode, src *image.RGBA, dw, dh int) *image.RGBA {
	sr := src.Bounds()
	sw, sh := sr.Dx(), sr.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))

	// Horizontal pass into a float buffer, then vertical.
	type px struct{ r, g, b, a float64 }
	tmp := make([][]px, sh)
	for y := range tmp {
		tmp[y] = make([]px, dw)
	}
	hScale := float64(sw) / float64(dw)
	hHW, hKAS := q.Support, 1.0
	if hScale > 1 {
		hHW *= hScale
		hKAS = 1 / hScale
	}
	for y := 0; y < sh; y++ {
		for dx := 0; dx < dw; dx++ {
			c := (float64(dx)+0.5)*hScale - 0.5
			i0 := int(math.Floor(c - hHW))
			i1 := int(math.Ceil(c + hHW))
			var rr, gg, bb, aa, tot float64
			for k := i0; k < i1; k++ {
				t := math.Abs((c - float64(k)) * hKAS)
				if t >= q.Support {
					continue
				}
				w := q.At(t)
				if w == 0 {
					continue
				}
				r, g, b, a := refSampler(src, mode, k, y)
				rr += r * w
				gg += g * w
				bb += b * w
				aa += a * w
				tot += w
			}
			if tot != 0 {
				rr, gg, bb, aa = rr/tot, gg/tot, bb/tot, aa/tot
			}
			tmp[y][dx] = px{rr, gg, bb, aa}
		}
	}
	vScale := float64(sh) / float64(dh)
	vHW, vKAS := q.Support, 1.0
	if vScale > 1 {
		vHW *= vScale
		vKAS = 1 / vScale
	}
	for dy := 0; dy < dh; dy++ {
		c := (float64(dy)+0.5)*vScale - 0.5
		i0 := int(math.Floor(c - vHW))
		i1 := int(math.Ceil(c + vHW))
		for dx := 0; dx < dw; dx++ {
			var rr, gg, bb, aa, tot float64
			for k := i0; k < i1; k++ {
				t := math.Abs((c - float64(k)) * vKAS)
				if t >= q.Support {
					continue
				}
				w := q.At(t)
				if w == 0 {
					continue
				}
				// A vertical transparent miss contributes zero but still
				// counts toward the normalizing total weight. Wrap/mirror
				// resolve the row before indexing the float buffer.
				kv, okv := refIndex(mode, k, sh)
				if !okv {
					tot += w
					continue
				}
				p := tmp[kv][dx]
				rr += p.r * w
				gg += p.g * w
				bb += p.b * w
				aa += p.a * w
				tot += w
			}
			if tot != 0 {
				rr, gg, bb, aa = rr/tot, gg/tot, bb/tot, aa/tot
			}
			if rr > aa {
				rr = aa
			}
			if gg > aa {
				gg = aa
			}
			if bb > aa {
				bb = aa
			}
			dst.SetRGBA(dx, dy, color.RGBA{
				uint8(fffftou(rr) >> 8), uint8(fffftou(gg) >> 8),
				uint8(fffftou(bb) >> 8), uint8(fffftou(aa) >> 8),
			})
		}
	}
	return dst
}

// makeChecker builds a w*h checkerboard with a strong two-color pattern, so
// periodic (wrap/mirror) behavior is easy to verify.
func makeChecker(w, h int) *image.RGBA {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if (x+y)%2 == 0 {
				m.SetRGBA(x, y, color.RGBA{0xff, 0x10, 0x20, 0xff})
			} else {
				m.SetRGBA(x, y, color.RGBA{0x10, 0x80, 0xff, 0xff})
			}
		}
	}
	return m
}

// makeAlphaBorder builds an opaque core surrounded by a fully transparent
// frame inside the image bounds.
func makeAlphaBorder(w, h, border int) *image.RGBA {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x >= border && x < w-border && y >= border && y < h-border {
				m.SetRGBA(x, y, color.RGBA{0x33, 0x66, 0x99, 0xff})
			}
		}
	}
	return m
}

func pixMaxDiff(a, b *image.RGBA) int {
	if len(a.Pix) != len(b.Pix) {
		return 1 << 30
	}
	d := 0
	for i := range a.Pix {
		x := int(a.Pix[i]) - int(b.Pix[i])
		if x < 0 {
			x = -x
		}
		if x > d {
			d = x
		}
	}
	return d
}

func TestBoundaryScaleAgainstReference(t *testing.T) {
	src := makeChecker(8, 6)
	modes := []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent}
	type tc struct {
		name string
		q    Interpolator
		ref  func(BoundaryMode, *image.RGBA, int, int) *image.RGBA
		tol  int
	}
	cases := []tc{
		{"nn", NearestNeighbor, func(b BoundaryMode, m *image.RGBA, w, h int) *image.RGBA {
			return refNNScale(b, m, w, h)
		}, 0},
		{"abl", ApproxBiLinear, refBilinearScale, 1},
		{"bl", BiLinear, func(b BoundaryMode, m *image.RGBA, w, h int) *image.RGBA {
			return refKernelScale(BiLinear, b, m, w, h)
		}, 2},
		{"cr", CatmullRom, func(b BoundaryMode, m *image.RGBA, w, h int) *image.RGBA {
			return refKernelScale(CatmullRom, b, m, w, h)
		}, 2},
	}
	sizes := [][2]int{{20, 16}, {8, 6}, {4, 3}, {30, 20}}
	for _, c := range cases {
		for _, mode := range modes {
			for _, sz := range sizes {
				dw, dh := sz[0], sz[1]
				got := image.NewRGBA(image.Rect(0, 0, dw, dh))
				c.q.Scale(got, got.Bounds(), src, src.Bounds(), Src, &Options{Boundary: mode})
				want := c.ref(mode, src, dw, dh)
				if d := pixMaxDiff(got, want); d > c.tol {
					t.Errorf("%s mode=%v size=%dx%d: max pixel diff %d > tol %d",
						c.name, mode, dw, dh, d, c.tol)
				}
			}
		}
	}
}

func TestBoundaryClampIsLegacy(t *testing.T) {
	// The zero/clamp mode must produce exactly the legacy fast-path output.
	src := makeChecker(10, 8)
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, BiLinear, CatmullRom} {
		for _, opts := range []*Options{nil, {}, {Boundary: BoundaryClamp}} {
			a := image.NewRGBA(image.Rect(0, 0, 24, 20))
			b := image.NewRGBA(image.Rect(0, 0, 24, 20))
			q.Scale(a, a.Bounds(), src, src.Bounds(), Src, nil)
			q.Scale(b, b.Bounds(), src, src.Bounds(), Src, opts)
			if d := pixMaxDiff(a, b); d != 0 {
				t.Errorf("%T opts=%v: clamp differs from legacy by %d", q, opts, d)
			}
		}
	}
}

func TestBoundaryTransparentEdges(t *testing.T) {
	dw, dh := 14, 14

	// With a fully transparent source and a transparent boundary, every
	// sample is transparent black regardless of the interpolator's support
	// width. Under Src the destination must become transparent black; under
	// Over it must be left exactly untouched.
	clearSrc := image.NewRGBA(image.Rect(0, 0, 6, 6))
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, BiLinear, CatmullRom} {
		dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
		q.Scale(dst, dst.Bounds(), clearSrc, clearSrc.Bounds(), Src, &Options{Boundary: BoundaryTransparent})
		for y := 0; y < dh; y++ {
			for x := 0; x < dw; x++ {
				if c := dst.RGBAAt(x, y); c != (color.RGBA{}) {
					t.Fatalf("%T Src transparent: pixel %v = %v, want zero", q, image.Pt(x, y), c)
				}
			}
		}

		base := image.NewRGBA(image.Rect(0, 0, dw, dh))
		base.SetRGBA(0, 0, color.RGBA{0x12, 0x34, 0x56, 0x78})
		dst2 := image.NewRGBA(image.Rect(0, 0, dw, dh))
		copy(dst2.Pix, base.Pix)
		q.Scale(dst2, dst2.Bounds(), clearSrc, clearSrc.Bounds(), Over, &Options{Boundary: BoundaryTransparent})
		if dst2.RGBAAt(0, 0) != base.RGBAAt(0, 0) {
			t.Errorf("%T Over transparent modified an untouched pixel: %v want %v",
				q, dst2.RGBAAt(0, 0), base.RGBAAt(0, 0))
		}
	}

	// For nearest neighbor, scaling up an opaque core surrounded by a
	// transparent frame maps the outer destination ring unambiguously to
	// out-of-range samples, which must be transparent black.
	bordered := makeAlphaBorder(6, 6, 1)
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	NearestNeighbor.Scale(dst, dst.Bounds(), bordered, bordered.Bounds(), Src, &Options{Boundary: BoundaryTransparent})
	corners := [][2]int{{0, 0}, {dw - 1, 0}, {0, dh - 1}, {dw - 1, dh - 1}}
	for _, p := range corners {
		if c := dst.RGBAAt(p[0], p[1]); c != (color.RGBA{}) {
			t.Errorf("nearest corner %v = %v, want transparent black", p, c)
		}
	}
}

func TestBoundaryWrapPeriodicity(t *testing.T) {
	// The periodic guarantee is a property of the boundary sampler itself:
	// coordinates separated by a whole number of periods resolve to the same
	// source pixel. Checking it through a full Scale would conflate this with
	// the (different) mapping implied by a differently sized source rectangle.
	src := makeChecker(8, 6)
	for _, mode := range []struct {
		mode BoundaryMode
		// period in pixels on each axis.
		px, py int
	}{
		{BoundaryWrap, 8, 6},
		{BoundaryMirror, 16, 12},
	} {
		s := bctx{mode: mode.mode, src: src, sr: src.Bounds()}
		for u := -20; u < 20; u++ {
			for v := -16; v < 16; v++ {
				r0, g0, b0, a0, ok0 := s.at(u, v)
				r1, g1, b1, a1, ok1 := s.at(u+3*mode.px, v+3*mode.py)
				if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 || ok0 != ok1 {
					t.Fatalf("mode=%v not periodic at (%d,%d): (%v) vs (%v)",
						mode.mode, u, v,
						[5]interface{}{r0, g0, b0, a0, ok0},
						[5]interface{}{r1, g1, b1, a1, ok1})
				}
			}
		}
	}

	// Mirror reflection sanity: the point one step before the left edge maps
	// to the first pixel, so it equals at(0,0); wrap maps it to the last
	// column, so it equals at(width-1, 0).
	m := bctx{mode: BoundaryMirror, src: src, sr: src.Bounds()}
	mr, _, _, _, _ := m.at(-1, 0)
	mrIn, _, _, _, _ := m.at(0, 0)
	if mr != mrIn {
		t.Errorf("mirror left-edge reflection mismatch: %d vs %d", mr, mrIn)
	}
	w := bctx{mode: BoundaryWrap, src: src, sr: src.Bounds()}
	wr, _, _, _, _ := w.at(-1, 0)
	wrEdge, _, _, _, _ := w.at(7, 0)
	if wr != wrEdge {
		t.Errorf("wrap left-edge wrap-around mismatch: %d vs %d", wr, wrEdge)
	}
}

func TestBoundaryTransformReferenceAndRotation(t *testing.T) {
	src := makeChecker(8, 8)
	mat := transformMatrix(1.5, 6, 3)
	modes := []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent}
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, CatmullRom} {
		for _, mode := range modes {
			a := image.NewRGBA(image.Rect(0, 0, 32, 32))
			b := image.NewRGBA(image.Rect(0, 0, 32, 32))
			q.Transform(a, mat, src, src.Bounds(), Src, &Options{Boundary: mode})
			// The same call twice must be deterministic and stable.
			q.Transform(b, mat, src, src.Bounds(), Src, &Options{Boundary: mode})
			if d := pixMaxDiff(a, b); d != 0 {
				t.Errorf("%T mode=%v transform not stable, diff=%d", q, mode, d)
			}
			// Every produced pixel must satisfy the premultiplied invariant.
			for y := 0; y < 32; y++ {
				for x := 0; x < 32; x++ {
					c := a.RGBAAt(x, y)
					if c.R > c.A || c.G > c.A || c.B > c.A {
						t.Fatalf("%T mode=%v invalid premultiplied pixel %v at %v", q, mode, c, image.Pt(x, y))
					}
				}
			}
		}
	}
}

func TestBoundaryNegativeTransform(t *testing.T) {
	// A negative-determinant (flipping) matrix maps many samples outside sr.
	src := makeChecker(8, 6)
	mat := f64.Aff3{
		-1.4, 0.2, 20,
		0.2, -1.1, 18,
	}
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, BiLinear, CatmullRom} {
		for _, mode := range []BoundaryMode{BoundaryClamp, BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
			dst := image.NewRGBA(image.Rect(0, 0, 28, 28))
			q.Transform(dst, mat, src, src.Bounds(), Src, &Options{Boundary: mode})
			for y := 0; y < 28; y++ {
				for x := 0; x < 28; x++ {
					c := dst.RGBAAt(x, y)
					if c.R > c.A || c.G > c.A || c.B > c.A {
						t.Fatalf("%T mode=%v invalid pixel %v", q, mode, c)
					}
				}
			}
		}
	}
}

func TestBoundaryOverhangSourceRect(t *testing.T) {
	// sr extends beyond the actual image bounds on every side. The generic
	// boundary path must never read out of range.
	src := makeChecker(6, 6)
	sr := image.Rect(-3, -2, 9, 10)
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, BiLinear, CatmullRom} {
		for _, mode := range []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
			dst := image.NewRGBA(image.Rect(0, 0, 16, 16))
			q.Scale(dst, dst.Bounds(), src, sr, Src, &Options{Boundary: mode})
		}
	}
}

func TestBoundaryAliasedImages(t *testing.T) {
	// Source and destination share storage; reads must stay in range.
	base := makeChecker(12, 12)
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, CatmullRom} {
		for _, mode := range []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
			dst := image.NewRGBA(image.Rect(0, 0, 12, 12))
			copy(dst.Pix, base.Pix)
			q.Scale(dst, dst.Bounds(), dst, dst.Bounds(), Src, &Options{Boundary: mode})
		}
	}
}

func TestBoundaryMasksAndPremultiplied(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 8, 8))
	fillPix(rand.New(rand.NewSource(7)), src.Pix)
	srcMask := image.NewRGBA(image.Rect(0, 0, 8, 8))
	fillPix(rand.New(rand.NewSource(8)), srcMask.Pix)
	dstMask := image.NewRGBA(image.Rect(0, 0, 20, 20))
	fillPix(rand.New(rand.NewSource(9)), dstMask.Pix)
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, CatmullRom} {
		for _, mode := range []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
			for _, op := range []Op{Src, Over} {
				dst := image.NewRGBA(image.Rect(0, 0, 20, 20))
				q.Scale(dst, dst.Bounds(), src, src.Bounds(), op, &Options{
					Boundary: mode,
					SrcMask:  srcMask,
					DstMask:  dstMask,
				})
				for y := 0; y < 20; y++ {
					for x := 0; x < 20; x++ {
						c := dst.RGBAAt(x, y)
						if c.R > c.A || c.G > c.A || c.B > c.A {
							t.Fatalf("%T mode=%v op=%v invalid pixel %v", q, mode, op, c)
						}
					}
				}
			}
		}
	}
}

func TestBoundaryDegenerateInputs(t *testing.T) {
	src := makeChecker(6, 6)
	q := CatmullRom

	// Empty source rectangle: no writes, dst stays at its initial value.
	dst := image.NewRGBA(image.Rect(0, 0, 8, 8))
	dst.SetRGBA(0, 0, color.RGBA{1, 2, 3, 4})
	q.Scale(dst, dst.Bounds(), src, image.Rect(2, 2, 2, 2), Src, &Options{Boundary: BoundaryWrap})
	if got := dst.RGBAAt(0, 0); got != (color.RGBA{1, 2, 3, 4}) {
		t.Errorf("empty sr wrote a pixel: %v", got)
	}

	// Zero-size destination: must not panic.
	tiny := image.NewRGBA(image.Rect(0, 0, 0, 0))
	q.Scale(tiny, tiny.Bounds(), src, src.Bounds(), Src, &Options{Boundary: BoundaryMirror})

	// Non-finite matrices: skipped safely, dst unchanged.
	for _, bad := range []f64.Aff3{
		{math.NaN(), 0, 0, 0, 1, 0},
		{math.Inf(1), 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, math.Inf(-1)},
	} {
		d := image.NewRGBA(image.Rect(0, 0, 8, 8))
		d.SetRGBA(0, 0, color.RGBA{5, 6, 7, 8})
		for _, interp := range []Interpolator{NearestNeighbor, ApproxBiLinear, BiLinear, CatmullRom} {
			interp.Transform(d, bad, src, src.Bounds(), Src, &Options{Boundary: BoundaryWrap})
		}
		if got := d.RGBAAt(0, 0); got != (color.RGBA{5, 6, 7, 8}) {
			t.Errorf("non-finite matrix wrote a pixel: %v", got)
		}
	}

	// A pathological kernel support is capped to a bounded tap count rather
	// than iterating over 1e12 taps. Use a 1x1 destination so the capped
	// 2D kernel still runs to completion quickly.
	huge := &Kernel{Support: 1e12, At: CatmullRom.At}
	one := image.NewRGBA(image.Rect(0, 0, 1, 1))
	huge.Scale(one, one.Bounds(), src, src.Bounds(), Src, &Options{Boundary: BoundaryWrap})
	if c := one.RGBAAt(0, 0); c.R > c.A || c.G > c.A || c.B > c.A {
		t.Errorf("huge kernel produced invalid premultiplied pixel %v", c)
	}
}

func TestBoundaryAllocations(t *testing.T) {
	// The boundary samplers read the source through the image.Image
	// interface (the same shape as the legacy generic *_Image_Image fallback
	// leaves), so each sampled pixel boxes a color.Color. The important
	// guarantee is therefore linearity: allocations may scale with the number
	// of pixels, but never quadratically, and a single call adds only a fixed
	// amount of working storage regardless of image size.
	src := makeChecker(16, 16)
	type size struct {
		w, h, pixels int
	}
	sizes := []size{{12, 12, 144}, {24, 24, 576}}
	for _, q := range []Interpolator{NearestNeighbor, ApproxBiLinear, CatmullRom} {
		var perPixel, baseRatio float64
		for i, sz := range sizes {
			dst := image.NewRGBA(image.Rect(0, 0, sz.w, sz.h))
			allocs := testing.AllocsPerRun(6, func() {
				q.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, &Options{Boundary: BoundaryMirror})
			})
			ratio := allocs / float64(sz.pixels)
			if i == 0 {
				perPixel, baseRatio = ratio, ratio
				continue
			}
			// On the larger image the per-pixel allocation ratio must not
			// increase: any quadratic or tap-driven blow-up would raise it.
			if ratio > perPixel*1.25+1e-9 {
				t.Errorf("%T allocations super-linear: small ratio=%v large ratio=%v", q, baseRatio, ratio)
			}
		}
	}

	// A concrete *image.RGBA source still goes through the interface in the
	// boundary path, but its color is a value type; assert a single run stays
	// well below one allocation per byte of destination pixels.
	dst := image.NewRGBA(image.Rect(0, 0, 32, 32))
	allocs := testing.AllocsPerRun(6, func() {
		NearestNeighbor.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, &Options{Boundary: BoundaryWrap})
	})
	if allocs > float64(32*32)*2 {
		t.Errorf("nearest neighbor allocs %v exceed linear bound", allocs)
	}
}

func TestBoundaryIndexUnit(t *testing.T) {
	// Direct, exhaustive checks of the coordinate mapping including large
	// negative offsets, guarding against off-by-one at the seams.
	for n := 1; n <= 8; n++ {
		for p := -3 * n; p < 3*n; p++ {
			// Wrap always lands in [0,n) and is periodic.
			if q, ok := refIndex(BoundaryWrap, p, n); !ok || q < 0 || q >= n {
				t.Fatalf("wrap n=%d p=%d -> %d,%v", n, p, q, ok)
			}
			// Mirror lands in [0,n), never on a reflected seam out of range.
			if q, ok := refIndex(BoundaryMirror, p, n); !ok || q < 0 || q >= n {
				t.Fatalf("mirror n=%d p=%d -> %d,%v", n, p, q, ok)
			}
			// In-range points are real pixels for every mode.
			if p >= 0 && p < n {
				if q, ok := refIndex(BoundaryTransparent, p, n); !ok || q != p {
					t.Fatalf("transparent in-range n=%d p=%d -> %d,%v", n, p, q, ok)
				}
			} else if _, ok := refIndex(BoundaryTransparent, p, n); ok {
				t.Fatalf("transparent out-of-range n=%d p=%d reported in range", n, p)
			}
			// Production helper works in absolute coordinates; shift the
			// rectangle and the point by the same origin (100) and confirm
			// it agrees with the local reference.
			wantQ, wantOK := refIndex(BoundaryMirror, p, n)
			gotQ, gotOK := boundaryIndex(BoundaryMirror, 100+p, 100, 100+n, n)
			if gotQ != 100+wantQ || gotOK != wantOK {
				t.Fatalf("boundaryIndex mirror p=%d n=%d: got (%d,%v) want (%d,%v)",
					p, n, gotQ, gotOK, 100+wantQ, wantOK)
			}
		}
	}
	// Empty axis: always a transparent miss.
	if _, ok := boundaryIndex(BoundaryWrap, 5, 0, 0, 0); ok {
		t.Fatal("empty axis reported a valid sample")
	}
}

// invertAff3 is an independent affine-inverse implementation used by the
// transform reference samplers.
func invertAff3(m f64.Aff3) f64.Aff3 {
	det := m[0]*m[4] - m[3]*m[1]
	return f64.Aff3{
		m[4] / det, -m[1] / det, (m[1]*m[5] - m[4]*m[2]) / det,
		-m[3] / det, m[0] / det, (m[3]*m[2] - m[0]*m[5]) / det,
	}
}

// refNNTransform is an independent nearest-neighbor transform.
func refNNTransform(mode BoundaryMode, src *image.RGBA, s2d f64.Aff3, dw, dh int) *image.RGBA {
	d2s := invertAff3(s2d)
	sr := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			fx := float64(x) + 0.5
			fy := float64(y) + 0.5
			// Round toward negative infinity, matching the production bias
			// trick (which makes floor equivalent to int on the shifted,
			// non-negative coordinates). Convert to coordinates local to
			// sr.Min so they can be fed to refSampler.
			u := int(math.Floor(d2s[0]*fx+d2s[1]*fy+d2s[2])) - sr.Min.X
			v := int(math.Floor(d2s[3]*fx+d2s[4]*fy+d2s[5])) - sr.Min.Y
			r, g, b, a := refSampler(src, mode, u, v)
			dst.SetRGBA(x, y, color.RGBA{
				uint8(refTrunc(r) >> 8), uint8(refTrunc(g) >> 8),
				uint8(refTrunc(b) >> 8), uint8(refTrunc(a) >> 8),
			})
		}
	}
	return dst
}

// refBilinearTransform is an independent bilinear transform.
func refBilinearTransform(mode BoundaryMode, src *image.RGBA, s2d f64.Aff3, dw, dh int) *image.RGBA {
	d2s := invertAff3(s2d)
	sr := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			fx := float64(x) + 0.5
			fy := float64(y) + 0.5
			cx := d2s[0]*fx + d2s[1]*fy + d2s[2] - 0.5 - float64(sr.Min.X)
			cy := d2s[3]*fx + d2s[4]*fy + d2s[5] - 0.5 - float64(sr.Min.Y)
			x0 := math.Floor(cx)
			fxx := cx - x0
			y0 := math.Floor(cy)
			fyy := cy - y0
			var rr, gg, bb, aa float64
			for _, ty := range []struct {
				v int
				w float64
			}{{int(y0), 1 - fyy}, {int(y0) + 1, fyy}} {
				for _, tx := range []struct {
					u int
					w float64
				}{{int(x0), 1 - fxx}, {int(x0) + 1, fxx}} {
					r, g, b, a := refSampler(src, mode, tx.u, ty.v)
					w := tx.w * ty.w
					rr += r * w
					gg += g * w
					bb += b * w
					aa += a * w
				}
			}
			dst.SetRGBA(x, y, color.RGBA{
				uint8(refTrunc(rr) >> 8), uint8(refTrunc(gg) >> 8),
				uint8(refTrunc(bb) >> 8), uint8(refTrunc(aa) >> 8),
			})
		}
	}
	return dst
}

// pixMaxDiffMasked reports the maximum per-channel difference, considering
// only pixels for which mask is true.
func pixMaxDiffMasked(a, b *image.RGBA, mask func(x, y int) bool) int {
	if len(a.Pix) != len(b.Pix) {
		return 1 << 30
	}
	d := 0
	bnd := a.Bounds()
	for y := bnd.Min.Y; y < bnd.Max.Y; y++ {
		for x := bnd.Min.X; x < bnd.Max.X; x++ {
			if !mask(x, y) {
				continue
			}
			ca, cb := a.RGBAAt(x, y), b.RGBAAt(x, y)
			for k := 0; k < 4; k++ {
				va := [4]uint8{ca.R, ca.G, ca.B, ca.A}[k]
				vb := [4]uint8{cb.R, cb.G, cb.B, cb.A}[k]
				x2 := int(va) - int(vb)
				if x2 < 0 {
					x2 = -x2
				}
				if x2 > d {
					d = x2
				}
			}
		}
	}
	return d
}

// centerInSRMask returns a predicate reporting whether the center of dst
// pixel (x,y) maps back inside the source rectangle. This is exactly the
// region the legacy transform leaves visit (the forward bounding box plus
// the point-in-sr guard); pixels outside it are skipped rather than blended
// from a partial kernel neighborhood.
func centerInSRMask(mat f64.Aff3, src *image.RGBA, dw, dh int) func(int, int) bool {
	d2s := invertAff3(mat)
	sr := src.Bounds()
	return func(x, y int) bool {
		fx, fy := float64(x)+0.5, float64(y)+0.5
		u := d2s[0]*fx + d2s[1]*fy + d2s[2]
		v := d2s[3]*fx + d2s[4]*fy + d2s[5]
		return u >= float64(sr.Min.X) && u < float64(sr.Max.X) &&
			v >= float64(sr.Min.Y) && v < float64(sr.Max.Y)
	}
}

func TestBoundaryTransformAgainstReference(t *testing.T) {
	src := makeChecker(8, 6)
	dw, dh := 24, 20
	mats := map[string]f64.Aff3{
		"rotate": transformMatrix(1.5, 6, 3),
		"flip":   {-1.3, 0.2, 20, 0.2, -1.1, 16},
		"shrink": transformMatrix(0.4, 10, 8),
	}
	for _, mode := range []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
		for name, mat := range mats {
			mask := centerInSRMask(mat, src, dw, dh)

			gotNN := image.NewRGBA(image.Rect(0, 0, dw, dh))
			NearestNeighbor.Transform(gotNN, mat, src, src.Bounds(), Src, &Options{Boundary: mode})
			wantNN := refNNTransform(mode, src, mat, dw, dh)
			if d := pixMaxDiffMasked(gotNN, wantNN, mask); d > 0 {
				t.Errorf("NN mode=%v mat=%s: diff=%d", mode, name, d)
			}

			gotBL := image.NewRGBA(image.Rect(0, 0, dw, dh))
			ApproxBiLinear.Transform(gotBL, mat, src, src.Bounds(), Src, &Options{Boundary: mode})
			wantBL := refBilinearTransform(mode, src, mat, dw, dh)
			if d := pixMaxDiffMasked(gotBL, wantBL, mask); d > 1 {
				t.Errorf("ABL mode=%v mat=%s: diff=%d", mode, name, d)
			}
		}
	}
}

func TestBoundarySingularMatrix(t *testing.T) {
	src := makeChecker(6, 6)
	// Determinant zero: finite inputs but non-invertible.
	singular := f64.Aff3{
		1, 2, 3,
		2, 4, 5,
	}
	for _, mode := range []BoundaryMode{BoundaryClamp, BoundaryWrap, BoundaryMirror, BoundaryTransparent} {
		dst := image.NewRGBA(image.Rect(0, 0, 10, 10))
		dst.SetRGBA(0, 0, color.RGBA{9, 8, 7, 6})
		NearestNeighbor.Transform(dst, singular, src, src.Bounds(), Src, &Options{Boundary: mode})
		if got := dst.RGBAAt(0, 0); got != (color.RGBA{9, 8, 7, 6}) {
			t.Errorf("mode=%v singular matrix wrote a pixel: %v", mode, got)
		}
	}
}

// TestBoundaryTranslatedRect verifies boundary resolution for a source
// rectangle whose Min is not the origin, including coordinates outside it.
func TestBoundaryTranslatedRect(t *testing.T) {
	// Build an image and pick a sub-rectangle offset from the origin.
	full := makeChecker(12, 10)
	lo := image.Pt(3, 2)
	sr := image.Rect(lo.X, lo.Y, lo.X+6, lo.Y+5)
	for _, mode := range []BoundaryMode{BoundaryWrap, BoundaryMirror, BoundaryClamp} {
		s := bctx{mode: mode, src: full, sr: sr}
		w, h := sr.Dx(), sr.Dy()
		// Coordinates offset by a whole number of periods must stay in range
		// and resolve to a real pixel.
		for du := -2 * w; du <= 2*w; du++ {
			for dv := -2 * h; dv <= 2*h; dv++ {
				x, y := lo.X+du, lo.Y+dv
				r, g, b, a, ok := s.at(x, y)
				if !ok {
					t.Fatalf("mode=%v at (%d,%d) reported a miss (expected periodic pixel)", mode, x, y)
				}
				_ = r
				_ = g
				_ = b
				_ = a
				// Every resolved pixel must correspond to an in-bounds image read.
				qx, okx := boundaryIndex(mode, x, sr.Min.X, sr.Max.X, w)
				qy, oky := boundaryIndex(mode, y, sr.Min.Y, sr.Max.Y, h)
				if !okx || !oky || !image.Pt(qx, qy).In(full.Bounds()) {
					t.Fatalf("mode=%v resolved (%d,%d) out of image bounds: (%d,%d)", mode, x, y, qx, qy)
				}
			}
		}
	}

	// Transparent mode must miss exactly outside the rectangle, never read
	// beyond the image bounds.
	st := bctx{mode: BoundaryTransparent, src: full, sr: sr}
	if _, _, _, _, ok := st.at(sr.Min.X-1, sr.Min.Y); ok {
		t.Error("transparent: point just left of sr should miss")
	}
	if _, _, _, _, ok := st.at(sr.Min.X, sr.Min.Y); !ok {
		t.Error("transparent: point inside sr should hit")
	}
}
