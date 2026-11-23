// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

// This file implements the configurable boundary modes (see BoundaryMode in
// scale.go). It is intentionally separate from the generated impl.go:
//
//   - The zero-value BoundaryClamp keeps using the generated, type-specialized
//     leaves, so existing results are unchanged bit for bit.
//   - BoundaryWrap, BoundaryMirror and BoundaryTransparent are opt-in and go
//     through the self-contained samplers in this file. Every source read is
//     funneled through boundaryIndex, which can never return an out-of-range
//     coordinate, so these paths cannot perform an out-of-bounds read even for
//     negative transforms, source rectangles that overhang the image, or
//     aliased source and destination images.
//
// The samplers operate on the image.Image and RGBA64Image interfaces rather
// than on concrete Pix fields, mirroring the generic *_Image_Image_* fallback
// leaves in impl.go.

import (
	"image"
	"image/color"
	"math"

	"golang.org/x/image/math/f64"
)

// bctx is a boundary-aware source sampler.
type bctx struct {
	mode    BoundaryMode
	src     image.Image
	sr      image.Rectangle
	srcMask image.Image
	smp     image.Point
}

// at returns the alpha-premultiplied color of the source at the absolute
// source-space point (x, y), resolving coordinates outside sr according to
// the configured mode. A BoundaryTransparent miss, or an empty source axis,
// yields transparent black and ok == false; the returned color is zero in
// that case.
func (s *bctx) at(x, y int) (r, g, b, a uint32, ok bool) {
	qx, okx := boundaryIndex(s.mode, x, s.sr.Min.X, s.sr.Max.X, s.sr.Dx())
	qy, oky := boundaryIndex(s.mode, y, s.sr.Min.Y, s.sr.Max.Y, s.sr.Dy())
	if !okx || !oky {
		return 0, 0, 0, 0, false
	}
	r, g, b, a = s.src.At(qx, qy).RGBA()
	if s.srcMask != nil {
		// The mask is evaluated at the resolved pixel, i.e. the pixel that
		// is actually read, so wrap and mirror stay self-consistent.
		_, _, _, ma := s.srcMask.At(s.smp.X+qx, s.smp.Y+qy).RGBA()
		r = r * ma / 0xffff
		g = g * ma / 0xffff
		b = b * ma / 0xffff
		a = a * ma / 0xffff
	}
	return r, g, b, a, true
}

// atf is at with the channels converted to float64, for the interpolating
// samplers. A transparent miss contributes zero.
func (s *bctx) atf(x, y int) (r, g, b, a float64) {
	ru, gu, bu, au, _ := s.at(x, y)
	return float64(ru), float64(gu), float64(bu), float64(au)
}

// compositor writes an alpha-premultiplied color (in 0..0xffff) to the
// destination, applying Op and an optional destination mask.
type compositor struct {
	dst     Image
	rgba64  RGBA64Image
	dstMask image.Image
	dmp     image.Point
	op      Op
	c64     color.RGBA64
}

func (c *compositor) read(x, y int) (r, g, b, a uint32) {
	if c.rgba64 != nil {
		q := c.rgba64.RGBA64At(x, y)
		return uint32(q.R), uint32(q.G), uint32(q.B), uint32(q.A)
	}
	return c.dst.At(x, y).RGBA()
}

func clampU16(v uint32) uint32 {
	if v > 0xffff {
		return 0xffff
	}
	return v
}

func (c *compositor) write(x, y int, r, g, b, a uint32) {
	r, g, b, a = clampU16(r), clampU16(g), clampU16(b), clampU16(a)
	if c.rgba64 != nil {
		c.rgba64.SetRGBA64(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
		return
	}
	c.c64 = color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
	c.dst.Set(x, y, &c.c64)
}

// put stores an alpha-premultiplied source color at destination (x, y),
// honoring Op and the destination mask.
func (c *compositor) put(x, y int, r, g, b, a uint32) {
	if c.op == Over {
		qr, qg, qb, qa := c.read(x, y)
		if c.dstMask != nil {
			_, _, _, ma := c.dstMask.At(c.dmp.X+x, c.dmp.Y+y).RGBA()
			r = r * ma / 0xffff
			g = g * ma / 0xffff
			b = b * ma / 0xffff
			a = a * ma / 0xffff
		}
		pa1 := 0xffff - a
		c.write(x, y, qr*pa1/0xffff+r, qg*pa1/0xffff+g, qb*pa1/0xffff+b, qa*pa1/0xffff+a)
		return
	}
	// Op Src.
	if c.dstMask == nil {
		c.write(x, y, r, g, b, a)
		return
	}
	qr, qg, qb, qa := c.read(x, y)
	_, _, _, ma := c.dstMask.At(c.dmp.X+x, c.dmp.Y+y).RGBA()
	pa1 := 0xffff - ma // Note that this is ma, not a.
	c.write(x, y,
		qr*pa1/0xffff+r*ma/0xffff,
		qg*pa1/0xffff+g*ma/0xffff,
		qb*pa1/0xffff+b*ma/0xffff,
		qa*pa1/0xffff+a*ma/0xffff,
	)
}

// prepBoundary factors out the setup shared by every boundary-mode Scale.
func prepBoundary(dst Image, dr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) (s bctx, c compositor, adr image.Rectangle, ok bool) {
	var o Options
	if opts != nil {
		o = *opts
	}

	adr = dst.Bounds().Intersect(dr)
	adr, o.DstMask = clipAffectedDestRect(adr, o.DstMask, o.DstMaskP)
	if adr.Empty() || sr.Empty() {
		return s, c, adr, false
	}
	if op == Over && o.SrcMask == nil && opaque(src) {
		op = Src
	}

	s = bctx{
		mode:    normalizeBoundary(o.Boundary),
		src:     src,
		sr:      sr,
		srcMask: o.SrcMask,
		smp:     o.SrcMaskP,
	}
	c = compositor{dst: dst, dstMask: o.DstMask, dmp: o.DstMaskP, op: op}
	if r64, is64 := dst.(RGBA64Image); is64 {
		c.rgba64 = r64
	}
	return s, c, adr, true
}

// prepTransformBoundary additionally builds the inverse mapping and the
// non-negative-coordinate bias used by the Transform leaves.
func prepTransformBoundary(dst Image, s2d f64.Aff3, src image.Image, sr image.Rectangle, op Op, opts *Options) (s bctx, c compositor, dr, adr image.Rectangle, d2s f64.Aff3, bias image.Point, ok bool) {
	// A non-finite matrix has no defined source coordinates; skip safely.
	if !aff3Finite(&s2d) {
		return s, c, dr, adr, d2s, bias, false
	}

	var o Options
	if opts != nil {
		o = *opts
	}
	mode := normalizeBoundary(o.Boundary)

	dr = transformRect(&s2d, &sr)
	if mode == BoundaryWrap || mode == BoundaryMirror {
		// Wrap and mirror tile the whole source plane, so every destination
		// pixel has a defined source coordinate regardless of where the
		// forward-mapped source rectangle lands. Visit the whole destination
		// rather than only the forward bounding box.
		adr = dst.Bounds()
	} else {
		adr = dst.Bounds().Intersect(dr)
	}
	adr, o.DstMask = clipAffectedDestRect(adr, o.DstMask, o.DstMaskP)
	if adr.Empty() || sr.Empty() {
		return s, c, dr, adr, d2s, bias, false
	}
	if op == Over && o.SrcMask == nil && opaque(src) {
		op = Src
	}

	d2s = invert(&s2d)
	// A singular (or otherwise non-invertible) matrix yields a non-finite
	// inverse, from which no source coordinate is well defined. Skip rather
	// than feed NaN/Inf into the integer coordinate conversion.
	if !aff3Finite(&d2s) {
		return s, c, dr, adr, d2s, bias, false
	}
	bias = transformRect(&d2s, &adr).Min
	bias.X--
	bias.Y--
	d2s[2] -= float64(bias.X)
	d2s[5] -= float64(bias.Y)
	adr = adr.Sub(dr.Min)

	s = bctx{
		mode:    mode,
		src:     src,
		sr:      sr,
		srcMask: o.SrcMask,
		smp:     o.SrcMaskP,
	}
	c = compositor{dst: dst, dstMask: o.DstMask, dmp: o.DstMaskP, op: op}
	if r64, is64 := dst.(RGBA64Image); is64 {
		c.rgba64 = r64
	}
	return s, c, dr, adr, d2s, bias, true
}

// truncU16 truncates a non-negative float64 to the uint32 channel range.
func truncU16(v float64) uint32 {
	if v <= 0 {
		return 0
	}
	if v >= 0xffff {
		return 0xffff
	}
	return uint32(v)
}

// effectiveHalfWidth returns the (downscale-broadened) kernel half width and
// the matching kernel-argument scale, capped at maxKernelSpan taps so that a
// pathological Kernel.Support or extreme scale ratio cannot trigger an
// unbounded allocation. A non-finite half width is likewise replaced by the
// cap, which is a safe, in-range degradation.
func effectiveHalfWidth(support, scale float64) (halfWidth, kernelArgScale float64) {
	halfWidth, kernelArgScale = support, 1.0
	if scale > 1 {
		halfWidth *= scale
		kernelArgScale = 1 / scale
	}
	if math.IsNaN(halfWidth) || math.IsInf(halfWidth, 0) ||
		int(math.Ceil(2*halfWidth)) > maxKernelSpan {
		halfWidth = float64(maxKernelSpan) / 2
	}
	return halfWidth, kernelArgScale
}

// kernelAt evaluates q at |t|, returning zero at or beyond the support.
func kernelAt(q *Kernel, t float64) float64 {
	if t >= q.Support {
		return 0
	}
	return q.At(t)
}

// clampPremultiplied enforces the alpha-premultiplied invariant r,g,b <= a.
func clampPremultiplied(r, g, b, a *float64) {
	if *r > *a {
		*r = *a
	}
	if *g > *a {
		*g = *a
	}
	if *b > *a {
		*b = *a
	}
}

// tap is one weighted source coordinate.
type tap struct {
	coord int
	w     float64
}

// ---------------------------------------------------------------------------
// Nearest neighbor
// ---------------------------------------------------------------------------

func nnScaleBoundary(dst Image, dr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, adr, ok := prepBoundary(dst, dr, src, sr, op, opts)
	if !ok {
		return
	}
	dw2 := uint64(dr.Dx()) * 2
	dh2 := uint64(dr.Dy()) * 2
	sw := uint64(sr.Dx())
	sh := uint64(sr.Dy())
	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		sy := sr.Min.Y + int((2*uint64(dy)+1)*sh/dh2)
		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			sx := sr.Min.X + int((2*uint64(dx)+1)*sw/dw2)
			r, g, b, a, _ := s.at(sx, sy)
			c.put(dr.Min.X+dx, dr.Min.Y+dy, r, g, b, a)
		}
	}
}

func nnTransformBoundary(dst Image, s2d f64.Aff3, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, dr, adr, d2s, bias, ok := prepTransformBoundary(dst, s2d, src, sr, op, opts)
	if !ok {
		return
	}
	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		dyf := float64(dr.Min.Y+dy) + 0.5
		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			dxf := float64(dr.Min.X+dx) + 0.5
			sx := int(float64(d2s[0]*dxf)+float64(d2s[1]*dyf)+d2s[2]) + bias.X
			sy := int(float64(d2s[3]*dxf)+float64(d2s[4]*dyf)+d2s[5]) + bias.Y
			r, g, b, a, _ := s.at(sx, sy)
			c.put(dr.Min.X+dx, dr.Min.Y+dy, r, g, b, a)
		}
	}
}

// ---------------------------------------------------------------------------
// Approximate bi-linear
// ---------------------------------------------------------------------------

func ablScaleBoundary(dst Image, dr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, adr, ok := prepBoundary(dst, dr, src, sr, op, opts)
	if !ok {
		return
	}
	sw, sh := sr.Dx(), sr.Dy()
	xscale := float64(sw) / float64(dr.Dx())
	yscale := float64(sh) / float64(dr.Dy())

	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		sy := (float64(dy)+0.5)*yscale - 0.5
		y0 := math.Floor(sy)
		fy := sy - y0
		sy0 := sr.Min.Y + int(y0)
		sy1 := sy0 + 1
		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			sx := (float64(dx)+0.5)*xscale - 0.5
			x0 := math.Floor(sx)
			fx := sx - x0
			sx0 := sr.Min.X + int(x0)
			sx1 := sx0 + 1

			s00r, s00g, s00b, s00a := s.atf(sx0, sy0)
			s10r, s10g, s10b, s10a := s.atf(sx1, sy0)
			s01r, s01g, s01b, s01a := s.atf(sx0, sy1)
			s11r, s11g, s11b, s11a := s.atf(sx1, sy1)

			wx0, wx1 := 1-fx, fx
			wy0, wy1 := 1-fy, fy
			pr := (s00r*wx0+s10r*wx1)*wy0 + (s01r*wx0+s11r*wx1)*wy1
			pg := (s00g*wx0+s10g*wx1)*wy0 + (s01g*wx0+s11g*wx1)*wy1
			pb := (s00b*wx0+s10b*wx1)*wy0 + (s01b*wx0+s11b*wx1)*wy1
			pa := (s00a*wx0+s10a*wx1)*wy0 + (s01a*wx0+s11a*wx1)*wy1
			c.put(dr.Min.X+dx, dr.Min.Y+dy, truncU16(pr), truncU16(pg), truncU16(pb), truncU16(pa))
		}
	}
}

func ablTransformBoundary(dst Image, s2d f64.Aff3, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, dr, adr, d2s, bias, ok := prepTransformBoundary(dst, s2d, src, sr, op, opts)
	if !ok {
		return
	}
	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		dyf := float64(dr.Min.Y+dy) + 0.5
		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			dxf := float64(dr.Min.X+dx) + 0.5
			sx := float64(d2s[0]*dxf) + float64(d2s[1]*dyf) + d2s[2] + float64(bias.X) - 0.5
			sy := float64(d2s[3]*dxf) + float64(d2s[4]*dyf) + d2s[5] + float64(bias.Y) - 0.5

			x0 := math.Floor(sx)
			fx := sx - x0
			sx0 := int(x0)
			sx1 := sx0 + 1
			y0 := math.Floor(sy)
			fy := sy - y0
			sy0 := int(y0)
			sy1 := sy0 + 1

			s00r, s00g, s00b, s00a := s.atf(sx0, sy0)
			s10r, s10g, s10b, s10a := s.atf(sx1, sy0)
			s01r, s01g, s01b, s01a := s.atf(sx0, sy1)
			s11r, s11g, s11b, s11a := s.atf(sx1, sy1)

			wx0, wx1 := 1-fx, fx
			wy0, wy1 := 1-fy, fy
			pr := (s00r*wx0+s10r*wx1)*wy0 + (s01r*wx0+s11r*wx1)*wy1
			pg := (s00g*wx0+s10g*wx1)*wy0 + (s01g*wx0+s11g*wx1)*wy1
			pb := (s00b*wx0+s10b*wx1)*wy0 + (s01b*wx0+s11b*wx1)*wy1
			pa := (s00a*wx0+s10a*wx1)*wy0 + (s01a*wx0+s11a*wx1)*wy1
			c.put(dr.Min.X+dx, dr.Min.Y+dy, truncU16(pr), truncU16(pg), truncU16(pb), truncU16(pa))
		}
	}
}

// ---------------------------------------------------------------------------
// Kernel (BiLinear, Catmull-Rom, and any custom Kernel)
// ---------------------------------------------------------------------------

func kernelScaleBoundary(q *Kernel, dst Image, dr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, adr, ok := prepBoundary(dst, dr, src, sr, op, opts)
	if !ok {
		return
	}
	xHalfWidth, xKAS := effectiveHalfWidth(q.Support, float64(sr.Dx())/float64(dr.Dx()))
	yHalfWidth, yKAS := effectiveHalfWidth(q.Support, float64(sr.Dy())/float64(dr.Dy()))

	xtaps := make([]tap, 0, int(math.Ceil(2*xHalfWidth))+1)
	ytaps := make([]tap, 0, int(math.Ceil(2*yHalfWidth))+1)

	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		cy := (float64(dy)+0.5)*float64(sr.Dy())/float64(dr.Dy()) - 0.5
		iy0 := int(math.Floor(cy - yHalfWidth))
		iy1 := int(math.Ceil(cy + yHalfWidth))
		ytaps = ytaps[:0]
		totalY := 0.0
		for ky := iy0; ky < iy1; ky++ {
			w := kernelAt(q, abs((cy-float64(ky))*yKAS))
			if w == 0 {
				continue
			}
			ytaps = append(ytaps, tap{sr.Min.Y + ky, w})
			totalY += w
		}

		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			cx := (float64(dx)+0.5)*float64(sr.Dx())/float64(dr.Dx()) - 0.5
			ix0 := int(math.Floor(cx - xHalfWidth))
			ix1 := int(math.Ceil(cx + xHalfWidth))
			xtaps = xtaps[:0]
			totalX := 0.0
			for kx := ix0; kx < ix1; kx++ {
				w := kernelAt(q, abs((cx-float64(kx))*xKAS))
				if w == 0 {
					continue
				}
				xtaps = append(xtaps, tap{sr.Min.X + kx, w})
				totalX += w
			}

			var pr, pg, pb, pa float64
			for _, ty := range ytaps {
				for _, tx := range xtaps {
					fr, fg, fb, fa := s.atf(tx.coord, ty.coord)
					w := tx.w * ty.w
					pr += fr * w
					pg += fg * w
					pb += fb * w
					pa += fa * w
				}
			}
			totalW := totalX * totalY
			if totalW != 0 {
				pr /= totalW
				pg /= totalW
				pb /= totalW
				pa /= totalW
			}
			clampPremultiplied(&pr, &pg, &pb, &pa)
			c.put(dr.Min.X+dx, dr.Min.Y+dy,
				uint32(fffftou(pr)), uint32(fffftou(pg)),
				uint32(fffftou(pb)), uint32(fffftou(pa)))
		}
	}
}

func kernelTransformBoundary(q *Kernel, dst Image, s2d f64.Aff3, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	s, c, dr, adr, d2s, bias, ok := prepTransformBoundary(dst, s2d, src, sr, op, opts)
	if !ok {
		return
	}
	xscale := abs(d2s[0])
	if v := abs(d2s[1]); xscale < v {
		xscale = v
	}
	yscale := abs(d2s[3])
	if v := abs(d2s[4]); yscale < v {
		yscale = v
	}
	xHalfWidth, xKAS := effectiveHalfWidth(q.Support, xscale)
	yHalfWidth, yKAS := effectiveHalfWidth(q.Support, yscale)

	xtaps := make([]tap, 0, int(math.Ceil(2*xHalfWidth))+1)
	ytaps := make([]tap, 0, int(math.Ceil(2*yHalfWidth))+1)

	for dy := adr.Min.Y; dy < adr.Max.Y; dy++ {
		dyf := float64(dr.Min.Y+dy) + 0.5
		for dx := adr.Min.X; dx < adr.Max.X; dx++ {
			dxf := float64(dr.Min.X+dx) + 0.5
			sx := float64(d2s[0]*dxf) + float64(d2s[1]*dyf) + d2s[2]
			sy := float64(d2s[3]*dxf) + float64(d2s[4]*dyf) + d2s[5]
			if math.IsNaN(sx) || math.IsNaN(sy) || math.IsInf(sx, 0) || math.IsInf(sy, 0) {
				continue
			}
			sx += float64(bias.X) - 0.5
			sy += float64(bias.Y) - 0.5

			ix0 := int(math.Floor(sx - xHalfWidth))
			ix1 := int(math.Ceil(sx + xHalfWidth))
			xtaps = xtaps[:0]
			totalX := 0.0
			for kx := ix0; kx < ix1; kx++ {
				w := kernelAt(q, abs((sx-float64(kx))*xKAS))
				if w == 0 {
					continue
				}
				xtaps = append(xtaps, tap{kx, w})
				totalX += w
			}

			iy0 := int(math.Floor(sy - yHalfWidth))
			iy1 := int(math.Ceil(sy + yHalfWidth))
			ytaps = ytaps[:0]
			totalY := 0.0
			for ky := iy0; ky < iy1; ky++ {
				w := kernelAt(q, abs((sy-float64(ky))*yKAS))
				if w == 0 {
					continue
				}
				ytaps = append(ytaps, tap{ky, w})
				totalY += w
			}

			var pr, pg, pb, pa float64
			for _, ty := range ytaps {
				for _, tx := range xtaps {
					fr, fg, fb, fa := s.atf(tx.coord, ty.coord)
					w := tx.w * ty.w
					pr += fr * w
					pg += fg * w
					pb += fb * w
					pa += fa * w
				}
			}
			totalW := totalX * totalY
			if totalW != 0 {
				pr /= totalW
				pg /= totalW
				pb /= totalW
				pa /= totalW
			}
			clampPremultiplied(&pr, &pg, &pb, &pa)
			c.put(dr.Min.X+dx, dr.Min.Y+dy,
				uint32(fffftou(pr)), uint32(fffftou(pg)),
				uint32(fffftou(pb)), uint32(fffftou(pa)))
		}
	}
}
