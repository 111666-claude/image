// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
	"image/color"

	"golang.org/x/image/math/f64"
)

// This file implements the additional Porter-Duff compositing operators
// (SrcIn, SrcOut, Atop and Xor) that this package adds to the Over and Src
// operators provided by the standard library's image/draw package.
//
// The operators are defined on alpha-premultiplied colors. If s and d are
// the source and destination colors and sa and da are their alpha values,
// with every channel in the range [0, 1], then the composited result is
//
//	s*fs + d*fd
//
// per channel, including the alpha channel, where the factors fs and fd
// depend on the operator:
//
//	SrcIn:  fs = da,     fd = 0
//	SrcOut: fs = 1 - da, fd = 0
//	Atop:   fs = da,     fd = 1 - sa
//	Xor:    fs = 1 - da, fd = 1 - sa
//
// As with the standard library's operators, a non-nil mask is applied to
// the source before compositing: the source is ``src in mask''. A nil mask
// is treated as opaque.
//
// If the source and destination are the same image and the source and
// destination rectangles overlap, each destination pixel is written only
// after its source and destination pixels have been read, iterating
// backwards when the source rectangle is above or to the left of the
// destination rectangle, so the result is the same as compositing from a
// copy of the source.

// porterDuffOp reports whether op is one of the additional Porter-Duff
// operators implemented by this file.
func porterDuffOp(op Op) bool {
	return op == SrcIn || op == SrcOut || op == Atop || op == Xor
}

// porterDuffFactors returns the source and destination blend factors fs and
// fd for op, given the source and destination alpha values sa and da in the
// range [0, 0xffff]. It is only called with the operators recognized by
// porterDuffOp.
func porterDuffFactors(sa, da uint32, op Op) (fs, fd uint32) {
	switch op {
	case SrcIn:
		return da, 0
	case SrcOut:
		return 0xffff - da, 0
	case Atop:
		return da, 0xffff - sa
	case Xor:
		return 0xffff - da, 0xffff - sa
	}
	return 0, 0
}

// porterDuffBlend64 composes the alpha-premultiplied source color s over
// the alpha-premultiplied destination color d using op.
func porterDuffBlend64(s, d color.RGBA64, op Op) color.RGBA64 {
	fs, fd := porterDuffFactors(uint32(s.A), uint32(d.A), op)
	return color.RGBA64{
		R: uint16(uint32(s.R)*fs/0xffff + uint32(d.R)*fd/0xffff),
		G: uint16(uint32(s.G)*fs/0xffff + uint32(d.G)*fd/0xffff),
		B: uint16(uint32(s.B)*fs/0xffff + uint32(d.B)*fd/0xffff),
		A: uint16(uint32(s.A)*fs/0xffff + uint32(d.A)*fd/0xffff),
	}
}

// drawMaskPorterDuff implements DrawMask for the additional Porter-Duff
// operators. It is a no-op for any other operator, matching the standard
// library's behavior for unknown operators.
func drawMaskPorterDuff(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	if !porterDuffOp(op) {
		return
	}
	clipPorterDuff(dst, &r, src, &sp, mask, &mp)
	if r.Empty() {
		return
	}

	// Fast paths for an *image.RGBA destination. If the source is the same
	// *image.RGBA and the rectangles overlap, fall through to the general
	// implementations below, which iterate in a direction that makes
	// aliasing safe.
	if dst0, ok := dst.(*image.RGBA); ok {
		aliased := false
		if src0, ok := src.(*image.RGBA); ok {
			aliased = src0 == dst0 && r.Overlaps(r.Add(sp.Sub(r.Min)))
		}
		if !aliased {
			if mask == nil {
				switch src0 := src.(type) {
				case *image.Uniform:
					sr, sg, sb, sa := src0.RGBA()
					drawPorterDuffFillRGBA(dst0, r, sr, sg, sb, sa, op)
					return
				case *image.RGBA:
					drawPorterDuffRGBA(dst0, r, src0, sp, op)
					return
				case *image.NRGBA:
					drawPorterDuffNRGBA(dst0, r, src0, sp, op)
					return
				}
			} else if mask0, ok := mask.(*image.Alpha); ok {
				switch src0 := src.(type) {
				case *image.Uniform:
					sr, sg, sb, sa := src0.RGBA()
					drawPorterDuffFillMaskRGBA(dst0, r, sr, sg, sb, sa, mask0, mp, op)
					return
				case *image.RGBA:
					drawPorterDuffMaskRGBA(dst0, r, src0, sp, mask0, mp, op)
					return
				}
			}
		}
	}

	x0, x1, dx := r.Min.X, r.Max.X, 1
	y0, y1, dy := r.Min.Y, r.Max.Y, 1
	if processBackwardPorterDuff(dst, r, src, sp) {
		x0, x1, dx = x1-1, x0-1, -1
		y0, y1, dy = y1-1, y0-1, -1
	}

	// Try the draw.RGBA64Image and image.RGBA64Image interfaces. These are
	// like the draw.Image and image.Image interfaces but they can avoid
	// allocations from converting concrete color types to the color.Color
	// interface type.
	if dst0, _ := dst.(RGBA64Image); dst0 != nil {
		if src0, _ := src.(image.RGBA64Image); src0 != nil {
			if mask == nil {
				sy := sp.Y + y0 - r.Min.Y
				for y := y0; y != y1; y, sy = y+dy, sy+dy {
					sx := sp.X + x0 - r.Min.X
					for x := x0; x != x1; x, sx = x+dx, sx+dx {
						s := src0.RGBA64At(sx, sy)
						d := dst0.RGBA64At(x, y)
						dst0.SetRGBA64(x, y, porterDuffBlend64(s, d, op))
					}
				}
				return
			}
			if mask0, _ := mask.(image.RGBA64Image); mask0 != nil {
				sy := sp.Y + y0 - r.Min.Y
				my := mp.Y + y0 - r.Min.Y
				for y := y0; y != y1; y, sy, my = y+dy, sy+dy, my+dy {
					sx := sp.X + x0 - r.Min.X
					mx := mp.X + x0 - r.Min.X
					for x := x0; x != x1; x, sx, mx = x+dx, sx+dx, mx+dx {
						s := src0.RGBA64At(sx, sy)
						ma := uint32(mask0.RGBA64At(mx, my).A)
						s.R = uint16(uint32(s.R) * ma / 0xffff)
						s.G = uint16(uint32(s.G) * ma / 0xffff)
						s.B = uint16(uint32(s.B) * ma / 0xffff)
						s.A = uint16(uint32(s.A) * ma / 0xffff)
						d := dst0.RGBA64At(x, y)
						dst0.SetRGBA64(x, y, porterDuffBlend64(s, d, op))
					}
				}
				return
			}
		}
	}

	// General fallback, using only the image.Image and draw.Image
	// interfaces, so that any destination and source types are supported.
	sy := sp.Y + y0 - r.Min.Y
	my := mp.Y + y0 - r.Min.Y
	for y := y0; y != y1; y, sy, my = y+dy, sy+dy, my+dy {
		sx := sp.X + x0 - r.Min.X
		mx := mp.X + x0 - r.Min.X
		for x := x0; x != x1; x, sx, mx = x+dx, sx+dx, mx+dx {
			sr, sg, sb, sa := src.At(sx, sy).RGBA()
			if mask != nil {
				_, _, _, ma := mask.At(mx, my).RGBA()
				sr = sr * ma / 0xffff
				sg = sg * ma / 0xffff
				sb = sb * ma / 0xffff
				sa = sa * ma / 0xffff
			}
			dr, dg, db, da := dst.At(x, y).RGBA()
			fs, fd := porterDuffFactors(sa, da, op)
			dst.Set(x, y, color.RGBA64{
				R: uint16(sr*fs/0xffff + dr*fd/0xffff),
				G: uint16(sg*fs/0xffff + dg*fd/0xffff),
				B: uint16(sb*fs/0xffff + db*fd/0xffff),
				A: uint16(sa*fs/0xffff + da*fd/0xffff),
			})
		}
	}
}

// clipPorterDuff clips r against dst's bounds, and against src and mask's
// bounds (translated by sp and mp), adjusting sp and mp to keep them
// aligned with r.Min. It mirrors the clip function in the standard
// library's image/draw package.
func clipPorterDuff(dst Image, r *image.Rectangle, src image.Image, sp *image.Point, mask image.Image, mp *image.Point) {
	orig := r.Min
	*r = r.Intersect(dst.Bounds())
	*r = r.Intersect(src.Bounds().Add(orig.Sub(*sp)))
	if mask != nil {
		*r = r.Intersect(mask.Bounds().Add(orig.Sub(*mp)))
	}
	dx := r.Min.X - orig.X
	dy := r.Min.Y - orig.Y
	if dx == 0 && dy == 0 {
		return
	}
	(*sp).X += dx
	(*sp).Y += dy
	if mask != nil {
		(*mp).X += dx
		(*mp).Y += dy
	}
}

// processBackwardPorterDuff reports whether the composition should iterate
// backwards over the destination so that an overlapping source that shares
// the destination's pixels is read before it is overwritten. It mirrors the
// processBackward function in the standard library's image/draw package.
func processBackwardPorterDuff(dst image.Image, r image.Rectangle, src image.Image, sp image.Point) bool {
	return dst == src &&
		r.Overlaps(r.Add(sp.Sub(r.Min))) &&
		(sp.Y < r.Min.Y || (sp.Y == r.Min.Y && sp.X < r.Min.X))
}

// drawPorterDuffFillRGBA composites a uniform alpha-premultiplied source
// color onto an *image.RGBA.
func drawPorterDuffFillRGBA(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32, op Op) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		d := dst.PixOffset(r.Min.X, y)
		for x := r.Min.X; x < r.Max.X; x, d = x+1, d+4 {
			dr := uint32(dst.Pix[d+0]) * 0x101
			dg := uint32(dst.Pix[d+1]) * 0x101
			db := uint32(dst.Pix[d+2]) * 0x101
			da := uint32(dst.Pix[d+3]) * 0x101
			fs, fd := porterDuffFactors(sa, da, op)
			dst.Pix[d+0] = uint8((sr*fs/0xffff + dr*fd/0xffff) >> 8)
			dst.Pix[d+1] = uint8((sg*fs/0xffff + dg*fd/0xffff) >> 8)
			dst.Pix[d+2] = uint8((sb*fs/0xffff + db*fd/0xffff) >> 8)
			dst.Pix[d+3] = uint8((sa*fs/0xffff + da*fd/0xffff) >> 8)
		}
	}
}

// drawPorterDuffRGBA composites an *image.RGBA source onto an *image.RGBA.
func drawPorterDuffRGBA(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point, op Op) {
	sy := sp.Y
	for y := r.Min.Y; y < r.Max.Y; y, sy = y+1, sy+1 {
		d := dst.PixOffset(r.Min.X, y)
		s := src.PixOffset(sp.X, sy)
		for x := r.Min.X; x < r.Max.X; x, d, s = x+1, d+4, s+4 {
			sr := uint32(src.Pix[s+0]) * 0x101
			sg := uint32(src.Pix[s+1]) * 0x101
			sb := uint32(src.Pix[s+2]) * 0x101
			sa := uint32(src.Pix[s+3]) * 0x101
			dr := uint32(dst.Pix[d+0]) * 0x101
			dg := uint32(dst.Pix[d+1]) * 0x101
			db := uint32(dst.Pix[d+2]) * 0x101
			da := uint32(dst.Pix[d+3]) * 0x101
			fs, fd := porterDuffFactors(sa, da, op)
			dst.Pix[d+0] = uint8((sr*fs/0xffff + dr*fd/0xffff) >> 8)
			dst.Pix[d+1] = uint8((sg*fs/0xffff + dg*fd/0xffff) >> 8)
			dst.Pix[d+2] = uint8((sb*fs/0xffff + db*fd/0xffff) >> 8)
			dst.Pix[d+3] = uint8((sa*fs/0xffff + da*fd/0xffff) >> 8)
		}
	}
}

// drawPorterDuffNRGBA composites an *image.NRGBA source onto an
// *image.RGBA, converting the source to alpha-premultiplied form.
func drawPorterDuffNRGBA(dst *image.RGBA, r image.Rectangle, src *image.NRGBA, sp image.Point, op Op) {
	sy := sp.Y
	for y := r.Min.Y; y < r.Max.Y; y, sy = y+1, sy+1 {
		d := dst.PixOffset(r.Min.X, y)
		s := src.PixOffset(sp.X, sy)
		for x := r.Min.X; x < r.Max.X; x, d, s = x+1, d+4, s+4 {
			sa := uint32(src.Pix[s+3]) * 0x101
			sr := uint32(src.Pix[s+0]) * sa / 0xff
			sg := uint32(src.Pix[s+1]) * sa / 0xff
			sb := uint32(src.Pix[s+2]) * sa / 0xff
			dr := uint32(dst.Pix[d+0]) * 0x101
			dg := uint32(dst.Pix[d+1]) * 0x101
			db := uint32(dst.Pix[d+2]) * 0x101
			da := uint32(dst.Pix[d+3]) * 0x101
			fs, fd := porterDuffFactors(sa, da, op)
			dst.Pix[d+0] = uint8((sr*fs/0xffff + dr*fd/0xffff) >> 8)
			dst.Pix[d+1] = uint8((sg*fs/0xffff + dg*fd/0xffff) >> 8)
			dst.Pix[d+2] = uint8((sb*fs/0xffff + db*fd/0xffff) >> 8)
			dst.Pix[d+3] = uint8((sa*fs/0xffff + da*fd/0xffff) >> 8)
		}
	}
}

// drawPorterDuffFillMaskRGBA composites a uniform alpha-premultiplied
// source color, modulated by an *image.Alpha mask, onto an *image.RGBA.
func drawPorterDuffFillMaskRGBA(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32, mask *image.Alpha, mp image.Point, op Op) {
	my := mp.Y
	for y := r.Min.Y; y < r.Max.Y; y, my = y+1, my+1 {
		d := dst.PixOffset(r.Min.X, y)
		mi := mask.PixOffset(mp.X, my)
		for x := r.Min.X; x < r.Max.X; x, d, mi = x+1, d+4, mi+1 {
			ma := uint32(mask.Pix[mi]) * 0x101
			sr0 := sr * ma / 0xffff
			sg0 := sg * ma / 0xffff
			sb0 := sb * ma / 0xffff
			sa0 := sa * ma / 0xffff
			dr := uint32(dst.Pix[d+0]) * 0x101
			dg := uint32(dst.Pix[d+1]) * 0x101
			db := uint32(dst.Pix[d+2]) * 0x101
			da := uint32(dst.Pix[d+3]) * 0x101
			fs, fd := porterDuffFactors(sa0, da, op)
			dst.Pix[d+0] = uint8((sr0*fs/0xffff + dr*fd/0xffff) >> 8)
			dst.Pix[d+1] = uint8((sg0*fs/0xffff + dg*fd/0xffff) >> 8)
			dst.Pix[d+2] = uint8((sb0*fs/0xffff + db*fd/0xffff) >> 8)
			dst.Pix[d+3] = uint8((sa0*fs/0xffff + da*fd/0xffff) >> 8)
		}
	}
}

// drawPorterDuffMaskRGBA composites an *image.RGBA source, modulated by an
// *image.Alpha mask, onto an *image.RGBA.
func drawPorterDuffMaskRGBA(dst *image.RGBA, r image.Rectangle, src *image.RGBA, sp image.Point, mask *image.Alpha, mp image.Point, op Op) {
	sy, my := sp.Y, mp.Y
	for y := r.Min.Y; y < r.Max.Y; y, sy, my = y+1, sy+1, my+1 {
		d := dst.PixOffset(r.Min.X, y)
		s := src.PixOffset(sp.X, sy)
		mi := mask.PixOffset(mp.X, my)
		for x := r.Min.X; x < r.Max.X; x, d, s, mi = x+1, d+4, s+4, mi+1 {
			ma := uint32(mask.Pix[mi]) * 0x101
			sr := uint32(src.Pix[s+0]) * 0x101 * ma / 0xffff
			sg := uint32(src.Pix[s+1]) * 0x101 * ma / 0xffff
			sb := uint32(src.Pix[s+2]) * 0x101 * ma / 0xffff
			sa := uint32(src.Pix[s+3]) * 0x101 * ma / 0xffff
			dr := uint32(dst.Pix[d+0]) * 0x101
			dg := uint32(dst.Pix[d+1]) * 0x101
			db := uint32(dst.Pix[d+2]) * 0x101
			da := uint32(dst.Pix[d+3]) * 0x101
			fs, fd := porterDuffFactors(sa, da, op)
			dst.Pix[d+0] = uint8((sr*fs/0xffff + dr*fd/0xffff) >> 8)
			dst.Pix[d+1] = uint8((sg*fs/0xffff + dg*fd/0xffff) >> 8)
			dst.Pix[d+2] = uint8((sb*fs/0xffff + db*fd/0xffff) >> 8)
			dst.Pix[d+3] = uint8((sa*fs/0xffff + da*fd/0xffff) >> 8)
		}
	}
}

// scalePorterDuff implements Scale for the additional Porter-Duff operators
// by scaling the source into a temporary image with the Src operator and
// then compositing the temporary image onto the destination. It is called
// by the generated interpolator implementations, so s.Scale with the Src
// operator does not recurse back into scalePorterDuff.
func scalePorterDuff(s Scaler, dst Image, dr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	var o Options
	if opts != nil {
		o = *opts
	}

	// adr is the affected destination pixels, in dst's coordinate space.
	adr := dst.Bounds().Intersect(dr)
	adr, o.DstMask = clipAffectedDestRect(adr, o.DstMask, o.DstMaskP)
	if adr.Empty() || sr.Empty() {
		return
	}

	tmp := image.NewRGBA(adr)
	s.Scale(tmp, dr, src, sr, Src, &Options{
		SrcMask:  o.SrcMask,
		SrcMaskP: o.SrcMaskP,
	})
	var mp image.Point
	if o.DstMask != nil {
		mp = o.DstMaskP.Add(adr.Min)
	}
	drawMaskPorterDuff(dst, adr, tmp, adr.Min, o.DstMask, mp, op)
}

// transformPorterDuff implements Transform for the additional Porter-Duff
// operators, in the same way that scalePorterDuff implements Scale.
func transformPorterDuff(t Transformer, dst Image, s2d f64.Aff3, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	var o Options
	if opts != nil {
		o = *opts
	}

	dr := transformRect(&s2d, &sr)
	// adr is the affected destination pixels, in dst's coordinate space.
	adr := dst.Bounds().Intersect(dr)
	adr, o.DstMask = clipAffectedDestRect(adr, o.DstMask, o.DstMaskP)
	if adr.Empty() || sr.Empty() {
		return
	}

	tmp := image.NewRGBA(adr)
	t.Transform(tmp, s2d, src, sr, Src, &Options{
		SrcMask:  o.SrcMask,
		SrcMaskP: o.SrcMaskP,
	})
	var mp image.Point
	if o.DstMask != nil {
		mp = o.DstMaskP.Add(adr.Min)
	}
	drawMaskPorterDuff(dst, adr, tmp, adr.Min, o.DstMask, mp, op)
}
