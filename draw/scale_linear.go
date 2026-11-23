// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

// This file implements the non-default Options.Transfer / Options.Alpha modes
// for Kernel.Scale and Kernel.Transform. Those modes resample in a different
// color space than the storage format, so they cannot use the generated
// type-specialized fast paths in impl.go (which filter stored, premultiplied
// sRGB values directly). The data flow is otherwise the same: filtering is
// separable, with the horizontal pass writing an intermediate [][4]float64
// buffer that the vertical pass consumes.
//
// The default modes (TransferSRGB, AlphaPremultiplied) never reach this file,
// so their output is bit-for-bit unchanged.

import (
	"image"
	"image/color"
	"math"
	"sync"

	"golang.org/x/image/math/f64"
)

// srgbDecodeTable maps each 8-bit sRGB-encoded value to linear light, i.e.
// srgbDecode(v/255). Sources stored as 8-bit channels are by far the most
// common, and decoding them through math.Pow dominates the cost of filtering
// otherwise. Building the table is deferred so that programs which never use
// TransferLinear do not pay for it.
var (
	srgbDecodeTableOnce sync.Once
	srgbDecodeTable     [256]float64
)

func decode8(v uint8) float64 {
	srgbDecodeTableOnce.Do(func() {
		for i := range srgbDecodeTable {
			srgbDecodeTable[i] = srgbDecode(float64(i) / 0xff)
		}
	})
	return srgbDecodeTable[v]
}

// srgbDecode converts an sRGB-encoded value to linear light.
//
// The standard piecewise sRGB curve is defined on [0, 1]. Kernel filters with
// negative lobes (such as CatmullRom) and out-of-gamut samples can produce
// values outside that range during filtering, so the curve is extended oddly:
// f(-v) = -f(v). This is monotonic and invertible everywhere, keeps zero fixed
// and is antisymmetric, so a negative excursion in the source produces a
// negative excursion in linear light rather than a discontinuity at zero.
// Values above 1 are evaluated by the formula, which continues the curve
// smoothly and stays monotonic.
//
// Non-finite inputs map to zero. That is a deliberately stable choice: a NaN
// cannot be ordered, and letting it flow through a whole neighborhood would
// turn one bad sample into a transparent (or fully opaque) region depending on
// the exact rounding of comparisons. Zero keeps the output finite.
func srgbDecode(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if v < 0 {
		return -srgbDecode(-v)
	}
	if v <= 0.04045 {
		return v / 12.92
	}
	return math.Pow((v+0.055)/1.055, 2.4)
}

// srgbEncode converts a linear-light value to the sRGB encoding.
func srgbEncode(v float64) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if v < 0 {
		return -srgbEncode(-v)
	}
	if v <= 0.0031308 {
		return 12.92 * v
	}
	return 1.055*math.Pow(v, 1/2.4) - 0.055
}

// clamp01 maps v to [0, 1], and maps NaN to 0.
func clamp01(v float64) float64 {
	if !(v > 0) { // false for NaN and for v <= 0
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// sampleStraight returns the straight (unassociated) sRGB-encoded color and
// alpha of the source pixel at (x, y), with each component in [0, 1].
//
// Premultiplied source formats are unassociated before the transfer function
// is applied: a stored premultiplied sRGB channel p represents alpha*encode(c),
// so recovering c requires dividing by alpha in encoded space first. A pixel
// with zero alpha is transparent black regardless of its stored channels.
// Out-of-bounds coordinates yield transparent black, matching image.Image.At
// for the standard image types.
func sampleStraight(src image.Image, x, y int) (r, g, b, a float64) {
	switch m := src.(type) {
	case *image.RGBA:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		i := m.PixOffset(x, y)
		av := float64(m.Pix[i+3])
		if av == 0 {
			return 0, 0, 0, 0
		}
		// Recover straight encoded channels by dividing by the 8-bit alpha.
		return clamp01(float64(m.Pix[i+0]) / av),
			clamp01(float64(m.Pix[i+1]) / av),
			clamp01(float64(m.Pix[i+2]) / av),
			av / 0xff

	case *image.NRGBA:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		i := m.PixOffset(x, y)
		return float64(m.Pix[i+0]) / 0xff,
			float64(m.Pix[i+1]) / 0xff,
			float64(m.Pix[i+2]) / 0xff,
			float64(m.Pix[i+3]) / 0xff

	case *image.RGBA64:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		i := m.PixOffset(x, y)
		av := float64(uint16(m.Pix[i+6])<<8 | uint16(m.Pix[i+7]))
		if av == 0 {
			return 0, 0, 0, 0
		}
		return clamp01(float64(uint16(m.Pix[i+0])<<8|uint16(m.Pix[i+1])) / av),
			clamp01(float64(uint16(m.Pix[i+2])<<8|uint16(m.Pix[i+3])) / av),
			clamp01(float64(uint16(m.Pix[i+4])<<8|uint16(m.Pix[i+5])) / av),
			av / 0xffff

	case *image.NRGBA64:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		i := m.PixOffset(x, y)
		return float64(uint16(m.Pix[i+0])<<8|uint16(m.Pix[i+1])) / 0xffff,
			float64(uint16(m.Pix[i+2])<<8|uint16(m.Pix[i+3])) / 0xffff,
			float64(uint16(m.Pix[i+4])<<8|uint16(m.Pix[i+5])) / 0xffff,
			float64(uint16(m.Pix[i+6])<<8|uint16(m.Pix[i+7])) / 0xffff

	case *image.Gray:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		v := float64(m.Pix[m.PixOffset(x, y)]) / 0xff
		return v, v, v, 1

	case *image.Gray16:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0
		}
		i := m.PixOffset(x, y)
		v := float64(uint16(m.Pix[i])<<8|uint16(m.Pix[i+1])) / 0xffff
		return v, v, v, 1
	}

	// Generic path. RGBA64At avoids an interface allocation for the types that
	// provide it; the values are alpha-premultiplied, as color.Color.RGBA()
	// specifies.
	var rv, gv, bv, av float64
	if m, ok := src.(image.RGBA64Image); ok {
		c := m.RGBA64At(x, y)
		rv, gv, bv, av = float64(c.R), float64(c.G), float64(c.B), float64(c.A)
	} else {
		r32, g32, b32, a32 := src.At(x, y).RGBA()
		rv, gv, bv, av = float64(r32), float64(g32), float64(b32), float64(a32)
	}
	a = clamp01(av / 0xffff)
	if a == 0 {
		return 0, 0, 0, 0
	}
	return clamp01((rv / 0xffff) / a),
		clamp01((gv / 0xffff) / a),
		clamp01((bv / 0xffff) / a),
		a
}

// linearize8 is the fast path for the most common linear-light sources:
// 8-bit straight-alpha NRGBA and 8-bit Gray. It decodes through the table and
// reports false for any other type or any non-default transfer function. The
// values returned are exact: decode8(v) == srgbDecode(v/0xff).
func linearize8(src image.Image, x, y int) (r, g, b, a float64, ok bool) {
	switch m := src.(type) {
	case *image.NRGBA:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0, true
		}
		i := m.PixOffset(x, y)
		return decode8(m.Pix[i+0]), decode8(m.Pix[i+1]), decode8(m.Pix[i+2]),
			float64(m.Pix[i+3]) / 0xff, true
	case *image.Gray:
		if !(image.Point{x, y}.In(m.Rect)) {
			return 0, 0, 0, 0, true
		}
		v := decode8(m.Pix[m.PixOffset(x, y)])
		return v, v, v, 1, true
	}
	return 0, 0, 0, 0, false
}

// linearize returns a source pixel in the working (filtering) space as a
// straight color, together with its alpha.
func linearize(src image.Image, x, y int, opts *Options) (r, g, b, a float64) {
	if opts.Transfer == TransferLinear {
		var ok bool
		if r, g, b, a, ok = linearize8(src, x, y); !ok {
			r, g, b, a = sampleStraight(src, x, y)
			r, g, b = srgbDecode(r), srgbDecode(g), srgbDecode(b)
		}
	} else {
		r, g, b, a = sampleStraight(src, x, y)
	}
	if opts.SrcMask != nil {
		_, _, _, ma := opts.SrcMask.At(opts.SrcMaskP.X+x, opts.SrcMaskP.Y+y).RGBA()
		a *= float64(ma) / 0xffff
		if a < 0 {
			a = 0
		} else if a > 1 {
			a = 1
		}
	}
	return r, g, b, a
}

// finish converts a filtered working-space color to premultiplied sRGB.
//
// In AlphaPremultiplied mode the accumulator holds sum(w*color*alpha) and
// sum(w*alpha), so the straight color is their ratio. In AlphaStraight mode it
// holds the straight color and alpha separately. In both cases the result is
// clamped to the legal premultiplied range (0 <= channel <= alpha <= 1) and a
// zero or negative alpha yields transparent black, which also makes negative
// lobes and non-finite accumulations degrade to a defined value.
func finish(pr, pg, pb, pa float64, premul bool, transfer TransferFunction) (r, g, b, a float64) {
	pa = clamp01(pa)
	a = pa
	if premul {
		if !(pa > 0) {
			return 0, 0, 0, 0
		}
		pr, pg, pb = pr/pa, pg/pa, pb/pa
	}
	pr, pg, pb = clamp01(pr), clamp01(pg), clamp01(pb)
	if transfer == TransferLinear {
		pr, pg, pb = srgbEncode(pr), srgbEncode(pg), srgbEncode(pb)
	}
	// clamp01 again: srgbEncode of a clamped input is in [0, 1], but be
	// explicit, and force channel <= alpha.
	return clamp01(pr * a), clamp01(pg * a), clamp01(pb * a), a
}

// rgba64Setter returns a function that writes one premultiplied color.RGBA64.
// RGBA64Image's SetRGBA64 avoids the color.Color interface boxing that
// Image.Set incurs.
func rgba64Setter(dst Image) func(x, y int, c color.RGBA64) {
	if s, ok := dst.(interface {
		SetRGBA64(x, y int, c color.RGBA64)
	}); ok {
		return s.SetRGBA64
	}
	return func(x, y int, c color.RGBA64) {
		dst.Set(x, y, c)
	}
}

// writeComposited applies the Porter-Duff operator to one filtered source
// pixel and writes the result.
//
// The composition is done on premultiplied sRGB storage values, exactly as the
// default (generated) paths do. Only the resampling of the source is performed
// in linear light: Over and Src therefore have the same relative order and the
// same arithmetic as when TransferSRGB is used. r, g, b and a are premultiplied
// sRGB in [0, 1].
func writeComposited(dst Image, set func(int, int, color.RGBA64), x, y int, r, g, b, a float64, op Op, opts *Options) {
	if opts.DstMask != nil {
		// A destination mask is coverage. For Over, the masked source is
		// composited normally: dst*(1 - srcA*mask) + src*mask. For Src, the
		// destination is interpolated toward the source by the mask coverage:
		// dst*(1 - mask) + src*mask. These are the semantics of the default
		// Image fallback paths (there is no typed fast path when DstMask is
		// set).
		_, _, _, ma := opts.DstMask.At(opts.DstMaskP.X+x, opts.DstMaskP.Y+y).RGBA()
		m := clamp01(float64(ma) / 0xffff)
		qr, qg, qb, qa := dst.At(x, y).RGBA()
		dr := clamp01(float64(qr) / 0xffff)
		dg := clamp01(float64(qg) / 0xffff)
		db := clamp01(float64(qb) / 0xffff)
		da := clamp01(float64(qa) / 0xffff)
		if op == Over {
			r, g, b, a = r*m, g*m, b*m, a*m
			ia := 1 - a
			r, g, b, a = r+dr*ia, g+dg*ia, b+db*ia, a+da*ia
		} else {
			im := 1 - m
			r, g, b, a = r*m+dr*im, g*m+dg*im, b*m+db*im, a*m+da*im
		}
	} else if op == Over {
		qr, qg, qb, qa := dst.At(x, y).RGBA()
		ia := 1 - a
		r = r + clamp01(float64(qr)/0xffff)*ia
		g = g + clamp01(float64(qg)/0xffff)*ia
		b = b + clamp01(float64(qb)/0xffff)*ia
		a = a + clamp01(float64(qa)/0xffff)*ia
	}
	set(x, y, color.RGBA64{
		R: ftou(clamp01(r)),
		G: ftou(clamp01(g)),
		B: ftou(clamp01(b)),
		A: ftou(clamp01(a)),
	})
}

// weight returns c.weight if it is finite and zero otherwise. A user-supplied
// Kernel.At that returns NaN or an infinity cannot be normalized meaningfully;
// dropping that contribution keeps concurrent results deterministic and
// finite.
func weight(w float64) float64 {
	if !finite(w) {
		return 0
	}
	return w
}

func (z *kernelScaler) scaleLinear(dst Image, dr, adr image.Rectangle, src image.Image, sr image.Rectangle, op Op, opts *Options) {
	var tmp [][4]float64
	if z.pool.New != nil {
		tmpp := z.pool.Get().(*[][4]float64)
		defer z.pool.Put(tmpp)
		tmp = *tmpp
	} else {
		tmp = z.makeTmpBuf()
	}

	premul := opts.Alpha == AlphaPremultiplied

	// Horizontal pass: sample each source pixel once per destination column,
	// convert to the working space, and (in premultiplied mode) premultiply
	// there. tmp holds one row per source row, one entry per destination
	// column, and is indexed the same way as the generated paths.
	t := 0
	for y := int32(0); y < z.sh; y++ {
		sy := sr.Min.Y + int(y)
		for _, s := range z.horizontal.sources {
			var pr, pg, pb, pa float64
			for _, c := range z.horizontal.contribs[s.i:s.j] {
				w := weight(c.weight)
				if w == 0 {
					continue
				}
				r, g, b, a := linearize(src, sr.Min.X+int(c.coord), sy, opts)
				if premul {
					r, g, b = r*a, g*a, b*a
				}
				pr += w * r
				pg += w * g
				pb += w * b
				pa += w * a
			}
			iw := s.invTotalWeight
			if !finite(iw) {
				iw = 0
			}
			tmp[t] = [4]float64{pr * iw, pg * iw, pb * iw, pa * iw}
			t++
		}
	}

	// Vertical pass and composition.
	set := rgba64Setter(dst)
	for dx := int32(adr.Min.X); dx < int32(adr.Max.X); dx++ {
		x := dr.Min.X + int(dx)
		for dy, s := range z.vertical.sources[adr.Min.Y:adr.Max.Y] {
			var pr, pg, pb, pa float64
			for _, c := range z.vertical.contribs[s.i:s.j] {
				w := weight(c.weight)
				if w == 0 {
					continue
				}
				p := &tmp[int(c.coord)*int(z.dw)+int(dx)]
				pr += w * p[0]
				pg += w * p[1]
				pb += w * p[2]
				pa += w * p[3]
			}
			iw := s.invTotalWeight
			if !finite(iw) {
				iw = 0
			}
			r, g, b, a := finish(pr*iw, pg*iw, pb*iw, pa*iw, premul, opts.Transfer)
			writeComposited(dst, set, x, dr.Min.Y+int(adr.Min.Y+dy), r, g, b, a, op, opts)
		}
	}
}

func (q *Kernel) transformLinear(dst Image, dr, adr image.Rectangle, d2s *f64.Aff3, src image.Image, sr image.Rectangle, bias image.Point, op Op, opts *Options) {
	// When shrinking, broaden the effective kernel support so that we still
	// visit every source pixel. This mirrors the generated transform paths.
	xscale := abs(d2s[0])
	if s := abs(d2s[1]); xscale < s {
		xscale = s
	}
	yscale := abs(d2s[3])
	if s := abs(d2s[4]); yscale < s {
		yscale = s
	}
	xHalfWidth, xKernelArgScale := q.Support, 1.0
	if xscale > 1 {
		xHalfWidth *= xscale
		xKernelArgScale = 1 / xscale
	}
	yHalfWidth, yKernelArgScale := q.Support, 1.0
	if yscale > 1 {
		yHalfWidth *= yscale
		yKernelArgScale = 1 / yscale
	}
	if !finite(xHalfWidth) {
		xHalfWidth = 0
	}
	if !finite(yHalfWidth) {
		yHalfWidth = 0
	}

	// Guard against NaN/Inf support or absurd scales before converting to int:
	// a non-finite half width would otherwise make make() panic.
	xw := 1 + 2*int(math.Ceil(clampMax(xHalfWidth)))
	yw := 1 + 2*int(math.Ceil(clampMax(yHalfWidth)))
	// ix/jx are clamped to [sr.Min.X, sr.Max.X), so no pass can need more
	// than sr.Dx() weights. Bounding the buffer by the source dimensions keeps
	// a pathological Support (or a tiny source scaled down enormously) from
	// allocating an enormous slice; the loop below clamps to len as a backstop.
	xw = max(1, min(xw, sr.Dx()))
	yw = max(1, min(yw, sr.Dy()))
	xWeights := make([]float64, xw)
	yWeights := make([]float64, yw)

	premul := opts.Alpha == AlphaPremultiplied
	set := rgba64Setter(dst)

	for dy := int32(adr.Min.Y); dy < int32(adr.Max.Y); dy++ {
		dyf := float64(dr.Min.Y+int(dy)) + 0.5
		for dx := int32(adr.Min.X); dx < int32(adr.Max.X); dx++ {
			dxf := float64(dr.Min.X+int(dx)) + 0.5
			sx := float64(d2s[0]*dxf) + float64(d2s[1]*dyf) + d2s[2]
			sy := float64(d2s[3]*dxf) + float64(d2s[4]*dyf) + d2s[5]
			// A singular or non-finite transform would otherwise convert
			// NaN/Inf to an implementation-defined int. Leave dst unchanged.
			if !finite(sx) || !finite(sy) {
				continue
			}
			if !(image.Point{int(sx) + bias.X, int(sy) + bias.Y}).In(sr) {
				continue
			}

			sx += float64(bias.X)
			sx -= 0.5
			ix := max(int(math.Floor(sx-xHalfWidth)), sr.Min.X)
			jx := min(int(math.Ceil(sx+xHalfWidth)), sr.Max.X)
			if jx-ix > len(xWeights) {
				// Only reachable with a non-finite or pathological Support;
				// clamp rather than panic or read out of bounds.
				jx = ix + len(xWeights)
			}

			totalXWeight := 0.0
			for kx := ix; kx < jx; kx++ {
				xWeight := 0.0
				if t := abs((sx - float64(kx)) * xKernelArgScale); t < q.Support {
					xWeight = weight(q.At(t))
				}
				xWeights[kx-ix] = xWeight
				totalXWeight += xWeight
			}
			if !finite(totalXWeight) {
				totalXWeight = 0
			}
			if totalXWeight != 0 {
				for x := range xWeights[:jx-ix] {
					xWeights[x] /= totalXWeight
				}
			}

			sy += float64(bias.Y)
			sy -= 0.5
			iy := max(int(math.Floor(sy-yHalfWidth)), sr.Min.Y)
			jy := min(int(math.Ceil(sy+yHalfWidth)), sr.Max.Y)
			if jy-iy > len(yWeights) {
				jy = iy + len(yWeights)
			}

			totalYWeight := 0.0
			for ky := iy; ky < jy; ky++ {
				yWeight := 0.0
				if t := abs((sy - float64(ky)) * yKernelArgScale); t < q.Support {
					yWeight = weight(q.At(t))
				}
				yWeights[ky-iy] = yWeight
				totalYWeight += yWeight
			}
			if !finite(totalYWeight) {
				totalYWeight = 0
			}
			if totalYWeight != 0 {
				for y := range yWeights[:jy-iy] {
					yWeights[y] /= totalYWeight
				}
			}

			var pr, pg, pb, pa float64
			for ky := iy; ky < jy; ky++ {
				yWeight := yWeights[ky-iy]
				if yWeight == 0 {
					continue
				}
				for kx := ix; kx < jx; kx++ {
					w := xWeights[kx-ix] * yWeight
					if w == 0 {
						continue
					}
					r, g, b, a := linearize(src, kx, ky, opts)
					if premul {
						r, g, b = r*a, g*a, b*a
					}
					pr += w * r
					pg += w * g
					pb += w * b
					pa += w * a
				}
			}
			r, g, b, a := finish(pr, pg, pb, pa, premul, opts.Transfer)
			writeComposited(dst, set, dr.Min.X+int(dx), dr.Min.Y+int(dy), r, g, b, a, op, opts)
		}
	}
}

// clampMax bounds a kernel half width so that converting the doubled weight
// buffer length to int cannot overflow. It is far above any practical Support.
func clampMax(v float64) float64 {
	const limit = 1 << 24
	if !(v > 0) {
		return 0
	}
	if v > limit {
		return limit
	}
	return v
}
