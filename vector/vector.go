// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go
//go:generate asmfmt -w acc_amd64.s

// asmfmt is https://github.com/klauspost/asmfmt

// Package vector provides a rasterizer for 2-D vector graphics.
package vector // import "golang.org/x/image/vector"

// The rasterizer's design follows
// https://medium.com/@raphlinus/inside-the-fastest-font-renderer-in-the-world-75ae5270c445
//
// Proof of concept code is in
// https://github.com/google/font-go
//
// See also:
// http://nothings.org/gamedev/rasterize/
// http://projects.tuxee.net/cl-vectors/section-the-cl-aa-algorithm
// https://people.gnome.org/~mathieu/libart/internals.html#INTERNALS-SCANLINE

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

// floatingPointMathThreshold is the width or height above which the rasterizer
// chooses to used floating point math instead of fixed point math.
//
// Both implementations of line segmentation rasterization (see raster_fixed.go
// and raster_floating.go) implement the same algorithm (in ideal, infinite
// precision math) but they perform differently in practice. The fixed point
// math version is roughtly 1.25x faster (on GOARCH=amd64) on the benchmarks,
// but at sufficiently large scales, the computations will overflow and hence
// show rendering artifacts. The floating point math version has more
// consistent quality over larger scales, but it is significantly slower.
//
// This constant determines when to use the faster implementation and when to
// use the better quality implementation.
//
// The rationale for this particular value is that TestRasterizePolygon in
// vector_test.go checks the rendering quality of polygon edges at various
// angles, inscribed in a circle of diameter 512. It may be that a higher value
// would still produce acceptable quality, but 512 seems to work.
const floatingPointMathThreshold = 512

const (
	// DefaultFlattenTolerance is the default value of
	// Rasterizer.FlattenTolerance: the maximum distance, in pixels, between
	// a Bézier curve and the polyline it is flattened into.
	//
	// The value is 1/(4·√3) ≈ 0.144 pixels, which reproduces the historical
	// flattening behavior of this package (subdividing when devSquared
	// reaches 1/3 into 1 + (3·devSquared)^(1/4) segments).
	DefaultFlattenTolerance = 0.14433756729740643

	// DefaultMaxFlattenPoints is the default value of
	// Rasterizer.MaxFlattenPoints: the maximum number of line segments (and
	// hence points) that a single QuadTo or CubeTo call is flattened into.
	DefaultMaxFlattenPoints = 4096
)

func lerp(t, px, py, qx, qy float32) (x, y float32) {
	return px + t*(qx-px), py + t*(qy-py)
}

func clamp(i, width int32) uint {
	if i < 0 {
		return 0
	}
	if i < width {
		return uint(i)
	}
	return uint(width)
}

// NewRasterizer returns a new Rasterizer whose rendered mask image is bounded
// by the given width and height.
func NewRasterizer(w, h int) *Rasterizer {
	z := &Rasterizer{}
	z.Reset(w, h)
	return z
}

// Raster is a 2-D vector graphics rasterizer.
//
// The zero value is usable, in that it is a Rasterizer whose rendered mask
// image has zero width and zero height. Call Reset to change its bounds.
type Rasterizer struct {
	// bufXxx are buffers of float32 or uint32 values, holding either the
	// individual or cumulative area values.
	//
	// We don't actually need both values at any given time, and to conserve
	// memory, the integration of the individual to the cumulative could modify
	// the buffer in place. In other words, we could use a single buffer, say
	// of type []uint32, and add some math.Float32bits and math.Float32frombits
	// calls to satisfy the compiler's type checking. As of Go 1.7, though,
	// there is a performance penalty between:
	//	bufF32[i] += x
	// and
	//	bufU32[i] = math.Float32bits(x + math.Float32frombits(bufU32[i]))
	//
	// See golang.org/issue/17220 for some discussion.
	bufF32 []float32
	bufU32 []uint32

	useFloatingPointMath bool

	size   image.Point
	firstX float32
	firstY float32
	penX   float32
	penY   float32

	// DrawOp is the operator used for the Draw method.
	//
	// The zero value is draw.Over.
	DrawOp draw.Op

	// FlattenTolerance is the maximum allowed distance, in pixels, between
	// a Bézier curve and the polyline it is flattened into before
	// rasterization. Smaller values give smoother curves at the cost of
	// more line segments. It applies uniformly to QuadTo and CubeTo; LineTo
	// and ClosePath are unaffected, so straight segments are never
	// subdivided and closed paths keep their direction.
	//
	// The tolerance is in the same coordinate space as the XxxTo arguments,
	// that is, device (raster) pixels after any user-supplied transform has
	// been applied. Callers rendering at a higher device resolution (such
	// as a HiDPI scale factor) should set this to the pixel error they can
	// accept at that final resolution; there is no need to adjust it per
	// curve.
	//
	// Non-positive or non-finite values are treated as
	// DefaultFlattenTolerance.
	FlattenTolerance float32

	// MaxFlattenPoints bounds the number of line segments (and hence LineTo
	// points) that a single QuadTo or CubeTo call is flattened into. It
	// caps the work done for extremely curvy or extremely large (possibly
	// out-of-bounds) geometry, which would otherwise be flattened into an
	// unbounded number of segments; such curves are approximated more
	// coarsely instead.
	//
	// Values less than 2 are treated as DefaultMaxFlattenPoints.
	MaxFlattenPoints int

	// TODO: an exported field equivalent to the mask point in the
	// draw.DrawMask function in the stdlib image/draw package?
}

// Reset resets a Rasterizer as if it was just returned by NewRasterizer.
//
// This includes setting z.DrawOp to draw.Over and z.FlattenTolerance and
// z.MaxFlattenPoints to their zero (default) values. No path or curve state
// is carried over from before the Reset.
func (z *Rasterizer) Reset(w, h int) {
	z.size = image.Point{w, h}
	z.firstX = 0
	z.firstY = 0
	z.penX = 0
	z.penY = 0
	z.DrawOp = draw.Over
	z.FlattenTolerance = 0
	z.MaxFlattenPoints = 0

	z.setUseFloatingPointMath(w > floatingPointMathThreshold || h > floatingPointMathThreshold)
}

func (z *Rasterizer) setUseFloatingPointMath(b bool) {
	z.useFloatingPointMath = b

	// Make z.bufF32 or z.bufU32 large enough to hold width * height samples.
	if z.useFloatingPointMath {
		if n := z.size.X * z.size.Y; n > cap(z.bufF32) {
			z.bufF32 = make([]float32, n)
		} else {
			z.bufF32 = z.bufF32[:n]
			for i := range z.bufF32 {
				z.bufF32[i] = 0
			}
		}
	} else {
		if n := z.size.X * z.size.Y; n > cap(z.bufU32) {
			z.bufU32 = make([]uint32, n)
		} else {
			z.bufU32 = z.bufU32[:n]
			for i := range z.bufU32 {
				z.bufU32[i] = 0
			}
		}
	}
}

// Size returns the width and height passed to NewRasterizer or Reset.
func (z *Rasterizer) Size() image.Point {
	return z.size
}

// Bounds returns the rectangle from (0, 0) to the width and height passed to
// NewRasterizer or Reset.
func (z *Rasterizer) Bounds() image.Rectangle {
	return image.Rectangle{Max: z.size}
}

// Pen returns the location of the path-drawing pen: the last argument to the
// most recent XxxTo call.
func (z *Rasterizer) Pen() (x, y float32) {
	return z.penX, z.penY
}

// ClosePath closes the current path.
func (z *Rasterizer) ClosePath() {
	z.LineTo(z.firstX, z.firstY)
}

// MoveTo starts a new path and moves the pen to (ax, ay).
//
// The coordinates are allowed to be out of the Rasterizer's bounds.
func (z *Rasterizer) MoveTo(ax, ay float32) {
	z.firstX = ax
	z.firstY = ay
	z.penX = ax
	z.penY = ay
}

// LineTo adds a line segment, from the pen to (bx, by), and moves the pen to
// (bx, by).
//
// The coordinates are allowed to be out of the Rasterizer's bounds.
func (z *Rasterizer) LineTo(bx, by float32) {
	if z.useFloatingPointMath {
		z.floatingLineTo(bx, by)
	} else {
		z.fixedLineTo(bx, by)
	}
}

// QuadTo adds a quadratic Bézier segment, from the pen via (bx, by) to (cx,
// cy), and moves the pen to (cx, cy).
//
// The coordinates are allowed to be out of the Rasterizer's bounds.
//
// The curve is flattened into at most z.MaxFlattenPoints line segments, with
// z.FlattenTolerance controlling the approximation error; see the
// documentation of those fields.
func (z *Rasterizer) QuadTo(bx, by, cx, cy float32) {
	ax, ay := z.penX, z.penY
	devsq := devSquared(ax, ay, bx, by, cx, cy)
	if n := z.flattenSegments(devsq); n > 1 {
		t, nInv := float32(0), 1/float32(n)
		for i := 0; i < n-1; i++ {
			t += nInv
			abx, aby := lerp(t, ax, ay, bx, by)
			bcx, bcy := lerp(t, bx, by, cx, cy)
			z.LineTo(lerp(t, abx, aby, bcx, bcy))
		}
	}
	z.LineTo(cx, cy)
}

// CubeTo adds a cubic Bézier segment, from the pen via (bx, by) and (cx, cy)
// to (dx, dy), and moves the pen to (dx, dy).
//
// The coordinates are allowed to be out of the Rasterizer's bounds.
//
// The curve is flattened into at most z.MaxFlattenPoints line segments, with
// z.FlattenTolerance controlling the approximation error; see the
// documentation of those fields.
func (z *Rasterizer) CubeTo(bx, by, cx, cy, dx, dy float32) {
	ax, ay := z.penX, z.penY
	devsq := devSquared(ax, ay, bx, by, dx, dy)
	if devsqAlt := devSquared(ax, ay, cx, cy, dx, dy); devsq < devsqAlt {
		devsq = devsqAlt
	}
	if n := z.flattenSegments(devsq); n > 1 {
		t, nInv := float32(0), 1/float32(n)
		for i := 0; i < n-1; i++ {
			t += nInv
			abx, aby := lerp(t, ax, ay, bx, by)
			bcx, bcy := lerp(t, bx, by, cx, cy)
			cdx, cdy := lerp(t, cx, cy, dx, dy)
			abcx, abcy := lerp(t, abx, aby, bcx, bcy)
			bcdx, bcdy := lerp(t, bcx, bcy, cdx, cdy)
			z.LineTo(lerp(t, abcx, abcy, bcdx, bcdy))
		}
	}
	z.LineTo(dx, dy)
}

// flattenSegments returns the number of line segments that a Bézier curve
// with the given devSquared measure is flattened into, honoring
// z.FlattenTolerance and z.MaxFlattenPoints.
//
// The result is deterministic and always in [1, maxN]: devsq may be NaN or
// +Inf for degenerate or huge coordinates, and the segment count estimate
// derived from a finite devsq may exceed the int range, but neither may
// cause unbounded work. There is no recursion here or in the callers, so
// flattening cannot overflow the goroutine stack.
func (z *Rasterizer) flattenSegments(devsq float32) int {
	if !(devsq <= math.MaxFloat32) {
		// devsq is NaN or +Inf, which means that some coordinate is
		// non-finite or so large (around 1e19 or more) that the curve is
		// hopelessly far off-screen. A single segment is deterministic,
		// bounded and visually identical.
		return 1
	}
	maxN := z.MaxFlattenPoints
	if maxN < 2 {
		maxN = DefaultMaxFlattenPoints
	}
	coeff, thresh := flattenCoefficients(z.FlattenTolerance)
	if !(float64(devsq) >= thresh) {
		return 1
	}
	// devsq is finite and positive here, but coeff*devsq can still be +Inf
	// for a tiny positive tolerance, so n can be +Inf. The comparison
	// below is false in that case, as it is for any n >= maxN-1.
	n := math.Sqrt(math.Sqrt(coeff * float64(devsq)))
	if !(n < float64(maxN-1)) {
		return maxN
	}
	return 1 + int(n)
}

// flattenCoefficients maps a flatten tolerance τ, in pixels, to the
// coefficient and devSquared threshold used to compute the segment count.
//
// For a quadratic Bézier, devSquared returns |D|² where D is the second
// difference of the control points, and the curve deviates from its chord by
// at most |D|/4. Splitting the curve into n equal-parameter segments scales
// D by 1/n², so the deviation bound becomes |D|/(4n²). Requiring this to be
// at most τ gives
//
//	n ≥ (|D|²/(16τ²))^(1/4),   subdividing only while |D|² ≥ 16τ².
//
// A tolerance of 1/(4·√3) ≈ 0.144 therefore recovers the historical
// constants coeff = 3 and thresh = 1/3. The default case below keeps those
// exact constants so that a zero FlattenTolerance is bit-for-bit compatible
// with previous versions of this package.
//
// Non-positive or non-finite tolerances (including NaN and ±Inf) map to the
// default, so every input yields a deterministic, finite segment count.
func flattenCoefficients(tolerance float32) (coeff, thresh float64) {
	if t := float64(tolerance); t > 0 && t <= math.MaxFloat32 {
		return 1 / (16 * t * t), 16 * t * t
	}
	return 3, 0.333
}

// devSquared returns a measure of how curvy the sequence (ax, ay) to (bx, by)
// to (cx, cy) is. It determines how many line segments will approximate a
// Bézier curve segment.
//
// http://lists.nongnu.org/archive/html/freetype-devel/2016-08/msg00080.html
// gives the rationale for this evenly spaced heuristic instead of a recursive
// de Casteljau approach:
//
// The reason for the subdivision by n is that I expect the "flatness"
// computation to be semi-expensive (it's done once rather than on each
// potential subdivision) and also because you'll often get fewer subdivisions.
// Taking a circular arc as a simplifying assumption (ie a spherical cow),
// where I get n, a recursive approach would get 2^⌈lg n⌉, which, if I haven't
// made any horrible mistakes, is expected to be 33% more in the limit.
func devSquared(ax, ay, bx, by, cx, cy float32) float32 {
	devx := ax - 2*bx + cx
	devy := ay - 2*by + cy
	return devx*devx + devy*devy
}

// Draw implements the Drawer interface from the standard library's image/draw
// package.
//
// The vector paths previously added via the XxxTo calls become the mask for
// drawing src onto dst.
func (z *Rasterizer) Draw(dst draw.Image, r image.Rectangle, src image.Image, sp image.Point) {
	// TODO: adjust r and sp (and mp?) if src.Bounds() doesn't contain
	// r.Add(sp.Sub(r.Min)).

	if src, ok := src.(*image.Uniform); ok {
		srcR, srcG, srcB, srcA := src.RGBA()
		switch dst := dst.(type) {
		case *image.Alpha:
			// Fast path for glyph rendering.
			if srcA == 0xffff {
				if z.DrawOp == draw.Over {
					z.rasterizeDstAlphaSrcOpaqueOpOver(dst, r)
				} else {
					z.rasterizeDstAlphaSrcOpaqueOpSrc(dst, r)
				}
				return
			}
		case *image.RGBA:
			if z.DrawOp == draw.Over {
				z.rasterizeDstRGBASrcUniformOpOver(dst, r, srcR, srcG, srcB, srcA)
			} else {
				z.rasterizeDstRGBASrcUniformOpSrc(dst, r, srcR, srcG, srcB, srcA)
			}
			return
		}
	}

	if z.DrawOp == draw.Over {
		z.rasterizeOpOver(dst, r, src, sp)
	} else {
		z.rasterizeOpSrc(dst, r, src, sp)
	}
}

func (z *Rasterizer) accumulateMask() {
	if z.useFloatingPointMath {
		if n := z.size.X * z.size.Y; n > cap(z.bufU32) {
			z.bufU32 = make([]uint32, n)
		} else {
			z.bufU32 = z.bufU32[:n]
		}
		if haveAccumulateSIMD {
			floatingAccumulateMaskSIMD(z.bufU32, z.bufF32)
		} else {
			floatingAccumulateMask(z.bufU32, z.bufF32)
		}
	} else {
		if haveAccumulateSIMD {
			fixedAccumulateMaskSIMD(z.bufU32)
		} else {
			fixedAccumulateMask(z.bufU32)
		}
	}
}

func (z *Rasterizer) rasterizeDstAlphaSrcOpaqueOpOver(dst *image.Alpha, r image.Rectangle) {
	// TODO: non-zero vs even-odd winding?
	if r == dst.Bounds() && r == z.Bounds() {
		// We bypass the z.accumulateMask step and convert straight from
		// z.bufF32 or z.bufU32 to dst.Pix.
		if z.useFloatingPointMath {
			if haveAccumulateSIMD {
				floatingAccumulateOpOverSIMD(dst.Pix, z.bufF32)
			} else {
				floatingAccumulateOpOver(dst.Pix, z.bufF32)
			}
		} else {
			if haveAccumulateSIMD {
				fixedAccumulateOpOverSIMD(dst.Pix, z.bufU32)
			} else {
				fixedAccumulateOpOver(dst.Pix, z.bufU32)
			}
		}
		return
	}

	z.accumulateMask()
	pix := dst.Pix[dst.PixOffset(r.Min.X, r.Min.Y):]
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			ma := z.bufU32[y*z.size.X+x]
			i := y*dst.Stride + x

			// This formula is like rasterizeOpOver's, simplified for the
			// concrete dst type and opaque src assumption.
			a := 0xffff - ma
			pix[i] = uint8((uint32(pix[i])*0x101*a/0xffff + ma) >> 8)
		}
	}
}

func (z *Rasterizer) rasterizeDstAlphaSrcOpaqueOpSrc(dst *image.Alpha, r image.Rectangle) {
	// TODO: non-zero vs even-odd winding?
	if r == dst.Bounds() && r == z.Bounds() {
		// We bypass the z.accumulateMask step and convert straight from
		// z.bufF32 or z.bufU32 to dst.Pix.
		if z.useFloatingPointMath {
			if haveAccumulateSIMD {
				floatingAccumulateOpSrcSIMD(dst.Pix, z.bufF32)
			} else {
				floatingAccumulateOpSrc(dst.Pix, z.bufF32)
			}
		} else {
			if haveAccumulateSIMD {
				fixedAccumulateOpSrcSIMD(dst.Pix, z.bufU32)
			} else {
				fixedAccumulateOpSrc(dst.Pix, z.bufU32)
			}
		}
		return
	}

	z.accumulateMask()
	pix := dst.Pix[dst.PixOffset(r.Min.X, r.Min.Y):]
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			ma := z.bufU32[y*z.size.X+x]

			// This formula is like rasterizeOpSrc's, simplified for the
			// concrete dst type and opaque src assumption.
			pix[y*dst.Stride+x] = uint8(ma >> 8)
		}
	}
}

func (z *Rasterizer) rasterizeDstRGBASrcUniformOpOver(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32) {
	z.accumulateMask()
	pix := dst.Pix[dst.PixOffset(r.Min.X, r.Min.Y):]
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			ma := z.bufU32[y*z.size.X+x]

			// This formula is like rasterizeOpOver's, simplified for the
			// concrete dst type and uniform src assumption.
			a := 0xffff - (sa * ma / 0xffff)
			i := y*dst.Stride + 4*x
			pix[i+0] = uint8(((uint32(pix[i+0])*0x101*a + sr*ma) / 0xffff) >> 8)
			pix[i+1] = uint8(((uint32(pix[i+1])*0x101*a + sg*ma) / 0xffff) >> 8)
			pix[i+2] = uint8(((uint32(pix[i+2])*0x101*a + sb*ma) / 0xffff) >> 8)
			pix[i+3] = uint8(((uint32(pix[i+3])*0x101*a + sa*ma) / 0xffff) >> 8)
		}
	}
}

func (z *Rasterizer) rasterizeDstRGBASrcUniformOpSrc(dst *image.RGBA, r image.Rectangle, sr, sg, sb, sa uint32) {
	z.accumulateMask()
	pix := dst.Pix[dst.PixOffset(r.Min.X, r.Min.Y):]
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			ma := z.bufU32[y*z.size.X+x]

			// This formula is like rasterizeOpSrc's, simplified for the
			// concrete dst type and uniform src assumption.
			i := y*dst.Stride + 4*x
			pix[i+0] = uint8((sr * ma / 0xffff) >> 8)
			pix[i+1] = uint8((sg * ma / 0xffff) >> 8)
			pix[i+2] = uint8((sb * ma / 0xffff) >> 8)
			pix[i+3] = uint8((sa * ma / 0xffff) >> 8)
		}
	}
}

func (z *Rasterizer) rasterizeOpOver(dst draw.Image, r image.Rectangle, src image.Image, sp image.Point) {
	z.accumulateMask()
	out := color.RGBA64{}
	outc := color.Color(&out)
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			sr, sg, sb, sa := src.At(sp.X+x, sp.Y+y).RGBA()
			ma := z.bufU32[y*z.size.X+x]

			// This algorithm comes from the standard library's image/draw
			// package.
			dr, dg, db, da := dst.At(r.Min.X+x, r.Min.Y+y).RGBA()
			a := 0xffff - (sa * ma / 0xffff)
			out.R = uint16((dr*a + sr*ma) / 0xffff)
			out.G = uint16((dg*a + sg*ma) / 0xffff)
			out.B = uint16((db*a + sb*ma) / 0xffff)
			out.A = uint16((da*a + sa*ma) / 0xffff)

			dst.Set(r.Min.X+x, r.Min.Y+y, outc)
		}
	}
}

func (z *Rasterizer) rasterizeOpSrc(dst draw.Image, r image.Rectangle, src image.Image, sp image.Point) {
	z.accumulateMask()
	out := color.RGBA64{}
	outc := color.Color(&out)
	for y, y1 := 0, r.Max.Y-r.Min.Y; y < y1; y++ {
		for x, x1 := 0, r.Max.X-r.Min.X; x < x1; x++ {
			sr, sg, sb, sa := src.At(sp.X+x, sp.Y+y).RGBA()
			ma := z.bufU32[y*z.size.X+x]

			// This algorithm comes from the standard library's image/draw
			// package.
			out.R = uint16(sr * ma / 0xffff)
			out.G = uint16(sg * ma / 0xffff)
			out.B = uint16(sb * ma / 0xffff)
			out.A = uint16(sa * ma / 0xffff)

			dst.Set(r.Min.X+x, r.Min.Y+y, outc)
		}
	}
}
