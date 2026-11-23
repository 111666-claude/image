// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

// This file contains the Rasterizer's coverage (alpha mask) export. Instead
// of compositing source colors onto a destination image (see Rasterizer.Draw),
// a client can export the per-pixel coverage of the vector paths into a
// reusable buffer for later, possibly repeated, post-processing.

import (
	"context"
	"errors"
	"image"
	"image/color"
)

// maxCoverageBufferBytes is the upper bound on the size of a CoverageBuffer's
// backing memory. It keeps a single buffer from requesting an unbounded amount
// of memory when given absurdly large dimensions or strides.
const maxCoverageBufferBytes = 1 << 30 // 1 GiB.

// Stable errors returned by NewCoverageBuffer and Rasterizer.Coverage.
var (
	// ErrInvalidCoverageRect is returned when the buffer rectangle is empty or
	// not well-formed.
	ErrInvalidCoverageRect = errors.New("vector: invalid coverage buffer rectangle")

	// ErrInvalidCoverageStride is returned when the stride is negative or
	// smaller than the rectangle's width.
	ErrInvalidCoverageStride = errors.New("vector: invalid coverage buffer stride")

	// ErrCoverageBufferTooSmall is returned when a caller-supplied buffer does
	// not have room for the rectangle and stride, or when the required size
	// exceeds maxCoverageBufferBytes.
	ErrCoverageBufferTooSmall = errors.New("vector: coverage buffer too small")

	// ErrNilCoverageBuffer is returned by Rasterizer.Coverage when given a nil
	// *CoverageBuffer.
	ErrNilCoverageBuffer = errors.New("vector: nil coverage buffer")
)

// CoverageBuffer is a reusable, read-mostly destination for a Rasterizer's
// per-pixel coverage values. Each value is an 8-bit alpha: 0x00 means the
// pixel is not covered at all and 0xff means it is fully covered. Values in
// between are partial coverage (anti-aliasing).
//
// A CoverageBuffer is not an *image.Alpha, but Alpha returns one backed by the
// same pixel storage, so the coverage can be fed straight into image/draw
// compositing (for example draw.DrawMask) without an extra copy.
//
// The zero value is not usable. Obtain one with NewCoverageBuffer.
//
// The pixel backing slice is intentionally unexported: callers access the
// contents through the image.Image methods or through the read-only
// *image.Alpha returned by Alpha, so the package never hands out a mutable
// internal slice of the Rasterizer itself.
type CoverageBuffer struct {
	pix    []byte
	stride int
	rect   image.Rectangle
}

// NewCoverageBuffer returns a CoverageBuffer whose logical rectangle is rect,
// using the given stride (the number of bytes between vertically adjacent
// rows). If stride is zero, the rectangle's width is used.
//
// If pix is non-nil it is used as the backing storage and must have len(pix)
// at least as large as the storage implied by rect and stride; otherwise the
// buffer allocates its own storage.
//
// NewCoverageBuffer does not panic on bad input. It returns one of the
// Err… sentinel errors instead:
//
//   - ErrInvalidCoverageRect if rect is empty or not well-formed.
//   - ErrInvalidCoverageStride if stride is negative or less than rect's width.
//   - ErrCoverageBufferTooSmall if pix is too short, or the implied storage
//     exceeds maxCoverageBufferBytes.
//
// On success the returned buffer's pixels are all zero.
func NewCoverageBuffer(rect image.Rectangle, stride int, pix []byte) (*CoverageBuffer, error) {
	w, h := rect.Dx(), rect.Dy()
	// Reject empty rectangles and non-canonical ones (Min greater than Max),
	// which image.Rect would otherwise silently canonicalize.
	if w <= 0 || h <= 0 ||
		rect.Min.X > rect.Max.X || rect.Min.Y > rect.Max.Y {
		return nil, ErrInvalidCoverageRect
	}
	if stride < 0 {
		return nil, ErrInvalidCoverageStride
	}
	if stride == 0 {
		stride = w
	}
	if stride < w {
		return nil, ErrInvalidCoverageStride
	}

	// Compute the required storage in int64 so that a pathological stride and
	// height cannot wrap around, then reject anything over the resource cap.
	need := int64(stride)*int64(h-1) + int64(w)
	if need <= 0 || need > maxCoverageBufferBytes {
		return nil, ErrCoverageBufferTooSmall
	}

	if pix != nil {
		if int64(len(pix)) < need {
			return nil, ErrCoverageBufferTooSmall
		}
		pix = pix[:need]
	} else {
		pix = make([]byte, need)
	}
	// make() yields zeroed storage; clear a caller-supplied buffer as well so
	// that NewCoverageBuffer always returns all-zero pixels.
	for i := range pix {
		pix[i] = 0
	}

	return &CoverageBuffer{
		pix:    pix,
		stride: stride,
		rect:   rect,
	}, nil
}

// Bounds implements the image.Image interface.
func (b *CoverageBuffer) Bounds() image.Rectangle { return b.rect }

// ColorModel implements the image.Image interface. It is color.AlphaModel.
func (b *CoverageBuffer) ColorModel() color.Model { return color.AlphaModel }

// At implements the image.Image interface. Points outside the buffer's bounds
// report a fully transparent color.
func (b *CoverageBuffer) At(x, y int) color.Color {
	return b.AlphaAt(x, y)
}

// AlphaAt returns the coverage at (x, y). Points outside the buffer's bounds
// report a fully transparent color, matching image.Alpha.AlphaAt.
func (b *CoverageBuffer) AlphaAt(x, y int) color.Alpha {
	if !(image.Point{x, y}.In(b.rect)) {
		return color.Alpha{}
	}
	return color.Alpha{A: b.pix[b.pixOffset(x, y)]}
}

// Stride returns the number of bytes between vertically adjacent rows.
func (b *CoverageBuffer) Stride() int { return b.stride }

// Alpha returns an *image.Alpha that shares the buffer's pixel storage, with
// the same bounds and stride. The returned image is an alias, not a copy:
// calling Clear, or another Rasterizer.Coverage into this buffer, changes what
// the returned image sees. Treat the returned image as read-only while this
// buffer may be re-used.
func (b *CoverageBuffer) Alpha() *image.Alpha {
	return &image.Alpha{
		Pix:    b.pix,
		Stride: b.stride,
		Rect:   b.rect,
	}
}

// Clear sets every pixel in the buffer to zero coverage. It is safe to call
// repeatedly, including on a freshly allocated buffer.
func (b *CoverageBuffer) Clear() {
	// Clear only the storage that Bounds and Alpha expose (each row's width
	// plus all complete rows), leaving any caller-supplied padding untouched.
	h := b.rect.Dy()
	w := b.rect.Dx()
	base := b.pixOffset(b.rect.Min.X, b.rect.Min.Y)
	for y := 0; y < h; y++ {
		row := b.pix[base+y*b.stride : base+y*b.stride+w]
		for i := range row {
			row[i] = 0
		}
	}
}

// pixOffset returns the index of (x, y), which must be inside b.rect.
func (b *CoverageBuffer) pixOffset(x, y int) int {
	return (y-b.rect.Min.Y)*b.stride + (x - b.rect.Min.X)
}

// Coverage writes the per-pixel coverage of the Rasterizer's vector paths into
// dst and returns dst.
//
// The coverage is the geometric, non-zero-winding coverage of the paths. It is
// independent of z.DrawOp and of any source image: there is no "transparent
// source" concept for coverage, so nothing about a source can change it, and
// neither the Rasterizer's destination nor its source (as used by Draw) are
// read or modified.
//
// The Rasterizer's coordinate space and the buffer's rectangle are mapped one
// to one by their shared origin: Rasterizer cell (x, y) corresponds to buffer
// pixel (x, y). Coverage is clipped to the intersection of z.Bounds() and
// dst.Bounds(); pixels of dst outside that intersection are left untouched
// (zero after NewCoverageBuffer or Clear). An empty intersection is legal and
// writes nothing. Empty paths (no path, or only degenerate segments) produce
// zero coverage, as does the zero-value Rasterizer.
//
// Coverage does not consume the paths or their intermediate area values. It is
// therefore safe to call Coverage repeatedly, and to interleave Coverage with
// Draw calls on the same Rasterizer, without calling Reset or re-feeding the
// paths. Reset (or a fresh NewRasterizer) starts a new, zeroed set of paths.
//
// A nil ctx is treated as context.Background. If ctx is already canceled,
// Coverage returns the context's error without writing. Cancellation is
// observed between scanlines; on cancellation it returns the context's error
// and the pixels already written are unspecified. A successful call always
// returns a nil error and never returns Err… values other than
// ErrNilCoverageBuffer (for a nil dst).
//
// Coverage is read-only with respect to the Rasterizer's path state, but a
// Rasterizer is not safe for concurrent path mutation (MoveTo, LineTo, Reset,
// Draw, Coverage). Multiple goroutines may concurrently read a CoverageBuffer
// that no goroutine is mutating.
func (z *Rasterizer) Coverage(ctx context.Context, dst *CoverageBuffer) (*CoverageBuffer, error) {
	if dst == nil {
		return nil, ErrNilCoverageBuffer
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Intersect the rasterizer's bounds (which start at the origin) with the
	// buffer's rectangle, in the rasterizer's cell coordinates.
	clip := z.Bounds().Intersect(dst.rect)
	if clip.Empty() {
		return dst, nil
	}

	z.accumulateMask()
	width := z.size.X

	for y := clip.Min.Y; y < clip.Max.Y; y++ {
		if (y-clip.Min.Y)&0x3f == 0x3f {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		rowBase := dst.pixOffset(clip.Min.X, y)
		maskRow := z.mask[y*width:]
		for x := clip.Min.X; x < clip.Max.X; x++ {
			// z.mask holds 16-bit coverage; the 8-bit value is the high byte.
			// This matches rasterizeDstAlphaSrcOpaqueOpSrc's "ma >> 8".
			dst.pix[rowBase+(x-clip.Min.X)] = uint8(maskRow[x] >> 8)
		}
	}
	return dst, nil
}
