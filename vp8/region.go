// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8

// This file implements decoding a rectangular region of a VP8 frame.
//
// VP8 is a raster-scan format: the predictors of a macroblock depend on the
// macroblocks above and to its left, and the loop filter of a macroblock reads
// the row above it. A region decode therefore still reconstructs every
// macroblock row from row zero down to the last requested row, but it keeps
// only two macroblock rows of pixels in memory: the "above" row, in the first
// bank of the working image, and the row being reconstructed, in the second.
//
// Two details make the windowing sound:
//
//   - The one-shot DecodeFrame loop-filters the whole frame only after every
//     macroblock has been reconstructed, so its predictors always see
//     unfiltered neighbours above. A streaming decode cannot defer the filter,
//     so it saves the unfiltered pixels of the row above in rawAboveY,
//     rawAboveCb and rawAboveCr and predicts from those.
//   - A macroblock's bottom edge is filtered by the row below it, so the band's
//     last row is finalized by reconstructing and filtering one more row. When
//     the band reaches the bottom of the frame, the frame boundary plays that
//     role instead.
//
// Rows outside the requested region never reach the returned image.

import (
	"context"
	"errors"
	"image"
	"io"
)

var (
	// ErrRegionTooLarge is returned when the requested region or the working
	// buffers needed to decode it exceed the memory limit.
	ErrRegionTooLarge = errors.New("vp8: region is too large")

	// ErrZeroSizeFrame is returned when a frame header declares a zero width
	// or height.
	ErrZeroSizeFrame = errors.New("vp8: zero-size frame")
)

// defaultRegionMemory is the memory cap applied by DecodeFrameRegion when the
// caller does not supply one.
const defaultRegionMemory = 256 << 20

// RegionOptions are the optional parameters to Decoder.DecodeFrameRegion.
type RegionOptions struct {
	// MaxMemory bounds the total number of bytes that DecodeFrameRegion may
	// allocate for its working buffers and its result. If zero, a 256 MiB
	// limit is used. Negative values are treated as zero.
	MaxMemory int64
}

// DecodeFrameRegion decodes the part of the current frame that intersects
// rect and returns it as a 4:2:0 YCbCr image whose Bounds equal the
// intersection of rect with the frame's visible rectangle.
//
// The decoder must already have been initialized with Init and have consumed
// the frame header with DecodeFrameHeader.
//
// An empty intersection is not an error: DecodeFrameRegion returns a zero-size
// image without decoding anything. Cancellation is checked once per macroblock
// row. The returned image owns freshly allocated pixel buffers; no pixel
// outside the returned Bounds is ever written to them.
//
// The returned pixels are identical to the corresponding pixels of the image
// that DecodeFrame would return for the same bitstream.
func (d *Decoder) DecodeFrameRegion(ctx context.Context, rect image.Rectangle, opts *RegionOptions) (*image.YCbCr, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if d.frameHeader.Width <= 0 || d.frameHeader.Height <= 0 {
		return nil, ErrZeroSizeFrame
	}

	full := image.Rect(0, 0, d.frameHeader.Width, d.frameHeader.Height)
	out := clipRect(rect, full)
	if out.Empty() {
		return &image.YCbCr{SubsampleRatio: image.YCbCrSubsampleRatio420}, nil
	}

	limit := int64(defaultRegionMemory)
	if opts != nil && opts.MaxMemory > 0 {
		limit = opts.MaxMemory
	}
	// Account for the working window and the result before allocating them.
	// The window is two macroblock rows tall and as wide as the frame; the
	// result is only as large as the requested rectangle. The decoder also
	// buffers the remaining input while parsing the header partitions, so the
	// unread input length is part of the peak footprint and is charged too.
	//
	// The chroma plane dimensions are computed exactly as image.NewYCbCr does
	// for a 4:2:0 image, because an odd-origin rectangle has one more chroma
	// row or column than a plain halving of its width and height.
	w := int64(16 * d.mbw)
	cw := int64((out.Max.X+1)/2 - out.Min.X/2)
	ch := int64((out.Max.Y+1)/2 - out.Min.Y/2)
	need := w*16*2 + // Two Y macroblock rows.
		2*(w/2)*8*2 + // Two Cb and two Cr macroblock rows.
		w*2 + // Unfiltered luma/chroma copies of the row above.
		int64(d.mbw)*64 + // Per-macroblock filter parameters.
		int64(out.Dx())*int64(out.Dy()) + 2*cw*ch + // The result: Y plus chroma.
		int64(d.r.n) // Header partitions read into memory.
	if need > limit {
		return nil, ErrRegionTooLarge
	}

	if err := d.parseOtherHeaders(); err != nil {
		return nil, err
	}

	mb0 := out.Min.Y >> 4
	mb1 := (out.Max.Y + 15) >> 4
	if mb1 > d.mbh {
		mb1 = d.mbh
	}
	decodeThrough := mb1
	if decodeThrough > d.mbh-1 {
		decodeThrough = d.mbh - 1
	}

	d.region = true
	d.rawAboveY = make([]byte, 16*d.mbw)
	d.rawAboveCb = make([]byte, 8*d.mbw)
	d.rawAboveCr = make([]byte, 8*d.mbw)
	d.img = image.NewYCbCr(
		image.Rect(0, 0, 16*d.mbw, 32),
		image.YCbCrSubsampleRatio420,
	)
	d.perMBFilterParams = make([]filterParam, d.mbw)
	d.upMB = make([]mb, d.mbw)
	dst := image.NewYCbCr(out, image.YCbCrSubsampleRatio420)

	err := d.decodeRegionRows(ctx, dst, mb0, mb1, decodeThrough)
	// Drop all per-frame state so that a failed decode cannot leak partially
	// decoded pixels into the next call.
	d.region = false
	d.img = nil
	d.rawAboveY, d.rawAboveCb, d.rawAboveCr = nil, nil, nil
	d.perMBFilterParams = nil
	d.upMB = nil
	if err != nil {
		return nil, err
	}
	return dst, nil
}

// decodeRegionRows reconstructs macroblock rows 0..decodeThrough in the
// two-bank window, filters them in raster order and copies the rows in
// [mb0, mb1) into dst.
func (d *Decoder) decodeRegionRows(ctx context.Context, dst *image.YCbCr, mb0, mb1, decodeThrough int) error {
	for mbx := 0; mbx < d.mbw; mbx++ {
		d.upMB[mbx] = mb{}
	}
	for mby := 0; mby <= decodeThrough; mby++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		d.leftMB = mb{}
		for mbx := 0; mbx < d.mbw; mbx++ {
			skip := d.reconstruct(mbx, mby)
			fs := d.filterParams[d.segment][btou(!d.usePredY16)]
			fs.inner = fs.inner || !skip
			d.perMBFilterParams[mbx] = fs
		}
		if d.fp.unexpectedEOF {
			return io.ErrUnexpectedEOF
		}
		for i := 0; i < d.nOP; i++ {
			if d.op[i].unexpectedEOF {
				return io.ErrUnexpectedEOF
			}
		}

		// Save the unfiltered bottom pixel row of this macroblock row for the
		// next row's predictors, then loop-filter this row in place. Filtering
		// also finalizes the bottom edge of the row above, which is in bank 0.
		// The row just reconstructed is in bank 1.
		copy(d.rawAboveY, d.img.Y[31*d.img.YStride:][:16*d.mbw])
		copy(d.rawAboveCb, d.img.Cb[15*d.img.CStride:][:8*d.mbw])
		copy(d.rawAboveCr, d.img.Cr[15*d.img.CStride:][:8*d.mbw])

		if d.filterHeader.level != 0 {
			if d.filterHeader.simple {
				d.simpleFilterRow(mby)
			} else {
				d.normalFilterRow(mby)
			}
		}

		if mby > 0 && mby-1 >= mb0 {
			d.copyRegionRow(dst, mby-1, 0)
		}
		// The just-filtered row becomes the next row's "above" row.
		d.shiftRegionRow()
	}
	// The deepest reconstructed row has no successor to filter its bottom
	// edge; its bottom edge is the frame boundary or the top of an unfiltered
	// row, and either way its pixels are final. It now sits in bank 0.
	if decodeThrough >= mb0 && decodeThrough == mb1-1 {
		d.copyRegionRow(dst, decodeThrough, 0)
	}
	return nil
}

// shiftRegionRow moves the filtered second bank of the window up into the
// first, so that it becomes the "above" row for the next macroblock row.
func (d *Decoder) shiftRegionRow() {
	copy(d.img.Y[:16*d.img.YStride], d.img.Y[16*d.img.YStride:32*d.img.YStride])
	copy(d.img.Cb[:8*d.img.CStride], d.img.Cb[8*d.img.CStride:16*d.img.CStride])
	copy(d.img.Cr[:8*d.img.CStride], d.img.Cr[8*d.img.CStride:16*d.img.CStride])
}

// copyRegionRow copies the visible part of the finalized macroblock row r from
// bank of the working window into dst.
func (d *Decoder) copyRegionRow(dst *image.YCbCr, r, bank int) {
	out := dst.Bounds()
	y0, y1 := 16*r, 16*r+16
	if y0 < out.Min.Y {
		y0 = out.Min.Y
	}
	if y1 > out.Max.Y {
		y1 = out.Max.Y
	}
	if y0 >= y1 {
		return
	}
	x0, x1 := out.Min.X, out.Max.X
	if x0 < 0 {
		x0 = 0
	}
	if x1 > d.frameHeader.Width {
		x1 = d.frameHeader.Width
	}
	if x0 >= x1 {
		return
	}
	c0 := x0 / 2
	nC := (x1+1)/2 - c0
	for y := y0; y < y1; y++ {
		srcY := d.img.Y[(bank*16+(y-16*r))*d.img.YStride:]
		dstY := dst.YOffset(x0, y)
		copy(dst.Y[dstY:dstY+x1-x0], srcY[x0:x1])

		// Writing each chroma row once per luma row is idempotent and copes
		// with a Bounds origin that is odd.
		dstC := dst.COffset(x0, y)
		src := d.img.Cb[(bank*8+(y-16*r)/2)*d.img.CStride+c0:]
		copy(dst.Cb[dstC:dstC+nC], src[:nC])
		src = d.img.Cr[(bank*8+(y-16*r)/2)*d.img.CStride+c0:]
		copy(dst.Cr[dstC:dstC+nC], src[:nC])
	}
}

// simpleFilterRow applies the simple loop filter to macroblock row mby, which
// is in the second bank of the working window. mby is the row's index in the
// full frame and is only used to decide whether a top edge exists.
func (d *Decoder) simpleFilterRow(mby int) {
	yBase := 16 * d.img.YStride
	for mbx := 0; mbx < d.mbw; mbx++ {
		f := d.perMBFilterParams[mbx]
		if f.level == 0 {
			continue
		}
		l := int(f.level)
		yi := yBase + 16*mbx
		if mbx > 0 {
			filter2(d.img.Y, l+4, yi, d.img.YStride, 1)
		}
		if f.inner {
			filter2(d.img.Y, l, yi+0x4, d.img.YStride, 1)
			filter2(d.img.Y, l, yi+0x8, d.img.YStride, 1)
			filter2(d.img.Y, l, yi+0xc, d.img.YStride, 1)
		}
		if mby > 0 {
			filter2(d.img.Y, l+4, yi, 1, d.img.YStride)
		}
		if f.inner {
			filter2(d.img.Y, l, yi+d.img.YStride*0x4, 1, d.img.YStride)
			filter2(d.img.Y, l, yi+d.img.YStride*0x8, 1, d.img.YStride)
			filter2(d.img.Y, l, yi+d.img.YStride*0xc, 1, d.img.YStride)
		}
	}
}

// normalFilterRow applies the normal loop filter to macroblock row mby, which
// is in the second bank of the working window.
func (d *Decoder) normalFilterRow(mby int) {
	yBase := 16 * d.img.YStride
	cBase := 8 * d.img.CStride
	for mbx := 0; mbx < d.mbw; mbx++ {
		f := d.perMBFilterParams[mbx]
		if f.level == 0 {
			continue
		}
		l, il, hl := int(f.level), int(f.ilevel), int(f.hlevel)
		yi := yBase + 16*mbx
		ci := cBase + 8*mbx
		if mbx > 0 {
			filter246(d.img.Y, 16, l+4, il, hl, yi, d.img.YStride, 1, false)
			filter246(d.img.Cb, 8, l+4, il, hl, ci, d.img.CStride, 1, false)
			filter246(d.img.Cr, 8, l+4, il, hl, ci, d.img.CStride, 1, false)
		}
		if f.inner {
			filter246(d.img.Y, 16, l, il, hl, yi+0x4, d.img.YStride, 1, true)
			filter246(d.img.Y, 16, l, il, hl, yi+0x8, d.img.YStride, 1, true)
			filter246(d.img.Y, 16, l, il, hl, yi+0xc, d.img.YStride, 1, true)
			filter246(d.img.Cb, 8, l, il, hl, ci+0x4, d.img.CStride, 1, true)
			filter246(d.img.Cr, 8, l, il, hl, ci+0x4, d.img.CStride, 1, true)
		}
		if mby > 0 {
			filter246(d.img.Y, 16, l+4, il, hl, yi, 1, d.img.YStride, false)
			filter246(d.img.Cb, 8, l+4, il, hl, ci, 1, d.img.CStride, false)
			filter246(d.img.Cr, 8, l+4, il, hl, ci, 1, d.img.CStride, false)
		}
		if f.inner {
			filter246(d.img.Y, 16, l, il, hl, yi+d.img.YStride*0x4, 1, d.img.YStride, true)
			filter246(d.img.Y, 16, l, il, hl, yi+d.img.YStride*0x8, 1, d.img.YStride, true)
			filter246(d.img.Y, 16, l, il, hl, yi+d.img.YStride*0xc, 1, d.img.YStride, true)
			filter246(d.img.Cb, 8, l, il, hl, ci+d.img.CStride*0x4, 1, d.img.CStride, true)
			filter246(d.img.Cr, 8, l, il, hl, ci+d.img.CStride*0x4, 1, d.img.CStride, true)
		}
	}
}

// clipRect clips r to bounds without computing r.Dx or r.Dy first, so that
// rectangles with extreme coordinates cannot overflow.
func clipRect(r, bounds image.Rectangle) image.Rectangle {
	if r.Min.X < bounds.Min.X {
		r.Min.X = bounds.Min.X
	}
	if r.Min.Y < bounds.Min.Y {
		r.Min.Y = bounds.Min.Y
	}
	if r.Max.X > bounds.Max.X {
		r.Max.X = bounds.Max.X
	}
	if r.Max.Y > bounds.Max.Y {
		r.Max.Y = bounds.Max.Y
	}
	if r.Min.X >= r.Max.X || r.Min.Y >= r.Max.Y {
		return image.Rectangle{}
	}
	return r
}
