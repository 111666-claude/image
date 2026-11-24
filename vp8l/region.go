// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

// This file implements decoding a rectangular region of a VP8L image.
//
// VP8L is a raster-scan format whose predictors, color cache and LZ77
// backwards references can depend on pixels anywhere earlier in the stream, so
// a region decoder must still consume every entropy-coded symbol. It does not,
// however, need to materialize the whole image: decoded pixels go into a fixed
// ring buffer, and every completed row is pushed through the inverse
// transforms and copied into the result only if the caller asked for it.
//
// The inverse transforms are applied one row at a time. Most of them (predictor,
// cross-color, subtract-green) are already row-local given the row above, and
// the color-index transform only widens the row. That keeps the peak allocation
// proportional to the image width rather than its area, and the result is
// proportional only to the requested rectangle.

import (
	"context"
	"errors"
	"image"
	"io"
)

// defaultRegionMemory is the memory cap used by DecodeRegion when the caller
// does not supply one.
const defaultRegionMemory = 256 << 20

// Options are the optional parameters to DecodeRegion. The zero value is
// valid.
type Options struct {
	// MaxMemory bounds the total number of bytes that DecodeRegion may
	// allocate, including the returned image. If zero, a 256 MiB limit is
	// used. Negative values are treated as zero.
	MaxMemory int64
}

// DecodeRegion decodes only the requested sub-rectangle of a VP8L (lossless
// WebP) image from r.
//
// The rectangle is given in the full image's coordinate space and is clipped
// to the image bounds. An empty clipped rectangle yields an empty *image.NRGBA
// without decoding pixel data and without allocating pixel storage. Pixels
// outside the requested rectangle are never exposed to the caller.
//
// The returned image's color semantics (non-alpha-premultiplied RGBA, byte
// order R, G, B, A) and its pixel values are identical to those of Decode.
//
// Cancellation is checked between decoded rows. A malformed or truncated
// bit-stream, or a stream that would need more than MaxMemory bytes, returns an
// error rather than panicking.
func DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, opts *Options) (*image.NRGBA, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limit := int64(defaultRegionMemory)
	if opts != nil && opts.MaxMemory > 0 {
		limit = opts.MaxMemory
	}

	d, w, h, err := decodeHeader(r)
	if err != nil {
		return nil, err
	}
	// VP8L headers encode 14-bit dimensions, so w and h are at most 16384 and
	// the pixel count at most 1<<28. Check the bounds explicitly so that no
	// later multiplication can overflow on a 32-bit platform.
	if w <= 0 || h <= 0 || w > 1<<14 || h > 1<<14 {
		return nil, errors.New("vp8l: invalid image dimensions")
	}
	acc := &memAccount{limit: limit}
	d.acc = acc
	d.ctx = ctx

	// Decode the transforms, mirroring Decode. The transform payloads (tile
	// modes, cross-color coefficients, palette) are small and are charged to
	// the memory limit as they are decoded.
	var (
		nTransforms int
		transforms  [nTransformTypes]transform
		seen        [nTransformTypes]bool
		originalW   = w
	)
	for {
		more, err := d.read(1)
		if err != nil {
			return nil, err
		}
		if more == 0 {
			break
		}
		var t transform
		t, w, err = d.decodeTransform(w, h)
		if err != nil {
			return nil, err
		}
		if seen[t.transformType] {
			return nil, errors.New("vp8l: repeated transform")
		}
		seen[t.transformType] = true
		transforms[nTransforms] = t
		nTransforms++
	}

	full := image.Rect(0, 0, int(originalW), int(h))
	out := rect.Intersect(full)
	if out.Empty() {
		return &image.NRGBA{}, nil
	}
	// The result is fully determined by the (clipped) request, never by the
	// image dimensions.
	if err := acc.charge(int64(out.Dx()) * int64(out.Dy()) * 4); err != nil {
		return nil, err
	}
	dst := image.NewNRGBA(out)

	p, err := newRegionPipeline(d, transforms[:nTransforms], w, h, dst, acc)
	if err != nil {
		return nil, err
	}
	if err := p.run(); err != nil {
		return nil, err
	}
	return dst, nil
}

// regionPipeline streams the inner (entropy-coded) image out of the bit-stream
// one row at a time, applies the inverse transforms, and copies requested
// pixels into the destination.
type regionPipeline struct {
	d          *decoder
	w, h       int32 // inner image dimensions
	transforms []transform
	dst        *image.NRGBA
	acc        *memAccount

	// Entropy state, mirroring decodePix.
	ccBits      uint32
	ccShift     uint32
	ccEntries   []uint32
	hGroups     []hGroup
	hPix        []byte
	hBits       uint32
	hMask       int32
	tilesPerRow int32

	// ring holds the most recently decoded inner pixels in RGBA order. It is
	// large enough for the format's practical maximum backwards distance, or
	// for the whole image when that is smaller.
	ring    []byte
	ringCap int64 // in pixels
	linear  int64 // pixels emitted so far

	// row holds the current row gathered from the ring; scratch holds the
	// output of each inverse transform stage.
	row     []byte
	scratch [nTransformTypes][]byte
	// prevPredRow holds the predictor stage's previous output row.
	prevPredRow []byte
}

func newRegionPipeline(d *decoder, transforms []transform, w, h int32, dst *image.NRGBA, acc *memAccount) (*regionPipeline, error) {
	p := &regionPipeline{
		d:          d,
		w:          w,
		h:          h,
		transforms: transforms,
		dst:        dst,
		acc:        acc,
	}

	// Color-cache parameters, same wire format as decodePix.
	useColorCache, err := d.read(1)
	if err != nil {
		return nil, err
	}
	if useColorCache != 0 {
		p.ccBits, err = d.read(4)
		if err != nil {
			return nil, err
		}
		if p.ccBits < 1 || 11 < p.ccBits {
			return nil, errors.New("vp8l: invalid color cache parameters")
		}
		p.ccShift = 32 - p.ccBits
		if err := acc.charge(int64(1<<p.ccBits) * 4); err != nil {
			return nil, err
		}
		p.ccEntries = make([]uint32, 1<<p.ccBits)
	}

	hGroups, hPix, hBits, err := d.decodeHuffmanGroups(w, h, true, p.ccBits)
	if err != nil {
		return nil, err
	}
	p.hGroups, p.hPix, p.hBits = hGroups, hPix, hBits
	if hBits != 0 {
		p.hMask, p.tilesPerRow = 1<<hBits-1, nTiles(w, hBits)
	}

	total := int64(w) * int64(h)
	if total <= 0 {
		return nil, errors.New("vp8l: invalid image dimensions")
	}
	ringPixels := total
	if ringPixels > maxRegionRingPixels {
		ringPixels = maxRegionRingPixels
	}
	if err := acc.charge(ringPixels * 4); err != nil {
		return nil, err
	}
	p.ring = make([]byte, ringPixels*4)
	p.ringCap = ringPixels
	return p, nil
}

// ringOffset returns the byte offset of pixel linear in the ring.
func (p *regionPipeline) ringOffset(linear int64) int {
	return int((linear % p.ringCap) * 4)
}

// run decodes every inner row, transforms it and copies the requested pixels
// into the destination image.
func (p *regionPipeline) run() error {
	var (
		x, y     int32
		hg       = &p.hGroups[0]
		lookupHG = p.hMask != 0
		total    = int64(p.w) * int64(p.h)
	)
	emit := func(r, g, b, a byte) error {
		o := p.ringOffset(p.linear)
		p.ring[o+0] = r
		p.ring[o+1] = g
		p.ring[o+2] = b
		p.ring[o+3] = a
		if p.ccBits != 0 {
			argb := uint32(r)<<16 | uint32(g)<<8 | uint32(b) | uint32(a)<<24
			p.ccEntries[(argb*colorCacheMultiplier)>>p.ccShift] = argb
		}
		p.linear++
		x++
		if x == p.w {
			if err := p.processRow(int32(y)); err != nil {
				return err
			}
			x, y = 0, y+1
		}
		return nil
	}

	for p.linear < total {
		if x == 0 {
			if err := p.d.ctx.Err(); err != nil {
				return err
			}
		}
		if lookupHG {
			i := 4 * (p.tilesPerRow*(y>>p.hBits) + (x >> p.hBits))
			idx := int(uint32(p.hPix[i])<<8 | uint32(p.hPix[i+1]))
			if idx >= len(p.hGroups) {
				return errors.New("vp8l: invalid Huffman group index")
			}
			hg = &p.hGroups[idx]
		}

		green, err := hg[huffGreen].next(p.d)
		if err != nil {
			return err
		}
		switch {
		case green < nLiteralCodes:
			red, err := hg[huffRed].next(p.d)
			if err != nil {
				return err
			}
			blue, err := hg[huffBlue].next(p.d)
			if err != nil {
				return err
			}
			alpha, err := hg[huffAlpha].next(p.d)
			if err != nil {
				return err
			}
			if err := emit(uint8(red), uint8(green), uint8(blue), uint8(alpha)); err != nil {
				return err
			}
			lookupHG = p.hMask != 0 && x&p.hMask == 0

		case green < nLiteralCodes+nLengthCodes:
			length, err := p.d.lz77Param(green - nLiteralCodes)
			if err != nil {
				return err
			}
			distSym, err := hg[huffDistance].next(p.d)
			if err != nil {
				return err
			}
			distCode, err := p.d.lz77Param(distSym)
			if err != nil {
				return err
			}
			dist := int64(distanceMap(p.w, distCode))
			// Every reference must point at an already-emitted pixel that is
			// still in the ring, and must not run past the end of the image.
			if dist < 1 || dist > p.linear || p.linear+int64(length) > total {
				return errors.New("vp8l: invalid LZ77 parameters")
			}
			if dist > p.ringCap {
				// The ring would have evicted the referenced pixel. Rather than
				// return different pixels than Decode, reject the stream.
				return errors.New("vp8l: LZ77 distance exceeds the region ring buffer")
			}
			for n := uint32(0); n < length; n++ {
				o := p.ringOffset(p.linear - dist)
				if err := emit(p.ring[o+0], p.ring[o+1], p.ring[o+2], p.ring[o+3]); err != nil {
					return err
				}
			}
			lookupHG = p.hMask != 0

		default:
			idx := green - nLiteralCodes - nLengthCodes
			if int(idx) >= len(p.ccEntries) {
				return errors.New("vp8l: invalid color cache index")
			}
			argb := p.ccEntries[idx]
			if err := emit(uint8(argb>>16), uint8(argb>>8), uint8(argb), uint8(argb>>24)); err != nil {
				return err
			}
			lookupHG = p.hMask != 0 && x&p.hMask == 0
		}
	}
	if y != p.h || x != 0 {
		return errors.New("vp8l: truncated pixel data")
	}
	return nil
}

// processRow gathers inner row y from the ring, applies every inverse
// transform in reverse order, and copies the requested span into the
// destination image.
func (p *regionPipeline) processRow(y int32) error {
	width := int(p.w)
	row, err := p.rowBuf(&p.row, width)
	if err != nil {
		return err
	}
	base := int64(y) * int64(width) % p.ringCap * 4
	mod := p.ringCap * 4
	for i := 0; i < width*4; i++ {
		row[i] = p.ring[(base+int64(i))%mod]
	}

	curWidth := width
	for i := len(p.transforms) - 1; i >= 0; i-- {
		t := &p.transforms[i]
		switch t.transformType {
		case transformTypeSubtractGreen:
			inverseSubtractGreenRow(row)

		case transformTypeCrossColor:
			inverseCrossColorRow(t, row, y, curWidth)

		case transformTypePredictor:
			next, err := p.inversePredictorRow(t, row, y, curWidth)
			if err != nil {
				return err
			}
			row = next

		case transformTypeColorIndexing:
			next, err := p.rowBuf(&p.scratch[transformTypeColorIndexing], int(t.oldWidth))
			if err != nil {
				return err
			}
			inverseColorIndexingRow(t, row, next, curWidth)
			row = next
			curWidth = int(t.oldWidth)
		}
	}
	out := p.dst.Bounds()
	yi := int(y)
	if yi < out.Min.Y || yi >= out.Max.Y {
		return nil
	}
	for x := out.Min.X; x < out.Max.X; x++ {
		si := x * 4
		di := p.dst.PixOffset(x, yi)
		p.dst.Pix[di+0] = row[si+0]
		p.dst.Pix[di+1] = row[si+1]
		p.dst.Pix[di+2] = row[si+2]
		p.dst.Pix[di+3] = row[si+3]
	}
	return nil
}

// rowBuf returns a reusable buffer of width pixels, growing it (and charging
// the growth) as needed.
func (p *regionPipeline) rowBuf(slot *[]byte, width int) ([]byte, error) {
	if b := *slot; len(b) >= width*4 {
		return b[:width*4], nil
	}
	if err := p.acc.charge(int64(width) * 4); err != nil {
		return nil, err
	}
	b := make([]byte, width*4)
	*slot = b
	return b, nil
}

// inversePredictorRow applies the predictor inverse to one row, using the
// previous reconstructed row as the row above.
func (p *regionPipeline) inversePredictorRow(t *transform, row []byte, y int32, width int) ([]byte, error) {
	mask := int32(1)<<t.bits - 1
	tilesPerRow := nTiles(t.oldWidth, t.bits)

	out, err := p.rowBuf(&p.scratch[transformTypePredictor], width)
	if err != nil {
		return nil, err
	}
	if p.prevPredRow == nil {
		// One extra pixel of capacity reproduces the whole-array decoder's
		// top-right read in the last column, which aliases the current row's
		// first pixel because both rows share one contiguous buffer.
		if err := p.acc.charge((int64(t.oldWidth) + 1) * 4); err != nil {
			return nil, err
		}
		p.prevPredRow = make([]byte, (t.oldWidth+1)*4)
	}
	top := p.prevPredRow

	if y == 0 {
		// The first pixel's predictor is mode 0 (opaque black); the rest of
		// the first row uses mode 1 (left).
		for k := 0; k < 4; k++ {
			out[k] = row[k]
		}
		out[3] += 0xff
		for x := 1; x < width; x++ {
			i := x * 4
			for k := 0; k < 4; k++ {
				out[i+k] = row[i+k] + out[i-4+k]
			}
		}
	} else {
		// The first column's predictor is mode 2 (top).
		for k := 0; k < 4; k++ {
			out[k] = row[k] + top[k]
		}
		// See the predictPixel doc comment: top's extra slot stands in for the
		// current row's first pixel, which is what the whole-array decoder
		// reads as the last column's top-right neighbour.
		copy(top[width*4:], out[:4])
		q := 4 * (y >> t.bits) * tilesPerRow
		mode := t.pix[q+1] & 0x0f
		q += 4
		for x := 1; x < width; x++ {
			i := x * 4
			if int32(x)&mask == 0 {
				mode = t.pix[q+1] & 0x0f
				q += 4
			}
			predictPixel(mode, row, out, top, i, x, width)
		}
	}
	copy(p.prevPredRow, out)
	return out, nil
}

// predictPixel reconstructs the pixel at byte offset i using one of the 14
// defined VP8L predictor modes. Modes 14 and 15 are unused by the format; like
// the whole-array inversePredictor, they leave the residual unchanged.
//
// top is the previous reconstructed row and must have one extra pixel of
// capacity past index width. The whole-array inversePredictor keeps the
// previous row and the current row in one contiguous buffer, so in the last
// column its top-right read aliases the first pixel of the current row; the
// caller stores that pixel in top's extra slot to reproduce the same values.
func predictPixel(mode byte, row, out, top []byte, i, x, width int) {
	l := i - 4
	tp := i
	switch mode {
	case 0:
		out[i+0] = row[i+0]
		out[i+1] = row[i+1]
		out[i+2] = row[i+2]
		out[i+3] = row[i+3] + 0xff
	case 1: // L
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + out[l+k]
		}
	case 2: // T
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[tp+k]
		}
	case 3: // TR
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[i+4+k]
		}
	case 4: // TL
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[tp-4+k]
		}
	case 5: // Average2(Average2(L, TR), T)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(avg2(out[l+k], top[i+4+k]), top[tp+k])
		}
	case 6: // Average2(L, TL)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(out[l+k], top[tp-4+k])
		}
	case 7: // Average2(L, T)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(out[l+k], top[tp+k])
		}
	case 8: // Average2(TL, T)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(top[tp-4+k], top[tp+k])
		}
	case 9: // Average2(T, TR)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(top[tp+k], top[i+4+k])
		}
	case 10: // Average2(Average2(L, TL), Average2(T, TR))
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(avg2(out[l+k], top[tp-4+k]), avg2(top[tp+k], top[i+4+k]))
		}
	case 11: // Select(L, T, TL)
		l0, l1, l2, l3 := int32(out[l+0]), int32(out[l+1]), int32(out[l+2]), int32(out[l+3])
		c0, c1, c2, c3 := int32(top[tp-4+0]), int32(top[tp-4+1]), int32(top[tp-4+2]), int32(top[tp-4+3])
		t0, t1, t2, t3 := int32(top[tp+0]), int32(top[tp+1]), int32(top[tp+2]), int32(top[tp+3])
		lv := abs(c0-t0) + abs(c1-t1) + abs(c2-t2) + abs(c3-t3)
		tv := abs(c0-l0) + abs(c1-l1) + abs(c2-l2) + abs(c3-l3)
		if lv < tv {
			out[i+0] = row[i+0] + uint8(l0)
			out[i+1] = row[i+1] + uint8(l1)
			out[i+2] = row[i+2] + uint8(l2)
			out[i+3] = row[i+3] + uint8(l3)
		} else {
			out[i+0] = row[i+0] + uint8(t0)
			out[i+1] = row[i+1] + uint8(t1)
			out[i+2] = row[i+2] + uint8(t2)
			out[i+3] = row[i+3] + uint8(t3)
		}
	case 12: // ClampAddSubtractFull(L, T, TL)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + clampAddSubtractFull(out[l+k], top[tp+k], top[tp-4+k])
		}
	case 13: // ClampAddSubtractHalf(Average2(L, T), TL)
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + clampAddSubtractHalf(avg2(out[l+k], top[tp+k]), top[tp-4+k])
		}
	default:
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k]
		}
	}
}

// inverseSubtractGreenRow applies the subtract-green inverse to one row.
func inverseSubtractGreenRow(row []byte) {
	for i := 0; i < len(row); i += 4 {
		g := row[i+1]
		row[i+0] += g
		row[i+2] += g
	}
}

// inverseCrossColorRow applies the cross-color inverse to one row. y is the
// row's index in the image, which selects the tile's coefficients.
func inverseCrossColorRow(t *transform, row []byte, y int32, width int) {
	mask := int32(1)<<t.bits - 1
	tilesPerRow := nTiles(t.oldWidth, t.bits)
	var greenToRed, greenToBlue, redToBlue int32
	q := 4 * (y >> t.bits) * tilesPerRow
	for x := 0; x < width; x++ {
		if int32(x)&mask == 0 {
			redToBlue = int32(int8(t.pix[q+0]))
			greenToBlue = int32(int8(t.pix[q+1]))
			greenToRed = int32(int8(t.pix[q+2]))
			q += 4
		}
		i := x * 4
		red := row[i+0]
		green := row[i+1]
		blue := row[i+2]
		red += uint8(uint32(greenToRed*int32(int8(green))) >> 5)
		blue += uint8(uint32(greenToBlue*int32(int8(green))) >> 5)
		blue += uint8(uint32(redToBlue*int32(int8(red))) >> 5)
		row[i+0] = red
		row[i+2] = blue
	}
}

// inverseColorIndexingRow expands one coded row of color indices into one
// outer-width row of palette pixels, mirroring inverseColorIndexing.
func inverseColorIndexingRow(t *transform, in, out []byte, inWidth int) {
	if t.bits == 0 {
		for x := 0; x < inWidth; x++ {
			idx := 4 * uint32(in[x*4+1])
			o := x * 4
			out[o+0] = t.pix[idx+0]
			out[o+1] = t.pix[idx+1]
			out[o+2] = t.pix[idx+2]
			out[o+3] = t.pix[idx+3]
		}
		return
	}

	vMask, xMask, bitsPerPixel := uint32(0), int32(0), uint32(8>>t.bits)
	switch t.bits {
	case 1:
		vMask, xMask = 0x0f, 0x01
	case 2:
		vMask, xMask = 0x03, 0x03
	case 3:
		vMask, xMask = 0x01, 0x07
	}
	p, d, v := 0, 0, uint32(0)
	for x := int32(0); x < t.oldWidth; x++ {
		// Only load a new coded pixel while the coded row still has one; the
		// padding bits of the final coded pixel are reused. This matches
		// inverseColorIndexing, whose coded row is exactly inWidth pixels wide
		// even when oldWidth is not a multiple of the packing.
		if x&xMask == 0 && int(x>>xMask) < inWidth {
			v = uint32(in[p+1])
			p += 4
		}
		idx := 4 * (v & vMask)
		out[d+0] = t.pix[idx+0]
		out[d+1] = t.pix[idx+1]
		out[d+2] = t.pix[idx+2]
		out[d+3] = t.pix[idx+3]
		d += 4
		v >>= bitsPerPixel
	}
}
