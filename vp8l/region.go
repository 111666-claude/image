// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8l

import (
	"context"
	"errors"
	"image"
	"io"
)

// defaultRegionMemoryLimit is the memory cap used by DecodeRegion when
// Options.MaxMemory is zero. It bounds every decoder-internal allocation
// (entropy ring buffer, color cache, Huffman trees, transform tiles and the
// returned image), independent of the (potentially hostile) image dimensions.
const defaultRegionMemoryLimit = 256 << 20 // 256 MiB

// Options are the optional parameters to DecodeRegion. The zero value is
// valid and uses sensible defaults.
type Options struct {
	// MaxMemory bounds the total number of bytes that DecodeRegion may
	// allocate, including the returned image's pixel buffer. If zero, a
	// 256 MiB limit is used. Negative values are treated as zero.
	MaxMemory int64
}

// DecodeRegion decodes only the requested sub-rectangle of a VP8L (lossless
// WebP) image from r.
//
// VP8L is a raster-scan format whose predictors, color cache and LZ77
// backwards references can depend on pixels anywhere earlier in the stream.
// DecodeRegion therefore still consumes the entire bit-stream sequentially,
// but it never allocates a full-image pixel array: decoded pixels live in a
// fixed-size ring buffer (bounded by the format's maximum LZ77 distance), and
// only the pixels inside r are materialized into the returned image.
//
// The rectangle r is given in the full image's coordinate space and is
// clipped to the image bounds. If the clipped rectangle is empty, an empty
// *image.NRGBA is returned and no pixel storage is allocated. Pixels outside
// the requested rectangle are never exposed to the caller.
//
// The returned image's color semantics (non-alpha-premultiplied RGBA, stored
// as R, G, B, A bytes) are identical to those of Decode.
//
// Cancellation (via ctx) and memory-limit breaches return the corresponding
// context or "vp8l: memory limit exceeded" errors; malformed or truncated
// bit-streams return descriptive errors without exceeding MaxMemory.
func DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, opts *Options) (*image.NRGBA, error) {
	limit := int64(defaultRegionMemoryLimit)
	if opts != nil && opts.MaxMemory > 0 {
		limit = opts.MaxMemory
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	d, w, h, err := decodeHeader(r)
	if err != nil {
		return nil, err
	}
	// VP8L headers encode at most 14 bits per dimension. Reject zero or
	// implausible dimensions before any multiplication can overflow.
	if w <= 0 || h <= 0 || w > 1<<14 || h > 1<<14 {
		return nil, errors.New("vp8l: invalid image dimensions")
	}
	// The largest possible pixel count is 1<<28, whose 4-byte stride fits in
	// both int32 and int on 32- and 64-bit platforms. Verify it explicitly so
	// later width*height multiplications cannot wrap.
	if int64(w) > (1<<31-1)/4/int64(h) {
		return nil, errors.New("vp8l: image dimensions overflow")
	}
	acc := &memAccount{limit: limit}
	d.acc = acc
	d.ctx = ctx

	// Decode the transforms, mirroring Decode.
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
		t, nw, err := d.decodeTransform(w, h)
		if err != nil {
			return nil, err
		}
		if seen[t.transformType] {
			return nil, errors.New("vp8l: repeated transform")
		}
		seen[t.transformType] = true
		transforms[nTransforms] = t
		nTransforms++
		w = nw
	}

	// Map the caller's rectangle through the forward transforms to the
	// coordinate space of the entropy-coded inner image. The result is not
	// used to skip entropy symbols (the stream is strictly sequential), but
	// the mapping validates the request against every transform's tile and
	// color-indexing boundaries.
	full := image.Rect(0, 0, int(originalW), int(h))
	out := rect.Intersect(full)
	if out.Empty() {
		return &image.NRGBA{Rect: image.Rectangle{}}, nil
	}
	if _, err := mapRegionRect(transforms[:nTransforms], w, out); err != nil {
		return nil, err
	}

	// The output buffer is charged up-front: its size is fully determined by
	// the (clipped) request, never by the image dimensions.
	if err := acc.charge(int64(out.Dx()) * int64(out.Dy()) * 4); err != nil {
		return nil, err
	}
	dst := image.NewNRGBA(out)

	// Run the streaming entropy decode into the ring buffer, applying the
	// inverse transforms one row at a time and copying only requested rows.
	pipe, err := newRegionPipeline(d, transforms[:nTransforms], w, h, out, acc, dst)
	if err != nil {
		return nil, err
	}
	if err := pipe.run(); err != nil {
		return nil, err
	}
	return dst, nil
}

// mapRegionRect maps a rectangle in the original (outer) image's coordinates
// through the forward transforms, yielding the rectangle in the entropy-coded
// inner image that must be decoded. Each transform may enlarge the required
// area: predictor and cross-color transforms work on fixed-size tiles, while
// color indexing packs multiple pixels into each coded byte.
func mapRegionRect(transforms []transform, innerWidth int32, r image.Rectangle) (image.Rectangle, error) {
	for _, t := range transforms {
		switch t.transformType {
		case transformTypePredictor, transformTypeCrossColor:
			// Tile modes are stored per (1<<bits)-sized tile. Include every
			// tile that the rectangle touches.
			tile := int(1) << t.bits
			x0 := floorDiv(r.Min.X, tile) * tile
			x1 := ceilDiv(r.Max.X, tile) * tile
			y0 := floorDiv(r.Min.Y, tile) * tile
			y1 := ceilDiv(r.Max.Y, tile) * tile
			r = image.Rect(x0, y0, x1, y1)

		case transformTypeColorIndexing:
			// Multiple outer pixels are packed into one coded pixel. The
			// inverse expands horizontally; rows are unaffected.
			switch t.bits {
			case 0:
				// One index per coded pixel.
			case 1: // 2 indices per byte.
				r = image.Rect(r.Min.X/2, r.Min.Y, (r.Max.X+1)/2, r.Max.Y)
			case 2: // 4 indices per byte.
				r = image.Rect(r.Min.X/4, r.Min.Y, (r.Max.X+3)/4, r.Max.Y)
			case 3: // 8 indices per byte.
				r = image.Rect(r.Min.X/8, r.Min.Y, (r.Max.X+7)/8, r.Max.Y)
			default:
				return image.Rectangle{}, errors.New("vp8l: invalid color indexing parameters")
			}
		}
	}
	// The Huffman-group meta-image partitions the inner image into tiles.
	return r.Intersect(image.Rect(0, 0, int(innerWidth), r.Max.Y)), nil
}

func floorDiv(x, d int) int { return x / d }
func ceilDiv(x, d int) int  { return (x + d - 1) / d }

// regionPipeline streams the inner image out of the bit-stream one row at a
// time, applies the inverse transforms incrementally, and copies requested
// pixels into the destination NRGBA image.
type regionPipeline struct {
	d          *decoder
	w, h       int32 // inner image dimensions
	transforms []transform
	out        image.Rectangle
	acc        *memAccount
	dst        *image.NRGBA

	// Entropy-decode state, mirroring decodePix.
	ccBits      uint32
	ccShift     uint32
	ccEntries   []uint32
	hGroups     []hGroup
	hPix        []byte
	hBits       uint32
	hMask       int32
	tilesPerRow int32

	// ring holds the most recently decoded inner pixels. Its capacity is the
	// whole image for small images or the format's maximum LZ77 distance for
	// large ones, so every legal backwards reference stays addressable.
	ring    []byte
	ringCap int64 // in pixels
	linear  int64 // number of pixels emitted so far

	// Reusable full-width row buffers, one per inverse-transform stage.
	// Slot 0 is also used for the row gathered from the ring: the inner
	// image width and the predictor's oldWidth coincide unless a color-
	// indexing transform widens the row, which happens in a later stage, so
	// the two uses never need different widths at the same time.
	rowBufs [nTransformTypes][]byte
	// prevPredRow retains the predictor stage's previous output row.
	prevPredRow []byte
}

func newRegionPipeline(d *decoder, transforms []transform, w, h int32,
	out image.Rectangle, acc *memAccount, dst *image.NRGBA) (*regionPipeline, error) {

	p := &regionPipeline{
		d:          d,
		w:          w,
		h:          h,
		transforms: transforms,
		out:        out,
		acc:        acc,
		dst:        dst,
	}

	// Decode color-cache parameters (same wire format as decodePix).
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
		p.ccEntries = make([]uint32, 1<<p.ccBits)
		if err := acc.charge(int64(len(p.ccEntries)) * 4); err != nil {
			return nil, err
		}
	}

	// Decode the Huffman groups (top-level meta-image).
	hGroups, hPix, hBits, err := d.decodeHuffmanGroups(w, h, true, p.ccBits)
	if err != nil {
		return nil, err
	}
	p.hGroups, p.hPix, p.hBits = hGroups, hPix, hBits
	if hBits != 0 {
		p.hMask, p.tilesPerRow = 1<<hBits-1, nTiles(w, hBits)
	}

	// Allocate the entropy ring buffer. Every allocation size is computed in
	// int64 and charged before the make, so hostile dimensions cannot trigger
	// an overflow or an unbounded allocation.
	totalPixels := int64(w) * int64(h)
	if totalPixels <= 0 {
		return nil, errors.New("vp8l: invalid image dimensions")
	}
	ringPixels := totalPixels
	if ringPixels > maxLZ77DistancePixels {
		ringPixels = maxLZ77DistancePixels
	}
	ringBytes := ringPixels * 4
	if err := acc.charge(ringBytes); err != nil {
		return nil, err
	}
	p.ring = make([]byte, ringBytes)
	p.ringCap = ringPixels
	return p, nil
}

// ringOffset returns the byte offset of pixel linear (plus byte k) in ring.
func (p *regionPipeline) ringOffset(linear int64, k int) int {
	return int((linear%p.ringCap)*4 + int64(k))
}

// insertColorCache inserts one decoded ARGB pixel into the color cache.
func (p *regionPipeline) insertColorCache(r, g, b, a byte) {
	if p.ccBits == 0 {
		return
	}
	argb := uint32(r)<<16 | uint32(g)<<8 | uint32(b) | uint32(a)<<24
	p.ccEntries[(argb*colorCacheMultiplier)>>p.ccShift] = argb
}

// run decodes every inner row, pushes each row through the inverse-transform
// pipeline and copies the requested outer rows into the destination.
func (p *regionPipeline) run() error {
	var (
		x, y        int32
		hg          = &p.hGroups[0]
		lookupHG    = p.hMask != 0
		totalPixels = int64(p.w) * int64(p.h)
	)
	// emit stores one decoded pixel in the ring, mirroring one iteration of
	// decodePix's pixel loop, and immediately drains a completed row through
	// the inverse-transform pipeline so that ring overwrites can never lose
	// data that a later LZ77 reference or transform might still need.
	emit := func(r, g, b, a byte) error {
		o := p.ringOffset(p.linear, 0)
		p.ring[o+0] = r
		p.ring[o+1] = g
		p.ring[o+2] = b
		p.ring[o+3] = a
		p.insertColorCache(r, g, b, a)
		p.linear++
		x++
		if x == p.w {
			if err := p.processRow(y * p.w); err != nil {
				return err
			}
			x, y = 0, y+1
		}
		return nil
	}

	for p.linear < totalPixels {
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
			// LZ77 backwards reference.
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
			dist := distanceMap(p.w, distCode)
			// dist must point at an already-emitted pixel (dist <= linear),
			// stay inside the image and run for at least one pixel. Because
			// the ring retains the most recent ringCap pixels and ringCap is
			// at least the largest encodable distance, any dist satisfying
			// dist <= linear is still addressable after a wrap.
			if dist < 1 || p.linear+int64(length) > totalPixels || int64(dist) > p.linear {
				return errors.New("vp8l: invalid LZ77 parameters")
			}
			for n := uint32(0); n < length; n++ {
				src := p.ringOffset(p.linear-int64(dist), 0)
				if err := emit(p.ring[src+0], p.ring[src+1], p.ring[src+2], p.ring[src+3]); err != nil {
					return err
				}
			}
			lookupHG = p.hMask != 0

		default:
			// Color-cache lookup.
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

// processRow gathers inner row y, applies every inverse transform in reverse
// order and copies the requested span into the destination image.
func (p *regionPipeline) processRow(rowStart int32) error {
	y := rowStart / p.w
	width := int(p.w)

	// Gather a contiguous copy of the row from the (possibly wrapping) ring.
	var err error
	row, err := p.chargedRowBuf(0, width)
	if err != nil {
		return err
	}
	base := (int64(rowStart) % p.ringCap) * 4
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
			curWidth = len(next) / 4

		case transformTypeColorIndexing:
			next, err := p.chargedRowBuf(nTransformTypes-1, int(t.oldWidth))
			if err != nil {
				return err
			}
			inverseColorIndexingRow(t, row, next, y, curWidth)
			row = next
			curWidth = int(t.oldWidth)
		}
	}

	// Copy the requested span. Coordinates are now in the outer image.
	if y >= int32(p.out.Min.Y) && y < int32(p.out.Max.Y) {
		for x := p.out.Min.X; x < p.out.Max.X; x++ {
			si := x * 4
			di := p.dst.PixOffset(x, int(y))
			p.dst.Pix[di+0] = row[si+0]
			p.dst.Pix[di+1] = row[si+1]
			p.dst.Pix[di+2] = row[si+2]
			p.dst.Pix[di+3] = row[si+3]
		}
	}
	return nil
}

// chargedRowBuf returns a reusable row buffer of width pixels, charged once.
func (p *regionPipeline) chargedRowBuf(slot, width int) ([]byte, error) {
	if b := p.rowBufs[slot]; len(b) >= width*4 {
		return b[:width*4], nil
	}
	if err := p.acc.charge(int64(width) * 4); err != nil {
		return nil, err
	}
	b := make([]byte, width*4)
	p.rowBufs[slot] = b
	return b, nil
}

// inverseSubtractGreenRow applies the subtract-green inverse to one row.
func inverseSubtractGreenRow(row []byte) {
	for i := 0; i < len(row); i += 4 {
		g := row[i+1]
		row[i+0] += g
		row[i+2] += g
	}
}

// inverseCrossColorRow applies the cross-color inverse to one row.
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

// inversePredictorRow applies the predictor inverse to one row, using
// prevPredRow as the row above and replacing it with the reconstructed row.
func (p *regionPipeline) inversePredictorRow(t *transform, row []byte, y int32, width int) ([]byte, error) {
	mask := int32(1)<<t.bits - 1
	tilesPerRow := nTiles(t.oldWidth, t.bits)

	if p.prevPredRow == nil {
		if err := p.acc.charge(int64(t.oldWidth) * 4); err != nil {
			return nil, err
		}
		p.prevPredRow = make([]byte, t.oldWidth*4)
	}
	top := p.prevPredRow
	out, err := p.chargedRowBuf(transformTypePredictor, width)
	if err != nil {
		return nil, err
	}

	if y == 0 {
		// First pixel: mode 0 (opaque black).
		out[0] = row[0]
		out[1] = row[1]
		out[2] = row[2]
		out[3] = row[3] + 0xff
		for x := 1; x < width; x++ {
			i := x * 4
			for k := 0; k < 4; k++ {
				out[i+k] = row[i+k] + out[i-4+k]
			}
		}
	} else {
		// First column: mode 2 (T).
		for k := 0; k < 4; k++ {
			out[k] = row[k] + top[k]
		}

		q := 4 * (y >> t.bits) * tilesPerRow
		mode := t.pix[q+1] & 0x0f
		q += 4
		for x := 1; x < width; x++ {
			i := x * 4
			if int32(x)&mask == 0 {
				mode = t.pix[q+1] & 0x0f
				q += 4
			}
			applyPredictorMode(mode, row, out, top, i, x, width)
		}
	}
	copy(p.prevPredRow, out)
	return out, nil
}

// applyPredictorMode reconstructs pixel at byte offset i using one of the 14
// VP8L predictor modes. Neighbors that would fall outside the image are
// clamped to the edge pixel, matching libwebp's border handling.
func applyPredictorMode(mode byte, row, out, top []byte, i, x, width int) {
	l := i - 4
	tp := i // same column in the row above
	switch mode {
	case 0: // Opaque black.
		out[i+0] = row[i+0]
		out[i+1] = row[i+1]
		out[i+2] = row[i+2]
		out[i+3] = row[i+3] + 0xff
	case 1: // L.
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + out[l+k]
		}
	case 2: // T.
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[tp+k]
		}
	case 3: // TR.
		tr := i + 4
		if x+1 >= width {
			tr = i
		}
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[tr+k]
		}
	case 4: // TL.
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + top[tp-4+k]
		}
	case 5: // Average2(Average2(L, TR), T).
		tr := i + 4
		if x+1 >= width {
			tr = i
		}
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(avg2(out[l+k], top[tr+k]), top[tp+k])
		}
	case 6: // Average2(L, TL).
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(out[l+k], top[tp-4+k])
		}
	case 7: // Average2(L, T).
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(out[l+k], top[tp+k])
		}
	case 8: // Average2(TL, T).
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(top[tp-4+k], top[tp+k])
		}
	case 9: // Average2(T, TR).
		tr := i + 4
		if x+1 >= width {
			tr = i
		}
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(top[tp+k], top[tr+k])
		}
	case 10: // Average2(Average2(L, TL), Average2(T, TR)).
		tr := i + 4
		if x+1 >= width {
			tr = i
		}
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + avg2(avg2(out[l+k], top[tp-4+k]), avg2(top[tp+k], top[tr+k]))
		}
	case 11: // Select(L, T, TL).
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
	case 12: // ClampAddSubtractFull(L, T, TL).
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + clampAddSubtractFull(out[l+k], top[tp+k], top[tp-4+k])
		}
	case 13: // ClampAddSubtractHalf(Average2(L, T), TL).
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k] + clampAddSubtractHalf(avg2(out[l+k], top[tp+k]), top[tp-4+k])
		}
	default:
		// Modes 14 and 15 are unused by the VP8L format. Mirror the whole-
		// array inversePredictor, whose switch has no matching case: the
		// residual passes through unchanged.
		for k := 0; k < 4; k++ {
			out[i+k] = row[i+k]
		}
	}
}

// inverseColorIndexingRow expands one coded row of color indices into one
// outer-width row of palette pixels, mirroring inverseColorIndexing.
func inverseColorIndexingRow(t *transform, in, out []byte, y int32, inWidth int) {
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
		// Only consume another coded pixel while the coded row holds one;
		// padding nibbles/bytes in the final coded pixel are re-read. This
		// mirrors inverseColorIndexing, whose coded row is exactly inWidth
		// pixels wide even when oldWidth is not a multiple of the packing.
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
