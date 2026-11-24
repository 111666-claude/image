// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

import (
	"image"
	"math"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// This file implements the optional vertical metrics tables ('vhea' and
// 'vmtx') that are used for vertical (top-to-bottom) layout of CJK and other
// ideographic scripts.
//
// https://www.microsoft.com/typography/otspec/vhea.htm
// https://www.microsoft.com/typography/otspec/vmtx.htm

// verticalData is the parsed, font-wide part of the vertical metrics. The
// per-glyph parts (advances and top side bearings) are read from the vmtx
// table on demand.
type verticalData struct {
	// valid reports whether the font has a usable vhea/vmtx pair. If it is
	// false and verticalErr is nil, the font has neither table and the
	// vertical metrics methods fall back to the horizontal metrics.
	valid bool

	// ascent, descent and lineGap are the vhea table's vertTypoAscender,
	// vertTypoDescender (typically negative) and vertTypoLineGap fields, in
	// font units.
	ascent, descent, lineGap int32

	// caretSlopeRise and caretSlopeRun are the vhea table's caretSlopeRise
	// and caretSlopeRun fields.
	caretSlopeRise, caretSlopeRun int32

	// numVMetrics is the vhea table's numOfLongVerMetrics field: the number
	// of (advance height, top side bearing) records at the start of the vmtx
	// table. It is in [1, numGlyphs].
	numVMetrics int32

	// vmtxOffset is the absolute file offset of the vmtx table and vmtxLength
	// is its length in bytes.
	vmtxOffset uint32
	vmtxLength uint32
}

// parseVertical parses the optional vhea and vmtx tables. Its signature
// mirrors the other parseXxx methods used in Font.initialize, but it never
// returns a non-nil buf.
//
// A malformed pair produces a non-nil error, which the caller caches rather
// than returns, so that horizontal layout still works. Fonts without either
// table produce a zero verticalData and a nil error.
func (f *Font) parseVertical(buf []byte, numGlyphs int32) (buf1 []byte, v verticalData, err error) {
	// A table that was not seen while parsing the table directory is the zero
	// table. Do not test just the offset: a dfont font's first table can be at
	// offset zero.
	hasVhea := f.vhea != (table{})
	hasVmtx := f.vmtx != (table{})
	if !hasVhea && !hasVmtx {
		return buf, verticalData{}, nil
	}
	// OpenType requires both tables for vertical layout. Report whichever is
	// missing.
	if !hasVhea {
		return buf, verticalData{}, errInvalidVheaTable
	}
	if !hasVmtx {
		return buf, verticalData{}, errInvalidVmtxTable
	}

	// https://www.microsoft.com/typography/otspec/vhea.htm says that the vhea
	// table is 36 bytes. Versions 1.0 and 1.1 have the same layout.
	if f.vhea.length != 36 {
		return buf, verticalData{}, errInvalidVheaTable
	}
	// Map any read error (for example a truncated table whose directory entry
	// claims more bytes than the file contains) to a stable error, so that
	// callers can distinguish vertical metrics failures from horizontal ones.
	fail := func(err error) ([]byte, verticalData, error) {
		return buf, verticalData{}, err
	}
	readU16 := func(i int) (uint16, bool) {
		x, err := f.src.u16(buf, f.vhea, i)
		if err != nil {
			return 0, false
		}
		return x, true
	}

	n, ok := readU16(34)
	if !ok {
		return fail(errInvalidVheaTable)
	}
	// numOfLongVerMetrics must be in [1, numGlyphs]. Zero would mean that no
	// glyph has an advance height and a value greater than numGlyphs is
	// inconsistent with the vmtx table.
	if n == 0 || int32(n) > numGlyphs {
		return fail(errInvalidVheaTable)
	}
	a, okA := readU16(4)
	d, okD := readU16(6)
	l, okL := readU16(8)
	ru, okRun := readU16(20)
	ri, okRise := readU16(18)
	if !okA || !okD || !okL || !okRun || !okRise {
		return fail(errInvalidVheaTable)
	}

	// https://www.microsoft.com/typography/otspec/vmtx.htm says that the table
	// is 4*numOfLongVerMetrics bytes plus a trailing array of 2-byte top side
	// bearings for the remaining glyphs. That trailing array is optional, so
	// (as for hmtx) accept either the full length or just the long metrics.
	//
	// The arithmetic is done in uint32: numGlyphs is at most 65535, so neither
	// term can overflow.
	fullLength := 4 * uint32(n)
	trailingLength := 2 * (uint32(numGlyphs) - uint32(n))
	if f.vmtx.length != fullLength && f.vmtx.length != fullLength+trailingLength {
		return fail(errInvalidVmtxTable)
	}

	return buf, verticalData{
		valid:          true,
		ascent:         int32(int16(a)),
		descent:        int32(int16(d)),
		lineGap:        int32(int16(l)),
		caretSlopeRise: int32(int16(ri)),
		caretSlopeRun:  int32(int16(ru)),
		numVMetrics:    int32(n),
		vmtxOffset:     f.vmtx.offset,
		vmtxLength:     f.vmtx.length,
	}, nil
}

// scaleVertical returns x (in font units) scaled to ppem, rounded to the
// nearest fixed.Int26_6 value. It returns an error instead of overflowing if x
// and ppem are too large to represent.
func scaleVertical(x int32, ppem fixed.Int26_6, unitsPerEm Units) (fixed.Int26_6, error) {
	if unitsPerEm <= 0 {
		return 0, errInvalidUnitsPerEm
	}
	p, u := int64(x)*int64(ppem), int64(unitsPerEm)
	var r int64
	if p >= 0 {
		r = (p + u/2) / u
	} else {
		r = (p - u/2) / u
	}
	if r < math.MinInt32 || r > math.MaxInt32 {
		return 0, errVerticalMetricsOverflow
	}
	return fixed.Int26_6(r), nil
}

// glyphVerticalMetrics returns the x'th glyph's advance height and top side
// bearing, in font units.
//
// The caller must have checked that x < NumGlyphs and that
// f.cached.verticalErr is nil. If the font has no vertical metrics, it returns
// ok == false.
func (f *Font) glyphVerticalMetrics(b *Buffer, x GlyphIndex) (advance uint16, topSideBearing int16, ok bool, err error) {
	v := &f.cached.vertical
	if !v.valid {
		return 0, 0, false, nil
	}

	// https://www.microsoft.com/typography/otspec/vmtx.htm says that "As an
	// optimization, the number of records can be less than the number of
	// glyphs, in which case the advance height value of the last record
	// applies to all remaining glyph IDs." The trailing top side bearing
	// array supplies the bearings for those glyphs.
	xi := int32(x)
	metricIndex := xi
	if metricIndex >= v.numVMetrics {
		metricIndex = v.numVMetrics - 1
	}
	buf, err := b.view(&f.src, int(v.vmtxOffset)+4*int(metricIndex), 4)
	if err != nil {
		return 0, 0, false, errInvalidVmtxTable
	}
	advance = u16(buf)
	topSideBearing = int16(u16(buf[2:]))

	if xi >= v.numVMetrics {
		// The trailing array is optional. If it is absent, use a zero top
		// side bearing for the remaining glyphs.
		if offset := 4*int(v.numVMetrics) + 2*int(xi-v.numVMetrics); offset+2 <= int(v.vmtxLength) {
			buf, err = b.view(&f.src, int(v.vmtxOffset)+offset, 2)
			if err != nil {
				return 0, 0, false, errInvalidVmtxTable
			}
			topSideBearing = int16(u16(buf))
		} else {
			topSideBearing = 0
		}
	}
	return advance, topSideBearing, true, nil
}

// verticalUnits returns the font-wide vertical ascent, descent and line gap in
// font units. The descent is non-positive, matching vhea version 1.1's
// vertTypoDescender field.
//
// If the font has no vertical metrics, the ideographic em-box is used: an
// ascent and descent of half the em each, and no line gap.
func (f *Font) verticalUnits() (ascent, descent, lineGap int32) {
	if v := &f.cached.vertical; v.valid {
		ascent, descent, lineGap = v.ascent, v.descent, v.lineGap
		// vhea version 1.0 stores "descent" as a positive distance from the
		// centreline, while version 1.1 (vertTypoDescender) stores it as a
		// non-positive offset. Normalize to the latter so that the sign is
		// unambiguous for callers.
		if descent > 0 {
			descent = -descent
		}
		return ascent, descent, lineGap
	}
	upem := int32(f.cached.unitsPerEm)
	ascent = upem / 2
	return ascent, -(upem - ascent), 0
}

// HasVerticalMetrics reports whether the font has a usable 'vhea'/'vmtx' table
// pair.
//
// If it returns true, GlyphVerticalAdvance, GlyphVerticalOrigin,
// GlyphVerticalBounds and VerticalMetrics use those tables. If it returns
// false, those methods use a one-em fallback instead. A font whose vertical
// metrics tables are malformed also returns false; the vertical metrics methods
// report the malformed table as an error.
func (f *Font) HasVerticalMetrics() bool {
	return f.cached.verticalErr == nil && f.cached.vertical.valid
}

// VerticalMetrics returns the metrics to use for vertical (top-to-bottom) text
// layout.
//
// The values come from the 'vhea' table when the font has a usable
// 'vhea'/'vmtx' pair. If the font has neither table, the ideographic em-box is
// used instead: the ascent and descent are each half of the em and the line
// gap is zero, so Height equals one em. This fallback matches the CJK
// guidelines' notion of one em per ideographic glyph.
//
// If the font has a malformed vertical metrics table pair, VerticalMetrics
// returns a non-nil error. The horizontal APIs are unaffected by such a font:
// a malformed vertical table pair is cached at Parse time and reported only by
// the vertical metrics methods.
//
// ppem is the number of pixels in 1 em. To get the metrics in font units, use
// ppem = fixed.Int26_6(f.UnitsPerEm()).
//
// XHeight and CapHeight are copied from the font's horizontal metrics, since
// vhea defines no vertical equivalents.
func (f *Font) VerticalMetrics(b *Buffer, ppem fixed.Int26_6, h font.Hinting) (font.Metrics, error) {
	if err := f.cached.verticalErr; err != nil {
		return font.Metrics{}, err
	}
	ascent, descent, lineGap := f.verticalUnits()

	height, err := scaleVertical(ascent-descent+lineGap, ppem, f.cached.unitsPerEm)
	if err != nil {
		return font.Metrics{}, err
	}
	a, err := scaleVertical(ascent, ppem, f.cached.unitsPerEm)
	if err != nil {
		return font.Metrics{}, err
	}
	d, err := scaleVertical(-descent, ppem, f.cached.unitsPerEm)
	if err != nil {
		return font.Metrics{}, err
	}
	xh, err := scaleVertical(f.cached.xHeight, ppem, f.cached.unitsPerEm)
	if err != nil {
		return font.Metrics{}, err
	}
	ch, err := scaleVertical(f.cached.capHeight, ppem, f.cached.unitsPerEm)
	if err != nil {
		return font.Metrics{}, err
	}

	m := font.Metrics{
		Height:     height,
		Ascent:     a,
		Descent:    d,
		XHeight:    xh,
		CapHeight:  ch,
		CaretSlope: f.verticalCaretSlope(),
	}
	if h == font.HintingFull {
		// Quantize up to a whole pixel.
		m.Height = (m.Height + 63) &^ 63
		m.Ascent = (m.Ascent + 63) &^ 63
		m.Descent = (m.Descent + 63) &^ 63
		m.XHeight = (m.XHeight + 63) &^ 63
		m.CapHeight = (m.CapHeight + 63) &^ 63
	}
	return m, nil
}

// verticalCaretSlope returns the caret slope vector for vertical layout, in
// the same convention as font.Metrics.CaretSlope: X is the run and Y is the
// rise, with the Y axis pointing up.
func (f *Font) verticalCaretSlope() image.Point {
	if v := &f.cached.vertical; v.valid {
		return image.Point{X: int(v.caretSlopeRun), Y: int(v.caretSlopeRise)}
	}
	// A vertical font has a horizontal caret; the same is a sensible fallback
	// for a font with no vertical metrics at all.
	return image.Point{X: 1, Y: 0}
}

// GlyphVerticalAdvance returns the advance height for the x'th glyph. ppem is
// the number of pixels in 1 em.
//
// The advance comes from the 'vmtx' table. If the font has no vertical metrics
// tables, the advance is one em.
//
// It returns ErrNotFound if the glyph index is out of range, or the cached
// error if the font's vertical metrics tables are malformed.
func (f *Font) GlyphVerticalAdvance(b *Buffer, x GlyphIndex, ppem fixed.Int26_6, h font.Hinting) (fixed.Int26_6, error) {
	if int(x) >= f.NumGlyphs() {
		return 0, ErrNotFound
	}
	if err := f.cached.verticalErr; err != nil {
		return 0, err
	}
	if b == nil {
		b = &Buffer{}
	}
	advance, _, ok, err := f.glyphVerticalMetrics(b, x)
	if err != nil {
		return 0, err
	}
	if !ok {
		// Fallback: one em per glyph.
		advance = uint16(f.cached.unitsPerEm)
	}
	adv, err := scaleVertical(int32(advance), ppem, f.cached.unitsPerEm)
	if err != nil {
		return 0, err
	}
	if h == font.HintingFull {
		// Quantize the fixed.Int26_6 value to the nearest pixel.
		adv = (adv + 32) &^ 63
	}
	return adv, nil
}

// glyphVerticalOrigin returns the position of the x'th glyph's vertical
// origin, relative to its horizontal origin, and the glyph's segments and
// advance height.
//
// When the font has vertical metrics, the vertical origin's Y coordinate is
// the glyph's top (the maximum Y of its bounding box) plus its top side
// bearing. In the returned (Y down) coordinate space it is therefore negative,
// above the horizontal origin. The X coordinate is half the horizontal
// advance: vmtx does not specify an X coordinate, and centering the glyph in
// its horizontal advance is the conventional choice.
//
// When the font has no vertical metrics, the glyph is centered in a one-em
// advance: the vertical advance is one em and the free space above and below
// the glyph is split evenly.
func (f *Font) glyphVerticalOrigin(b *Buffer, x GlyphIndex, ppem fixed.Int26_6, h font.Hinting) (origin fixed.Point26_6, segments Segments, advance fixed.Int26_6, err error) {
	if int(x) >= f.NumGlyphs() {
		return fixed.Point26_6{}, nil, 0, ErrNotFound
	}
	if err := f.cached.verticalErr; err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	if f.cached.unitsPerEm <= 0 {
		return fixed.Point26_6{}, nil, 0, errInvalidUnitsPerEm
	}
	if b == nil {
		b = &Buffer{}
	}

	// Read everything from b before LoadGlyph: the returned segments are only
	// valid until b is re-used.
	hAdvance, err := f.GlyphAdvance(b, x, ppem, h)
	if err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	advanceUnits, tsb, ok, err := f.glyphVerticalMetrics(b, x)
	if err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	hasTsb := ok
	if !ok {
		// Fallback: one em of vertical advance, with the glyph centered.
		advanceUnits = uint16(f.cached.unitsPerEm)
	}
	advance, err = scaleVertical(int32(advanceUnits), ppem, f.cached.unitsPerEm)
	if err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	if h == font.HintingFull {
		advance = (advance + 32) &^ 63
	}

	segments, err = f.LoadGlyph(b, x, ppem, nil)
	if err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	bounds := segments.Bounds()

	if !hasTsb {
		// With no top side bearing to place the glyph, center it in the
		// one-em advance: leave equal free space above and below. The vertical
		// origin is the top of that advance, so in these (Y down) coordinates
		// the box runs from Y = 0 to Y = advance.
		free := advance - (bounds.Max.Y - bounds.Min.Y)
		return fixed.Point26_6{
			X: hAdvance / 2,
			Y: bounds.Min.Y - free/2,
		}, segments, advance, nil
	}

	tsbScaled, err := scaleVertical(int32(tsb), ppem, f.cached.unitsPerEm)
	if err != nil {
		return fixed.Point26_6{}, nil, 0, err
	}
	// The vertical origin is tsbScaled above the glyph's topmost point.
	return fixed.Point26_6{
		X: hAdvance / 2,
		Y: bounds.Min.Y - tsbScaled,
	}, segments, advance, nil
}

// GlyphVerticalOrigin returns the position of the x'th glyph's vertical origin
// relative to its horizontal origin, with Y increasing down. ppem is the
// number of pixels in 1 em.
//
// The Y coordinate is -(top side bearing + top of the glyph's bounding box).
// The X coordinate is half the glyph's horizontal advance, as vmtx does not
// specify an X coordinate.
//
// It returns ErrNotFound if the glyph index is out of range, or the cached
// error if the font's vertical metrics tables are malformed.
func (f *Font) GlyphVerticalOrigin(b *Buffer, x GlyphIndex, ppem fixed.Int26_6, h font.Hinting) (fixed.Point26_6, error) {
	if b == nil {
		b = &Buffer{}
	}
	origin, _, _, err := f.glyphVerticalOrigin(b, x, ppem, h)
	return origin, err
}

// GlyphVerticalBounds returns the bounding box of the x'th glyph for vertical
// layout, relative to that glyph's vertical origin, and its advance height.
// ppem is the number of pixels in 1 em.
//
// In the returned Rectangle26_6, the vertical origin is at (0, 0) and the Y
// axis increases down. The top of the glyph is therefore at Y =
// top side bearing, which is negative if the glyph extends above the vertical
// origin, and the bottom is at Y = top side bearing + the glyph's height. The X
// coordinates are relative to the vertical baseline: they are the horizontal
// bounds shifted by half the horizontal advance.
//
// The glyph's bounds are calculated from its segments, as for GlyphBounds,
// rather than from the bounds recorded in a 'glyf' table, which are not
// available for compound or CFF glyphs.
//
// It returns ErrNotFound if the glyph index is out of range, or the cached
// error if the font's vertical metrics tables are malformed.
func (f *Font) GlyphVerticalBounds(b *Buffer, x GlyphIndex, ppem fixed.Int26_6, h font.Hinting) (bounds fixed.Rectangle26_6, advance fixed.Int26_6, err error) {
	if b == nil {
		b = &Buffer{}
	}
	origin, segments, advance, err := f.glyphVerticalOrigin(b, x, ppem, h)
	if err != nil {
		return fixed.Rectangle26_6{}, 0, err
	}
	return segments.Bounds().Sub(origin), advance, nil
}
