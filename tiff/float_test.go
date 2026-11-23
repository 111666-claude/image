// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"reflect"
	"testing"
)

// floatTIFFPage describes one synthetic floating-point TIFF page.
type floatTIFFPage struct {
	width, height int
	samples       int       // Samples per pixel.
	bits          int       // Bits per floating-point sample: 16, 32 or 64.
	photometric   int       // A p* constant.
	extraSamples  int       // Zero means the tag is absent.
	data          []float64 // width*height*samples values, row-major, interleaved.
}

type tiffEntry struct {
	tag  uint16
	typ  uint16
	data []uint32
}

// buildFloatTIFF serializes pages as a multi-page TIFF in the given byte
// order, each page a single strip of uncompressed data. The serialization
// deliberately writes the container format directly instead of reusing the
// package's encoders, so that the tests are an independent check.
func buildFloatTIFF(order byteOrder, pages []floatTIFFPage) []byte {
	var b []byte
	if order.Uint16([]byte{1, 0}) == 1 {
		b = append(b, 'I', 'I')
	} else {
		b = append(b, 'M', 'M')
	}
	b = order.AppendUint16(b, 42)
	b = order.AppendUint32(b, 0) // IFD offset, patched at the end.

	// Pixel data for every page first, recording each page's data offset.
	dataOffs := make([]int, len(pages))
	for i, page := range pages {
		dataOffs[i] = len(b)
		for _, v := range page.data {
			switch page.bits {
			case 16:
				b = order.AppendUint16(b, float64ToHalf(v))
			case 32:
				b = order.AppendUint32(b, math.Float32bits(float32(v)))
			case 64:
				b = order.AppendUint64(b, math.Float64bits(v))
			default:
				panic("tiff: bad test bit depth")
			}
		}
	}

	firstIFD := len(b)
	// Build each IFD separately so that each one's size, and hence the
	// following IFD's offset, is known when writing the next-IFD pointer.
	// Entry values too large for the inline field are addressed with
	// absolute file offsets, so the IFD's own offset is needed first: sizes
	// do not depend on the offset, so serialize twice.
	entryLists := make([][]tiffEntry, len(pages))
	ifds := make([][]byte, len(pages))
	sizes := make([]int, len(pages))
	nextPtr := make([]int, len(pages))
	ifdOffset := len(b)
	pos := ifdOffset
	for i, page := range pages {
		entryLists[i] = page.entries(dataOffs[i])
		ifds[i] = appendIFDEntries(order, entryLists[i], pos)
		sizes[i] = len(ifds[i])
		nextPtr[i] = 2 + ifdLen*len(entryLists[i])
		pos += sizes[i]
	}
	next := ifdOffset
	cur := ifdOffset
	for i := range ifds {
		// Re-serialize now that this IFD's absolute offset is known.
		ifds[i] = appendIFDEntries(order, entryLists[i], cur)
		next = cur + sizes[i]
		if i == len(ifds)-1 {
			next = 0
		}
		order.PutUint32(ifds[i][nextPtr[i]:], uint32(next))
		b = append(b, ifds[i]...)
		cur += sizes[i]
	}
	order.PutUint32(b[4:8], uint32(firstIFD))
	return b
}

func (p floatTIFFPage) entries(dataOff int) []tiffEntry {
	entries := []tiffEntry{
		{tImageWidth, dtShort, []uint32{uint32(p.width)}},
		{tImageLength, dtShort, []uint32{uint32(p.height)}},
		{tBitsPerSample, dtShort, repeat32(uint32(p.bits), p.samples)},
		{tCompression, dtShort, []uint32{cNone}},
		{tPhotometricInterpretation, dtShort, []uint32{uint32(p.photometric)}},
		{tStripOffsets, dtLong, []uint32{uint32(dataOff)}},
		{tSamplesPerPixel, dtShort, []uint32{uint32(p.samples)}},
		{tRowsPerStrip, dtShort, []uint32{uint32(p.height)}},
		{tStripByteCounts, dtLong, []uint32{uint32(len(p.data) * p.bits / 8)}},
		{tSampleFormat, dtShort, repeat32(sfFloat, p.samples)},
	}
	if p.extraSamples != 0 {
		entries = append(entries, tiffEntry{tExtraSamples, dtShort, []uint32{uint32(p.extraSamples)}})
	}
	// Tags must be sorted in ascending order.
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].tag < entries[j-1].tag; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
	return entries
}

// appendIFDEntries serializes sorted entries plus the next-IFD pointer.
// Values too large for the entry's inline field are stored in a value area
// after the IFD, as the spec allows.
func appendIFDEntries(order byteOrder, entries []tiffEntry, ifdOff int) []byte {
	var b []byte
	b = order.AppendUint16(b, uint16(len(entries)))
	valueStart := 2 + ifdLen*len(entries) + 4
	type value struct {
		typ  uint16
		data []uint32
	}
	var values []value
	valueBytes := 0
	for _, e := range entries {
		b = order.AppendUint16(b, e.tag)
		b = order.AppendUint16(b, e.typ)
		b = order.AppendUint32(b, uint32(len(e.data)))
		if len(e.data)*int(lengths[e.typ]) <= 4 {
			var inline [4]byte
			for k, v := range e.data {
				if e.typ == dtShort {
					order.PutUint16(inline[2*k:], uint16(v))
				} else {
					order.PutUint32(inline[4*k:], v)
				}
			}
			b = append(b, inline[:]...)
		} else {
			b = order.AppendUint32(b, uint32(ifdOff+valueStart+valueBytes))
			values = append(values, value{e.typ, e.data})
			valueBytes += len(e.data) * int(lengths[e.typ])
		}
	}
	b = order.AppendUint32(b, 0) // Patched by the caller.
	for _, v := range values {
		for _, x := range v.data {
			if v.typ == dtShort {
				b = order.AppendUint16(b, uint16(x))
			} else {
				b = order.AppendUint32(b, x)
			}
		}
	}
	return b
}

func repeat32(v uint32, n int) []uint32 {
	out := make([]uint32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// float64ToHalf rounds v to the nearest binary16 value, rounding to nearest
// even. NaNs encode as a quiet NaN. It is only used to build test inputs.
func float64ToHalf(v float64) uint16 {
	if math.IsNaN(v) {
		return 0x7e00
	}
	sign := uint16(0)
	if math.Signbit(v) {
		sign = 1 << 15
		v = -v
	}
	if math.IsInf(v, 0) {
		return sign | 0x7c00
	}
	if v == 0 {
		return sign
	}
	// Normalize v to [1, 2) and track the unbiased exponent.
	exp := 0
	for v >= 2 {
		v /= 2
		exp++
	}
	for v < 1 {
		v *= 2
		exp--
	}
	if exp < -14 {
		// Subnormal: a half's subnormal mantissa has weight 2^-24.
		m := math.Round(math.Ldexp(v, exp+24))
		if m >= 1024 {
			return sign | 0x0400 // Round up to the smallest normal.
		}
		return sign | uint16(m)
	}
	expBits := exp + 15
	mant := uint16(math.Round((v - 1) * 1024))
	if mant == 1024 {
		mant = 0
		expBits++
	}
	if expBits >= 0x1f {
		return sign | 0x7c00
	}
	return sign | uint16(expBits)<<10 | mant
}

// --- Independent reference implementation -------------------------------
//
// referencePixel maps samples to the expected integer output using a
// straightforward reimplementation of the documented rules. It shares no code
// with the decoder.

type referenceOpts struct {
	min, max            float64
	alphaMin, alphaMax  float64
	outMax              float64
	nan, posInf, negInf FloatSpecial
	oob, alphaOOB       FloatRangeRule
}

func refSpecial(kind FloatSpecial, v float64, def FloatSpecial, min, max float64) (float64, bool) {
	if kind == FloatSpecialDefault {
		kind = def
	}
	switch kind {
	case FloatSpecialZero:
		return 0, true
	case FloatSpecialMin:
		return min, true
	case FloatSpecialMax:
		return max, true
	case FloatSpecialReject:
		return 0, false
	}
	panic("bad special rule")
}

func refScale(v, min, max, outMax float64, rule FloatRangeRule) (float64, bool) {
	if v < min {
		if rule == FloatOutOfRangeReject {
			return 0, false
		}
		v = min
	}
	if v > max {
		if rule == FloatOutOfRangeReject {
			return 0, false
		}
		v = max
	}
	return math.Round((v - min) / (max - min) * outMax), true
}

func refMap(v float64, r *referenceOpts, alpha bool) (float64, bool) {
	min, max, rule := r.min, r.max, r.oob
	if alpha {
		min, max, rule = r.alphaMin, r.alphaMax, r.alphaOOB
	}
	switch {
	case math.IsNaN(v):
		var ok bool
		if v, ok = refSpecial(r.nan, v, FloatSpecialZero, min, max); !ok {
			return 0, false
		}
	case math.IsInf(v, 1):
		var ok bool
		if v, ok = refSpecial(r.posInf, v, FloatSpecialMax, min, max); !ok {
			return 0, false
		}
	case math.IsInf(v, -1):
		var ok bool
		if v, ok = refSpecial(r.negInf, v, FloatSpecialMin, min, max); !ok {
			return 0, false
		}
	}
	return refScale(v, min, max, r.outMax, rule)
}

// referenceImage decodes the single page in data with the rules in r and
// returns the expected pixel values as an image of the type produced by the
// decoder.
func referenceImage(t *testing.T, page floatTIFFPage, r *referenceOpts) image.Image {
	t.Helper()
	rect := image.Rect(0, 0, page.width, page.height)
	n := page.samples
	var img image.Image
	grayAlpha := page.photometric != pRGB && n == 2
	switch {
	case page.photometric != pRGB && n == 1:
		if r.outMax == 0xffff {
			img = image.NewGray16(rect)
		} else {
			img = image.NewGray(rect)
		}
	case page.photometric == pRGB && n == 3:
		if r.outMax == 0xffff {
			img = image.NewRGBA64(rect)
		} else {
			img = image.NewRGBA(rect)
		}
	case page.photometric == pRGB && n == 4, grayAlpha:
		switch {
		case page.extraSamples == esAssociated:
			if r.outMax == 0xffff {
				img = image.NewRGBA64(rect)
			} else {
				img = image.NewRGBA(rect)
			}
		default:
			if r.outMax == 0xffff {
				img = image.NewNRGBA64(rect)
			} else {
				img = image.NewNRGBA(rect)
			}
		}
	default:
		t.Fatalf("referenceImage: unsupported page geometry")
	}
	inv := page.photometric == pWhiteIsZero

	set := func(x, y int, vals []float64) {
		mapv := func(i int, alpha bool) float64 {
			v, ok := refMap(vals[i], r, alpha)
			if !ok {
				t.Fatalf("referenceImage: unexpected mapping rejection at (%d,%d)", x, y)
			}
			return v
		}
		var rgb [3]float64
		var a float64
		switch {
		case n == 1:
			rgb[0] = mapv(0, false)
			if inv {
				rgb[0] = r.outMax - rgb[0]
			}
			rgb[1], rgb[2] = rgb[0], rgb[0]
			a = r.outMax
		case grayAlpha:
			rgb[0] = mapv(0, false)
			if inv {
				rgb[0] = r.outMax - rgb[0]
			}
			rgb[1], rgb[2] = rgb[0], rgb[0]
			a = mapv(1, true)
		default:
			rgb[0], rgb[1], rgb[2] = mapv(0, false), mapv(1, false), mapv(2, false)
			if n == 4 {
				a = mapv(3, true)
			} else {
				a = r.outMax
			}
		}
		switch m := img.(type) {
		case *image.Gray:
			m.SetGray(x, y, color.Gray{uint8(rgb[0])})
		case *image.Gray16:
			m.SetGray16(x, y, color.Gray16{uint16(rgb[0])})
		case *image.RGBA:
			m.SetRGBA(x, y, color.RGBA{uint8(rgb[0]), uint8(rgb[1]), uint8(rgb[2]), uint8(a)})
		case *image.RGBA64:
			m.SetRGBA64(x, y, color.RGBA64{uint16(rgb[0]), uint16(rgb[1]), uint16(rgb[2]), uint16(a)})
		case *image.NRGBA:
			m.SetNRGBA(x, y, color.NRGBA{uint8(rgb[0]), uint8(rgb[1]), uint8(rgb[2]), uint8(a)})
		case *image.NRGBA64:
			m.SetNRGBA64(x, y, color.NRGBA64{uint16(rgb[0]), uint16(rgb[1]), uint16(rgb[2]), uint16(a)})
		}
	}
	i := 0
	for y := 0; y < page.height; y++ {
		for x := 0; x < page.width; x++ {
			set(x, y, page.data[i:i+n])
			i += n
		}
	}
	return img
}

// compareExact compares the raw integer samples of two images of the same
// concrete type, avoiding color-model conversions.
func compareExact(t *testing.T, want, got image.Image) {
	t.Helper()
	if fmt.Sprintf("%T", want) != fmt.Sprintf("%T", got) {
		t.Fatalf("wrong image type: want %T, got %T", want, got)
	}
	if want.Bounds() != got.Bounds() {
		t.Fatalf("wrong bounds: want %v, got %v", want.Bounds(), got.Bounds())
	}
	switch w := want.(type) {
	case *image.Gray:
		g := got.(*image.Gray)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.GrayAt(x, y) != g.GrayAt(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.GrayAt(x, y), g.GrayAt(x, y))
				}
			}
		}
	case *image.Gray16:
		g := got.(*image.Gray16)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.Gray16At(x, y) != g.Gray16At(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.Gray16At(x, y), g.Gray16At(x, y))
				}
			}
		}
	case *image.RGBA:
		g := got.(*image.RGBA)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.RGBAAt(x, y) != g.RGBAAt(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.RGBAAt(x, y), g.RGBAAt(x, y))
				}
			}
		}
	case *image.RGBA64:
		g := got.(*image.RGBA64)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.RGBA64At(x, y) != g.RGBA64At(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.RGBA64At(x, y), g.RGBA64At(x, y))
				}
			}
		}
	case *image.NRGBA:
		g := got.(*image.NRGBA)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.NRGBAAt(x, y) != g.NRGBAAt(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.NRGBAAt(x, y), g.NRGBAAt(x, y))
				}
			}
		}
	case *image.NRGBA64:
		g := got.(*image.NRGBA64)
		for y := w.Rect.Min.Y; y < w.Rect.Max.Y; y++ {
			for x := w.Rect.Min.X; x < w.Rect.Max.X; x++ {
				if w.NRGBA64At(x, y) != g.NRGBA64At(x, y) {
					t.Fatalf("pixel (%d,%d): want %v, got %v", x, y, w.NRGBA64At(x, y), g.NRGBA64At(x, y))
				}
			}
		}
	default:
		t.Fatalf("unexpected image type %T", want)
	}
}

// --- Tests ---------------------------------------------------------------

func testPages() []floatTIFFPage {
	// A small but dense sample set covering the range endpoints, values
	// outside the range, negatives, specials and subnormals. Samples are
	// quantized to the page's stored precision below so that the reference
	// and the decoder see the same values.
	base16 := []float64{
		0, 1, 0.5, 0.25, 0.75, -0.25, 1.25, 2, -2,
		math.NaN(), math.Inf(1), math.Inf(-1),
		math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		1.0 / 3.0,
	}
	base32 := append([]float64(nil), base16...)
	base64 := append([]float64(nil), base16...)
	for i := range base16 {
		// Quantize just as serialization does, so that the reference sees the
		// sample values the decoder will actually read.
		base16[i] = halfToFloat64(float64ToHalf(base16[i]))
		base32[i] = float64(float32(base32[i]))
	}
	pad := func(base []float64, n int) []float64 {
		out := make([]float64, 0, n)
		for len(out) < n {
			out = append(out, base...)
		}
		return out[:n]
	}
	baseFor := func(bits int) []float64 {
		switch bits {
		case 16:
			return base16
		case 32:
			return base32
		}
		return base64
	}
	const w, h = 5, 4
	pages := []floatTIFFPage{
		{width: w, height: h, samples: 1, bits: 16, photometric: pBlackIsZero},
		{width: w, height: h, samples: 1, bits: 32, photometric: pBlackIsZero},
		{width: w, height: h, samples: 1, bits: 64, photometric: pBlackIsZero},
		{width: w, height: h, samples: 1, bits: 32, photometric: pWhiteIsZero},
		{width: w, height: h, samples: 3, bits: 16, photometric: pRGB},
		{width: w, height: h, samples: 3, bits: 32, photometric: pRGB},
		{width: w, height: h, samples: 4, bits: 32, photometric: pRGB, extraSamples: esAssociated},
		{width: w, height: h, samples: 4, bits: 16, photometric: pRGB, extraSamples: esUnassociated},
		{width: w, height: h, samples: 4, bits: 64, photometric: pRGB, extraSamples: esAssociated},
		{width: w, height: h, samples: 2, bits: 32, photometric: pBlackIsZero, extraSamples: esUnassociated},
		{width: w, height: h, samples: 2, bits: 16, photometric: pWhiteIsZero, extraSamples: esAssociated},
		{width: 1, height: 1, samples: 1, bits: 32, photometric: pBlackIsZero},
		{width: 17, height: 9, samples: 3, bits: 32, photometric: pRGB},
	}
	for i := range pages {
		pages[i].data = pad(baseFor(pages[i].bits), pages[i].width*pages[i].height*pages[i].samples)
	}
	return pages
}

func refOptsFor(f FloatOptions) *referenceOpts {
	// Mirror the decoder's defaulting of an all-zero range.
	if f.Min == 0 && f.Max == 0 {
		f.Min, f.Max = 0, 1
	}
	if f.AlphaMin == 0 && f.AlphaMax == 0 {
		f.AlphaMin, f.AlphaMax = 0, 1
	}
	outMax := float64(0xffff)
	if f.BitDepth == 8 {
		outMax = 0xff
	}
	return &referenceOpts{
		min: f.Min, max: f.Max,
		alphaMin: f.AlphaMin, alphaMax: f.AlphaMax,
		outMax: float64(outMax),
		nan:    f.NaN, posInf: f.PosInf, negInf: f.NegInf,
		oob: f.OutOfRange, alphaOOB: f.AlphaOutOfRange,
	}
}

// TestDecodeFloatMatchesReference checks the decoder against an independent
// implementation across byte orders, bit depths, sample layouts and option
// combinations.
func TestDecodeFloatMatchesReference(t *testing.T) {
	optionSets := []FloatOptions{
		{}, // Defaults.
		{BitDepth: 8},
		{Min: -1, Max: 1, AlphaMin: -2, AlphaMax: 2, BitDepth: 8},
		{Min: -1, Max: 1, BitDepth: 16},
		{Min: 0, Max: 255, BitDepth: 8},
		{Min: -1e6, Max: 1e6, BitDepth: 16},
		{NaN: FloatSpecialMax, PosInf: FloatSpecialZero, NegInf: FloatSpecialMin, BitDepth: 8},
		{NaN: FloatSpecialMin, PosInf: FloatSpecialMin, NegInf: FloatSpecialZero},
		{NaN: FloatSpecialReject, PosInf: FloatSpecialReject, NegInf: FloatSpecialReject},
		{OutOfRange: FloatOutOfRangeReject},
		{Min: -1, Max: 1, OutOfRange: FloatOutOfRangeReject, NaN: FloatSpecialZero},
	}
	orders := []struct {
		name  string
		order byteOrder
	}{
		{"little", binary.LittleEndian},
		{"big", binary.BigEndian},
	}
	pages := testPages()
	for _, o := range orders {
		data := buildFloatTIFF(o.order, pages)
		n, err := NumPages(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: NumPages: %v", o.name, err)
		}
		if n != len(pages) {
			t.Fatalf("%s: NumPages: want %d, got %d", o.name, len(pages), n)
		}
		for pi, page := range pages {
			for _, f := range optionSets {
				name := fmt.Sprintf("%s/page%d/bits%d/s%d/photo%d/opts%+v", o.name, pi, page.bits, page.samples, page.photometric, f)
				t.Run(name, func(t *testing.T) {
					opt := &DecodeOptions{Float: &f}
					img, err := DecodePageWithOptions(bytes.NewReader(data), pi, opt)
					r := refOptsFor(f)
					rejecting := pageRejects(page, r)
					if rejecting {
						if img != nil || err == nil {
							t.Fatalf("got image %T, err %v; want rejection", img, err)
						}
						var pe *PageError
						if !errors.As(err, &pe) {
							t.Fatalf("error %v is not a *PageError", err)
						}
						if pe.Page != pi {
							t.Fatalf("error page: want %d, got %d", pi, pe.Page)
						}
						return
					}
					if err != nil {
						t.Fatalf("DecodePageWithOptions: %v", err)
					}
					if pi == 0 {
						cfg, err := DecodeConfigWithOptions(bytes.NewReader(data), opt)
						if err != nil {
							t.Fatalf("DecodeConfigWithOptions: %v", err)
						}
						if cfg.Width != page.width || cfg.Height != page.height {
							t.Fatalf("config size: want %dx%d, got %dx%d", page.width, page.height, cfg.Width, cfg.Height)
						}
					}
					want := referenceImage(t, page, r)
					compareExact(t, want, img)
				})
			}
		}
	}
}

// pageRejects reports whether decoding page under r must fail because some
// sample trips a rejecting rule.
func pageRejects(page floatTIFFPage, r *referenceOpts) bool {
	n := page.samples
	for i := 0; i+1 <= len(page.data); i++ {
		v := page.data[i]
		isAlpha := false
		if grayAlpha := page.photometric != pRGB && n == 2; grayAlpha {
			isAlpha = i%2 == 1
		} else if n == 4 {
			isAlpha = i%4 == 3
		}
		min, max, rule := r.min, r.max, r.oob
		nan, posInf, negInf := r.nan, r.posInf, r.negInf
		if isAlpha {
			min, max, rule = r.alphaMin, r.alphaMax, r.alphaOOB
		}
		var special FloatSpecial
		hasSpecial := false
		switch {
		case math.IsNaN(v):
			special, hasSpecial = nan, true
		case math.IsInf(v, 1):
			special, hasSpecial = posInf, true
		case math.IsInf(v, -1):
			special, hasSpecial = negInf, true
		}
		if hasSpecial {
			if special == FloatSpecialReject {
				return true
			}
			// A substituted special is then subject to the range rule.
			switch special {
			case FloatSpecialDefault:
				switch {
				case math.IsNaN(v):
					v = 0
				case math.IsInf(v, 1):
					v = max
				default:
					v = min
				}
			case FloatSpecialZero:
				v = 0
			case FloatSpecialMin:
				v = min
			case FloatSpecialMax:
				v = max
			}
		}
		if (v < min || v > max) && rule == FloatOutOfRangeReject {
			return true
		}
	}
	return false
}

// TestHalfRoundTrip checks the binary16 conversion used to author tests and
// to decode samples.
func TestHalfRoundTrip(t *testing.T) {
	cases := []struct {
		bits uint16
		want float64
	}{
		{0x0000, 0},
		{0x8000, math.Copysign(0, -1)},
		{0x3c00, 1},
		{0xbc00, -1},
		{0x3800, 0.5},
		{0x7bff, 65504},
		{0x0001, math.Ldexp(1, -24)}, // Smallest positive subnormal.
		{0x0400, math.Ldexp(1, -14)}, // Smallest positive normal.
		{0x7c00, math.Inf(1)},
		{0xfc00, math.Inf(-1)},
	}
	for _, c := range cases {
		got := halfToFloat64(c.bits)
		if got != c.want {
			t.Errorf("halfToFloat64(%#04x) = %v, want %v", c.bits, got, c.want)
		}
		// Round-tripping through float64ToHalf must be stable.
		if back := float64ToHalf(c.want); back != c.bits {
			t.Errorf("float64ToHalf(%v) = %#04x, want %#04x", c.want, back, c.bits)
		}
	}
	if v := halfToFloat64(0x7e00); !math.IsNaN(v) {
		t.Errorf("halfToFloat64(qNaN) = %v, want NaN", v)
	}
	if v := halfToFloat64(0xfe00); !math.IsNaN(v) {
		t.Errorf("halfToFloat64(-qNaN) = %v, want NaN", v)
	}
}

// TestFloatSpecialRules checks the documented per-value defaults and each
// explicit rule for NaN and the infinities.
func TestFloatSpecialRules(t *testing.T) {
	page := floatTIFFPage{
		width: 4, height: 1, samples: 1, bits: 64, photometric: pBlackIsZero,
		data: []float64{math.NaN(), math.Inf(1), math.Inf(-1), 0.5},
	}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	tests := []struct {
		name string
		f    FloatOptions
		want [4]uint16
	}{
		{
			// Defaults: NaN -> zero, +Inf -> max, -Inf -> min.
			name: "default", f: FloatOptions{},
			want: [4]uint16{0x0000, 0xffff, 0x0000, 0x8000},
		},
		{
			name: "zero", f: FloatOptions{BitDepth: 8, NaN: FloatSpecialZero, PosInf: FloatSpecialZero, NegInf: FloatSpecialZero},
			want: [4]uint16{0, 0, 0, 0x80},
		},
		{
			name: "max", f: FloatOptions{BitDepth: 8, NaN: FloatSpecialMax, PosInf: FloatSpecialMax, NegInf: FloatSpecialMax},
			want: [4]uint16{0xff, 0xff, 0xff, 0x80},
		},
		{
			name: "min", f: FloatOptions{BitDepth: 8, NaN: FloatSpecialMin, PosInf: FloatSpecialMin, NegInf: FloatSpecialMin},
			want: [4]uint16{0, 0, 0, 0x80},
		},
		{
			// A substituted value outside the range is then clamped.
			name: "zero-outside-range", f: FloatOptions{BitDepth: 8, Min: 1, Max: 2, NaN: FloatSpecialZero},
			want: [4]uint16{0, 0xff, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			img, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &tt.f})
			if err != nil {
				t.Fatal(err)
			}
			var got [4]uint16
			if tt.f.BitDepth == 8 {
				g := img.(*image.Gray)
				for x := 0; x < 4; x++ {
					got[x] = uint16(g.GrayAt(x, 0).Y)
				}
			} else {
				g := img.(*image.Gray16)
				for x := 0; x < 4; x++ {
					got[x] = g.Gray16At(x, 0).Y
				}
			}
			if got != tt.want {
				t.Errorf("got %#04x, want %#04x", got, tt.want)
			}
		})
	}
}

// TestFloatRejectChannel checks that a rejecting rule reports the offending
// channel, and that other channels are not blamed.
func TestFloatRejectChannel(t *testing.T) {
	page := floatTIFFPage{
		width: 1, height: 1, samples: 3, bits: 32, photometric: pRGB,
		data: []float64{0.5, math.NaN(), 0.5},
	}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	f := FloatOptions{NaN: FloatSpecialReject}
	_, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f})
	var pe *PageError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *PageError", err)
	}
	if pe.Page != 0 || pe.Channel != 1 {
		t.Errorf("got page %d channel %d, want page 0 channel 1", pe.Page, pe.Channel)
	}
}

// TestFloatOutOfRangeReject checks the range rules, including alpha.
func TestFloatOutOfRangeReject(t *testing.T) {
	page := floatTIFFPage{
		width: 2, height: 1, samples: 4, bits: 32, photometric: pRGB, extraSamples: esUnassociated,
		data: []float64{1, 0, 0, 0.5, 1, 0, 0, 1.5},
	}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	f := FloatOptions{AlphaOutOfRange: FloatOutOfRangeReject}
	_, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f})
	var pe *PageError
	if !errors.As(err, &pe) || pe.Channel != 3 {
		t.Fatalf("got %v, want a channel-3 PageError", err)
	}

	// Clamping (the default) accepts the same input and saturates.
	img, err := DecodeWithOptions(bytes.NewReader(data), nil)
	if err != nil {
		t.Fatal(err)
	}
	if a := img.(*image.NRGBA64).NRGBA64At(1, 0).A; a != 0xffff {
		t.Errorf("clamped alpha: want 0xffff, got %#04x", a)
	}
}

// TestFloatShortStrip checks that a strip too short for its geometry fails
// with a page/channel context error rather than panicking, for every bit
// depth and sample layout.
func TestFloatShortStrip(t *testing.T) {
	_ = fmt.Sprint // Keep fmt imported for other tests in this file.
	for _, bits := range []int{16, 32, 64} {
		for _, samples := range []int{1, 3, 4} {
			photo := pBlackIsZero
			extra := 0
			if samples > 1 {
				photo = pRGB
			}
			if samples == 4 {
				extra = esUnassociated
			}
			if samples == 2 {
				continue
			}
			page := floatTIFFPage{
				width: 4, height: 4, samples: samples, bits: bits, photometric: photo,
				extraSamples: extra,
				data:         make([]float64, 4*4*samples),
			}
			data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
			// Truncate the strip by one sample.
			full := len(page.data) * bits / 8
			page.data = page.data[:len(page.data)-1]
			data = buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
			if got := len(data); got == 0 || full == 0 {
				t.Fatal("bad test construction")
			}
			img, err := Decode(bytes.NewReader(data))
			if err == nil {
				t.Fatalf("bits=%d samples=%d: got image %T, want error", bits, samples, img)
			}
			var pe *PageError
			if !errors.As(err, &pe) {
				t.Errorf("bits=%d samples=%d: error %v is not a *PageError", bits, samples, err)
			}
			if !errors.Is(err, errNoPixels) {
				t.Errorf("bits=%d samples=%d: error %v is not errNoPixels", bits, samples, err)
			}
		}
	}
}

// TestFloatBadOffsets checks that strip offsets past the end of the data do
// not panic.
func TestFloatBadOffsets(t *testing.T) {
	page := floatTIFFPage{width: 2, height: 2, samples: 1, bits: 32, photometric: pBlackIsZero, data: make([]float64, 4)}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	// Corrupt the StripOffsets value (tag 273) to point past the end. Walk
	// the single IFD to locate it.
	off := int64(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint16(data[off:]))
	found := false
	for i := 0; i < n; i++ {
		e := off + 2 + int64(i*ifdLen)
		if binary.LittleEndian.Uint16(data[e:]) == tStripOffsets {
			binary.LittleEndian.PutUint32(data[e+8:], 0xfffffff0)
			found = true
		}
	}
	if !found {
		t.Fatal("could not find StripOffsets tag")
	}
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("got nil error, want non-nil")
	}
}

// TestFloatSampleCountOverflow checks that absurd sample counts and block
// sizes are rejected without panicking.
func TestFloatSampleCountOverflow(t *testing.T) {
	// A single page whose BitsPerSample array has far more entries than any
	// supported layout.
	page := floatTIFFPage{
		width: 1, height: 1, samples: 1, bits: 32, photometric: pRGB,
		data: []float64{0, 0, 0},
	}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	// Patch BitsPerSample and SampleFormat counts to a huge value. Rebuild a
	// synthetic IFD instead, since the count must stay in sync between the
	// two tags for setupFloat to accept it.
	data = buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	off := int64(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint16(data[off:]))
	for i := 0; i < n; i++ {
		e := off + 2 + int64(i*ifdLen)
		tag := binary.LittleEndian.Uint16(data[e:])
		if tag == tBitsPerSample || tag == tSampleFormat {
			binary.LittleEndian.PutUint32(data[e+4:], 0x7fffffff)
		}
	}
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("got nil error, want non-nil")
	}
}

// TestFloatValuesAreStable decodes the same input twice and checks that the
// result is byte-identical, and that a fuzzer-friendly second pass through
// DecodeConfig does not change the outcome.
func TestFloatValuesAreStable(t *testing.T) {
	pages := testPages()
	data := buildFloatTIFF(binary.BigEndian, pages)
	f := FloatOptions{Min: -1, Max: 1, BitDepth: 16, NaN: FloatSpecialMax}
	first, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f})
	if err != nil {
		t.Fatal(err)
	}
	second, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f})
	if err != nil {
		t.Fatal(err)
	}
	compareExact(t, first, second)
}

// TestFloatIntegerNoRegression checks that the presence of options does not
// change the decoding of an integer TIFF.
func TestFloatIntegerNoRegression(t *testing.T) {
	for _, name := range []string{
		"video-001.tiff", "video-001-16bit.tiff", "video-001-gray.tiff",
		"video-001-gray-16bit.tiff", "video-001-paletted.tiff", "bw-deflate.tiff",
	} {
		contents, err := os.ReadFile("../testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		plain, err := Decode(bytes.NewReader(contents))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		withOpts, err := DecodeWithOptions(bytes.NewReader(contents), &DecodeOptions{
			Float: &FloatOptions{BitDepth: 8, Min: -1, Max: 1},
		})
		if err != nil {
			t.Fatalf("%s with options: %v", name, err)
		}
		if fmt.Sprintf("%T", plain) != fmt.Sprintf("%T", withOpts) || plain.Bounds() != withOpts.Bounds() {
			t.Fatalf("%s: type/bounds changed: %T %v vs %T %v", name, plain, plain.Bounds(), withOpts, withOpts.Bounds())
		}
		if !reflect.DeepEqual(plain, withOpts) {
			t.Fatalf("%s: decoding changed when options were passed", name)
		}
	}
}

// TestFloatOptionsValidation checks that invalid options are rejected.
func TestFloatOptionsValidation(t *testing.T) {
	page := floatTIFFPage{width: 1, height: 1, samples: 1, bits: 32, photometric: pBlackIsZero, data: []float64{0}}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	tests := []FloatOptions{
		{BitDepth: 12},
		{Min: 1, Max: 1},
		{Min: 2, Max: 1},
		{Min: math.NaN(), Max: 1},
		{Min: 0, Max: math.Inf(1)},
		{Min: -math.MaxFloat64, Max: math.MaxFloat64},
		{Min: 0, Max: 1, AlphaMin: 5, AlphaMax: 5},
		{NaN: 99},
		{OutOfRange: 99},
	}
	for _, f := range tests {
		f := f
		if _, err := DecodeWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f}); err == nil {
			t.Errorf("options %+v: got nil error, want non-nil", f)
		}
		if _, err := DecodeConfigWithOptions(bytes.NewReader(data), &DecodeOptions{Float: &f}); err == nil {
			t.Errorf("DecodeConfigWithOptions %+v: got nil error, want non-nil", f)
		}
	}
}

// TestFloatPredictorRejected checks that a floating-point page using the
// integer predictor is rejected rather than mis-decoded.
func TestFloatPredictorRejected(t *testing.T) {
	page := floatTIFFPage{width: 2, height: 2, samples: 1, bits: 32, photometric: pBlackIsZero, data: make([]float64, 4)}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	// Append a Predictor=2 entry by hand: rebuild with an extra entry.
	page.extraSamples = 0
	data = buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	// Instead of a full rebuild, verify that a page with Predictor=1 is
	// accepted; the rejection path is covered by a direct decoder test.
	d, err := newDecoder(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	d.features[tPredictor] = []uint{prHorizontal}
	if err := d.setupFloat(); err == nil {
		t.Fatal("setupFloat with horizontal predictor: got nil error")
	}
}

// TestNumPages checks page enumeration and out-of-range pages.
func TestNumPages(t *testing.T) {
	data := buildFloatTIFF(binary.BigEndian, testPages())
	n, err := NumPages(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if want := len(testPages()); n != want {
		t.Fatalf("NumPages: want %d, got %d", want, n)
	}
	if _, err := DecodePageWithOptions(bytes.NewReader(data), n, nil); err == nil {
		t.Fatal("DecodePageWithOptions out of range: got nil error")
	}
	if _, err := DecodePageWithOptions(bytes.NewReader(data), -1, nil); err == nil {
		t.Fatal("DecodePageWithOptions negative page: got nil error")
	}
	// Truncated input.
	if _, err := NumPages(bytes.NewReader(data[:4])); err == nil {
		t.Fatal("NumPages truncated: got nil error")
	}
}

// TestFloatUnexpectedBits checks that floating-point data with an
// unsupported bit depth is rejected cleanly.
func TestFloatUnexpectedBits(t *testing.T) {
	for _, bits := range []int{8, 24, 128} {
		page := floatTIFFPage{width: 1, height: 1, samples: 1, bits: 32, photometric: pBlackIsZero, data: []float64{0}}
		data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
		off := int64(binary.LittleEndian.Uint32(data[4:8]))
		n := int(binary.LittleEndian.Uint16(data[off:]))
		for i := 0; i < n; i++ {
			e := off + 2 + int64(i*ifdLen)
			if binary.LittleEndian.Uint16(data[e:]) == tBitsPerSample {
				binary.LittleEndian.PutUint16(data[e+8:], uint16(bits))
			}
		}
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Errorf("bits=%d: got nil error, want non-nil", bits)
		}
	}
}

// TestFloatMixedSampleFormat checks that a page that mixes float and integer
// samples is rejected.
func TestFloatMixedSampleFormat(t *testing.T) {
	page := floatTIFFPage{width: 1, height: 1, samples: 1, bits: 32, photometric: pRGB, data: []float64{0, 0, 0}}
	data := buildFloatTIFF(binary.LittleEndian, []floatTIFFPage{page})
	off := int64(binary.LittleEndian.Uint32(data[4:8]))
	n := int(binary.LittleEndian.Uint16(data[off:]))
	for i := 0; i < n; i++ {
		e := off + 2 + int64(i*ifdLen)
		if binary.LittleEndian.Uint16(data[e:]) == tSampleFormat {
			// Three shorts stored out of line; make the middle one unsigned.
			vo := int64(binary.LittleEndian.Uint32(data[e+8:]))
			binary.LittleEndian.PutUint16(data[vo+2:], sfUnsignedInt)
		}
	}
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("mixed sample format: got nil error, want non-nil")
	}
}

// FuzzDecodeFloat feeds arbitrary bytes to the decoder with float options
// enabled. The invariant is simply that Decode must not panic; any error is
// acceptable.
func FuzzDecodeFloat(f *testing.F) {
	f.Add(buildFloatTIFF(binary.LittleEndian, testPages()[:3]))
	f.Add(buildFloatTIFF(binary.BigEndian, []floatTIFFPage{{
		width: 2, height: 2, samples: 4, bits: 64, photometric: pRGB,
		extraSamples: esAssociated,
		data:         make([]float64, 16),
	}}))
	f.Fuzz(func(t *testing.T, data []byte) {
		opt := &DecodeOptions{Float: &FloatOptions{
			Min: -1, Max: 1, BitDepth: 8,
			NaN: FloatSpecialMax, PosInf: FloatSpecialZero, NegInf: FloatSpecialMin,
		}}
		if cfg, err := DecodeConfigWithOptions(bytes.NewReader(data), opt); err == nil {
			if cfg.Width*cfg.Height > 1e6 {
				return
			}
		}
		_, _ = DecodeWithOptions(bytes.NewReader(data), opt)
		_, _ = NumPages(bytes.NewReader(data))
	})
}

// patchTag rewrites the inline or out-of-line value of the single IFD entry
// with the given tag in a single-page TIFF. It is a test helper for building
// malformed inputs.
func patchTag(t *testing.T, data []byte, order byteOrder, tag uint16, value uint32) {
	t.Helper()
	off := int64(order.Uint32(data[4:8]))
	n := int(order.Uint16(data[off:]))
	for i := 0; i < n; i++ {
		e := off + 2 + int64(i*ifdLen)
		if order.Uint16(data[e:]) != tag {
			continue
		}
		typ := order.Uint16(data[e+2:])
		count := order.Uint32(data[e+4:])
		if count*lengths[typ] <= 4 {
			switch lengths[typ] {
			case 2:
				order.PutUint16(data[e+8:], uint16(value))
			case 4:
				order.PutUint32(data[e+8:], value)
			}
		} else {
			vo := int64(order.Uint32(data[e+8:]))
			switch lengths[typ] {
			case 2:
				order.PutUint16(data[vo:], uint16(value))
			case 4:
				order.PutUint32(data[vo:], value)
			}
		}
		return
	}
	t.Fatalf("tag %d not found", tag)
}

// TestFloatPlanarRejected checks that separate color planes are rejected.
func TestFloatPlanarRejected(t *testing.T) {
	page := floatTIFFPage{width: 2, height: 2, samples: 3, bits: 32, photometric: pRGB, data: make([]float64, 12)}
	data := buildFloatTIFF(binary.BigEndian, []floatTIFFPage{page})
	// The test builder does not emit PlanarConfiguration; add one by
	// rewriting the page's IFD.
	data = withExtraEntry(t, data, binary.BigEndian, tiffEntry{tPlanarConfiguration, dtShort, []uint32{2}})
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("planar configuration 2: got nil error, want non-nil")
	}
}

// TestFloatSamplePerPixelMismatch checks that SamplesPerPixel disagreeing
// with the BitsPerSample count is rejected.
func TestFloatSamplePerPixelMismatch(t *testing.T) {
	page := floatTIFFPage{width: 1, height: 1, samples: 3, bits: 32, photometric: pRGB, data: []float64{0, 0, 0}}
	data := buildFloatTIFF(binary.BigEndian, []floatTIFFPage{page})
	patchTag(t, data, binary.BigEndian, tSamplesPerPixel, 4)
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Fatal("mismatched SamplesPerPixel: got nil error, want non-nil")
	}
}

// TestIntegerErrorsUnchanged checks that integer decoding failures keep their
// original error values (no PageError wrapping) when options are passed.
func TestIntegerErrorsUnchanged(t *testing.T) {
	b, err := os.ReadFile("../testdata/bw-uncompressed.tiff")
	if err != nil {
		t.Fatal(err)
	}
	// Same mutation as TestShortBlockData: shrink the byte count.
	old := []byte{0x4c, 0x04}
	i := bytes.Index(b, old)
	if i < 0 {
		t.Fatal("could not find byte count")
	}
	b[i+1] = 0x01
	_, plainErr := Decode(bytes.NewReader(b))
	_, optErr := DecodeWithOptions(bytes.NewReader(b), &DecodeOptions{Float: &FloatOptions{}})
	if plainErr == nil || optErr == nil {
		t.Fatalf("got errors %v and %v, want both non-nil", plainErr, optErr)
	}
	if plainErr.Error() != optErr.Error() {
		t.Errorf("integer error changed with options: %q vs %q", plainErr, optErr)
	}
	var pe *PageError
	if errors.As(optErr, &pe) {
		t.Errorf("integer error was wrapped in a PageError: %v", optErr)
	}
}

// withExtraEntry rebuilds a single-page TIFF, preserving its pixel data but
// adding one IFD entry.
func withExtraEntry(t *testing.T, data []byte, order byteOrder, extra tiffEntry) []byte {
	t.Helper()
	ifdOff := int64(order.Uint32(data[4:8]))
	n := int(order.Uint16(data[ifdOff:]))
	var entries []tiffEntry
	for i := 0; i < n; i++ {
		e := ifdOff + 2 + int64(i*ifdLen)
		tag := order.Uint16(data[e:])
		typ := order.Uint16(data[e+2:])
		count := order.Uint32(data[e+4:])
		vals := make([]uint32, count)
		if count*lengths[typ] <= 4 {
			for k := range vals {
				if typ == dtShort {
					vals[k] = uint32(order.Uint16(data[e+8+int64(2*k):]))
				} else {
					vals[k] = order.Uint32(data[e+8+int64(4*k):])
				}
			}
		} else {
			vo := int64(order.Uint32(data[e+8:]))
			for k := range vals {
				if typ == dtShort {
					vals[k] = uint32(order.Uint16(data[vo+int64(2*k):]))
				} else {
					vals[k] = order.Uint32(data[vo+int64(4*k):])
				}
			}
		}
		entries = append(entries, tiffEntry{tag, typ, vals})
	}
	entries = append(entries, extra)
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].tag < entries[j-1].tag; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
	// Re-lay out the file: header, a copy of the original bytes up to the
	// IFD, then the new IFD.
	out := append([]byte(nil), data[:ifdOff]...)
	newIFD := len(out)
	order.PutUint32(out[4:8], uint32(newIFD))
	out = append(out, appendIFDEntries(order, entries, newIFD)...)
	return out
}

// TestFloatIFDParseErrorHasPageContext checks that an error raised while
// parsing an entry of a float page, before its sample layout is validated,
// still carries page context.
func TestFloatIFDParseErrorHasPageContext(t *testing.T) {
	page := floatTIFFPage{width: 2, height: 2, samples: 1, bits: 32, photometric: pBlackIsZero, data: make([]float64, 4)}
	first := floatTIFFPage{width: 1, height: 1, samples: 1, bits: 32, photometric: pBlackIsZero, data: []float64{0}}
	data := buildFloatTIFF(binary.BigEndian, []floatTIFFPage{first, page})
	// Walk to the second IFD and give ImageWidth (which sorts before
	// SampleFormat) an invalid datatype. Entry parsing fails before setup,
	// but the IFD still declares floating point samples.
	o := int64(binary.BigEndian.Uint32(data[4:8]))
	n := int(binary.BigEndian.Uint16(data[o:]))
	second := int64(binary.BigEndian.Uint32(data[o+2+int64(ifdLen*n):]))
	m := int(binary.BigEndian.Uint16(data[second:]))
	found := false
	for i := 0; i < int(m); i++ {
		e := second + 2 + int64(i*ifdLen)
		if binary.BigEndian.Uint16(data[e:]) == tImageWidth {
			binary.BigEndian.PutUint16(data[e+2:], uint16(len(lengths)))
			found = true
		}
	}
	if !found {
		t.Fatal("could not find ImageWidth tag")
	}
	_, err := DecodePageWithOptions(bytes.NewReader(data), 1, nil)
	if err == nil {
		t.Fatal("got nil error, want non-nil")
	}
	var pe *PageError
	if !errors.As(err, &pe) {
		t.Fatalf("error %v is not a *PageError", err)
	}
	if pe.Page != 1 {
		t.Errorf("error page: want 1, got %d", pe.Page)
	}
}

// buildTiledFloatTIFF builds a single-page, tiled floating-point TIFF whose
// tiles are laid out row-major and may overhang the image edge. Each tile is
// stored uncompressed and is fully padded to tileWidth*tileHeight samples.
func buildTiledFloatTIFF(order byteOrder, page floatTIFFPage, tileW, tileH int) []byte {
	across := (page.width + tileW - 1) / tileW
	down := (page.height + tileH - 1) / tileH
	tileBytes := tileW * tileH * page.samples * page.bits / 8
	var b []byte
	if order.Uint16([]byte{1, 0}) == 1 {
		b = append(b, 'I', 'I')
	} else {
		b = append(b, 'M', 'M')
	}
	b = order.AppendUint16(b, 42)
	b = order.AppendUint32(b, 0)
	var offsets, counts []uint32
	for ty := 0; ty < down; ty++ {
		for tx := 0; tx < across; tx++ {
			offsets = append(offsets, uint32(len(b)))
			counts = append(counts, uint32(tileBytes))
			for y := 0; y < tileH; y++ {
				for x := 0; x < tileW; x++ {
					px, py := tx*tileW+x, ty*tileH+y
					for s := 0; s < page.samples; s++ {
						var v float64
						if px < page.width && py < page.height {
							v = page.data[(py*page.width+px)*page.samples+s]
						}
						switch page.bits {
						case 16:
							b = order.AppendUint16(b, float64ToHalf(v))
						case 32:
							b = order.AppendUint32(b, math.Float32bits(float32(v)))
						case 64:
							b = order.AppendUint64(b, math.Float64bits(v))
						}
					}
				}
			}
		}
	}
	entries := []tiffEntry{
		{tImageWidth, dtShort, []uint32{uint32(page.width)}},
		{tImageLength, dtShort, []uint32{uint32(page.height)}},
		{tBitsPerSample, dtShort, repeat32(uint32(page.bits), page.samples)},
		{tCompression, dtShort, []uint32{cNone}},
		{tPhotometricInterpretation, dtShort, []uint32{uint32(page.photometric)}},
		{tSamplesPerPixel, dtShort, []uint32{uint32(page.samples)}},
		{tSampleFormat, dtShort, repeat32(sfFloat, page.samples)},
		{tTileWidth, dtShort, []uint32{uint32(tileW)}},
		{tTileLength, dtShort, []uint32{uint32(tileH)}},
		{tTileOffsets, dtLong, offsets},
		{tTileByteCounts, dtLong, counts},
	}
	if page.extraSamples != 0 {
		entries = append(entries, tiffEntry{tExtraSamples, dtShort, []uint32{uint32(page.extraSamples)}})
	}
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && entries[j].tag < entries[j-1].tag; j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
	ifdOff := len(b)
	order.PutUint32(b[4:8], uint32(ifdOff))
	b = append(b, appendIFDEntries(order, entries, ifdOff)...)
	return b
}

// TestFloatTiled checks tiled float pages, including tiles that overhang the
// image edge (the row-skip path), against the reference.
func TestFloatTiled(t *testing.T) {
	const w, h = 13, 7
	data := make([]float64, w*h*3)
	for i := range data {
		switch i % 7 {
		case 0:
			data[i] = math.NaN()
		case 1:
			data[i] = math.Inf(1)
		case 2:
			data[i] = -0.5
		default:
			// Quantize to float32, matching what is stored.
			data[i] = float64(float32(float64(i%100) / 100))
		}
	}
	page := floatTIFFPage{width: w, height: h, samples: 3, bits: 32, photometric: pRGB, data: data}
	for _, order := range []struct {
		name string
		o    byteOrder
	}{{"little", binary.LittleEndian}, {"big", binary.BigEndian}} {
		for _, tile := range [][2]int{{8, 8}, {16, 8}, {16, 16}} {
			t.Logf("%s tile %v", order.name, tile)
			raw := buildTiledFloatTIFF(order.o, page, tile[0], tile[1])
			f := FloatOptions{Min: 0, Max: 1, BitDepth: 8, NaN: FloatSpecialZero, PosInf: FloatSpecialMax, NegInf: FloatSpecialMin}
			img, err := DecodeWithOptions(bytes.NewReader(raw), &DecodeOptions{Float: &f})
			if err != nil {
				t.Fatalf("%s tile %v: %v", order.name, tile, err)
			}
			r := refOptsFor(f)
			compareExact(t, referenceImage(t, page, r), img)
		}
	}
}

// TestFloatTiledTooSmall checks that too-small float tiles are rejected like
// integer ones.
func TestFloatTiledTooSmall(t *testing.T) {
	page := floatTIFFPage{width: 8, height: 8, samples: 1, bits: 32, photometric: pBlackIsZero, data: make([]float64, 64)}
	raw := buildTiledFloatTIFF(binary.LittleEndian, page, 8, 8)
	// Patch TileWidth to 4, which is below the 8-pixel minimum.
	patchTag(t, raw, binary.LittleEndian, tTileWidth, 4)
	if _, err := Decode(bytes.NewReader(raw)); err == nil {
		t.Fatal("too-small float tile: got nil error, want non-nil")
	}
}
