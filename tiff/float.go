// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
)

// FloatSpecial selects what a floating-point decoder does with a special
// sample value (NaN, positive infinity or negative infinity).
type FloatSpecial uint8

const (
	// FloatSpecialDefault uses the conventional default for the particular
	// special value: NaN is treated as zero, positive infinity as the maximum
	// of the configured range, and negative infinity as the minimum. This is
	// the zero value.
	FloatSpecialDefault FloatSpecial = iota

	// FloatSpecialZero substitutes zero, which is then subject to the
	// configured range and out-of-range rule.
	FloatSpecialZero

	// FloatSpecialMin substitutes the minimum of the configured range.
	FloatSpecialMin

	// FloatSpecialMax substitutes the maximum of the configured range.
	FloatSpecialMax

	// FloatSpecialReject causes decoding to fail.
	FloatSpecialReject
)

// FloatRangeRule selects how finite floating-point samples outside the
// configured range are handled.
type FloatRangeRule uint8

const (
	// FloatOutOfRangeClamp clamps samples to the configured range. This is the
	// zero value.
	FloatOutOfRangeClamp FloatRangeRule = iota

	// FloatOutOfRangeReject causes decoding to fail.
	FloatOutOfRangeReject
)

// FloatOptions controls how IEEE floating-point TIFF samples (SampleFormat 3)
// are converted to the integer image types of the standard library.
//
// Color samples are mapped linearly from [Min, Max] to the full output range
// selected by BitDepth. The alpha sample, if any, is mapped independently from
// [AlphaMin, AlphaMax]. Mapping is deterministic: a value v is scaled to
// (v-Min)/(Max-Min), clamped according to OutOfRange, multiplied by 255 or
// 65535, and rounded to the nearest integer (ties away from zero).
//
// The zero value of FloatOptions is valid and selects the defaults described
// on each field.
type FloatOptions struct {
	// Min and Max define the linear range of color samples that is mapped to
	// the full output range. If both are zero, the default range [0, 1] is
	// used. Max must be greater than Min, both must be finite, and Max-Min
	// must be finite: an enormous range whose width overflows float64 is
	// rejected.
	Min, Max float64

	// AlphaMin and AlphaMax define the linear range of the alpha sample. If
	// both are zero, the default range [0, 1] is used. The same validity
	// rules as for Min and Max apply.
	AlphaMin, AlphaMax float64

	// BitDepth selects 8-bit (*image.Gray, *image.RGBA, *image.NRGBA) or
	// 16-bit (*image.Gray16, *image.RGBA64, *image.NRGBA64) output. A value of
	// zero selects 16-bit output. No other value is allowed.
	BitDepth int

	// NaN, PosInf and NegInf select how the corresponding special sample
	// values are handled. The same rules apply to alpha samples.
	NaN, PosInf, NegInf FloatSpecial

	// OutOfRange selects how finite color samples outside [Min, Max] are
	// handled. The zero value clamps to the range, so, for example, negative
	// samples below Min become Min.
	OutOfRange FloatRangeRule

	// AlphaOutOfRange is like OutOfRange but applies to the alpha sample.
	AlphaOutOfRange FloatRangeRule
}

// DecodeOptions configures optional decoder behavior. The zero value selects
// the package defaults.
type DecodeOptions struct {
	// Float configures decoding of IEEE floating-point samples. A nil Float
	// uses the zero FloatOptions, whose defaults are documented on that type.
	Float *FloatOptions
}

// PageError reports an error encountered while decoding a page of a TIFF
// image. Channel is the zero-based sample index within a pixel, or -1 if the
// error is not specific to a channel.
type PageError struct {
	Page    int
	Channel int
	Err     error
}

func (e *PageError) Error() string {
	if e.Channel >= 0 {
		return fmt.Sprintf("tiff: page %d: channel %d: %v", e.Page, e.Channel, e.Err)
	}
	return fmt.Sprintf("tiff: page %d: %v", e.Page, e.Err)
}

func (e *PageError) Unwrap() error { return e.Err }

// pageError adds page and channel context to err. It does not add the context
// twice.
func (d *decoder) pageError(channel int, err error) error {
	return makePageError(d.page, channel, err)
}

// makePageError adds page and channel context to err. It does not add the
// context twice.
func makePageError(page, channel int, err error) error {
	if err == nil {
		return nil
	}
	var pe *PageError
	if errors.As(err, &pe) {
		return err
	}
	return &PageError{Page: page, Channel: channel, Err: err}
}

// normalizeFloatOptions returns a copy of o with the zero-value defaults
// filled in and validates the result.
func normalizeFloatOptions(o *FloatOptions) (FloatOptions, error) {
	var f FloatOptions
	if o != nil {
		f = *o
	}
	if f.Min == 0 && f.Max == 0 {
		f.Min, f.Max = 0, 1
	}
	if f.AlphaMin == 0 && f.AlphaMax == 0 {
		f.AlphaMin, f.AlphaMax = 0, 1
	}
	if f.BitDepth == 0 {
		f.BitDepth = 16
	}
	if f.BitDepth != 8 && f.BitDepth != 16 {
		return f, fmt.Errorf("tiff: invalid floating point BitDepth %d: must be 8 or 16", f.BitDepth)
	}
	if err := validateFloatRange("color", f.Min, f.Max); err != nil {
		return f, err
	}
	if err := validateFloatRange("alpha", f.AlphaMin, f.AlphaMax); err != nil {
		return f, err
	}
	if f.NaN > FloatSpecialReject || f.PosInf > FloatSpecialReject || f.NegInf > FloatSpecialReject {
		return f, errors.New("tiff: invalid floating point special-value rule")
	}
	if f.OutOfRange > FloatOutOfRangeReject || f.AlphaOutOfRange > FloatOutOfRangeReject {
		return f, errors.New("tiff: invalid floating point out-of-range rule")
	}
	return f, nil
}

func validateFloatRange(name string, min, max float64) error {
	if math.IsNaN(min) || math.IsNaN(max) || math.IsInf(min, 0) || math.IsInf(max, 0) {
		return fmt.Errorf("tiff: %s floating point range is not finite", name)
	}
	if !(max > min) {
		return fmt.Errorf("tiff: %s floating point range is empty or inverted", name)
	}
	if math.IsInf(max-min, 0) {
		return fmt.Errorf("tiff: %s floating point range is too large", name)
	}
	return nil
}

// halfToFloat64 converts an IEEE 754 binary16 value to float64.
func halfToFloat64(h uint16) float64 {
	sign := h >> 15
	exp := (h >> 10) & 0x1f
	mant := h & 0x3ff

	var v float64
	switch exp {
	case 0:
		// Zero or subnormal.
		v = math.Ldexp(float64(mant), -24)
	case 0x1f:
		if mant == 0 {
			v = math.Inf(1)
		} else {
			v = math.NaN()
		}
	default:
		v = math.Ldexp(float64(1024+mant), int(exp)-25)
	}
	if sign != 0 {
		v = -v
	}
	return v
}

// applyFloatSpecial substitutes v according to rule. def is the default
// interpretation when rule is FloatSpecialDefault.
func applyFloatSpecial(name string, rule, def FloatSpecial, min, max float64) (float64, error) {
	if rule == FloatSpecialDefault {
		rule = def
	}
	switch rule {
	case FloatSpecialZero:
		return 0, nil
	case FloatSpecialMin:
		return min, nil
	case FloatSpecialMax:
		return max, nil
	case FloatSpecialReject:
		return 0, fmt.Errorf("non-finite floating point sample (%s)", name)
	default:
		return 0, errors.New("invalid floating point special-value rule")
	}
}

// mapFloatSample maps a floating-point sample to [0, outMax] according to the
// decoder's float options. alpha selects the alpha range and out-of-range
// rule.
func (d *decoder) mapFloatSample(v float64, alpha bool) (uint32, error) {
	min, max := d.floatOpts.Min, d.floatOpts.Max
	rule := d.floatOpts.OutOfRange
	if alpha {
		min, max = d.floatOpts.AlphaMin, d.floatOpts.AlphaMax
		rule = d.floatOpts.AlphaOutOfRange
	}

	var err error
	switch {
	case math.IsNaN(v):
		v, err = applyFloatSpecial("NaN", d.floatOpts.NaN, FloatSpecialZero, min, max)
	case math.IsInf(v, +1):
		v, err = applyFloatSpecial("+Inf", d.floatOpts.PosInf, FloatSpecialMax, min, max)
	case math.IsInf(v, -1):
		v, err = applyFloatSpecial("-Inf", d.floatOpts.NegInf, FloatSpecialMin, min, max)
	}
	if err != nil {
		return 0, err
	}

	// A special-value rule can substitute a value outside the range (zero, in
	// particular), so apply the range rule afterwards.
	if v < min {
		if rule == FloatOutOfRangeReject {
			return 0, fmt.Errorf("sample %v is below minimum %v", v, min)
		}
		v = min
	} else if v > max {
		if rule == FloatOutOfRangeReject {
			return 0, fmt.Errorf("sample %v is above maximum %v", v, max)
		}
		v = max
	}

	scaled := (v - min) / (max - min) * float64(d.outMax)
	// A huge but finite range can make the scaling overflow to +Inf; treat
	// that as the maximum rather than relying on the float-to-uint32
	// conversion for out-of-range values. !(scaled > 0) also catches a
	// negative zero result.
	if !(scaled > 0) {
		return 0, nil
	}
	if scaled >= float64(d.outMax) {
		return d.outMax, nil
	}
	return uint32(math.Round(scaled)), nil
}

// readFloatSamples reads len(out) consecutive samples of bps bytes each from
// d.buf, advancing d.off. On a short buffer it reports the channel that could
// not be read.
func (d *decoder) readFloatSamples(out []float64, bps int) error {
	need, ok := checkedMul(int64(len(out)), int64(bps))
	if !ok || need > int64(len(d.buf)-d.off) || d.off < 0 || d.off > len(d.buf) {
		avail := len(d.buf) - d.off
		if avail < 0 {
			avail = 0
		}
		ch := avail / bps
		if ch >= len(out) {
			ch = len(out) - 1
		}
		if ch < 0 {
			ch = 0
		}
		return d.pageError(ch, errNoPixels)
	}
	for i := range out {
		p := d.buf[d.off:]
		switch bps {
		case 2:
			out[i] = halfToFloat64(d.byteOrder.Uint16(p))
		case 4:
			out[i] = float64(math.Float32frombits(d.byteOrder.Uint32(p)))
		case 8:
			out[i] = math.Float64frombits(d.byteOrder.Uint64(p))
		default:
			return d.pageError(i, UnsupportedError("floating point BitsPerSample"))
		}
		d.off += bps
	}
	return nil
}

// checkedMul returns a*b and whether the multiplication overflowed int64. It
// assumes a and b are non-negative.
func checkedMul(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a != 0 && b > math.MaxInt64/a {
		return 0, false
	}
	return a * b, true
}

// decodeFloat decodes a block of floating-point samples into dst.
func (d *decoder) decodeFloat(dst image.Image, xmin, ymin, xmax, ymax int) error {
	d.off = 0
	if dst == nil {
		return d.pageError(-1, UnsupportedError("floating point color model"))
	}
	if xmax < xmin || ymax < ymin {
		return d.pageError(-1, FormatError("invalid block bounds"))
	}
	if d.samples < 1 || d.samples > 4 {
		return d.pageError(-1, FormatError("unsupported sample count"))
	}

	pixelBytes, ok := checkedMul(int64(d.samples), int64(d.floatBytes))
	if !ok {
		return d.pageError(-1, FormatError("sample count overflow"))
	}
	rowBytes, ok := checkedMul(int64(xmax-xmin), pixelBytes)
	if !ok || int64(int(rowBytes)) != rowBytes {
		return d.pageError(-1, FormatError("sample count overflow"))
	}

	rMaxX := minInt(xmax, dst.Bounds().Max.X)
	rMaxY := minInt(ymax, dst.Bounds().Max.Y)
	out16 := d.outBits == 16

	var samples [4]float64
	n := d.samples
	for y := ymin; y < rMaxY; y++ {
		for x := xmin; x < xmax; x++ {
			if err := d.readFloatSamples(samples[:n], d.floatBytes); err != nil {
				return err
			}
			if x >= rMaxX {
				// A tile can extend past the image edge. Consume, but do not
				// store, the overhanging pixels: the block rows are fully
				// padded.
				continue
			}
			if err := d.storeFloatPixel(dst, x, y, samples[:n], out16); err != nil {
				return err
			}
		}
	}
	return nil
}

// storeFloatPixel maps samples and writes one pixel to dst.
func (d *decoder) storeFloatPixel(dst image.Image, x, y int, samples []float64, out16 bool) error {
	switch d.mode {
	case mGray, mGrayInvert:
		v, err := d.mapFloatSample(samples[0], false)
		if err != nil {
			return d.pageError(0, err)
		}
		if d.mode == mGrayInvert {
			v = d.outMax - v
		}
		if out16 {
			dst.(*image.Gray16).SetGray16(x, y, color.Gray16{uint16(v)})
		} else {
			dst.(*image.Gray).SetGray(x, y, color.Gray{uint8(v)})
		}

	case mRGB:
		r, err := d.mapFloatSample(samples[0], false)
		if err != nil {
			return d.pageError(0, err)
		}
		g, err := d.mapFloatSample(samples[1], false)
		if err != nil {
			return d.pageError(1, err)
		}
		b, err := d.mapFloatSample(samples[2], false)
		if err != nil {
			return d.pageError(2, err)
		}
		if out16 {
			dst.(*image.RGBA64).SetRGBA64(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), 0xffff})
		} else {
			dst.(*image.RGBA).SetRGBA(x, y, color.RGBA{uint8(r), uint8(g), uint8(b), 0xff})
		}

	case mRGBA, mNRGBA:
		var r, g, b uint32
		alphaChannel := 3
		var err error
		if d.grayAlpha {
			// Gray with alpha: replicate the mapped gray sample across R, G
			// and B, applying WhiteIsZero inversion first.
			alphaChannel = 1
			if r, err = d.mapFloatSample(samples[0], false); err != nil {
				return d.pageError(0, err)
			}
			if d.whiteIsZero {
				r = d.outMax - r
			}
			g, b = r, r
		} else {
			if r, err = d.mapFloatSample(samples[0], false); err != nil {
				return d.pageError(0, err)
			}
			if g, err = d.mapFloatSample(samples[1], false); err != nil {
				return d.pageError(1, err)
			}
			if b, err = d.mapFloatSample(samples[2], false); err != nil {
				return d.pageError(2, err)
			}
		}
		a, err := d.mapFloatSample(samples[alphaChannel], true)
		if err != nil {
			return d.pageError(alphaChannel, err)
		}
		if out16 {
			if d.mode == mRGBA {
				dst.(*image.RGBA64).SetRGBA64(x, y, color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
			} else {
				dst.(*image.NRGBA64).SetNRGBA64(x, y, color.NRGBA64{uint16(r), uint16(g), uint16(b), uint16(a)})
			}
		} else {
			if d.mode == mRGBA {
				dst.(*image.RGBA).SetRGBA(x, y, color.RGBA{uint8(r), uint8(g), uint8(b), uint8(a)})
			} else {
				dst.(*image.NRGBA).SetNRGBA(x, y, color.NRGBA{uint8(r), uint8(g), uint8(b), uint8(a)})
			}
		}

	default:
		return d.pageError(-1, UnsupportedError("floating point color model"))
	}
	return nil
}
