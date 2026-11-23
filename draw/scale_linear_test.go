// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"sync"
	"testing"

	"golang.org/x/image/math/f64"
)

// box is a box kernel of support 0.5. When downscaling by exactly 2, it visits
// exactly two source pixels with equal weights, which makes it convenient for
// analytic references.
var box = &Kernel{0.5, func(t float64) float64 { return 1 }}

// decodeEncoded is an independent sRGB decode used by the tests.
func decodeEncoded(v float64) float64 {
	if v <= 0.04045 {
		return v / 12.92
	}
	return math.Pow((v+0.055)/1.055, 2.4)
}

// encodeLinear is an independent sRGB encode used by the tests.
func encodeLinear(v float64) float64 {
	if v <= 0.0031308 {
		return 12.92 * v
	}
	return 1.055*math.Pow(v, 1/2.4) - 0.055
}

func float64Of(c color.RGBA64) [4]float64 {
	return [4]float64{
		float64(c.R) / 0xffff,
		float64(c.G) / 0xffff,
		float64(c.B) / 0xffff,
		float64(c.A) / 0xffff,
	}
}

func maxAbsDiff(a, b *image.RGBA64) float64 {
	worst := 0.0
	for y := a.Rect.Min.Y; y < a.Rect.Max.Y; y++ {
		for x := a.Rect.Min.X; x < a.Rect.Max.X; x++ {
			ca := float64Of(a.RGBA64At(x, y))
			cb := float64Of(b.RGBA64At(x-a.Rect.Min.X+b.Rect.Min.X, y-a.Rect.Min.Y+b.Rect.Min.Y))
			for i := 0; i < 4; i++ {
				if d := math.Abs(ca[i] - cb[i]); d > worst {
					worst = d
				}
			}
		}
	}
	return worst
}

// oneQuantum is the tolerance for one 16-bit rounding step.
const oneQuantum = 1.0 / 0xffff

// TestLinearTransferRoundTrip checks that the extended transfer curve is
// monotonic and that encode(decode(v)) == v for the whole legal range,
// including values outside [0, 1] used by negative-lobe kernels.
func TestLinearTransferRoundTrip(t *testing.T) {
	for v := -2.0; v <= 2.0; v += 0.0009765625 {
		dec := srgbDecode(v)
		enc := srgbEncode(dec)
		if d := math.Abs(enc - v); d > 1e-12 {
			t.Fatalf("round trip at %v: got %v, diff %v", v, enc, d)
		}
	}
	// Odd extension: f(-v) == -f(v), and zero is fixed.
	if got := srgbDecode(-0.5); got != -srgbDecode(0.5) {
		t.Errorf("decode odd symmetry: got %v", got)
	}
	if got := srgbEncode(-0.5); got != -srgbEncode(0.5) {
		t.Errorf("encode odd symmetry: got %v", got)
	}
	if srgbDecode(0) != 0 || srgbEncode(0) != 0 {
		t.Errorf("zero is not fixed")
	}
	// Non-finite inputs are stable and finite.
	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got := srgbDecode(v); math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("decode(%v) = %v, want finite", v, got)
		}
		if got := srgbEncode(v); math.IsNaN(got) || math.IsInf(got, 0) {
			t.Errorf("encode(%v) = %v, want finite", v, got)
		}
	}
	// Monotonic on a coarse grid over a wide range.
	prev := math.Inf(-1)
	for v := -1.5; v <= 1.5; v += 0.01 {
		got := srgbDecode(v)
		if got < prev {
			t.Fatalf("decode not monotonic at %v: %v < %v", v, got, prev)
		}
		prev = got
	}
}

// TestLinearLuminanceLadder is the core brightness test: a grayscale ladder
// downscaled 2:1 with a box kernel must average light, not encoded values.
//
// Averaging black and white in sRGB encoding gives 0.5 encoded, which is about
// 0.21 in linear light. Averaging in linear light gives 0.5 linear, which is
// about 0.735 encoded. The linear path must land near the latter.
func TestLinearLuminanceLadder(t *testing.T) {
	const n = 8
	src := image.NewNRGBA(image.Rect(0, 0, n, 1))
	for x := 0; x < n; x++ {
		v := uint8(x * 0xff / (n - 1))
		src.SetNRGBA(x, 0, color.NRGBA{v, v, v, 0xff})
	}

	run := func(opts *Options) *image.RGBA64 {
		dst := image.NewRGBA64(image.Rect(0, 0, n/2, 1))
		box.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, opts)
		return dst
	}
	enc := run(nil)
	lin := run(&Options{Transfer: TransferLinear})

	// Independent references: each dst pixel is the mean of its source pair,
	// computed both in encoded space and in linear light.
	var errEncDecoded, errLinDecoded float64
	for x := 0; x < n/2; x++ {
		e0 := float64(src.Pix[4*(2*x)+0]) / 0xff
		e1 := float64(src.Pix[4*(2*x+1)+0]) / 0xff
		wantEncoded := (e0 + e1) / 2
		wantLinear := (decodeEncoded(e0) + decodeEncoded(e1)) / 2
		wantLinearEncoded := encodeLinear(wantLinear)

		gotEnc := float64(enc.RGBA64At(x, 0).R) / 0xffff
		gotLin := float64(lin.RGBA64At(x, 0).R) / 0xffff
		if d := math.Abs(gotEnc - wantEncoded); d > oneQuantum {
			t.Errorf("x=%d: sRGB output %v, want encoded mean %v (diff %v)", x, gotEnc, wantEncoded, d)
		}
		if d := math.Abs(gotLin - wantLinearEncoded); d > oneQuantum {
			t.Errorf("x=%d: linear output %v, want %v (diff %v)", x, gotLin, wantLinearEncoded, d)
		}
		errEncDecoded += math.Abs(decodeEncoded(gotEnc) - wantLinear)
		errLinDecoded += math.Abs(decodeEncoded(gotLin) - wantLinear)
	}
	// The linear path must be far closer to the light-linear reference.
	if !(errLinDecoded < errEncDecoded/4) {
		t.Errorf("linear-light error %v not much smaller than sRGB error %v", errLinDecoded, errEncDecoded)
	}
	if errLinDecoded > 0.01*float64(n/2) {
		t.Errorf("linear-light absolute error too large: %v", errLinDecoded)
	}

	// The black/white pair specifically: 0.5 linear is about 0.7354 encoded,
	// while 0.5 encoded (the sRGB result) is only about 0.21 linear.
	bw := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	bw.SetNRGBA(0, 0, color.NRGBA{0x00, 0x00, 0x00, 0xff})
	bw.SetNRGBA(1, 0, color.NRGBA{0xff, 0xff, 0xff, 0xff})
	bwDst := func(opts *Options) *image.RGBA64 {
		d := image.NewRGBA64(image.Rect(0, 0, 1, 1))
		box.Scale(d, d.Bounds(), bw, bw.Bounds(), Src, opts)
		return d
	}
	if r := float64(bwDst(&Options{Transfer: TransferLinear}).RGBA64At(0, 0).R) / 0xffff; r < 0.70 || r > 0.77 {
		t.Errorf("linear black+white mean = %v, want near 0.735", r)
	}
	if r := float64(bwDst(nil).RGBA64At(0, 0).R) / 0xffff; math.Abs(r-0.5) > oneQuantum {
		t.Errorf("sRGB black+white mean = %v, want 0.5", r)
	}
}

// refContrib is one source coordinate and its normalized weight.
type refContrib struct {
	coord int
	w     float64
}

// refWeights recomputes a separable kernel's weights from the documented
// definition, independently of newDistrib. Coordinates are relative to the
// source rectangle.
func refWeights(srcN, dstN int, q *Kernel) [][]refContrib {
	scale := float64(srcN) / float64(dstN)
	halfWidth, argScale := q.Support, 1.0
	if scale > 1 {
		halfWidth *= scale
		argScale = 1 / scale
	}
	out := make([][]refContrib, dstN)
	for d := 0; d < dstN; d++ {
		center := (float64(d)+0.5)*scale - 0.5
		lo := int(math.Floor(center - halfWidth))
		hi := int(math.Ceil(center + halfWidth))
		if lo < 0 {
			lo = 0
		}
		if hi > srcN {
			hi = srcN
		}
		total := 0.0
		var cs []refContrib
		for c := lo; c < hi; c++ {
			tt := math.Abs(center-float64(c)) * argScale
			if tt >= q.Support {
				continue
			}
			w := q.At(tt)
			if w == 0 || math.IsNaN(w) || math.IsInf(w, 0) {
				continue
			}
			cs = append(cs, refContrib{c, w})
			total += w
		}
		if total != 0 {
			for i := range cs {
				cs[i].w /= total
			}
		}
		out[d] = cs
	}
	return out
}

// refFinish mirrors the documented output contract: clamp, optionally encode,
// premultiply, and force channel <= alpha.
func refFinish(r, g, b, a float64, premul bool, transfer TransferFunction) [4]float64 {
	a = clamp01(a)
	if premul {
		if a <= 0 {
			return [4]float64{}
		}
		r, g, b = r/a, g/a, b/a
	}
	r, g, b = clamp01(r), clamp01(g), clamp01(b)
	if transfer == TransferLinear {
		r, g, b = encodeLinear(r), encodeLinear(g), encodeLinear(b)
	}
	return [4]float64{clamp01(r * a), clamp01(g * a), clamp01(b * a), a}
}

func setRGBA64FromFloats(m *image.RGBA64, x, y int, f [4]float64) {
	m.SetRGBA64(x, y, color.RGBA64{
		uint16(ftou(f[0])), uint16(ftou(f[1])), uint16(ftou(f[2])), uint16(ftou(f[3])),
	})
}

// refScaleNRGBA filters a straight 8-bit source with the given options and
// returns the premultiplied, encoded result for the Src operator onto a
// transparent destination.
func refScaleNRGBA(src *image.NRGBA, sr image.Rectangle, dw, dh int, q *Kernel, opts *Options) *image.RGBA64 {
	cols := refWeights(sr.Dx(), dw, q)
	rows := refWeights(sr.Dy(), dh, q)
	premul := opts.Alpha == AlphaPremultiplied
	dst := image.NewRGBA64(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			var pr, pg, pb, pa float64
			for _, rc := range rows[y] {
				for _, cc := range cols[x] {
					w := rc.w * cc.w
					c := src.NRGBAAt(sr.Min.X+cc.coord, sr.Min.Y+rc.coord)
					r := float64(c.R) / 0xff
					g := float64(c.G) / 0xff
					b := float64(c.B) / 0xff
					a := float64(c.A) / 0xff
					if opts.Transfer == TransferLinear {
						r, g, b = decodeEncoded(r), decodeEncoded(g), decodeEncoded(b)
					}
					if premul {
						r, g, b = r*a, g*a, b*a
					}
					pr += w * r
					pg += w * g
					pb += w * b
					pa += w * a
				}
			}
			setRGBA64FromFloats(dst, x, y, refFinish(pr, pg, pb, pa, premul, opts.Transfer))
		}
	}
	return dst
}

func makeTestNRGBA(w, h int, seed uint32) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	s := seed
	next := func() uint8 {
		s = s*1664525 + 1013904223
		return uint8(s >> 24)
	}
	for i := 0; i < len(m.Pix); i += 4 {
		m.Pix[i+0] = next()
		m.Pix[i+1] = next()
		m.Pix[i+2] = next()
		m.Pix[i+3] = next()
	}
	// Include some fully transparent and fully opaque pixels.
	for x := 0; x < w; x++ {
		m.SetNRGBA(x, 0, color.NRGBA{uint8(x * 37), uint8(x * 91), uint8(x * 53), 0x00})
		m.SetNRGBA(x, h-1, color.NRGBA{uint8(x * 17), uint8(x * 29), uint8(x * 43), 0xff})
	}
	return m
}

// TestLinearScaleMatchesReference compares Scale in every transfer/alpha
// combination against the independent float reference, for several kernels,
// native and non-zero source rectangles, and both up- and down-scaling.
func TestLinearScaleMatchesReference(t *testing.T) {
	kernels := map[string]*Kernel{"box": box, "bilinear": BiLinear, "catmullrom": CatmullRom}
	cases := []struct {
		name   string
		sr     image.Rectangle
		dw, dh int
	}{
		{"down", image.Rect(0, 0, 12, 10), 5, 4},
		{"up", image.Rect(0, 0, 5, 4), 11, 9},
		{"offset", image.Rect(2, 3, 14, 13), 6, 5},
		{"same", image.Rect(0, 0, 3, 3), 3, 3},
		{"thin", image.Rect(1, 1, 15, 3), 7, 2},
	}
	for kname, q := range kernels {
		for _, opts := range []*Options{
			{Transfer: TransferLinear},
			{Alpha: AlphaStraight},
			{Transfer: TransferLinear, Alpha: AlphaStraight},
			{Transfer: TransferSRGB, Alpha: AlphaPremultiplied},
		} {
			for _, c := range cases {
				src := makeTestNRGBA(16, 16, 12345)
				want := refScaleNRGBA(src, c.sr, c.dw, c.dh, q, opts)
				got := image.NewRGBA64(image.Rect(0, 0, c.dw, c.dh))
				q.Scale(got, got.Bounds(), src, c.sr, Src, opts)
				if d := maxAbsDiff(got, want); d > 2*oneQuantum {
					t.Errorf("kernel=%s opts=%+v case=%s: max diff %v", kname, *opts, c.name, d)
				}
			}
		}
	}
}

// TestLinearDefaultUnchanged asserts that the new options do not perturb the
// default path, at the byte level, for every Kernel. This is the compatibility
// guarantee: a nil *Options and the zero-valued fields must agree exactly.
func TestLinearDefaultUnchanged(t *testing.T) {
	src := makeTestNRGBA(16, 16, 999)
	base := image.NewRGBA(image.Rect(0, 0, 9, 7))
	fillPixForTest(base)
	m := transformMatrix(0.75, 3, 2)
	for _, q := range []*Kernel{BiLinear, CatmullRom, box} {
		for _, op := range []Op{Src, Over} {
			run := func(opts *Options) []byte {
				d := image.NewRGBA(base.Rect)
				copy(d.Pix, base.Pix)
				q.Scale(d, d.Bounds(), src, src.Bounds(), op, opts)
				return append([]byte(nil), d.Pix...)
			}
			a := run(nil)
			b := run(&Options{})
			c := run(&Options{Transfer: TransferSRGB, Alpha: AlphaPremultiplied})
			if !bytes.Equal(a, b) || !bytes.Equal(a, c) {
				t.Errorf("Scale default output changed for kernel Support=%v op=%v", q.Support, op)
			}

			runT := func(opts *Options) []byte {
				d := image.NewRGBA(base.Rect)
				copy(d.Pix, base.Pix)
				q.Transform(d, m, src, src.Bounds(), op, opts)
				return append([]byte(nil), d.Pix...)
			}
			ta := runT(nil)
			tb := runT(&Options{Transfer: TransferSRGB, Alpha: AlphaPremultiplied})
			if !bytes.Equal(ta, tb) {
				t.Errorf("Transform default output changed for kernel Support=%v op=%v", q.Support, op)
			}
		}
	}
}

func fillPixForTest(m *image.RGBA) {
	for i := range m.Pix {
		m.Pix[i] = uint8(i*7 + 13)
	}
	// Keep it a legal premultiplied image.
	for i := 0; i < len(m.Pix); i += 4 {
		for j := 0; j < 3; j++ {
			if m.Pix[i+j] > m.Pix[i+3] {
				m.Pix[i+j] = m.Pix[i+3]
			}
		}
	}
}

// TestLinearImageTypes checks that RGBA, NRGBA, RGBA64, Gray and generic
// (interface-only) images all produce the same result for equivalent content.
// The wrappers hide the concrete fast paths so the generic branch is exercised.
func TestLinearImageTypes(t *testing.T) {
	const w, h, dw, dh = 8, 6, 4, 3
	opts := &Options{Transfer: TransferLinear}
	scale := func(src image.Image) *image.RGBA64 {
		dst := image.NewRGBA64(image.Rect(0, 0, dw, dh))
		BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, opts)
		return dst
	}

	// Opaque grayscale content keeps the storage types comparable.
	gray := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gray.SetGray(x, y, color.Gray{uint8((x*17 + y*31) & 0xff)})
		}
	}
	nrgba := image.NewNRGBA(gray.Rect)
	rgba := image.NewRGBA(gray.Rect)
	rgba64 := image.NewRGBA64(gray.Rect)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := gray.GrayAt(x, y).Y
			nrgba.SetNRGBA(x, y, color.NRGBA{v, v, v, 0xff})
			rgba.SetRGBA(x, y, color.RGBA{v, v, v, 0xff})
			rgba64.SetRGBA64(x, y, color.RGBA64{
				uint16(v) * 0x101, uint16(v) * 0x101, uint16(v) * 0x101, 0xffff,
			})
		}
	}

	want := scale(gray)
	for name, src := range map[string]image.Image{
		"nrgba":     nrgba,
		"rgba":      rgba,
		"rgba64":    rgba64,
		"generic":   srcWrapper{nrgba},
		"generic64": srcWrapper{rgba64},
	} {
		if d := maxAbsDiff(scale(src), want); d > 2*oneQuantum {
			t.Errorf("%s: max diff %v from Gray", name, d)
		}
	}

	// Non-zero Bounds: the same pixels with an offset rectangle must give the
	// same output as the zero-origin version.
	off := image.NewNRGBA(image.Rect(5, 7, 5+w, 7+h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off.SetNRGBA(5+x, 7+y, nrgba.NRGBAAt(x, y))
		}
	}
	dst := image.NewRGBA64(image.Rect(3, 4, 3+dw, 4+dh))
	BiLinear.Scale(dst, dst.Bounds(), off, off.Bounds(), Src, opts)
	ref := scale(nrgba)
	if d := maxAbsDiff(dst, ref); d > 2*oneQuantum {
		t.Errorf("non-zero bounds: max diff %v", d)
	}
}

// TestTransparentEndpoints checks that fully transparent source pixels cannot
// tint their neighbors in AlphaPremultiplied mode, and that they can in
// AlphaStraight mode (which is the documented reason to choose it).
func TestTransparentEndpoints(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	src.SetNRGBA(0, 0, color.NRGBA{0xff, 0x00, 0x00, 0x00}) // transparent "red"
	src.SetNRGBA(1, 0, color.NRGBA{0x00, 0x00, 0xff, 0xff}) // opaque blue

	run := func(opts *Options) color.RGBA64 {
		dst := image.NewRGBA64(image.Rect(0, 0, 1, 1))
		box.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, opts)
		return dst.RGBA64At(0, 0)
	}

	premul := run(&Options{Transfer: TransferLinear})
	if premul.R != 0 {
		t.Errorf("premultiplied: red leaked from transparent pixel: %v", premul)
	}
	if a := float64(premul.A) / 0xffff; math.Abs(a-0.5) > oneQuantum {
		t.Errorf("premultiplied: alpha = %v, want 0.5", a)
	}

	straight := run(&Options{Transfer: TransferLinear, Alpha: AlphaStraight})
	if straight.R == 0 {
		t.Errorf("straight: expected the transparent pixel's color to contribute, got %v", straight)
	}

	// A transparent copy of any color must not change the premultiplied result
	// of an opaque sample.
	for _, c := range []color.NRGBA{{0xff, 0xff, 0xff, 0x00}, {0x00, 0xff, 0x00, 0x00}} {
		s2 := image.NewNRGBA(image.Rect(0, 0, 2, 1))
		s2.SetNRGBA(0, 0, c)
		s2.SetNRGBA(1, 0, color.NRGBA{0x00, 0x00, 0xff, 0xff})
		dst := image.NewRGBA64(image.Rect(0, 0, 1, 1))
		box.Scale(dst, dst.Bounds(), s2, s2.Bounds(), Src, &Options{Transfer: TransferLinear})
		if dst.RGBA64At(0, 0) != premul {
			t.Errorf("transparent pixel color %v affected premultiplied result: got %v want %v",
				c, dst.RGBA64At(0, 0), premul)
		}
	}
}

// TestLinearOverSrcOrder checks the composition order: filtering happens
// first, in linear light, and Porter-Duff composition happens afterwards on
// premultiplied sRGB values, exactly as for the default path.
func TestLinearOverSrcOrder(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			src.SetNRGBA(x, y, color.NRGBA{uint8(x * 0x40), uint8(y * 0x40), 0x80, 0x80})
		}
	}
	opts := &Options{Transfer: TransferLinear}
	const dw, dh = 2, 2

	// Src onto transparent gives the filtered premultiplied source directly.
	p := image.NewRGBA64(image.Rect(0, 0, dw, dh))
	BiLinear.Scale(p, p.Bounds(), src, src.Bounds(), Src, opts)

	// Src onto a non-transparent destination must ignore the destination.
	dirty := image.NewRGBA64(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			dirty.SetRGBA64(x, y, color.RGBA64{0x1234, 0x5678, 0x9abc, 0xffff})
		}
	}
	BiLinear.Scale(dirty, dirty.Bounds(), src, src.Bounds(), Src, opts)
	if d := maxAbsDiff(dirty, p); d > 0 {
		t.Errorf("Src did not ignore destination: max diff %v", d)
	}

	// Over onto opaque dst: result = src + dst*(1 - srcA).
	dst := image.NewRGBA64(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			dst.SetRGBA64(x, y, color.RGBA64{0x4000, 0x8000, 0xc000, 0xffff})
		}
	}
	BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), Over, opts)
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			s := float64Of(p.RGBA64At(x, y))
			q := float64Of(color.RGBA64{0x4000, 0x8000, 0xc000, 0xffff})
			want := [4]float64{
				clamp01(s[0] + q[0]*(1-s[3])),
				clamp01(s[1] + q[1]*(1-s[3])),
				clamp01(s[2] + q[2]*(1-s[3])),
				clamp01(s[3] + q[3]*(1-s[3])),
			}
			got := float64Of(dst.RGBA64At(x, y))
			for i := 0; i < 4; i++ {
				if d := math.Abs(got[i] - want[i]); d > 2*oneQuantum {
					t.Errorf("Over at (%d,%d) chan %d: got %v want %v", x, y, i, got[i], want[i])
				}
			}
		}
	}
}

// refWeightsAt is the Transform analogue of refWeights: given the continuous
// source-space position of a destination pixel center, it returns the clamped,
// normalized weights around that position.
func refWeightsAt(center, halfWidth, argScale float64, lo, hi int, q *Kernel) []refContrib {
	i := int(math.Floor(center - halfWidth))
	if i < lo {
		i = lo
	}
	j := int(math.Ceil(center + halfWidth))
	if j > hi {
		j = hi
	}
	var cs []refContrib
	total := 0.0
	for k := i; k < j; k++ {
		tt := math.Abs(center-float64(k)) * argScale
		if tt >= q.Support {
			continue
		}
		w := q.At(tt)
		if w == 0 || math.IsNaN(w) || math.IsInf(w, 0) {
			continue
		}
		cs = append(cs, refContrib{k, w})
		total += w
	}
	if total != 0 {
		for i := range cs {
			cs[i].w /= total
		}
	}
	return cs
}

// TestLinearTranslucentColorDifference downsamples a pair of translucent
// color blocks and compares both modes against a float reference in linear
// light. It uses a CIELAB-style color difference so that it tests perceived
// color, not just per-channel error, and it also checks that alpha is filtered
// independently of color.
func TestLinearTranslucentColorDifference(t *testing.T) {
	// A left half of 50%-alpha red and a right half of 50%-alpha blue. Equal
	// alpha means the premultiplied and straight results agree on alpha; the
	// question is what hue the boundary takes.
	const w, h = 8, 4
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < w/2 {
				src.SetNRGBA(x, y, color.NRGBA{0xff, 0x00, 0x00, 0x80})
			} else {
				src.SetNRGBA(x, y, color.NRGBA{0x00, 0x00, 0xff, 0x80})
			}
		}
	}
	const dw, dh = 4, 2

	dst := func(opts *Options) *image.RGBA64 {
		d := image.NewRGBA64(image.Rect(0, 0, dw, dh))
		BiLinear.Scale(d, d.Bounds(), src, src.Bounds(), Src, opts)
		return d
	}
	enc := dst(nil)
	lin := dst(&Options{Transfer: TransferLinear})
	want := refScaleNRGBA(src, src.Bounds(), dw, dh, BiLinear, &Options{Transfer: TransferLinear})

	// Linear-RGB color difference between two encoded colors: decode, then
	// compare componentwise (a stand-in for ΔE that is sensitive to the same
	// darkening error).
	colorDiff := func(a, b color.RGBA64) float64 {
		fa, fb := float64Of(a), float64Of(b)
		worst := 0.0
		for i := 0; i < 3; i++ {
			la, lb := decodeEncoded(fa[i]), decodeEncoded(fb[i])
			if d := math.Abs(la - lb); d > worst {
				worst = d
			}
		}
		return worst
	}

	var diffEnc, diffLin float64
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			g := want.RGBA64At(x, y)
			diffEnc += colorDiff(enc.RGBA64At(x, y), g)
			diffLin += colorDiff(lin.RGBA64At(x, y), g)

			// Alpha must be filtered the same way regardless of transfer.
			if enc.RGBA64At(x, y).A != lin.RGBA64At(x, y).A {
				t.Errorf("alpha differs between modes at (%d,%d): sRGB %#x, linear %#x",
					x, y, enc.RGBA64At(x, y).A, lin.RGBA64At(x, y).A)
			}
			// The premultiplied alpha of a uniform 50% block is 0x8080.
			if a := lin.RGBA64At(x, y).A; a != 0x8080 {
				t.Errorf("alpha at (%d,%d) = %#x, want 0x8080", x, y, a)
			}
		}
	}
	// The linear path must be substantially closer to the linear reference.
	if !(diffLin < diffEnc/4) {
		t.Errorf("linear color difference %v not much smaller than sRGB %v", diffLin, diffEnc)
	}
	if diffLin > 0.01 {
		t.Errorf("linear color difference from reference too large: %v", diffLin)
	}
}

// TestLinearStraightAlphaBlocks checks AlphaStraight on a fully transparent
// block next to an opaque one: the straight mode must carry the transparent
// block's color into the filtered result, while the premultiplied mode must
// not, and both must agree on alpha.
func TestLinearStraightAlphaBlocks(t *testing.T) {
	const w, h = 8, 4
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < w/2 {
				src.SetNRGBA(x, y, color.NRGBA{0xff, 0xff, 0x00, 0x00}) // transparent yellow
			} else {
				src.SetNRGBA(x, y, color.NRGBA{0x00, 0x00, 0xff, 0xff}) // opaque blue
			}
		}
	}
	const dw, dh = 4, 2
	run := func(opts *Options) *image.RGBA64 {
		d := image.NewRGBA64(image.Rect(0, 0, dw, dh))
		BiLinear.Scale(d, d.Bounds(), src, src.Bounds(), Src, opts)
		return d
	}
	premul := run(&Options{Transfer: TransferLinear})
	straight := run(&Options{Transfer: TransferLinear, Alpha: AlphaStraight})

	// Alpha is identical in both modes.
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			if premul.RGBA64At(x, y).A != straight.RGBA64At(x, y).A {
				t.Fatalf("alpha differs at (%d,%d)", x, y)
			}
		}
	}
	// At the boundary, premultiplied must have no red/yellow contribution from
	// the transparent block (R == 0 since the only opaque color is blue).
	if c := premul.RGBA64At(1, 0); c.R != 0 {
		t.Errorf("premultiplied boundary has red from transparent block: %v", c)
	}
	// Straight must pick up the yellow block's red.
	if c := straight.RGBA64At(1, 0); c.R == 0 {
		t.Errorf("straight boundary ignored transparent block's color: %v", c)
	}
}

// that recomputes the inverse mapping and kernel weights independently. It
// covers rotation plus scaling, a non-zero source rectangle, both operators,
// and every transfer/alpha combination. Only pixels whose continuous source
// position is comfortably inside sr are compared, so that the production and
// reference in-bounds tests cannot disagree at the boundary.
func TestLinearTransformMatchesReference(t *testing.T) {
	src := makeTestNRGBA(32, 32, 424242)
	m := transformMatrix(1.5, 8, 8) // scale and rotate by 30 degrees
	sr := image.Rect(4, 4, 28, 28)
	const dw, dh = 48, 48

	for _, q := range []*Kernel{box, BiLinear, CatmullRom} {
		for _, opts := range []*Options{
			{Transfer: TransferLinear},
			{Alpha: AlphaStraight},
			{Transfer: TransferLinear, Alpha: AlphaStraight},
		} {
			got := image.NewRGBA64(image.Rect(0, 0, dw, dh))
			q.Transform(got, m, src, sr, Src, opts)

			inv := invert(&m)
			premul := opts.Alpha == AlphaPremultiplied
			xscale := math.Max(abs(inv[0]), abs(inv[1]))
			yscale := math.Max(abs(inv[3]), abs(inv[4]))
			xHalf, xArg := q.Support, 1.0
			if xscale > 1 {
				xHalf, xArg = xHalf*xscale, 1/xscale
			}
			yHalf, yArg := q.Support, 1.0
			if yscale > 1 {
				yHalf, yArg = yHalf*yscale, 1/yscale
			}
			// Only compare pixels whose kernel footprint lies entirely inside
			// sr (in source pixel units, with a margin so the clamped weight
			// ranges are identical to the unclamped ones).
			margin := xHalf
			if yHalf > margin {
				margin = yHalf
			}
			margin += 1

			for dy := 0; dy < dh; dy++ {
				for dx := 0; dx < dw; dx++ {
					dxf, dyf := float64(dx)+0.5, float64(dy)+0.5
					cx := inv[0]*dxf + inv[1]*dyf + inv[2] - 0.5
					cy := inv[3]*dxf + inv[4]*dyf + inv[5] - 0.5
					if cx < float64(sr.Min.X)+margin || cx > float64(sr.Max.X-1)-margin ||
						cy < float64(sr.Min.Y)+margin || cy > float64(sr.Max.Y-1)-margin {
						continue
					}
					xcs := refWeightsAt(cx, xHalf, xArg, sr.Min.X, sr.Max.X, q)
					ycs := refWeightsAt(cy, yHalf, yArg, sr.Min.Y, sr.Max.Y, q)
					var pr, pg, pb, pa float64
					for _, rc := range ycs {
						for _, cc := range xcs {
							w := rc.w * cc.w
							c := src.NRGBAAt(cc.coord, rc.coord)
							r := float64(c.R) / 0xff
							g := float64(c.G) / 0xff
							b := float64(c.B) / 0xff
							a := float64(c.A) / 0xff
							if opts.Transfer == TransferLinear {
								r, g, b = decodeEncoded(r), decodeEncoded(g), decodeEncoded(b)
							}
							if premul {
								r, g, b = r*a, g*a, b*a
							}
							pr += w * r
							pg += w * g
							pb += w * b
							pa += w * a
						}
					}
					want := refFinish(pr, pg, pb, pa, premul, opts.Transfer)
					g := float64Of(got.RGBA64At(dx, dy))
					for i := 0; i < 4; i++ {
						if d := math.Abs(g[i] - want[i]); d > 2*oneQuantum {
							t.Errorf("kernel Support=%v opts=%+v at (%d,%d) chan %d: got %v want %v",
								q.Support, *opts, dx, dy, i, g[i], want[i])
							break
						}
					}
				}
			}
		}
	}
}

// TestLinearNegativeAndOutOfRange checks that CatmullRom's negative lobes and
// out-of-range intermediate values never yield an invalid premultiplied color
// and never yield non-finite output.
func TestLinearNegativeAndOutOfRange(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if (x+y)&1 == 0 {
				src.SetNRGBA(x, y, color.NRGBA{0xff, 0xff, 0xff, 0xff})
			} else {
				src.SetNRGBA(x, y, color.NRGBA{0x00, 0x00, 0x00, 0xff})
			}
		}
	}
	// A source with invalid premultiplied channels (R > A) exercises the
	// unassociation clamp.
	bad := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for i := 0; i < len(bad.Pix); i += 4 {
		bad.Pix[i+0], bad.Pix[i+1], bad.Pix[i+2], bad.Pix[i+3] = 0xff, 0x80, 0x00, 0x40
	}

	check := func(name string, m *image.RGBA64) {
		for y := m.Rect.Min.Y; y < m.Rect.Max.Y; y++ {
			for x := m.Rect.Min.X; x < m.Rect.Max.X; x++ {
				f := float64Of(m.RGBA64At(x, y))
				for i := 0; i < 4; i++ {
					if math.IsNaN(f[i]) || math.IsInf(f[i], 0) {
						t.Fatalf("%s: non-finite at (%d,%d)", name, x, y)
					}
				}
				if f[0] > f[3]+oneQuantum || f[1] > f[3]+oneQuantum || f[2] > f[3]+oneQuantum {
					t.Fatalf("%s: invalid premultiplied color at (%d,%d): %v", name, x, y, f)
				}
			}
		}
	}

	for _, opts := range []*Options{
		{Transfer: TransferLinear},
		{Transfer: TransferLinear, Alpha: AlphaStraight},
	} {
		for _, q := range []*Kernel{CatmullRom, box} {
			dst := image.NewRGBA64(image.Rect(0, 0, 11, 9))
			q.Scale(dst, dst.Bounds(), src, src.Bounds(), Over, opts)
			check("scale", dst)
			dst = image.NewRGBA64(image.Rect(0, 0, 11, 9))
			q.Transform(dst, transformMatrix(0.6, 2, 2), src, src.Bounds(), Src, opts)
			check("transform", dst)
			dst = image.NewRGBA64(image.Rect(0, 0, 6, 6))
			q.Scale(dst, dst.Bounds(), bad, bad.Bounds(), Src, opts)
			check("invalid-premul", dst)
		}
	}
}

// TestLinearNonFiniteInputs checks that NaN or infinite kernel weights and
// transform matrices produce stable, finite output instead of propagating NaN.
func TestLinearNonFiniteInputs(t *testing.T) {
	src := makeTestNRGBA(8, 8, 7)
	opts := &Options{Transfer: TransferLinear}

	nanKernel := &Kernel{1, func(t float64) float64 {
		if t < 0.5 {
			return math.NaN()
		}
		return math.Inf(1)
	}}
	dst := image.NewRGBA64(image.Rect(0, 0, 4, 4))
	nanKernel.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, opts)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			f := float64Of(dst.RGBA64At(x, y))
			for i := 0; i < 4; i++ {
				if math.IsNaN(f[i]) || math.IsInf(f[i], 0) {
					t.Fatalf("NaN kernel: non-finite at (%d,%d): %v", x, y, f)
				}
			}
		}
	}

	// Degenerate transform matrices: a singular matrix makes the inverse
	// non-finite. The destination must be left untouched, not corrupted.
	for _, m := range []f64.Aff3{
		{0, 0, 0, 0, 0, 0},
		{math.Inf(1), 0, 0, 0, 1, 0},
		{math.NaN(), 0, 0, 0, 1, 0},
	} {
		d := image.NewRGBA64(image.Rect(0, 0, 4, 4))
		for i := range d.Pix {
			d.Pix[i] = 0x5a
		}
		before := append([]byte(nil), d.Pix...)
		BiLinear.Transform(d, m, src, src.Bounds(), Src, opts)
		if !bytes.Equal(d.Pix, before) {
			t.Errorf("degenerate matrix %v modified destination", m)
		}
	}
}

// TestLinearDegenerateSize exercises empty images and extreme scale factors.
func TestLinearDegenerateSize(t *testing.T) {
	opts := &Options{Transfer: TransferLinear}
	src := makeTestNRGBA(8, 8, 1)

	// Empty destination or source rectangle: return without panic and without
	// writing anything.
	dst := image.NewRGBA64(image.Rect(0, 0, 4, 4))
	for i := range dst.Pix {
		dst.Pix[i] = 0x11
	}
	before := append([]byte(nil), dst.Pix...)
	box.Scale(dst, image.Rect(0, 0, 0, 0), src, src.Bounds(), Src, opts)
	box.Scale(dst, dst.Bounds(), src, image.Rect(0, 0, 0, 0), Src, opts)
	if !bytes.Equal(dst.Pix, before) {
		t.Errorf("empty scale wrote to destination")
	}

	// A 1x1 destination from a large source, and a large destination from a
	// 1x1 source, must both be finite and fully specified.
	for _, q := range []*Kernel{box, BiLinear, CatmullRom} {
		d1 := image.NewRGBA64(image.Rect(0, 0, 1, 1))
		q.Scale(d1, d1.Bounds(), src, src.Bounds(), Src, opts)
		if f := float64Of(d1.RGBA64At(0, 0)); math.IsNaN(f[0]) || math.IsInf(f[0], 0) {
			t.Errorf("1x1 downscale produced non-finite output")
		}

		one := image.NewNRGBA(image.Rect(0, 0, 1, 1))
		one.SetNRGBA(0, 0, color.NRGBA{0x80, 0x40, 0x20, 0xc0})
		d2 := image.NewRGBA64(image.Rect(0, 0, 9, 7))
		q.Scale(d2, d2.Bounds(), one, one.Bounds(), Src, opts)
		for y := 0; y < 7; y++ {
			for x := 0; x < 9; x++ {
				f := float64Of(d2.RGBA64At(x, y))
				for i := 0; i < 4; i++ {
					if math.IsNaN(f[i]) || math.IsInf(f[i], 0) {
						t.Fatalf("up scale from 1x1: non-finite at (%d,%d)", x, y)
					}
				}
			}
		}
		// Upscaling a single color must reproduce that color.
		if d := math.Abs(float64(d2.RGBA64At(4, 3).A)/0xffff - float64(0xc0)/0xff); d > oneQuantum {
			t.Errorf("up scale from 1x1 changed alpha: got %v", float64(d2.RGBA64At(4, 3).A)/0xffff)
		}
	}

	// A 1x1 source rectangle inside a larger image with non-zero bounds.
	big := image.NewNRGBA(image.Rect(10, 20, 30, 40))
	big.SetNRGBA(15, 25, color.NRGBA{0x10, 0x20, 0x30, 0x40})
	d := image.NewRGBA64(image.Rect(0, 0, 3, 3))
	box.Scale(d, d.Bounds(), big, image.Rect(15, 25, 16, 26), Src, opts)
	if got := d.RGBA64At(1, 1).A; got != 0x4040 {
		t.Errorf("1x1 sr: got alpha %#x, want 0x4040", got)
	}

	// An enormous up-scale factor with a tiny source (the distribution spans
	// the whole source) and a tiny down-scale factor must stay finite.
	huge := image.NewRGBA64(image.Rect(0, 0, 2, 2))
	q := CatmullRom
	q.Scale(huge, huge.Bounds(), src, image.Rect(0, 0, 7, 7), Src, opts)
	if f := float64Of(huge.RGBA64At(0, 0)); math.IsNaN(f[0]) {
		t.Errorf("extreme downscale produced NaN")
	}
}

// TestLinearNoMutation asserts that filtering does not modify the source
// image, the destination outside the affected rectangle, or the Kernel.
func TestLinearNoMutation(t *testing.T) {
	src := makeTestNRGBA(12, 10, 555)
	srcPix := append([]byte(nil), src.Pix...)

	q := &Kernel{1, func(t float64) float64 { return 1 - t }}
	qSupport := q.Support
	qAt0 := q.At(0)
	opts := &Options{Transfer: TransferLinear}
	optsCopy := *opts

	dst := image.NewRGBA64(image.Rect(0, 0, 12, 12))
	dstPix := append([]byte(nil), dst.Pix...)
	dr := image.Rect(1, 1, 7, 6)
	q.Scale(dst, dr, src, image.Rect(2, 1, 10, 9), Src, opts)
	if !bytes.Equal(src.Pix, srcPix) {
		t.Errorf("Scale modified the source image")
	}
	for y := 0; y < 12; y++ {
		for x := 0; x < 12; x++ {
			if (image.Point{x, y}).In(dr) {
				continue
			}
			bi := (y-dst.Rect.Min.Y)*dst.Stride + (x-dst.Rect.Min.X)*8
			if !bytes.Equal(dst.Pix[bi:bi+8], dstPix[bi:bi+8]) {
				t.Errorf("Scale wrote outside dr at (%d,%d)", x, y)
			}
		}
	}
	if q.Support != qSupport || q.At(0) != qAt0 {
		t.Errorf("Kernel was modified")
	}
	if opts.Transfer != optsCopy.Transfer || opts.Alpha != optsCopy.Alpha {
		t.Errorf("Options were modified")
	}

	dst2 := image.NewRGBA64(image.Rect(0, 0, 12, 12))
	q.Transform(dst2, transformMatrix(0.8, 2, 2), src, src.Bounds(), Src, opts)
	if !bytes.Equal(src.Pix, srcPix) {
		t.Errorf("Transform modified the source image")
	}
}

// TestLinearConcurrentReuse runs one reusable Scaler from many goroutines and
// checks that results match a sequential run, i.e. the pooled temporary buffer
// is not shared incorrectly and no state is mutated.
func TestLinearConcurrentReuse(t *testing.T) {
	src := makeTestNRGBA(16, 14, 2024)
	opts := &Options{Transfer: TransferLinear}
	const dw, dh = 9, 8

	want := image.NewRGBA64(image.Rect(0, 0, dw, dh))
	CatmullRom.Scale(want, want.Bounds(), src, src.Bounds(), Over, opts)
	wantPix := append([]byte(nil), want.Pix...)

	scaler := CatmullRom.NewScaler(dw, dh, src.Bounds().Dx(), src.Bounds().Dy())
	var wg sync.WaitGroup
	errs := make(chan string, 32)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				dst := image.NewRGBA64(image.Rect(0, 0, dw, dh))
				scaler.Scale(dst, dst.Bounds(), src, src.Bounds(), Over, opts)
				if !bytes.Equal(dst.Pix, wantPix) {
					errs <- "concurrent result differed from sequential result"
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	if !bytes.Equal(src.Pix, makeTestNRGBA(16, 14, 2024).Pix) {
		t.Errorf("concurrent use modified the source")
	}
}

// TestLinearMasks checks that source and destination masks still apply when
// filtering in linear light: a zero source mask makes the source transparent,
// and a destination mask blends according to coverage.
func TestLinearMasks(t *testing.T) {
	src := image.NewNRGBA(image.Rect(0, 0, 4, 1))
	for x := 0; x < 4; x++ {
		src.SetNRGBA(x, 0, color.NRGBA{0xff, 0x00, 0x00, 0xff})
	}

	// A fully transparent source mask yields nothing.
	srcMask := image.NewAlpha(image.Rect(0, 0, 4, 1))
	dst := image.NewRGBA64(image.Rect(0, 0, 2, 1))
	BiLinear.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, &Options{
		Transfer: TransferLinear, SrcMask: srcMask,
	})
	for x := 0; x < 2; x++ {
		if c := dst.RGBA64At(x, 0); c.A != 0 {
			t.Errorf("zero src mask produced alpha %v at x=%d", c.A, x)
		}
	}

	// A destination mask blends with the old destination according to
	// coverage: mask 0xff replaces, mask 0 leaves the destination untouched.
	dstMask := image.NewAlpha(image.Rect(0, 0, 2, 1))
	dstMask.SetAlpha(0, 0, color.Alpha{0xff})
	dstMask.SetAlpha(1, 0, color.Alpha{0x00})
	green := color.RGBA64{0x0000, 0xffff, 0x0000, 0xffff}
	dst2 := image.NewRGBA64(image.Rect(0, 0, 2, 1))
	dst2.SetRGBA64(0, 0, green)
	dst2.SetRGBA64(1, 0, green)
	BiLinear.Scale(dst2, dst2.Bounds(), src, src.Bounds(), Src, &Options{
		Transfer: TransferLinear, DstMask: dstMask,
	})
	if c := dst2.RGBA64At(1, 0); c != green {
		t.Errorf("zero dst mask changed destination: %v", c)
	}
	if c := dst2.RGBA64At(0, 0); c.R != 0xffff || c.G != 0 || c.A != 0xffff {
		t.Errorf("opaque dst mask should replace with source: %v", c)
	}

	// Over with a zero destination mask must leave the destination untouched,
	// even where the source is opaque.
	dst3 := image.NewRGBA64(image.Rect(0, 0, 2, 1))
	dst3.SetRGBA64(0, 0, green)
	dst3.SetRGBA64(1, 0, green)
	zeroMask := image.NewAlpha(image.Rect(0, 0, 2, 1))
	BiLinear.Scale(dst3, dst3.Bounds(), src, src.Bounds(), Over, &Options{
		Transfer: TransferLinear, DstMask: zeroMask,
	})
	for x := 0; x < 2; x++ {
		if c := dst3.RGBA64At(x, 0); c != green {
			t.Errorf("Over with zero dst mask changed destination at x=%d: %v", x, c)
		}
	}
}

// TestLinearSourcesOutsideBounds checks that a source rectangle extending
// beyond the image is treated as transparent black outside the image rather
// than panicking or reading out-of-bounds memory.
func TestLinearSourcesOutsideBounds(t *testing.T) {
	src := makeTestNRGBA(4, 4, 3)
	for _, q := range []*Kernel{box, BiLinear, CatmullRom} {
		// Overlapping sr: exercises bounds checking inside the kernel footprint.
		dst := image.NewRGBA64(image.Rect(0, 0, 5, 4))
		q.Scale(dst, dst.Bounds(), src, image.Rect(-3, -2, 7, 6), Src, &Options{Transfer: TransferLinear})
		for y := 0; y < 4; y++ {
			for x := 0; x < 5; x++ {
				f := float64Of(dst.RGBA64At(x, y))
				for i := 0; i < 4; i++ {
					if math.IsNaN(f[i]) || math.IsInf(f[i], 0) {
						t.Fatalf("outside-bounds: non-finite at (%d,%d)", x, y)
					}
				}
			}
		}

		// Entirely disjoint sr: every sample is out of bounds, so the output
		// must be fully transparent.
		dst2 := image.NewRGBA64(image.Rect(0, 0, 3, 3))
		q.Scale(dst2, dst2.Bounds(), src, image.Rect(-10, -10, -6, -6), Src, &Options{Transfer: TransferLinear})
		for y := 0; y < 3; y++ {
			for x := 0; x < 3; x++ {
				if c := dst2.RGBA64At(x, y); c.A != 0 {
					t.Errorf("disjoint sr: got alpha %v at (%d,%d), want 0", c.A, x, y)
				}
			}
		}
	}
}

func BenchmarkLinearScaleDown(b *testing.B) {
	src := makeTestNRGBA(1024, 768, 1)
	dst := image.NewRGBA64(image.Rect(0, 0, 200, 150))
	opts := &Options{Transfer: TransferLinear}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, opts)
	}
}

func BenchmarkLinearScaleDownDefault(b *testing.B) {
	src := makeTestNRGBA(1024, 768, 1)
	dst := image.NewRGBA64(image.Rect(0, 0, 200, 150))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), Src, nil)
	}
}
