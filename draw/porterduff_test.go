// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"bytes"
	"image"
	"image/color"
	stddraw "image/draw"
	"sync"
	"testing"
)

// pdOps are the additional Porter-Duff operators implemented by this
// package, on top of the standard library's Over and Src.
var pdOps = []Op{SrcIn, SrcOut, Atop, Xor}

func pdMod(i, n int) int {
	i %= n
	if i < 0 {
		i += n
	}
	return i
}

// pdColor returns a deterministic alpha-premultiplied color for (x, y),
// covering transparent, extreme and intermediate alpha values.
func pdColor(x, y int) color.RGBA64 {
	alphas := [6]uint32{0x0000, 0x0001, 0x1234, 0x8000, 0xfffe, 0xffff}
	fracs := [5]uint32{0x0000, 0x4000, 0x8000, 0xc000, 0xffff}
	a := alphas[pdMod(x*7+y*3, len(alphas))]
	return color.RGBA64{
		R: uint16(a * fracs[pdMod(x*5+y*11, len(fracs))] / 0xffff),
		G: uint16(a * fracs[pdMod(x*3+y*7, len(fracs))] / 0xffff),
		B: uint16(a * fracs[pdMod(x*11+y*2, len(fracs))] / 0xffff),
		A: uint16(a),
	}
}

// pdMaskAlpha returns a deterministic mask alpha for (x, y).
func pdMaskAlpha(x, y int) uint8 {
	as := [5]uint8{0x00, 0x40, 0x80, 0xc0, 0xff}
	return as[pdMod(x*3+y*5, len(as))]
}

// pdRefBlend is an independent, float64-based reference implementation of
// the Porter-Duff composition “(s in mask) OP d”, used to check the
// production implementation. s and d are alpha-premultiplied colors and ma
// is the mask alpha (0xffff for a nil mask).
func pdRefBlend(s, d color.RGBA64, ma uint32, op Op) color.RGBA64 {
	const M = 65535.0
	fm := float64(ma) / M
	sa := float64(s.A) / M * fm
	da := float64(d.A) / M
	var fs, fd float64
	switch op {
	case SrcIn:
		fs, fd = da, 0
	case SrcOut:
		fs, fd = 1-da, 0
	case Atop:
		fs, fd = da, 1-sa
	case Xor:
		fs, fd = 1-da, 1-sa
	default:
		panic("bad op")
	}
	blend := func(sc, dc uint16) uint16 {
		s := float64(sc) / M * fm
		d := float64(dc) / M
		v := s*fs + d*fd
		if v > 1 {
			v = 1
		}
		return uint16(v*M + 0.5)
	}
	a := sa*fs + da*fd
	if a > 1 {
		a = 1
	}
	return color.RGBA64{
		R: blend(s.R, d.R),
		G: blend(s.G, d.G),
		B: blend(s.B, d.B),
		A: uint16(a*M + 0.5),
	}
}

func pdAlmostEqual(got, want color.RGBA64, tol uint32) bool {
	diff := func(a, b uint16) uint32 {
		if a > b {
			return uint32(a) - uint32(b)
		}
		return uint32(b) - uint32(a)
	}
	return diff(got.R, want.R) <= tol &&
		diff(got.G, want.G) <= tol &&
		diff(got.B, want.B) <= tol &&
		diff(got.A, want.A) <= tol
}

func pdRGBA64(c color.Color) color.RGBA64 {
	r, g, b, a := c.RGBA()
	return color.RGBA64{uint16(r), uint16(g), uint16(b), uint16(a)}
}

// pdClipRect is an independent re-implementation of the clipping that
// DrawMask performs, used to know which destination pixels may be
// affected.
func pdClipRect(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point) image.Rectangle {
	orig := r.Min
	r = r.Intersect(dst.Bounds())
	r = r.Intersect(src.Bounds().Add(orig.Sub(sp)))
	if mask != nil {
		r = r.Intersect(mask.Bounds().Add(orig.Sub(mp)))
	}
	return r
}

// TestPorterDuffOps checks the additional Porter-Duff operators against an
// independent float64 reference formula, over a matrix of destination,
// source and mask types, using images with non-zero bounds and
// semi-transparent, transparent and extreme-alpha pixels.
func TestPorterDuffOps(t *testing.T) {
	var (
		db = image.Rect(10, 20, 18, 27) // Destination bounds: non-zero Min.
		sb = image.Rect(-4, 3, 6, 12)   // Source bounds: non-zero Min.
		mb = image.Rect(0, 0, 4, 3)     // Mask bounds: smaller than r.
		r  = image.Rect(11, 21, 17, 26)
		// sp and mp place the source and mask rectangles partially
		// outside the source and mask bounds, exercising clipping.
		sp = image.Point{-5, 2}
		mp = image.Point{-1, -1}
	)

	fillRGBA64 := func(m *image.RGBA64) {
		for y := m.Bounds().Min.Y; y < m.Bounds().Max.Y; y++ {
			for x := m.Bounds().Min.X; x < m.Bounds().Max.X; x++ {
				m.SetRGBA64(x, y, pdColor(x, y))
			}
		}
	}
	fillRGBA := func(m *image.RGBA) {
		for y := m.Bounds().Min.Y; y < m.Bounds().Max.Y; y++ {
			for x := m.Bounds().Min.X; x < m.Bounds().Max.X; x++ {
				m.SetRGBA64(x, y, pdColor(x, y))
			}
		}
	}
	fillNRGBA := func(m *image.NRGBA) {
		for y := m.Bounds().Min.Y; y < m.Bounds().Max.Y; y++ {
			for x := m.Bounds().Min.X; x < m.Bounds().Max.X; x++ {
				c := pdColor(x, y)
				m.SetNRGBA(x, y, color.NRGBAModel.Convert(c).(color.NRGBA))
			}
		}
	}

	newRGBA := func() *image.RGBA { m := image.NewRGBA(db); fillRGBA(m); return m }
	newRGBA64 := func() *image.RGBA64 { m := image.NewRGBA64(db); fillRGBA64(m); return m }

	dsts := []struct {
		name string
		new  func() Image
		tol  uint32
	}{
		{"RGBA", func() Image { return newRGBA() }, 0x140},
		{"RGBA64", func() Image { return newRGBA64() }, 8},
		{"Wrapper", func() Image { return dstWrapper{newRGBA()} }, 0x140},
	}

	srcs := []struct {
		name string
		new  func() image.Image
	}{
		{"RGBA", func() image.Image {
			m := image.NewRGBA(sb)
			fillRGBA(m)
			return m
		}},
		{"NRGBA", func() image.Image {
			m := image.NewNRGBA(sb)
			fillNRGBA(m)
			return m
		}},
		{"RGBA64", func() image.Image {
			m := image.NewRGBA64(sb)
			fillRGBA64(m)
			return m
		}},
		{"Uniform", func() image.Image {
			return image.NewUniform(color.RGBA64{0x6000, 0xa000, 0x2000, 0xc000})
		}},
		{"Wrapper", func() image.Image {
			m := image.NewRGBA(sb)
			fillRGBA(m)
			return srcWrapper{m}
		}},
	}

	newAlphaMask := func() *image.Alpha {
		m := image.NewAlpha(mb)
		for y := mb.Min.Y; y < mb.Max.Y; y++ {
			for x := mb.Min.X; x < mb.Max.X; x++ {
				m.SetAlpha(x, y, color.Alpha{pdMaskAlpha(x, y)})
			}
		}
		return m
	}
	masks := []struct {
		name string
		new  func() image.Image
	}{
		{"Nil", func() image.Image { return nil }},
		{"Alpha", func() image.Image { return newAlphaMask() }},
		{"Wrapper", func() image.Image { return srcWrapper{newAlphaMask()} }},
	}

	for _, op := range pdOps {
		for _, dstf := range dsts {
			for _, srcf := range srcs {
				for _, maskf := range masks {
					dst := dstf.new()
					before := dstf.new()
					src := srcf.new()
					mask := maskf.new()

					DrawMask(dst, r, src, sp, mask, mp, op)

					cr := pdClipRect(dst, r, src, sp, mask, mp)
					for y := db.Min.Y; y < db.Max.Y; y++ {
						for x := db.Min.X; x < db.Max.X; x++ {
							p := image.Point{x, y}
							got := pdRGBA64(dst.At(x, y))
							b := pdRGBA64(before.At(x, y))
							if !p.In(cr) {
								if got != b {
									t.Errorf("op=%v dst=%s src=%s mask=%s: pixel %v outside clipped rect changed: got %v, was %v",
										op, dstf.name, srcf.name, maskf.name, p, got, b)
								}
								continue
							}
							s := pdRGBA64(src.At(sp.X+x-r.Min.X, sp.Y+y-r.Min.Y))
							ma := uint32(0xffff)
							if mask != nil {
								ma = uint32(pdRGBA64(mask.At(mp.X+x-r.Min.X, mp.Y+y-r.Min.Y)).A)
							}
							want := pdRefBlend(s, b, ma, op)
							if !pdAlmostEqual(got, want, dstf.tol) {
								t.Errorf("op=%v dst=%s src=%s mask=%s: pixel %v: got %v, want %v (src %v, dst %v, mask alpha %#x)",
									op, dstf.name, srcf.name, maskf.name, p, got, want, s, b, ma)
							}
						}
					}
				}
			}
		}
	}
}

// TestPorterDuffNRGBADst checks the operators on a non-premultiplied
// *image.NRGBA destination, comparing in non-premultiplied space.
func TestPorterDuffNRGBADst(t *testing.T) {
	db := image.Rect(3, 4, 11, 12)
	sb := image.Rect(0, 0, 8, 8)
	r := db
	sp := image.Point{}

	nrgbaColors := []color.NRGBA{
		{0xff, 0x00, 0x00, 0xff},
		{0x00, 0xff, 0x80, 0x80},
		{0x40, 0x40, 0x40, 0xff},
		{0x00, 0x00, 0x00, 0x00},
	}
	newDst := func() *image.NRGBA {
		m := image.NewNRGBA(db)
		for y := db.Min.Y; y < db.Max.Y; y++ {
			for x := db.Min.X; x < db.Max.X; x++ {
				m.SetNRGBA(x, y, nrgbaColors[pdMod(x+y*3, len(nrgbaColors))])
			}
		}
		return m
	}
	src := image.NewRGBA(sb)
	for y := sb.Min.Y; y < sb.Max.Y; y++ {
		for x := sb.Min.X; x < sb.Max.X; x++ {
			src.SetRGBA64(x, y, pdColor(x+2, y+5))
		}
	}

	for _, op := range pdOps {
		dst := newDst()
		before := newDst()
		DrawMask(dst, r, src, sp, nil, image.Point{}, op)
		for y := db.Min.Y; y < db.Max.Y; y++ {
			for x := db.Min.X; x < db.Max.X; x++ {
				s := pdRGBA64(src.At(sp.X+x-r.Min.X, sp.Y+y-r.Min.Y))
				b := pdRGBA64(before.At(x, y))
				want := color.NRGBA64Model.Convert(pdRefBlend(s, b, 0xffff, op)).(color.NRGBA64)
				got := color.NRGBA64Model.Convert(dst.At(x, y)).(color.NRGBA64)
				// Converting to and from non-premultiplied colors
				// loses precision at low alpha, so the tolerance is
				// looser than for premultiplied destinations.
				if !pdAlmostEqual(color.RGBA64{uint16(got.R), uint16(got.G), uint16(got.B), uint16(got.A)},
					color.RGBA64{uint16(want.R), uint16(want.G), uint16(want.B), uint16(want.A)}, 0x500) {
					t.Errorf("op=%v: pixel (%d, %d): got %v, want %v", op, x, y, got, want)
				}
			}
		}
	}
}

// TestPorterDuffAliasing checks that drawing from a destination onto
// itself, with overlapping source and destination rectangles, gives the
// same result as drawing from a copy of the source.
func TestPorterDuffAliasing(t *testing.T) {
	b := image.Rect(0, 0, 20, 20)
	r := image.Rect(5, 5, 15, 15)
	sps := []image.Point{
		{3, 3}, // Up and left: requires backward iteration.
		{3, 8}, // Left only.
		{5, 2}, // Up only.
		{7, 7}, // Down and right: forward iteration.
		{5, 5}, // Identical rectangles.
	}
	newBase := func() *image.RGBA {
		m := image.NewRGBA(b)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				m.SetRGBA64(x, y, pdColor(x, y))
			}
		}
		return m
	}

	for _, op := range pdOps {
		for _, sp := range sps {
			got := newBase()
			Draw(got, r, got, sp, op)

			srcCopy := newBase()
			want := newBase()
			Draw(want, r, srcCopy, sp, op)

			if !bytes.Equal(got.Pix, want.Pix) {
				t.Errorf("op=%v sp=%v: aliased draw differs from draw from a copy", op, sp)
			}
		}
	}
}

// TestPorterDuffEdgeCases checks empty and out-of-bounds rectangles, huge
// rectangles, unknown operators and zero-alpha masks.
func TestPorterDuffEdgeCases(t *testing.T) {
	db := image.Rect(0, 0, 16, 16)
	newDst := func() *image.RGBA {
		m := image.NewRGBA(db)
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				m.SetRGBA64(x, y, pdColor(x, y))
			}
		}
		return m
	}
	src := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			src.SetRGBA64(x, y, pdColor(x+1, y+2))
		}
	}
	unchanged := func(dst, before *image.RGBA) bool {
		return bytes.Equal(dst.Pix, before.Pix)
	}

	for _, op := range pdOps {
		// An empty rectangle is a no-op.
		dst, before := newDst(), newDst()
		Draw(dst, image.Rect(4, 4, 4, 4), src, image.Point{}, op)
		if !unchanged(dst, before) {
			t.Errorf("op=%v: empty rectangle changed dst", op)
		}

		// A rectangle entirely outside the destination is a no-op.
		dst, before = newDst(), newDst()
		Draw(dst, image.Rect(100, 100, 200, 200), src, image.Point{}, op)
		if !unchanged(dst, before) {
			t.Errorf("op=%v: out-of-bounds rectangle changed dst", op)
		}

		// A huge rectangle is clipped to the destination and source
		// bounds, and gives the same result as the clipped rectangle.
		dst = newDst()
		huge := image.Rect(-1<<29, -1<<29, 1<<29, 1<<29)
		Draw(dst, huge, src, huge.Min, op)
		want := newDst()
		Draw(want, db, src, image.Point{}, op)
		if !bytes.Equal(dst.Pix, want.Pix) {
			t.Errorf("op=%v: huge rectangle differs from clipped rectangle", op)
		}

		// A zero-alpha mask leaves the destination unchanged for Atop
		// and Xor, and clears the affected rectangle for SrcIn and
		// SrcOut.
		dst = newDst()
		zero := image.NewAlpha(image.Rect(0, 0, 16, 16))
		DrawMask(dst, db, src, image.Point{}, zero, image.Point{}, op)
		switch op {
		case Atop, Xor:
			if !unchanged(dst, before) {
				t.Errorf("op=%v: zero mask changed dst", op)
			}
		case SrcIn, SrcOut:
			for i, v := range dst.Pix {
				if v != 0 {
					t.Errorf("op=%v: zero mask: dst.Pix[%d] = %d, want 0", op, i, v)
					break
				}
			}
		}
	}

	// Unknown operators behave exactly as they do in the standard library
	// (which, for an *image.RGBA destination, treats them like Src).
	for _, op := range []Op{Op(-1), Op(42)} {
		dst, want := newDst(), newDst()
		Draw(dst, db, src, image.Point{}, op)
		stddraw.Draw(want, db, src, image.Point{}, stddraw.Op(op))
		if !bytes.Equal(dst.Pix, want.Pix) {
			t.Errorf("op=%v: unknown op differs from standard library", int(op))
		}
	}

	// A large destination does not panic.
	dst := image.NewRGBA(image.Rect(0, 0, 1024, 1024))
	u := image.NewUniform(color.RGBA64{0x4000, 0x8000, 0x2000, 0xa000})
	for _, op := range pdOps {
		Draw(dst, dst.Bounds(), u, image.Point{}, op)
	}

	// A paletted destination (no alpha-premultiplied fast path) does not
	// panic.
	pal := image.NewPaletted(db, color.Palette{
		color.RGBA{0x00, 0x00, 0x00, 0x00},
		color.RGBA{0xff, 0x00, 0x00, 0xff},
		color.RGBA{0x00, 0x00, 0xff, 0xff},
	})
	for _, op := range pdOps {
		Draw(pal, db, src, image.Point{}, op)
	}
}

// TestPorterDuffOverSrcUnchanged checks that the Over and Src operators
// still produce exactly the standard library's results.
func TestPorterDuffOverSrcUnchanged(t *testing.T) {
	db := image.Rect(0, 0, 12, 12)
	sb := image.Rect(0, 0, 12, 12)
	r := image.Rect(2, 3, 10, 11)
	sp := image.Point{1, 2}
	mp := image.Point{0, 1}

	srcs := []image.Image{
		func() image.Image {
			m := image.NewRGBA(sb)
			for y := 0; y < 12; y++ {
				for x := 0; x < 12; x++ {
					m.SetRGBA64(x, y, pdColor(x, y))
				}
			}
			return m
		}(),
		func() image.Image {
			m := image.NewNRGBA(sb)
			for y := 0; y < 12; y++ {
				for x := 0; x < 12; x++ {
					m.SetNRGBA(x, y, color.NRGBAModel.Convert(pdColor(x, y)).(color.NRGBA))
				}
			}
			return m
		}(),
		image.NewUniform(color.RGBA{0x60, 0xa0, 0x20, 0xc0}),
	}
	masks := []image.Image{
		nil,
		func() image.Image {
			m := image.NewAlpha(sb)
			for y := 0; y < 12; y++ {
				for x := 0; x < 12; x++ {
					m.SetAlpha(x, y, color.Alpha{pdMaskAlpha(x, y)})
				}
			}
			return m
		}(),
	}

	for _, op := range []Op{Over, Src} {
		for _, src := range srcs {
			for _, mask := range masks {
				dst0 := image.NewRGBA(db)
				dst1 := image.NewRGBA(db)
				for i := range dst0.Pix {
					dst0.Pix[i] = uint8(i * 37)
					dst1.Pix[i] = uint8(i * 37)
				}
				DrawMask(dst0, r, src, sp, mask, mp, op)
				stddraw.DrawMask(dst1, r, src, sp, mask, mp, stddraw.Op(op))
				if !bytes.Equal(dst0.Pix, dst1.Pix) {
					t.Errorf("op=%v src=%T mask=%T: result differs from standard library", op, src, mask)
				}
			}
		}
	}
}

// TestPorterDuffScaleTransform checks that the Scaler and Transformer
// implementations support the additional operators, including destination
// masks, by comparing against a manual two-step composition.
func TestPorterDuffScaleTransform(t *testing.T) {
	db := image.Rect(0, 0, 40, 30)
	sb := image.Rect(0, 0, 24, 18)
	dr := image.Rect(4, 3, 36, 27)

	newDst := func() *image.RGBA {
		m := image.NewRGBA(db)
		for y := db.Min.Y; y < db.Max.Y; y++ {
			for x := db.Min.X; x < db.Max.X; x++ {
				m.SetRGBA64(x, y, pdColor(x, y))
			}
		}
		return m
	}
	src := image.NewRGBA(sb)
	for y := sb.Min.Y; y < sb.Max.Y; y++ {
		for x := sb.Min.X; x < sb.Max.X; x++ {
			src.SetRGBA64(x, y, pdColor(x+3, y+1))
		}
	}

	qs := []Interpolator{NearestNeighbor, ApproxBiLinear, CatmullRom}

	for _, op := range pdOps {
		for _, q := range qs {
			// A same-size scale with no destination mask is exactly a
			// DrawMask.
			dst0 := newDst()
			q.Scale(dst0, image.Rect(4, 3, 28, 21), src, sb, op, nil)
			dst1 := newDst()
			DrawMask(dst1, image.Rect(4, 3, 28, 21), src, sb.Min, nil, image.Point{}, op)
			if !bytes.Equal(dst0.Pix, dst1.Pix) {
				t.Errorf("op=%v q=%T: same-size scale differs from DrawMask", op, q)
			}

			// A scale with a rectangle destination mask equals scaling
			// into a temporary image and compositing the clipped
			// rectangle.
			maskRect := image.Rect(8, 8, 30, 20)
			dst0 = newDst()
			q.Scale(dst0, dr, src, sb, op, &Options{DstMask: maskRect})
			dst1 = newDst()
			adr := db.Intersect(dr).Intersect(maskRect)
			tmp := image.NewRGBA(adr)
			q.Scale(tmp, dr, src, sb, Src, nil)
			DrawMask(dst1, adr, tmp, adr.Min, nil, image.Point{}, op)
			if !bytes.Equal(dst0.Pix, dst1.Pix) {
				t.Errorf("op=%v q=%T: scale with rectangle DstMask differs from two-step composition", op, q)
			}

			// A scale with an *image.Alpha destination mask.
			alphaMask := image.NewAlpha(image.Rect(-2, -2, 38, 28))
			for y := alphaMask.Bounds().Min.Y; y < alphaMask.Bounds().Max.Y; y++ {
				for x := alphaMask.Bounds().Min.X; x < alphaMask.Bounds().Max.X; x++ {
					alphaMask.SetAlpha(x, y, color.Alpha{pdMaskAlpha(x, y)})
				}
			}
			dst0 = newDst()
			q.Scale(dst0, dr, src, sb, op, &Options{DstMask: alphaMask, DstMaskP: image.Point{-2, -2}})
			dst1 = newDst()
			adr = db.Intersect(dr)
			tmp = image.NewRGBA(adr)
			q.Scale(tmp, dr, src, sb, Src, nil)
			DrawMask(dst1, adr, tmp, adr.Min, alphaMask, image.Point{-2, -2}.Add(adr.Min), op)
			if !bytes.Equal(dst0.Pix, dst1.Pix) {
				t.Errorf("op=%v q=%T: scale with alpha DstMask differs from two-step composition", op, q)
			}

			// A transform equals transforming into a temporary image
			// and compositing.
			m := transformMatrix(1.5, 3, 2)
			dst0 = newDst()
			q.Transform(dst0, m, src, sb, op, nil)
			dst1 = newDst()
			adr = db.Intersect(transformRect(&m, &sb))
			tmp = image.NewRGBA(adr)
			q.Transform(tmp, m, src, sb, Src, nil)
			DrawMask(dst1, adr, tmp, adr.Min, nil, image.Point{}, op)
			if !bytes.Equal(dst0.Pix, dst1.Pix) {
				t.Errorf("op=%v q=%T: transform differs from two-step composition", op, q)
			}
		}
	}
}

// TestPorterDuffRace exercises the additional operators concurrently, for
// the race detector. Each goroutine draws to its own images; the
// interpolators are documented as safe for concurrent use.
func TestPorterDuffRace(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			dst := image.NewRGBA(image.Rect(0, 0, 32, 32))
			src := image.NewRGBA(image.Rect(0, 0, 32, 32))
			mask := image.NewAlpha(image.Rect(0, 0, 32, 32))
			for y := 0; y < 32; y++ {
				for x := 0; x < 32; x++ {
					dst.SetRGBA64(x, y, pdColor(x+i, y))
					src.SetRGBA64(x, y, pdColor(x, y+i))
					mask.SetAlpha(x, y, color.Alpha{pdMaskAlpha(x+i, y)})
				}
			}
			for _, op := range pdOps {
				DrawMask(dst, dst.Bounds(), src, image.Point{3, 5}, mask, image.Point{1, 2}, op)
				NearestNeighbor.Scale(dst, image.Rect(0, 0, 16, 16), src, src.Bounds(), op, nil)
				CatmullRom.Scale(dst, image.Rect(16, 16, 32, 32), src, src.Bounds(), op, nil)
				ApproxBiLinear.Transform(dst, transformMatrix(1.25, 2, 3), src, src.Bounds(), op, nil)
			}
		}(i)
	}
	wg.Wait()
}
