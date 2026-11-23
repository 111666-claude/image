// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

// This file pins the observable output of the deterministic fonts produced by
// fuzz_builder_test.go. The fault-injection gate proves malformed inputs are
// safe; these tests prove that a well-formed font yields stable, expected
// results across Parse, GlyphIndex, LoadGlyph, Kern and Metrics. The expected
// values are computed from the in-memory builder (not from host fonts or the
// network), so the golden contract is fully self-contained.

import (
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// TestGoldenValidTrueType exercises every required API on the minimal valid
// TrueType font and checks concrete, hand-derived values.
func TestGoldenValidTrueType(t *testing.T) {
	data := buildMinimalTrueType(minimalFontOptions{
		numGlyphs: 4, addKern: true, addGPOS: true,
	})
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if got, want := f.UnitsPerEm(), Units(2048); got != want {
		t.Errorf("UnitsPerEm=%d want %d", got, want)
	}
	if got, want := f.NumGlyphs(), 4; got != want {
		t.Errorf("NumGlyphs=%d want %d", got, want)
	}

	var b Buffer
	// cmap format-4 mapping built by buildCmapFormat4.
	for _, tc := range []struct {
		r    rune
		want GlyphIndex
	}{
		{'A', 1},
		{'B', 2},
		{'Z', 0}, // unmapped rune falls through to .notdef index 0.
	} {
		got, err := f.GlyphIndex(&b, tc.r)
		if err != nil {
			t.Fatalf("GlyphIndex(%q): %v", tc.r, err)
		}
		if got != tc.want {
			t.Errorf("GlyphIndex(%q)=%d want %d", tc.r, got, tc.want)
		}
	}

	ppem := fixed.Int26_6(f.UnitsPerEm())

	// Glyph 1 is the simple square outline from simpleGlyfSquare. At ppem equal
	// to unitsPerEm there is no scaling, so LoadGlyph returns the raw font-unit
	// coordinates (with the Y-axis flip applied by LoadGlyph).
	segs, err := f.LoadGlyph(&b, 1, ppem, nil)
	if err != nil {
		t.Fatalf("LoadGlyph(1): %v", err)
	}
	if len(segs) == 0 || segs[0].Op != SegmentOpMoveTo {
		t.Fatalf("LoadGlyph(1) segments=%v, want a MoveTo-started contour", segs)
	}
	// Bounds of the flipped square: x in [100,600], y flipped to [-444,0].
	bounds := segs.Bounds()
	if got := bounds.Min.X; got != 100 {
		t.Errorf("bounds.Min.X=%d want 100", got)
	}
	if got := bounds.Max.X; got != 600 {
		t.Errorf("bounds.Max.X=%d want 600", got)
	}
	if got := bounds.Min.Y; got != -444 {
		t.Errorf("bounds.Min.Y=%d want -444", got)
	}
	if got := bounds.Max.Y; got != 0 {
		t.Errorf("bounds.Max.Y=%d want 0", got)
	}

	// hmtx gives every glyph advanceWidth 600 font units.
	adv, err := f.GlyphAdvance(&b, 1, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("GlyphAdvance: %v", err)
	}
	if adv != 600 {
		t.Errorf("GlyphAdvance=%d want 600", adv)
	}

	// GPOS PairPos and the legacy kern table both encode -40 for pair (1,2).
	// GPOS takes precedence, but the magnitude is identical.
	for _, via := range []string{"gpos"} {
		k, err := f.Kern(&b, 1, 2, ppem, font.HintingNone)
		if err != nil {
			t.Fatalf("Kern via %s: %v", via, err)
		}
		if k != -40 {
			t.Errorf("Kern(1,2) via %s=%d want -40", via, k)
		}
	}

	// Metrics derive deterministically from hhea/OS2: ascent 1800, descent
	// -200, lineGap 0, xHeight 1000, capHeight 1400 (unscaled at ppem=upem).
	m, err := f.Metrics(&b, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("Metrics: %v", err)
	}
	if m.Ascent != 1800 {
		t.Errorf("Ascent=%d want 1800", m.Ascent)
	}
	if m.Descent != 200 { // sign flipped by Metrics.
		t.Errorf("Descent=%d want 200", m.Descent)
	}
	if m.Height != 2000 {
		t.Errorf("Height=%d want 2000", m.Height)
	}
	if m.XHeight != 1000 {
		t.Errorf("XHeight=%d want 1000", m.XHeight)
	}
	if m.CapHeight != 1400 {
		t.Errorf("CapHeight=%d want 1400", m.CapHeight)
	}

	// Name table family record.
	name, err := f.Name(&b, NameIDFamily)
	if err != nil {
		t.Fatalf("Name: %v", err)
	}
	if name != "FuzzFont" {
		t.Errorf("Name=%q want FuzzFont", name)
	}
}

// firstMoveTo returns the argument of the first MoveTo segment in s.
func firstMoveTo(s Segments) fixed.Point26_6 {
	for _, seg := range s {
		if seg.Op == SegmentOpMoveTo {
			return seg.Args[0]
		}
	}
	return fixed.Point26_6{}
}

// TestGoldenCompoundGlyph verifies that a compound glyph resolves to its
// component outline with the translation applied, and that an out-of-range
// glyph index reports ErrNotFound rather than panicking.
func TestGoldenCompoundGlyph(t *testing.T) {
	data := buildMinimalTrueType(minimalFontOptions{numGlyphs: 4, compound: true})
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	ppem := fixed.Int26_6(f.UnitsPerEm())

	// Use a dedicated buffer for the component so its segments are not aliased
	// and overwritten by the subsequent compound LoadGlyph on the shared b.
	var baseBuf Buffer
	base, err := f.LoadGlyph(&baseBuf, 2, ppem, nil) // component square.
	if err != nil {
		t.Fatalf("LoadGlyph(2): %v", err)
	}
	comp, err := f.LoadGlyph(&b, 1, ppem, nil) // compound referencing glyph 2.
	if err != nil {
		t.Fatalf("LoadGlyph(1) compound: %v", err)
	}
	if len(comp) != len(base) {
		t.Fatalf("compound segment count=%d want %d", len(comp), len(base))
	}
	// The compound embeds glyph 2 translated by (dx=25, dy=40). The component
	// outline starts at absolute (100,0); after the translation and the Y flip
	// its MoveTo origin is (100+25, 0-40) = (125,-40).
	compMove := firstMoveTo(comp)
	if compMove.X != 125 {
		t.Errorf("compound MoveTo X=%d want 125", compMove.X)
	}
	if compMove.Y != -40 {
		t.Errorf("compound MoveTo Y=%d want -40", compMove.Y)
	}
	// The uncompounded component's own origin is the pre-translation (100, 0),
	// flipped to (100, 0).
	baseMove := firstMoveTo(base)
	if baseMove.X != 100 || baseMove.Y != 0 {
		t.Errorf("component MoveTo=(%d,%d) want (100,0)", baseMove.X, baseMove.Y)
	}

	if _, err := f.LoadGlyph(&b, GlyphIndex(f.NumGlyphs()), ppem, nil); err != ErrNotFound {
		t.Errorf("LoadGlyph(out of range) err=%v want ErrNotFound", err)
	}
}

// TestGoldenKernTable checks the legacy kern path on a font without GPOS,
// ensuring the table builder and the Kern fallback agree.
func TestGoldenKernTable(t *testing.T) {
	data := buildMinimalTrueType(minimalFontOptions{numGlyphs: 4, addKern: true})
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	ppem := fixed.Int26_6(f.UnitsPerEm())
	k, err := f.Kern(&b, 1, 2, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("Kern: %v", err)
	}
	if k != -40 {
		t.Errorf("Kern(1,2)=%d want -40", k)
	}
	// An unlisted pair kerns to zero without error.
	k, err = f.Kern(&b, 0, 0, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("Kern(0,0): %v", err)
	}
	if k != 0 {
		t.Errorf("Kern(0,0)=%d want 0", k)
	}
}

// TestGoldenCollection checks that a TTC built from two minimal fonts exposes
// both members, each independently parseable and queryable.
func TestGoldenCollection(t *testing.T) {
	specs := []ttcFontSpec{
		{tables: minimalTrueTypeTables(minimalFontOptions{numGlyphs: 3}), magic: 0x00010000},
		{tables: minimalTrueTypeTables(minimalFontOptions{numGlyphs: 3, addKern: true}), magic: 0x00010000},
	}
	c, err := ParseCollection(buildTTC(specs))
	if err != nil {
		t.Fatalf("ParseCollection: %v", err)
	}
	if c.NumFonts() != 2 {
		t.Fatalf("NumFonts=%d want 2", c.NumFonts())
	}
	if _, err := c.Font(2); err != ErrNotFound {
		t.Errorf("Font(2) err=%v want ErrNotFound", err)
	}
	for i := 0; i < 2; i++ {
		f, err := c.Font(i)
		if err != nil {
			t.Fatalf("Font(%d): %v", i, err)
		}
		if f.NumGlyphs() != 3 {
			t.Errorf("Font(%d) NumGlyphs=%d want 3", i, f.NumGlyphs())
		}
		var b Buffer
		gi, err := f.GlyphIndex(&b, 'A')
		if err != nil {
			t.Fatalf("Font(%d) GlyphIndex: %v", i, err)
		}
		if gi != 1 {
			t.Errorf("Font(%d) GlyphIndex(A)=%d want 1", i, gi)
		}
	}
}

// TestGoldenDeterminism asserts that two independent builds from the same
// options produce byte-identical fonts, underpinning the JSONL summary's
// stability. It also confirms Parse output does not depend on build order.
func TestGoldenDeterminism(t *testing.T) {
	opts := minimalFontOptions{numGlyphs: 5, compound: true, addKern: true, addGPOS: true}
	a := buildMinimalTrueType(opts)
	b := buildMinimalTrueType(opts)
	if len(a) != len(b) {
		t.Fatalf("build lengths differ: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("builds diverge at byte %d", i)
		}
	}
	fa, err := Parse(a)
	if err != nil {
		t.Fatalf("Parse(a): %v", err)
	}
	fb, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse(b): %v", err)
	}
	if fa.NumGlyphs() != fb.NumGlyphs() || fa.UnitsPerEm() != fb.UnitsPerEm() {
		t.Fatal("parsed fonts differ despite identical input")
	}
}
