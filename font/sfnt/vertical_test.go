// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

func readVerticalTestFile(t *testing.T, filename string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.FromSlash("../testdata/" + filename))
	if err != nil {
		t.Fatalf("ReadFile(%q): %v", filename, err)
	}
	return data
}

// sfntTableEntry returns the byte offset of the 16-byte table directory entry
// and of the table data itself for the named table.
func sfntTableEntry(t *testing.T, data []byte, tag string) (entryOffset, tableOffset, length int) {
	t.Helper()
	numTables := int(binary.BigEndian.Uint16(data[4:]))
	for i := 0; i < numTables; i++ {
		e := 12 + 16*i
		if string(data[e:e+4]) == tag {
			return e, int(binary.BigEndian.Uint32(data[e+8:])), int(binary.BigEndian.Uint32(data[e+12:]))
		}
	}
	t.Fatalf("table %q not found", tag)
	return 0, 0, 0
}

func patchU16(data []byte, offset int, x uint16) {
	binary.BigEndian.PutUint16(data[offset:], x)
}

func patchU32(data []byte, offset int, x uint32) {
	binary.BigEndian.PutUint32(data[offset:], x)
}

// TestVerticalMetrics checks the font-wide vertical metrics of a TrueType font
// with vhea/vmtx, the equivalent CFF font and a real (subset) CJK font.
func TestVerticalMetrics(t *testing.T) {
	testCases := []struct {
		name string
		want font.Metrics
	}{
		// verticalTest: vhea ascent 500, descent -500, line gap 50, caret
		// slope rise 0 and run 1.
		{"verticalTest.ttf", font.Metrics{
			Height: 1050, Ascent: 500, Descent: 500,
			XHeight: 500, CapHeight: 700,
			CaretSlope: image.Point{X: 1, Y: 0},
		}},
		// The CFF flavor has the same metrics.
		{"verticalTestCFF.otf", font.Metrics{
			Height: 1050, Ascent: 500, Descent: 500,
			XHeight: 500, CapHeight: 700,
			CaretSlope: image.Point{X: 1, Y: 0},
		}},
		// mplus1p-subset.ttf is a subset of M PLUS 1p Regular, a real CJK
		// font. Its vhea ascent is 903, descent -615 and line gap 90. Its
		// OS/2 table is version 1, which has no x-height or cap-height
		// fields, and the glyphs 'x' and 'H' are not in the subset, so both
		// heights are zero.
		{"mplus1p-subset.ttf", font.Metrics{
			Height: 1608, Ascent: 903, Descent: 615,
			XHeight: 0, CapHeight: 0,
			CaretSlope: image.Point{X: 1, Y: 0},
		}},
	}
	for _, tc := range testCases {
		f, err := Parse(readVerticalTestFile(t, tc.name))
		if err != nil {
			t.Errorf("name=%q: Parse: %v", tc.name, err)
			continue
		}
		if !f.HasVerticalMetrics() {
			t.Errorf("name=%q: HasVerticalMetrics: got false, want true", tc.name)
			continue
		}
		// ppem equal to unitsPerEm makes the result measured in font units.
		ppem := fixed.Int26_6(f.UnitsPerEm())
		got, err := f.VerticalMetrics(nil, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("name=%q: VerticalMetrics: %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("name=%q: VerticalMetrics:\ngot  %v\nwant %v", tc.name, got, tc.want)
		}

		// XHeight and CapHeight are shared with the horizontal metrics.
		hm, err := f.Metrics(nil, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("name=%q: Metrics: %v", tc.name, err)
			continue
		}
		if got.XHeight != hm.XHeight || got.CapHeight != hm.CapHeight {
			t.Errorf("name=%q: XHeight/CapHeight differ from horizontal metrics: %v vs %v", tc.name, got, hm)
		}
	}
}

// TestVerticalMetricsFallback checks the synthesis used when a font has no
// vertical metrics tables: the ideographic em-box, one em tall.
func TestVerticalMetricsFallback(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTestNoVhea.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if f.HasVerticalMetrics() {
		t.Fatalf("HasVerticalMetrics: got true, want false")
	}
	ppem := fixed.Int26_6(f.UnitsPerEm())
	got, err := f.VerticalMetrics(nil, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("VerticalMetrics: %v", err)
	}
	want := font.Metrics{
		Height:     1000,
		Ascent:     500,
		Descent:    500,
		XHeight:    500,
		CapHeight:  700,
		CaretSlope: image.Point{X: 1, Y: 0},
	}
	if got != want {
		t.Errorf("VerticalMetrics:\ngot  %v\nwant %v", got, want)
	}

	// An empty (zero) ppem scales everything to zero rather than dividing by
	// zero.
	zero, err := f.VerticalMetrics(nil, 0, font.HintingNone)
	if err != nil {
		t.Fatalf("VerticalMetrics(ppem=0): %v", err)
	}
	if zero.Height != 0 || zero.Ascent != 0 || zero.Descent != 0 {
		t.Errorf("VerticalMetrics(ppem=0): got %v, want zero height/ascent/descent", zero)
	}
}

// TestVerticalMetricsScaling checks that a non-zero ppem other than
// unitsPerEm scales the vertical metrics, and that HintingFull quantizes them.
func TestVerticalMetricsScaling(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTest.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	// unitsPerEm is 1000. At 16 ppem, 1 font unit is 16/1000 pixel. The
	// expectation is computed the same way as scaleVertical: round to the
	// nearest 1/64 pixel.
	ppem := fixed.I(16)
	got, err := f.VerticalMetrics(nil, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("VerticalMetrics: %v", err)
	}
	px := func(units int64) fixed.Int26_6 {
		return fixed.Int26_6((units*16*64 + 1000/2) / 1000)
	}
	want := font.Metrics{
		Height:     px(1050), // 16.8 px
		Ascent:     px(500),  // 8 px
		Descent:    px(500),  // 8 px
		XHeight:    px(500),  // 8 px
		CapHeight:  px(700),  // 11.2 px
		CaretSlope: image.Point{X: 1, Y: 0},
	}
	if got != want {
		t.Errorf("VerticalMetrics:\ngot  %v\nwant %v", got, want)
	}

	hinted, err := f.VerticalMetrics(nil, ppem, font.HintingFull)
	if err != nil {
		t.Fatalf("VerticalMetrics(HintingFull): %v", err)
	}
	wantHinted := font.Metrics{
		Height:     17 * 64, // 16.8 px rounds up to 17.
		Ascent:     8 * 64,
		Descent:    8 * 64,
		XHeight:    8 * 64,
		CapHeight:  12 * 64, // 11.2 px rounds up to 12.
		CaretSlope: image.Point{X: 1, Y: 0},
	}
	if hinted != wantHinted {
		t.Errorf("VerticalMetrics(HintingFull):\ngot  %v\nwant %v", hinted, wantHinted)
	}
}

// TestGlyphVerticalAdvance checks the per-glyph advance heights, including the
// trailing-array default when numOfLongVerMetrics is smaller than numGlyphs.
func TestGlyphVerticalAdvance(t *testing.T) {
	testCases := []struct {
		name  string
		wants []fixed.Int26_6
	}{{
		// The first four glyphs have full vmtx records. The remaining four
		// reuse the last advance height (1000).
		name:  "verticalTest.ttf",
		wants: []fixed.Int26_6{1100, 1000, 900, 1000, 1000, 1000, 1000, 1000},
	}, {
		name:  "verticalTestCFF.otf",
		wants: []fixed.Int26_6{1100, 1000, 900, 1000, 1000, 1000, 1000, 1000},
	}, {
		// A font without vertical metrics gets one em per glyph.
		name:  "verticalTestNoVhea.ttf",
		wants: []fixed.Int26_6{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000},
	}}
	var b Buffer
	for _, tc := range testCases {
		f, err := Parse(readVerticalTestFile(t, tc.name))
		if err != nil {
			t.Errorf("name=%q: Parse: %v", tc.name, err)
			continue
		}
		ppem := fixed.Int26_6(f.UnitsPerEm())
		for i, want := range tc.wants {
			got, err := f.GlyphVerticalAdvance(&b, GlyphIndex(i), ppem, font.HintingNone)
			if err != nil {
				t.Errorf("name=%q, i=%d: GlyphVerticalAdvance: %v", tc.name, i, err)
				continue
			}
			if got != want {
				t.Errorf("name=%q, i=%d: GlyphVerticalAdvance: got %v, want %v", tc.name, i, got, want)
			}
		}

		if _, err := f.GlyphVerticalAdvance(&b, GlyphIndex(f.NumGlyphs()), ppem, font.HintingNone); err != ErrNotFound {
			t.Errorf("name=%q: out-of-range GlyphVerticalAdvance: got %v, want %v", tc.name, err, ErrNotFound)
		}
		if _, err := f.GlyphVerticalAdvance(&b, 0xffff, ppem, font.HintingNone); err != ErrNotFound {
			t.Errorf("name=%q: out-of-range GlyphVerticalAdvance(0xffff): got %v, want %v", tc.name, err, ErrNotFound)
		}
	}
}

// TestGlyphVerticalAdvanceScaling checks non-zero ppem scaling and hinting.
func TestGlyphVerticalAdvanceScaling(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTest.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	const a = 2 // 'A', whose advance height is 900 font units.

	testCases := []struct {
		ppem fixed.Int26_6
		h    font.Hinting
		want fixed.Int26_6
	}{
		// 900 * 16/1000 pixels = 14.4 px = 14:25.6, rounded to 14:26.
		{fixed.I(16), font.HintingNone, 922},
		// Hinting rounds to the nearest whole pixel: 14 px.
		{fixed.I(16), font.HintingFull, 14 * 64},
		// Half of unitsPerEm: 900 * 0.5 = 450 font-pixel units.
		{fixed.Int26_6(500), font.HintingNone, 450},
	}
	var b Buffer
	for _, tc := range testCases {
		got, err := f.GlyphVerticalAdvance(&b, a, tc.ppem, tc.h)
		if err != nil {
			t.Errorf("ppem=%v: %v", tc.ppem, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ppem=%v, h=%v: got %v, want %v", tc.ppem, tc.h, got, tc.want)
		}
	}
}

// TestGlyphVerticalOrigin checks the vertical origin: half the horizontal
// advance to the right, and the top side bearing above the glyph's top.
func TestGlyphVerticalOrigin(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTest.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ppem := fixed.Int26_6(f.UnitsPerEm())

	testCases := []struct {
		x    GlyphIndex
		want fixed.Point26_6
	}{{
		// .notdef: horizontal advance 1000, top side bearing 880, yMax 700.
		x:    0,
		want: pt(500, -(880 + 700)),
	}, {
		// A: horizontal advance 600, and a *negative* top side bearing of
		// -100, so the origin is below the glyph's top.
		x:    2,
		want: pt(300, -(-100 + 700)),
	}, {
		// uni4E00: horizontal advance 1000.
		x:    3,
		want: pt(500, -(550 + 450)),
	}, {
		// uni3001: 400-wide advance, negative top side bearing -50.
		x:    6,
		want: pt(500, -(-50 + 100)),
	}, {
		// glyph7 has no full vmtx record: its advance comes from the last
		// full record and its top side bearing (400) from the trailing
		// array.
		x:    7,
		want: pt(400, -(400 + 600)),
	}}
	var b Buffer
	for _, tc := range testCases {
		got, err := f.GlyphVerticalOrigin(&b, tc.x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalOrigin: %v", tc.x, err)
			continue
		}
		if got != tc.want {
			t.Errorf("x=%d: GlyphVerticalOrigin: got %v, want %v", tc.x, got, tc.want)
		}
	}

	if _, err := f.GlyphVerticalOrigin(&b, GlyphIndex(f.NumGlyphs()), ppem, font.HintingNone); err != ErrNotFound {
		t.Errorf("out-of-range GlyphVerticalOrigin: got %v, want %v", err, ErrNotFound)
	}
}

// TestGlyphVerticalBounds checks the vertical bounds, which are relative to the
// vertical origin: the top is the top side bearing and the X coordinates are
// shifted by half the horizontal advance.
func TestGlyphVerticalBounds(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTest.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ppem := fixed.Int26_6(f.UnitsPerEm())

	testCases := []struct {
		x       GlyphIndex
		want    fixed.Rectangle26_6
		wantAdv fixed.Int26_6
	}{{
		// A: horizontal bounds [0,600]x[-700,0], advance 600, top side
		// bearing -100.
		x: 2,
		want: fixed.Rectangle26_6{
			Min: pt(0-300, -100),
			Max: pt(600-300, -100+700),
		},
		wantAdv: 900,
	}, {
		// uni4E00: horizontal bounds [100,900]x[-450,-350], advance 1000, top
		// side bearing 550.
		x: 3,
		want: fixed.Rectangle26_6{
			Min: pt(100-500, 550),
			Max: pt(900-500, 550+100),
		},
		wantAdv: 1000,
	}, {
		// uni3001: horizontal bounds [400,600]x[-100,120], advance 1000, top
		// side bearing -50.
		x: 6,
		want: fixed.Rectangle26_6{
			Min: pt(400-500, -50),
			Max: pt(600-500, -50+220),
		},
		wantAdv: 1000,
	}, {
		// glyph7: horizontal bounds [100,900]x[-600,0], advance 800, top side
		// bearing 400 from the trailing array.
		x: 7,
		want: fixed.Rectangle26_6{
			Min: pt(100-400, 400),
			Max: pt(900-400, 400+600),
		},
		wantAdv: 1000,
	}}
	var b Buffer
	for _, tc := range testCases {
		got, gotAdv, err := f.GlyphVerticalBounds(&b, tc.x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalBounds: %v", tc.x, err)
			continue
		}
		if got != tc.want {
			t.Errorf("x=%d: GlyphVerticalBounds:\ngot  %v\nwant %v", tc.x, got, tc.want)
		}
		if gotAdv != tc.wantAdv {
			t.Errorf("x=%d: GlyphVerticalBounds advance: got %v, want %v", tc.x, gotAdv, tc.wantAdv)
		}
	}

	if _, _, err := f.GlyphVerticalBounds(&b, GlyphIndex(f.NumGlyphs()), ppem, font.HintingNone); err != ErrNotFound {
		t.Errorf("out-of-range GlyphVerticalBounds: got %v, want %v", err, ErrNotFound)
	}
}

// TestVerticalBoundsInvariants cross-checks every glyph's vertical bounds and
// origin against the horizontal GlyphBounds, which the vertical code must not
// disturb. It runs over the TrueType, CFF and CJK fixtures.
func TestVerticalBoundsInvariants(t *testing.T) {
	for _, name := range []string{"verticalTest.ttf", "verticalTestCFF.otf", "mplus1p-subset.ttf"} {
		f, err := Parse(readVerticalTestFile(t, name))
		if err != nil {
			t.Errorf("name=%q: Parse: %v", name, err)
			continue
		}
		ppem := fixed.Int26_6(f.UnitsPerEm())
		var b Buffer
		for x := GlyphIndex(0); int(x) < f.NumGlyphs(); x++ {
			hb, hAdv, err := f.GlyphBounds(&b, x, ppem, font.HintingNone)
			if err != nil {
				t.Errorf("name=%q, x=%d: GlyphBounds: %v", name, x, err)
				continue
			}
			origin, err := f.GlyphVerticalOrigin(&b, x, ppem, font.HintingNone)
			if err != nil {
				t.Errorf("name=%q, x=%d: GlyphVerticalOrigin: %v", name, x, err)
				continue
			}
			vb, vAdv, err := f.GlyphVerticalBounds(&b, x, ppem, font.HintingNone)
			if err != nil {
				t.Errorf("name=%q, x=%d: GlyphVerticalBounds: %v", name, x, err)
				continue
			}
			if want := hAdv / 2; origin.X != want {
				t.Errorf("name=%q, x=%d: origin.X: got %v, want %v (half the horizontal advance)", name, x, origin.X, want)
			}
			// The X extent is unchanged, just shifted by the origin's X.
			if want := hb.Min.X - origin.X; vb.Min.X != want {
				t.Errorf("name=%q, x=%d: bounds.Min.X: got %v, want %v", name, x, vb.Min.X, want)
			}
			if want := hb.Max.X - origin.X; vb.Max.X != want {
				t.Errorf("name=%q, x=%d: bounds.Max.X: got %v, want %v", name, x, vb.Max.X, want)
			}
			// The Y extent is unchanged, just shifted by the origin's Y.
			if want := hb.Min.Y - origin.Y; vb.Min.Y != want {
				t.Errorf("name=%q, x=%d: bounds.Min.Y: got %v, want %v", name, x, vb.Min.Y, want)
			}
			if want := hb.Max.Y - origin.Y; vb.Max.Y != want {
				t.Errorf("name=%q, x=%d: bounds.Max.Y: got %v, want %v", name, x, vb.Max.Y, want)
			}
			if vb.Min.Y >= vb.Max.Y && !hb.Empty() {
				t.Errorf("name=%q, x=%d: vertical bounds are inverted: %v", name, x, vb)
			}
			wantAdv, err := f.GlyphVerticalAdvance(&b, x, ppem, font.HintingNone)
			if err != nil {
				t.Errorf("name=%q, x=%d: GlyphVerticalAdvance: %v", name, x, err)
			} else if vAdv != wantAdv {
				t.Errorf("name=%q, x=%d: GlyphVerticalBounds advance: got %v, want %v", name, x, vAdv, wantAdv)
			}
		}
	}
}

// TestVerticalCJK checks specific values from a real CJK font (a subset of
// M PLUS 1p Regular). All of its ideographic glyphs share a vertical origin
// 860 units above the baseline.
func TestVerticalCJK(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "mplus1p-subset.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ppem := fixed.Int26_6(f.UnitsPerEm()) // 1000
	var b Buffer

	testCases := []struct {
		r      rune
		tsb    fixed.Int26_6 // top side bearing
		origin fixed.Int26_6 // the Y coordinate of the vertical origin
	}{
		{'A', 130, -860},
		{'一', 440, -860},
		{'中', 43, -860},
		{'文', 30, -860},
		{'漢', 37, -860},
		{'あ', 57, -860},
		{'。', 632, -860},
	}
	for _, tc := range testCases {
		x, err := f.GlyphIndex(&b, tc.r)
		if err != nil || x == 0 {
			t.Errorf("r=%q: GlyphIndex: x=%d, err=%v", tc.r, x, err)
			continue
		}
		origin, err := f.GlyphVerticalOrigin(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("r=%q: GlyphVerticalOrigin: %v", tc.r, err)
			continue
		}
		if origin.Y != tc.origin {
			t.Errorf("r=%q: origin.Y: got %v, want %v", tc.r, origin.Y, tc.origin)
		}
		bounds, adv, err := f.GlyphVerticalBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("r=%q: GlyphVerticalBounds: %v", tc.r, err)
			continue
		}
		if adv != 1000 {
			t.Errorf("r=%q: advance: got %v, want 1000", tc.r, adv)
		}
		// The top of the vertical bounds is the top side bearing.
		if bounds.Min.Y != tc.tsb {
			t.Errorf("r=%q: bounds.Min.Y: got %v, want %v", tc.r, bounds.Min.Y, tc.tsb)
		}
	}
}

// TestVerticalCJKGlyphs checks every glyph of the real CJK subset against
// values read independently with fontTools. It verifies that the advance
// height and top side bearing come from vmtx, and that the vertical origin is
// at half the horizontal advance.
func TestVerticalCJKGlyphs(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "mplus1p-subset.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	ppem := fixed.Int26_6(f.UnitsPerEm()) // 1000
	var b Buffer

	// (advance height, top side bearing), copied from fontTools for
	// mplus1p-subset.ttf's glyph order.
	testCases := []struct {
		advance, tsb fixed.Int26_6
	}{
		{1000, 666},
		{1000, 860},
		{1000, 538},
		{1000, 720},
		{1000, 120},
		{1000, 130},
		{1000, 120},
		{1000, 651},
		{1000, 632},
		{1000, 57},
		{1000, 462},
		{1000, 440},
		{1000, 43},
		{1000, 30},
		{1000, 30},
		{1000, 37},
		{1000, 80},
		{1000, 61},
		{1000, 93},
	}
	if got, want := f.NumGlyphs(), len(testCases); got != want {
		t.Fatalf("NumGlyphs: got %d, want %d", got, want)
	}
	for i, tc := range testCases {
		x := GlyphIndex(i)
		adv, err := f.GlyphVerticalAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalAdvance: %v", i, err)
			continue
		}
		if adv != tc.advance {
			t.Errorf("x=%d: GlyphVerticalAdvance: got %v, want %v", i, adv, tc.advance)
		}
		hAdv, err := f.GlyphAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphAdvance: %v", i, err)
			continue
		}
		origin, err := f.GlyphVerticalOrigin(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalOrigin: %v", i, err)
			continue
		}
		if want := hAdv / 2; origin.X != want {
			t.Errorf("x=%d: origin.X: got %v, want %v", i, origin.X, want)
		}
		bounds, vbAdv, err := f.GlyphVerticalBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalBounds: %v", i, err)
			continue
		}
		if vbAdv != tc.advance {
			t.Errorf("x=%d: advance: got %v, want %v", i, vbAdv, tc.advance)
		}
		if bounds.Min.Y != tc.tsb {
			t.Errorf("x=%d: bounds.Min.Y: got %v, want %v (top side bearing)", i, bounds.Min.Y, tc.tsb)
		}
		// X is the horizontal bounds shifted by half the horizontal advance.
		hb, _, err := f.GlyphBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphBounds: %v", i, err)
			continue
		}
		if bounds.Min.X != hb.Min.X-origin.X || bounds.Max.X != hb.Max.X-origin.X {
			t.Errorf("x=%d: vertical bounds X: got [%v,%v], want [%v,%v]", i,
				bounds.Min.X, bounds.Max.X, hb.Min.X-origin.X, hb.Max.X-origin.X)
		}
	}
}

// TestVerticalCollection checks that each face of a font collection has its own
// vertical metrics, and that a face without them falls back to one em.
func TestVerticalCollection(t *testing.T) {
	c, err := ParseCollection(readVerticalTestFile(t, "verticalTest.ttc"))
	if err != nil {
		t.Fatalf("ParseCollection: %v", err)
	}
	if got, want := c.NumFonts(), 2; got != want {
		t.Fatalf("NumFonts: got %d, want %d", got, want)
	}
	var b Buffer
	ppem := fixed.Int26_6(1000)

	f0, err := c.Font(0)
	if err != nil {
		t.Fatalf("Font(0): %v", err)
	}
	if !f0.HasVerticalMetrics() {
		t.Errorf("Font(0): HasVerticalMetrics: got false, want true")
	}
	m0, err := f0.VerticalMetrics(&b, ppem, font.HintingNone)
	if err != nil {
		t.Errorf("Font(0): VerticalMetrics: %v", err)
	} else if m0.Height != 1050 {
		t.Errorf("Font(0): Height: got %v, want 1050", m0.Height)
	}
	if adv, err := f0.GlyphVerticalAdvance(&b, 2, ppem, font.HintingNone); err != nil {
		t.Errorf("Font(0): GlyphVerticalAdvance: %v", err)
	} else if adv != 900 {
		t.Errorf("Font(0): GlyphVerticalAdvance: got %v, want 900", adv)
	}

	f1, err := c.Font(1)
	if err != nil {
		t.Fatalf("Font(1): %v", err)
	}
	if f1.HasVerticalMetrics() {
		t.Errorf("Font(1): HasVerticalMetrics: got true, want false")
	}
	m1, err := f1.VerticalMetrics(&b, ppem, font.HintingNone)
	if err != nil {
		t.Errorf("Font(1): VerticalMetrics: %v", err)
	} else if m1.Height != 1000 || m1.Ascent != 500 || m1.Descent != 500 {
		t.Errorf("Font(1): VerticalMetrics: got %v, want the one-em fallback", m1)
	}
	if adv, err := f1.GlyphVerticalAdvance(&b, 2, ppem, font.HintingNone); err != nil {
		t.Errorf("Font(1): GlyphVerticalAdvance: %v", err)
	} else if adv != 1000 {
		t.Errorf("Font(1): GlyphVerticalAdvance: got %v, want 1000", adv)
	}

	// The vertical tables on face 0 must not leak into face 1, and vice
	// versa.
	if m0 == m1 {
		t.Errorf("Font(0) and Font(1) have identical vertical metrics: %v", m0)
	}
}

// TestVerticalHorizontalUnaffected checks that the presence of vertical tables
// does not change any horizontal API result. verticalTest.ttf and
// verticalTestNoVhea.ttf have byte-identical horizontal tables.
func TestVerticalHorizontalUnaffected(t *testing.T) {
	withV, err := Parse(readVerticalTestFile(t, "verticalTest.ttf"))
	if err != nil {
		t.Fatalf("Parse(verticalTest): %v", err)
	}
	withoutV, err := Parse(readVerticalTestFile(t, "verticalTestNoVhea.ttf"))
	if err != nil {
		t.Fatalf("Parse(verticalTestNoVhea): %v", err)
	}
	if withV.NumGlyphs() != withoutV.NumGlyphs() {
		t.Fatalf("NumGlyphs differ: %d vs %d", withV.NumGlyphs(), withoutV.NumGlyphs())
	}
	var b Buffer
	ppem := fixed.Int26_6(1000)

	mWith, err := withV.Metrics(&b, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("Metrics(verticalTest): %v", err)
	}
	mWithout, err := withoutV.Metrics(&b, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("Metrics(verticalTestNoVhea): %v", err)
	}
	if mWith != mWithout {
		t.Errorf("horizontal Metrics differ with and without vertical tables:\n%v\n%v", mWith, mWithout)
	}

	for x := GlyphIndex(0); int(x) < withV.NumGlyphs(); x++ {
		advWith, err := withV.GlyphAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Fatalf("x=%d: GlyphAdvance(verticalTest): %v", x, err)
		}
		advWithout, err := withoutV.GlyphAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Fatalf("x=%d: GlyphAdvance(verticalTestNoVhea): %v", x, err)
		}
		if advWith != advWithout {
			t.Errorf("x=%d: GlyphAdvance differs with and without vertical tables: %v vs %v", x, advWith, advWithout)
		}
		bWith, _, err := withV.GlyphBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Fatalf("x=%d: GlyphBounds(verticalTest): %v", x, err)
		}
		bWithout, _, err := withoutV.GlyphBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Fatalf("x=%d: GlyphBounds(verticalTestNoVhea): %v", x, err)
		}
		if bWith != bWithout {
			t.Errorf("x=%d: GlyphBounds differs with and without vertical tables: %v vs %v", x, bWith, bWithout)
		}
	}
}

// TestVerticalTruncatedVmtx checks that a vmtx table which omits the optional
// trailing top side bearing array is accepted, and that the missing bearings
// default to zero. (This mirrors the hmtx behavior noted in golang/go#28379.)
func TestVerticalTruncatedVmtx(t *testing.T) {
	data := readVerticalTestFile(t, "verticalTest.ttf")
	vmtxEntry, _, vmtxLen := sfntTableEntry(t, data, "vmtx")
	_, vheaOff, _ := sfntTableEntry(t, data, "vhea")
	numVMetrics := binary.BigEndian.Uint16(data[vheaOff+34:])
	fullLen := 4 * int(numVMetrics)
	if vmtxLen <= fullLen {
		t.Fatalf("fixture vmtx length %d is already compressed (full metrics %d)", vmtxLen, fullLen)
	}
	// Claim only the long-metric records are present.
	patchU32(data, vmtxEntry+12, uint32(fullLen))

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if !f.HasVerticalMetrics() {
		t.Fatalf("HasVerticalMetrics: got false, want true")
	}
	var b Buffer
	ppem := fixed.Int26_6(f.UnitsPerEm())

	// The advances are still the last full record's advance.
	adv, err := f.GlyphVerticalAdvance(&b, 7, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("GlyphVerticalAdvance(7): %v", err)
	}
	if adv != 1000 {
		t.Errorf("GlyphVerticalAdvance(7): got %v, want 1000", adv)
	}
	// The top side bearing, which only the trailing array would have given,
	// defaults to zero. glyph7's yMax is 600, so the origin is 600 units above
	// the horizontal origin.
	origin, err := f.GlyphVerticalOrigin(&b, 7, ppem, font.HintingNone)
	if err != nil {
		t.Fatalf("GlyphVerticalOrigin(7): %v", err)
	}
	if want := pt(400, -600); origin != want {
		t.Errorf("GlyphVerticalOrigin(7): got %v, want %v", origin, want)
	}
}

// TestVerticalMalformed checks that malformed vertical metrics tables produce a
// stable error from the vertical APIs while leaving the horizontal APIs usable
// and Parse successful.
func TestVerticalMalformed(t *testing.T) {
	testCases := []struct {
		name   string
		patch  func(t *testing.T, data []byte)
		broken bool // whether HasVerticalMetrics should report false
	}{{
		name: "vhea length 34",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vhea")
			patchU32(data, e+12, 34)
		},
		broken: true,
	}, {
		name: "vhea length 0",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vhea")
			patchU32(data, e+12, 0)
		},
		broken: true,
	}, {
		name: "numOfLongVerMetrics zero",
		patch: func(t *testing.T, data []byte) {
			_, o, _ := sfntTableEntry(t, data, "vhea")
			patchU16(data, o+34, 0)
		},
		broken: true,
	}, {
		name: "numOfLongVerMetrics exceeds numGlyphs",
		patch: func(t *testing.T, data []byte) {
			_, o, _ := sfntTableEntry(t, data, "vhea")
			patchU16(data, o+34, 9) // numGlyphs is 8.
		},
		broken: true,
	}, {
		name: "numOfLongVerMetrics huge",
		patch: func(t *testing.T, data []byte) {
			_, o, _ := sfntTableEntry(t, data, "vhea")
			patchU16(data, o+34, 0xffff)
		},
		broken: true,
	}, {
		name: "vmtx length inconsistent with numOfLongVerMetrics",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vmtx")
			patchU32(data, e+12, 20)
		},
		broken: true,
	}, {
		name: "vhea without vmtx",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vmtx")
			patchU32(data, e+8, 0)  // offset
			patchU32(data, e+12, 0) // length
		},
		broken: true,
	}, {
		name: "vmtx without vhea",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vhea")
			patchU32(data, e+8, 0)  // offset
			patchU32(data, e+12, 0) // length
		},
		broken: true,
	}}

	for _, tc := range testCases {
		// Parse takes ownership of the bytes, so patch a private copy.
		data := append([]byte(nil), readVerticalTestFile(t, "verticalTest.ttf")...)
		tc.patch(t, data)

		f, err := Parse(data)
		if err != nil {
			t.Errorf("name=%q: Parse: %v", tc.name, err)
			continue
		}
		var b Buffer
		ppem := fixed.Int26_6(1000)

		// The vertical APIs report an error, but do not panic.
		if _, err := f.VerticalMetrics(&b, ppem, font.HintingNone); err == nil {
			t.Errorf("name=%q: VerticalMetrics: got nil error, want non-nil", tc.name)
		}
		if _, err := f.GlyphVerticalAdvance(&b, 0, ppem, font.HintingNone); err == nil {
			t.Errorf("name=%q: GlyphVerticalAdvance: got nil error, want non-nil", tc.name)
		}
		if _, err := f.GlyphVerticalOrigin(&b, 0, ppem, font.HintingNone); err == nil {
			t.Errorf("name=%q: GlyphVerticalOrigin: got nil error, want non-nil", tc.name)
		}
		if _, _, err := f.GlyphVerticalBounds(&b, 0, ppem, font.HintingNone); err == nil {
			t.Errorf("name=%q: GlyphVerticalBounds: got nil error, want non-nil", tc.name)
		}
		if got := f.HasVerticalMetrics(); got != !tc.broken {
			t.Errorf("name=%q: HasVerticalMetrics: got %v, want %v", tc.name, got, !tc.broken)
		}

		// The horizontal APIs still work, and return the same results as the
		// unpatched font.
		if _, err := f.Metrics(&b, ppem, font.HintingNone); err != nil {
			t.Errorf("name=%q: horizontal Metrics: %v", tc.name, err)
		}
		if _, err := f.GlyphAdvance(&b, 0, ppem, font.HintingNone); err != nil {
			t.Errorf("name=%q: horizontal GlyphAdvance: %v", tc.name, err)
		}
		if _, _, err := f.GlyphBounds(&b, 0, ppem, font.HintingNone); err != nil {
			t.Errorf("name=%q: horizontal GlyphBounds: %v", tc.name, err)
		}

		// The error is stable: repeated calls return the same non-nil error.
		_, err1 := f.VerticalMetrics(&b, ppem, font.HintingNone)
		_, err2 := f.VerticalMetrics(&b, ppem, font.HintingNone)
		if err1 == nil || err1 != err2 {
			t.Errorf("name=%q: VerticalMetrics error is unstable: %v vs %v", tc.name, err1, err2)
		}
	}
}

// TestVerticalErrorKinds checks that the specific malformed cases map to the
// specific errors, rather than all collapsing to one.
func TestVerticalErrorKinds(t *testing.T) {
	testCases := []struct {
		name  string
		patch func(t *testing.T, data []byte)
		want  error
	}{{
		name: "bad vhea length",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vhea")
			patchU32(data, e+12, 34)
		},
		want: errInvalidVheaTable,
	}, {
		name: "bad vmtx length",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vmtx")
			patchU32(data, e+12, 20)
		},
		want: errInvalidVmtxTable,
	}, {
		name: "missing vhea",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vhea")
			patchU32(data, e+8, 0)
			patchU32(data, e+12, 0)
		},
		want: errInvalidVheaTable,
	}, {
		name: "missing vmtx",
		patch: func(t *testing.T, data []byte) {
			e, _, _ := sfntTableEntry(t, data, "vmtx")
			patchU32(data, e+8, 0)
			patchU32(data, e+12, 0)
		},
		want: errInvalidVmtxTable,
	}}
	for _, tc := range testCases {
		data := append([]byte(nil), readVerticalTestFile(t, "verticalTest.ttf")...)
		tc.patch(t, data)
		f, err := Parse(data)
		if err != nil {
			t.Errorf("name=%q: Parse: %v", tc.name, err)
			continue
		}
		if _, err := f.VerticalMetrics(nil, 1000, font.HintingNone); !errors.Is(err, tc.want) {
			t.Errorf("name=%q: VerticalMetrics error: got %v, want %v", tc.name, err, tc.want)
		}
	}
}

// TestVerticalOverflow checks that an abnormal units-per-em value combined with
// a very large ppem yields an error rather than a silently wrapped result.
func TestVerticalOverflow(t *testing.T) {
	data := append([]byte(nil), readVerticalTestFile(t, "verticalTest.ttf")...)
	e, o, _ := sfntTableEntry(t, data, "head")
	_ = e
	// head.unitsPerEm is at offset 18 and is a uint16. One is accepted by
	// parseHead (only zero is rejected), but makes scaling overflow.
	patchU16(data, o+18, 1)

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got, want := f.UnitsPerEm(), Units(1); got != want {
		t.Fatalf("UnitsPerEm: got %d, want %d", got, want)
	}
	// 500 font units at (1 << 30) pixels-per-em is about 5*10^11, far beyond
	// the range of a fixed.Int26_6.
	if _, err := f.VerticalMetrics(nil, fixed.Int26_6(1<<30), font.HintingNone); err == nil {
		t.Errorf("VerticalMetrics: got nil error, want an overflow error")
	}
}

// TestVerticalNilBuffer and TestVerticalReaderAt check that the vertical APIs
// work with a nil *Buffer and on a font parsed from an io.ReaderAt.
func TestVerticalNilBufferAndReaderAt(t *testing.T) {
	data := readVerticalTestFile(t, "verticalTest.ttf")

	testCases := []struct {
		name string
		f    *Font
	}{}
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	testCases = append(testCases, struct {
		name string
		f    *Font
	}{"Parse", f})

	g, err := ParseReaderAt(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("ParseReaderAt: %v", err)
	}
	testCases = append(testCases, struct {
		name string
		f    *Font
	}{"ParseReaderAt", g})

	ppem := fixed.Int26_6(1000)
	for _, tc := range testCases {
		// A nil *Buffer is valid.
		adv, err := tc.f.GlyphVerticalAdvance(nil, 2, ppem, font.HintingNone)
		if err != nil || adv != 900 {
			t.Errorf("%s: GlyphVerticalAdvance(nil): got %v, %v; want 900, nil", tc.name, adv, err)
		}
		origin, err := tc.f.GlyphVerticalOrigin(nil, 2, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("%s: GlyphVerticalOrigin(nil): %v", tc.name, err)
		} else if want := pt(300, -600); origin != want {
			t.Errorf("%s: GlyphVerticalOrigin(nil): got %v, want %v", tc.name, origin, want)
		}
		bounds, _, err := tc.f.GlyphVerticalBounds(nil, 2, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("%s: GlyphVerticalBounds(nil): %v", tc.name, err)
		} else if bounds != (fixed.Rectangle26_6{Min: pt(-300, -100), Max: pt(300, 600)}) {
			t.Errorf("%s: GlyphVerticalBounds(nil): got %v", tc.name, bounds)
		}
		m, err := tc.f.VerticalMetrics(nil, ppem, font.HintingNone)
		if err != nil || m.Height != 1050 {
			t.Errorf("%s: VerticalMetrics(nil): got %v, %v; want Height 1050, nil", tc.name, m, err)
		}
	}
}

// TestVerticalZeroGlyphFont checks that a font with vertical tables but no
// glyphs is rejected rather than panicking.
func TestVerticalZeroGlyphFont(t *testing.T) {
	data := append([]byte(nil), readVerticalTestFile(t, "verticalTest.ttf")...)
	_, maxpOff, _ := sfntTableEntry(t, data, "maxp")
	// maxp.numGlyphs is at offset 4.
	patchU16(data, maxpOff+4, 0)

	// A font with zero glyphs is already rejected by the glyph data parsing
	// (loca/maxp disagreement), so Parse may fail. Either way it must not
	// panic.
	if f, err := Parse(data); err == nil {
		if _, err := f.VerticalMetrics(nil, 1000, font.HintingNone); err == nil {
			t.Errorf("VerticalMetrics on a zero-glyph font: got nil error, want non-nil")
		}
	}
}

// TestVerticalEmpty checks that a font with neither table reports an empty,
// non-error result and uses the fallback for every glyph.
func TestVerticalEmpty(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTestNoVhea.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if f.HasVerticalMetrics() {
		t.Fatalf("HasVerticalMetrics: got true, want false")
	}
	var b Buffer
	ppem := fixed.Int26_6(f.UnitsPerEm())
	// Every glyph has the same one-em advance, and the vertical origin puts
	// the glyph's center in the middle of that em.
	for x := GlyphIndex(0); int(x) < f.NumGlyphs(); x++ {
		adv, err := f.GlyphVerticalAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalAdvance: %v", x, err)
			continue
		}
		if adv != 1000 {
			t.Errorf("x=%d: GlyphVerticalAdvance: got %v, want 1000", x, adv)
		}
		bounds, _, err := f.GlyphVerticalBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalBounds: %v", x, err)
			continue
		}
		if bounds.Empty() {
			// An empty glyph (e.g. a space) has no extent to center.
			continue
		}
		// In vertical coordinates the origin is the top of the one-em box, so
		// the box runs from Y = 0 to Y = 1000 (the advance). The free space
		// above and below the glyph is equal.
		above := bounds.Min.Y
		below := 1000 - bounds.Max.Y
		if above < 0 || below < 0 || above != below {
			t.Errorf("x=%d: glyph not centered in the one-em box: bounds=%v, above=%v, below=%v", x, bounds, above, below)
		}
	}
}

// TestVerticalFallbackScaling checks the no-table fallback at a ppem other than
// unitsPerEm. The vertical origin and bounds must scale consistently, so that
// the glyph is still centered in the em.
func TestVerticalFallbackScaling(t *testing.T) {
	f, err := Parse(readVerticalTestFile(t, "verticalTestNoVhea.ttf"))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	// unitsPerEm is 1000 and the ppem is 20, so 1 font unit is 20/1000 px and
	// the advance is 20 px.
	ppem := fixed.I(20)
	const advance = 20 * 64
	for x := GlyphIndex(0); int(x) < f.NumGlyphs(); x++ {
		got, err := f.GlyphVerticalAdvance(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalAdvance: %v", x, err)
			continue
		}
		if got != advance {
			t.Errorf("x=%d: GlyphVerticalAdvance: got %v, want %v", x, got, fixed.Int26_6(advance))
		}
		bounds, gotAdv, err := f.GlyphVerticalBounds(&b, x, ppem, font.HintingNone)
		if err != nil {
			t.Errorf("x=%d: GlyphVerticalBounds: %v", x, err)
			continue
		}
		if gotAdv != advance {
			t.Errorf("x=%d: advance: got %v, want %v", x, gotAdv, fixed.Int26_6(advance))
		}
		if bounds.Empty() {
			continue
		}
		// Within a pixel of rounding, the free space above and below the
		// glyph in the 20-pixel box is equal.
		above := bounds.Min.Y
		below := advance - bounds.Max.Y
		if above < 0 || below < 0 {
			t.Errorf("x=%d: glyph escapes the one-em box: bounds=%v", x, bounds)
			continue
		}
		if d := above - below; d < -64 || d > 64 {
			t.Errorf("x=%d: glyph not centered at ppem 20: bounds=%v, above=%v, below=%v", x, bounds, above, below)
		}
	}
}
