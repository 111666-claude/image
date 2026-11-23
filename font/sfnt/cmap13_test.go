// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"fmt"
	"sync"
	"testing"
	"unicode"
)

// This file tests cmap format 13 (many-to-one range mappings) using
// hand-constructed SFNT fonts. The fonts carry only the tables that the
// parser requires to get as far as parseCmap; the glyph outlines themselves
// are empty.

// cmapGroup is one (start, end, glyph) triple. For format 12 it is a
// one-to-one mapping starting at glyph; for format 13 every code point in the
// range maps to exactly glyph. Fields are uint32 (not rune) so malformed
// groups beyond unicode.MaxRune can be expressed.
type cmapGroup struct {
	start, end uint32
	glyph      GlyphIndex
}

func putU16(b []byte, v uint16) {
	b[0] = byte(v >> 8)
	b[1] = byte(v >> 0)
}

func putU32(b []byte, v uint32) {
	b[0] = byte(v >> 24)
	b[1] = byte(v >> 16)
	b[2] = byte(v >> 8)
	b[3] = byte(v >> 0)
}

// buildFormat12Or13Subtable builds a format 12 or format 13 cmap subtable.
// The two formats share the 16-byte header and 12-byte group layout.
func buildFormat12Or13Subtable(format uint16, groups []cmapGroup) []byte {
	const headerSize, groupSize = 16, 12
	sub := make([]byte, headerSize+groupSize*len(groups))
	putU16(sub[0:], format)
	putU16(sub[2:], 0) // reserved
	putU32(sub[4:], uint32(len(sub)))
	putU32(sub[8:], 0) // language
	putU32(sub[12:], uint32(len(groups)))
	for i, g := range groups {
		o := headerSize + groupSize*i
		putU32(sub[o+0:], g.start)
		putU32(sub[o+4:], g.end)
		putU32(sub[o+8:], uint32(g.glyph))
	}
	return sub
}

// buildFormat4Subtable builds a BMP-only format 4 subtable whose segments all
// use idDelta (idRangeOffset = 0).
func buildFormat4Subtable(groups []cmapGroup) []byte {
	segCount := len(groups) + 1 // +1 for the mandatory 0xFFFF sentinel.
	segCountX2 := uint16(2 * segCount)
	searchRange := uint16(2)
	for searchRange*2 <= segCountX2 {
		searchRange *= 2
	}
	entrySelector := uint16(0)
	for 1<<(entrySelector+1) <= segCount {
		entrySelector++
	}
	rangeShift := segCountX2 - searchRange

	const headerSize = 14
	size := headerSize + 8*int(segCount)
	sub := make([]byte, size)
	putU16(sub[0:], 4)
	putU16(sub[2:], uint16(size))
	putU16(sub[4:], 0) // language
	putU16(sub[6:], segCountX2)
	putU16(sub[8:], searchRange)
	putU16(sub[10:], entrySelector)
	putU16(sub[12:], rangeShift)

	// endCodes: segCountX2 bytes at offset 14.
	// reservedPad: 2 bytes.
	// startCodes: segCountX2 bytes.
	// idDelta: segCountX2 bytes.
	// idRangeOffset: segCountX2 bytes, all zero.
	endCodes := sub[14:]
	startCodes := sub[14+2*segCount+2:]
	idDeltas := sub[14+4*segCount+2:]
	for i, g := range groups {
		putU16(endCodes[2*i:], uint16(g.end))
		putU16(startCodes[2*i:], uint16(g.start))
		// Glyph index = (code + idDelta) mod 65536.
		putU16(idDeltas[2*i:], uint16(int32(g.glyph)-int32(g.start)))
	}
	putU16(endCodes[2*(segCount-1):], 0xFFFF)
	putU16(startCodes[2*(segCount-1):], 0xFFFF)
	putU16(idDeltas[2*(segCount-1):], 1)
	return sub
}

type cmapSubtable struct {
	pid, psid uint16
	data      []byte
}

// buildCmap wraps one or more subtables in a cmap table.
func buildCmap(subs []cmapSubtable) []byte {
	num := len(subs)
	headerSize := 4 + 8*num
	total := headerSize
	for _, s := range subs {
		total += len(s.data)
	}
	out := make([]byte, headerSize, total)
	putU16(out[0:], 0) // version
	putU16(out[2:], uint16(num))

	offset := uint32(headerSize)
	for i, s := range subs {
		rec := out[4+8*i : 4+8*(i+1)]
		putU16(rec[0:], s.pid)
		putU16(rec[2:], s.psid)
		putU32(rec[4:], offset)
		offset += uint32(len(s.data))
		out = append(out, s.data...)
	}
	return out
}

// buildMinimalFont assembles a parseable TrueType font with the given cmap
// table and numGlyphs (the glyphs are empty).
func buildMinimalFont(cmapData []byte, numGlyphs uint16) []byte {
	// head: exactly 54 bytes, long loca format.
	head := make([]byte, 54)
	putU32(head[0:], 0x00010000)  // version
	putU32(head[4:], 0x00010000)  // fontRevision
	putU32(head[12:], 0x5F0F3CF5) // magicNumber
	putU16(head[18:], 1000)       // unitsPerEm
	putU16(head[50:], 1)          // indexToLocFormat = long

	// hhea: exactly 36 bytes, one horizontal metric.
	hhea := make([]byte, 36)
	putU32(hhea[0:], 0x00010000)
	putU16(hhea[34:], 1) // numberOfHMetrics

	// maxp: exactly 32 bytes (version 1.0).
	maxp := make([]byte, 32)
	putU32(maxp[0:], 0x00010000)
	putU16(maxp[4:], numGlyphs)

	// hmtx: one metric.
	hmtx := make([]byte, 4)
	putU16(hmtx[0:], 500) // advanceWidth
	putU16(hmtx[2:], 0)   // leftSideBearing

	// loca: long format, numGlyphs+1 zeros; glyf is empty.
	loca := make([]byte, 4*(numGlyphs+1))
	glyf := []byte{}

	// post: version 3.0, 32 bytes, no glyph names.
	post := make([]byte, 32)
	putU32(post[0:], 0x00030000)

	type ttab struct {
		tag  uint32
		data []byte
	}
	tables := []ttab{
		{0x636d6170, cmapData}, // "cmap"
		{0x676c7966, glyf},     // "glyf"
		{0x68656164, head},     // "head"
		{0x68686561, hhea},     // "hhea"
		{0x686d7478, hmtx},     // "hmtx"
		{0x6c6f6361, loca},     // "loca"
		{0x6d617870, maxp},     // "maxp"
		{0x706f7374, post},     // "post"
	}

	numTables := uint16(len(tables))
	searchRange := uint16(16)
	for searchRange*2 <= 16*numTables {
		searchRange *= 2
	}
	entrySelector := uint16(0)
	for 1<<(entrySelector+1) <= numTables {
		entrySelector++
	}
	rangeShift := 16*numTables - searchRange

	header := make([]byte, 12+16*numTables)
	putU32(header[0:], 0x00010000)
	putU16(header[4:], numTables)
	putU16(header[6:], searchRange)
	putU16(header[8:], entrySelector)
	putU16(header[10:], rangeShift)

	tableOffset := uint32(len(header))
	body := []byte{}
	for i, t := range tables {
		rec := header[12+16*i : 28+16*i]
		putU32(rec[0:], t.tag)
		putU32(rec[4:], 0) // checksum, ignored
		putU32(rec[8:], tableOffset)
		putU32(rec[12:], uint32(len(t.data)))

		body = append(body, t.data...)
		tableOffset += uint32(len(t.data))
		if len(t.data)&3 != 0 {
			pad := 4 - len(t.data)&3
			body = append(body, make([]byte, pad)...)
			tableOffset += uint32(pad)
		}
	}

	var font bytes.Buffer
	font.Write(header)
	font.Write(body)
	return font.Bytes()
}

// parseFont13 builds and parses a font whose cmap contains a single
// Windows-UCS4 (width-4) format 13 subtable.
func parseFont13(t *testing.T, groups []cmapGroup, numGlyphs uint16) *Font {
	t.Helper()
	sub := buildFormat12Or13Subtable(13, groups)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), numGlyphs)
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	return f
}

func TestCmapFormat13GlyphIndex(t *testing.T) {
	const numGlyphs = 16
	groups := []cmapGroup{
		{0x0000, 0x001F, 1},                    // controls -> glyph 1
		{'A', 'A', 7},                          // single-point group
		{'B', 'D', 8},                          // 3 code points share one glyph
		{0xFFFF, 0xFFFF, 9},                    // BMP boundary
		{0x10000, 0x10000, 10},                 // first non-BMP code point
		{0x1F4A9, 0x1F4A9, 11},                 // pile of poo
		{0x2FA00, 0x2FAFF, 12},                 // wide CJK compatibility range
		{unicode.MaxRune, unicode.MaxRune, 13}, // Unicode boundary
	}
	f := parseFont13(t, groups, numGlyphs)

	testCases := []struct {
		r    rune
		want GlyphIndex
	}{
		// Inside the control range: many-to-one, including the endpoints.
		{0x0000, 1},
		{0x001F, 1},
		{0x0010, 1},

		// Single point.
		{'A', 7},

		// Every code point in a many-to-one range yields the same glyph.
		{'B', 8},
		{'C', 8},
		{'D', 8},

		// BMP / non-BMP boundaries.
		{0xFFFF, 9},
		{0x10000, 10},
		{0x1F4A9, 11},

		// Wide range endpoints and interior.
		{0x2FA00, 12},
		{0x2FA7D, 12},
		{0x2FAFF, 12},

		// Highest valid Unicode code point.
		{unicode.MaxRune, 13},

		// Fallback: gaps, just-outside-range points, unassigned planes,
		// out-of-Unicode and negative runes all map to glyph 0 with no error.
		{'E', 0},
		{0x0020, 0},
		{0xFFFE, 0},
		{0x10001, 0},
		{0x2FB00, 0},
		{unicode.MaxRune + 1, 0},
		{-1, 0},
	}

	var b Buffer
	for _, tc := range testCases {
		got, err := f.GlyphIndex(&b, tc.r)
		if err != nil {
			t.Errorf("r=%#x: GlyphIndex: %v", tc.r, err)
			continue
		}
		if got != tc.want {
			t.Errorf("r=%#x: got glyph %d, want %d", tc.r, got, tc.want)
		}
	}
}

func TestCmapFormat13EmptyGroups(t *testing.T) {
	// A zero-group subtable is structurally valid and simply maps everything
	// to glyph 0.
	f := parseFont13(t, nil, 4)
	var b Buffer
	for _, r := range []rune{0, 'A', 0xFFFF, 0x10000, unicode.MaxRune} {
		got, err := f.GlyphIndex(&b, r)
		if err != nil {
			t.Errorf("r=%#x: %v", r, err)
		} else if got != 0 {
			t.Errorf("r=%#x: got %d, want 0", r, got)
		}
	}
}

func TestCmapFormat13Errors(t *testing.T) {
	const numGlyphs = 16

	tests := []struct {
		desc   string
		groups []cmapGroup
		want   error
		substr string
	}{{
		desc: "start greater than end",
		groups: []cmapGroup{
			{0x100, 0x0FF, 1},
		},
		want:   errInvalidCmapTable,
		substr: "group 0",
	}, {
		desc: "overlapping adjacent groups",
		groups: []cmapGroup{
			{0x100, 0x200, 1},
			{0x200, 0x300, 2},
		},
		want:   errInvalidCmapTable,
		substr: "overlaps",
	}, {
		desc: "overlapping contained groups",
		groups: []cmapGroup{
			{0x100, 0x400, 1},
			{0x200, 0x200, 2},
		},
		want:   errInvalidCmapTable,
		substr: "group 1",
	}, {
		desc: "unsorted groups",
		groups: []cmapGroup{
			{0x200, 0x300, 1},
			{0x100, 0x150, 2},
		},
		want:   errInvalidCmapTable,
		substr: "overlaps",
	}, {
		desc: "start beyond Unicode max",
		groups: []cmapGroup{
			{unicode.MaxRune + 1, unicode.MaxRune + 1, 1},
		},
		want:   errInvalidCmapTable,
		substr: "beyond Unicode max",
	}, {
		desc: "end beyond Unicode max",
		groups: []cmapGroup{
			{unicode.MaxRune, 0xFFFFFFFF, 1},
		},
		want:   errInvalidCmapTable,
		substr: "beyond Unicode max",
	}, {
		desc: "glyph index equals NumGlyphs",
		groups: []cmapGroup{
			{'A', 'A', numGlyphs},
		},
		want:   errInvalidCmapTable,
		substr: "font only has",
	}, {
		desc: "glyph index far beyond NumGlyphs",
		groups: []cmapGroup{
			{'A', 'A', 0xFFFF},
		},
		want:   errInvalidCmapTable,
		substr: fmt.Sprintf("group 0"),
	}, {
		desc: "error context names later group",
		groups: []cmapGroup{
			{'A', 'A', 1},
			{'B', 'B', 2},
			{'C', 'C', numGlyphs},
		},
		want:   errInvalidCmapTable,
		substr: "group 2",
	}}

	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			sub := buildFormat12Or13Subtable(13, tc.groups)
			data := buildMinimalFont(buildCmap([]cmapSubtable{
				{pidWindows, psidWindowsUCS4, sub},
			}), numGlyphs)
			_, err := Parse(data)
			if err == nil {
				t.Fatalf("Parse: want error containing %q, got nil", tc.substr)
			}
			if !errors.Is(err, tc.want) {
				t.Errorf("errors.Is: got %v, want wrapping %v", err, tc.want)
			}
			if !bytes.Contains([]byte(err.Error()), []byte(tc.substr)) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.substr)
			}
		})
	}
}

func TestCmapFormat13TooManyGroups(t *testing.T) {
	// A subtable claiming more than maxCmapSegments groups is rejected, and the
	// error carries both the advertised count and the limit.
	groups := make([]cmapGroup, maxCmapSegments+1)
	for i := range groups {
		groups[i] = cmapGroup{uint32(i), uint32(i), 1}
	}
	sub := buildFormat12Or13Subtable(13, groups)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 4)
	_, err := Parse(data)
	if !errors.Is(err, errUnsupportedNumberOfCmapSegments) {
		t.Fatalf("got %v, want %v", err, errUnsupportedNumberOfCmapSegments)
	}
	msg := err.Error()
	if !bytes.Contains([]byte(msg), []byte(fmt.Sprintf("numGroups=%d", maxCmapSegments+1))) {
		t.Errorf("error %q missing advertised numGroups", msg)
	}
	if !bytes.Contains([]byte(msg), []byte(fmt.Sprintf("limit %d", maxCmapSegments))) {
		t.Errorf("error %q missing limit", msg)
	}
}

func TestCmapFormat13Truncated(t *testing.T) {
	groups := []cmapGroup{{'A', 'Z', 1}}
	sub := buildFormat12Or13Subtable(13, groups)

	// Claim more groups than bytes are present: patch numGroups from 1 to 2 so
	// the subtable's declared length is inconsistent with its data.
	putU32(sub[12:], 2)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 4)
	if _, err := Parse(data); !errors.Is(err, errInvalidCmapTable) {
		t.Fatalf("inconsistent length: got %v, want %v", err, errInvalidCmapTable)
	}

	// Physically truncate the font inside the group records.
	sub = buildFormat12Or13Subtable(13, groups)
	data = buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 4)
	if _, err := Parse(data[:len(data)-4]); err == nil {
		t.Fatalf("truncated font: want error, got nil")
	}

	// Shrink the cmap table's recorded length (in its table directory record)
	// so that fewer than the 16-byte format 13 header remain after the
	// subtable offset. The subtable's own format bytes are still readable, so
	// the table gets selected and the failure occurs inside the format 13
	// maker.
	sub = buildFormat12Or13Subtable(13, groups)
	cmapData := buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	})
	data = buildMinimalFont(cmapData, 4)
	// The cmap table is the first directory record, starting at font offset 12;
	// its length field is at record offset 12.
	putU32(data[24:], 4+8+4)
	if _, err := Parse(data); !errors.Is(err, errInvalidCmapTable) {
		t.Fatalf("short header: got %v, want %v", err, errInvalidCmapTable)
	}

	// The subtable's declared u32 length claims more bytes than the cmap table
	// has.
	sub = buildFormat12Or13Subtable(13, groups)
	putU32(sub[4:], 0xFFFFFFFF)
	data = buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 4)
	if _, err := Parse(data); !errors.Is(err, errInvalidCmapTable) {
		t.Fatalf("oversized length: got %v, want %v", err, errInvalidCmapTable)
	}
}

func TestCmapSubtablePriority(t *testing.T) {
	const numGlyphs = 16

	// Width first: a width-2 format 4 and a width-4 format 13 coexist; the
	// format 13 subtable wins even though it is listed second.
	t.Run("format 4 and format 13 coexist", func(t *testing.T) {
		f4 := buildFormat4Subtable([]cmapGroup{
			{'A', 'A', 5},
			{'B', 'B', 6},
		})
		f13 := buildFormat12Or13Subtable(13, []cmapGroup{
			{'A', 'A', 7},
			{0x1F4A9, 0x1F4A9, 11},
		})
		data := buildMinimalFont(buildCmap([]cmapSubtable{
			{pidWindows, psidWindowsUCS2, f4},
			{pidWindows, psidWindowsUCS4, f13},
		}), numGlyphs)
		f, err := Parse(data)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		var b Buffer
		if got, _ := f.GlyphIndex(&b, 'A'); got != 7 {
			t.Errorf("'A': got %d, want 7 from width-4 format 13", got)
		}
		if got, _ := f.GlyphIndex(&b, 0x1F4A9); got != 11 {
			t.Errorf("non-BMP: got %d, want 11", got)
		}
		// 'B' is only in the (losing) format 4 subtable: fallback to 0.
		if got, _ := f.GlyphIndex(&b, 'B'); got != 0 {
			t.Errorf("'B': got %d, want 0 (format 4 subtable not selected)", got)
		}
	})

	// Equal width: record order breaks the tie (first width-4 subtable wins).
	t.Run("format 12 and format 13 equal width order", func(t *testing.T) {
		f12 := buildFormat12Or13Subtable(12, []cmapGroup{{'A', 'C', 6}})
		f13 := buildFormat12Or13Subtable(13, []cmapGroup{{'A', 'C', 9}})
		data := buildMinimalFont(buildCmap([]cmapSubtable{
			{pidWindows, psidWindowsUCS4, f12},
			{pidUnicode, psidUnicode2FullRepertoire, f13},
		}), numGlyphs)
		f, err := Parse(data)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		var b Buffer
		// Format 12 is one-to-one: A=6, B=7, C=8.
		for r, want := range map[rune]GlyphIndex{'A': 6, 'B': 7, 'C': 8} {
			if got, _ := f.GlyphIndex(&b, r); got != want {
				t.Errorf("r=%q: got %d, want %d (format 12 should win)", r, got, want)
			}
		}
	})

	// Same again with format 13 first to prove order, not format number, wins.
	t.Run("format 13 first wins", func(t *testing.T) {
		f12 := buildFormat12Or13Subtable(12, []cmapGroup{{'A', 'C', 6}})
		f13 := buildFormat12Or13Subtable(13, []cmapGroup{{'A', 'C', 9}})
		data := buildMinimalFont(buildCmap([]cmapSubtable{
			{pidUnicode, psidUnicode2FullRepertoire, f13},
			{pidWindows, psidWindowsUCS4, f12},
		}), numGlyphs)
		f, err := Parse(data)
		if err != nil {
			t.Fatalf("Parse: %v", err)
		}
		var b Buffer
		for _, r := range []rune{'A', 'B', 'C'} {
			if got, _ := f.GlyphIndex(&b, r); got != 9 {
				t.Errorf("r=%q: got %d, want 9 (format 13 should win)", r, got)
			}
		}
	})
}

func TestCmapFormat4ForcedFromMixedFont(t *testing.T) {
	// Restricting supportedCmapFormat to format 4 must reproduce the legacy
	// BMP-only semantics against a font that also carries format 13.
	originalSupportedCmapFormat := supportedCmapFormat
	defer func() { supportedCmapFormat = originalSupportedCmapFormat }()
	supportedCmapFormat = func(format, pid, psid uint16) bool {
		return format == 4 && originalSupportedCmapFormat(format, pid, psid)
	}

	f4 := buildFormat4Subtable([]cmapGroup{
		{'A', 'A', 5},
		{'B', 'B', 6},
	})
	f13 := buildFormat12Or13Subtable(13, []cmapGroup{
		{'A', 'A', 7},
		{0x1F4A9, 0x1F4A9, 11},
	})
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS2, f4},
		{pidWindows, psidWindowsUCS4, f13},
	}), 16)
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	if got, _ := f.GlyphIndex(&b, 'A'); got != 5 {
		t.Errorf("'A': got %d, want 5 from forced format 4", got)
	}
	if got, _ := f.GlyphIndex(&b, 'B'); got != 6 {
		t.Errorf("'B': got %d, want 6 from forced format 4", got)
	}
	if got, _ := f.GlyphIndex(&b, 0x1F4A9); got != 0 {
		t.Errorf("non-BMP under format 4: got %d, want 0", got)
	}
}

func TestCmapFormat12Synthetic(t *testing.T) {
	// The format 12 path must keep its original one-to-one semantics on a
	// constructed font.
	groups := []cmapGroup{
		{'A', 'C', 6}, // A=6, B=7, C=8
		{0x1F4A9, 0x1F4A9, 11},
		{unicode.MaxRune, unicode.MaxRune, 13},
	}
	sub := buildFormat12Or13Subtable(12, groups)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 16)
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	for r, want := range map[rune]GlyphIndex{
		'A':                 6,
		'B':                 7,
		'C':                 8,
		'D':                 0,
		0x1F4A9:             11,
		unicode.MaxRune:     13,
		unicode.MaxRune + 1: 0,
	} {
		if got, err := f.GlyphIndex(&b, r); err != nil || got != want {
			t.Errorf("r=%#x: got (%d, %v), want (%d, nil)", r, got, err, want)
		}
	}
}

func TestCmapFormat13Concurrent(t *testing.T) {
	groups := []cmapGroup{
		{'A', 'Z', 1},
		{0x1F400, 0x1F4FF, 2},
	}
	f := parseFont13(t, groups, 8)

	const goroutines = 32
	const iterations = 200
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed rune) {
			defer wg.Done()
			var b Buffer
			for i := 0; i < iterations; i++ {
				r := 'A' + (seed+int32(i))%26
				got, err := f.GlyphIndex(&b, r)
				if err != nil || got != 1 {
					t.Errorf("BMP r=%#x: got (%d, %v), want (1, nil)", r, got, err)
					return
				}
				got, err = f.GlyphIndex(&b, 0x1F400+rune(i%0x100))
				if err != nil || got != 2 {
					t.Errorf("non-BMP iteration %d: got (%d, %v), want (2, nil)", i, got, err)
					return
				}
				if got, _ := f.GlyphIndex(&b, '0'); got != 0 {
					t.Errorf("fallback: got %d, want 0", got)
					return
				}
			}
		}(rune(g))
	}
	wg.Wait()
}

func TestCmapFormat13DoesNotModifySource(t *testing.T) {
	groups := []cmapGroup{{'A', 'Z', 1}, {0x1F4A9, 0x1F4A9, 2}}
	sub := buildFormat12Or13Subtable(13, groups)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 8)
	before := sha256.Sum256(data)

	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	var b Buffer
	for _, r := range []rune{'A', 'M', 'Z', 0x1F4A9, '0', unicode.MaxRune} {
		if _, err := f.GlyphIndex(&b, r); err != nil {
			t.Fatalf("GlyphIndex(%#x): %v", r, err)
		}
	}
	after := sha256.Sum256(data)
	if before != after {
		t.Fatalf("source bytes were modified by Parse/GlyphIndex")
	}
}

func TestCmapFormat13ReaderAt(t *testing.T) {
	// The io.ReaderAt source path re-uses Buffer scratch space differently
	// (source.view returns writable buffers), so exercise it explicitly.
	groups := []cmapGroup{
		{'A', 'Z', 1},
		{0x10000, 0x10002, 2},
	}
	sub := buildFormat12Or13Subtable(13, groups)
	data := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, sub},
	}), 8)

	f, err := ParseReaderAt(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("ParseReaderAt: %v", err)
	}
	var b Buffer
	for r, want := range map[rune]GlyphIndex{
		'@':     0,
		'A':     1,
		'Z':     1,
		'[':     0,
		0xFFFF:  0,
		0x10000: 2,
		0x10001: 2,
		0x10002: 2,
		0x10003: 0,
	} {
		got, err := f.GlyphIndex(&b, r)
		if err != nil || got != want {
			t.Errorf("r=%#x: got (%d, %v), want (%d, nil)", r, got, err, want)
		}
	}

	// A malformed format 13 table must fail equally through ParseReaderAt.
	bad := buildMinimalFont(buildCmap([]cmapSubtable{
		{pidWindows, psidWindowsUCS4, buildFormat12Or13Subtable(13, []cmapGroup{
			{0x200, 0x100, 1},
		})},
	}), 4)
	if _, err := ParseReaderAt(bytes.NewReader(bad)); !errors.Is(err, errInvalidCmapTable) {
		t.Errorf("malformed: got %v, want %v", err, errInvalidCmapTable)
	}
}
