// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

// This file is the structured fault-injection quality gate for the sfnt
// parser. Unlike single-point "does it error" tests, it generates a fixed,
// reproducible corpus of truncated, out-of-bounds and oversized-count font
// samples covering the cmap, name, GPOS, glyf/CFF and collection stages, then
// asserts for every sample that:
//
//   - Parse / ParseCollection never panic and never run past a deadline,
//   - GlyphIndex, LoadGlyph, Kern and Metrics never panic on a font that Parse
//     accepted,
//   - any fault is reported as an error rather than silently swallowed,
//   - allocation stays under a per-sample budget,
//
// The run emits one stable JSONL record per sample and compares it against a
// checked-in golden summary (fuzz_golden.jsonl). Everything is synthesized in
// memory; there is no network or host-font dependency.

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

var (
	// fuzzSeed is the single fixed seed for all randomized mutation. Changing it
	// changes the corpus; the default keeps the JSONL summary reproducible.
	fuzzSeed = flag.Uint64("fuzzSeed", 0x5f6e74, "fixed seed for structured sfnt fault injection")
	// fuzzIters is the number of randomized mutations per family.
	fuzzIters = flag.Int("fuzzIters", 24, "number of randomized fault samples per family")
	// fuzzTimeout bounds how long a single sample may take. It is deliberately
	// generous under -race, which slows the watchdog target considerably.
	fuzzTimeout = flag.Duration("fuzzTimeout", 5*time.Second, "per-sample parse/query deadline")
	// fuzzAllocCap is the default allocation ceiling (bytes) for a sample whose
	// declared budget is zero.
	fuzzAllocCap = flag.Int64("fuzzAllocCap", 4<<20, "default per-sample allocation budget in bytes")
	// fuzzSummary optionally writes the JSONL summary to this path in addition
	// to the test log. Empty means only log output and the golden comparison.
	fuzzSummary = flag.String("fuzzSummary", "", "optional path to write the JSONL summary")
	// fuzzUpdate overwrites the checked-in golden summary instead of comparing.
	fuzzUpdate = flag.Bool("fuzzUpdate", false, "regenerate fuzz_golden.jsonl")
)

// fuzzStage identifies the parser stage a sample is designed to exercise.
type fuzzStage string

const (
	stageCollection fuzzStage = "collection"
	stageDirectory  fuzzStage = "directory"
	stageHead       fuzzStage = "head"
	stageMaxp       fuzzStage = "maxp"
	stageGlyf       fuzzStage = "glyf"
	stageLoca       fuzzStage = "loca"
	stageCmap       fuzzStage = "cmap"
	stageKern       fuzzStage = "kern"
	stageGPOS       fuzzStage = "gpos"
	stageHhea       fuzzStage = "hhea"
	stageHmtx       fuzzStage = "hmtx"
	stageOS2        fuzzStage = "os2"
	stageName       fuzzStage = "name"
	stagePost       fuzzStage = "post"
	stageCFF        fuzzStage = "cff"
)

// fuzzFamily groups samples by the table or structure they mutate.
type fuzzFamily string

const (
	famCollection fuzzFamily = "collection"
	famCmap       fuzzFamily = "cmap"
	famName       fuzzFamily = "name"
	famGPOS       fuzzFamily = "gpos"
	famGlyf       fuzzFamily = "glyf"
	famCFF        fuzzFamily = "cff"
	famDirectory  fuzzFamily = "directory"
	famRandom     fuzzFamily = "random"
)

// fuzzRecord is one stable JSONL line. Fields are chosen so that the record is
// deterministic across runs and machines: no wall-clock durations, no paths.
type fuzzRecord struct {
	ID          string `json:"id"`
	Family      string `json:"family"`
	Stage       string `json:"stage"`
	Tag         string `json:"tag"`
	Offset      int    `json:"offset"`
	Length      int    `json:"length"`
	Kind        string `json:"kind"`
	DeclaredCap int64  `json:"declared_alloc_cap"`
	AllocBytes  int64  `json:"alloc_bytes"`
	Allocs      uint64 `json:"allocs"`
	Accepted    bool   `json:"accepted"`
	WantReject  bool   `json:"want_reject"`
	Panicked    bool   `json:"panicked"`
	TimedOut    bool   `json:"timed_out"`
	ParseErr    string `json:"parse_err,omitempty"`
	QueryErr    string `json:"query_err,omitempty"`
	API         string `json:"api"`
}

// fuzzSample is a single generated font plus the metadata describing it.
type fuzzSample struct {
	id     string
	family fuzzFamily
	stage  fuzzStage
	tag    string
	offset int // absolute file offset the mutation targets, when known.
	kind   string
	data   []byte
	// collection marks TTC data exercised through ParseCollection.
	collection bool
	// wantReject asserts that the fault is observable: for single fonts Parse
	// must return an error, for collections either ParseCollection or a member
	// Font call must. It is set on the deterministic truncation / out-of-bounds
	// / oversized-count / bad-magic samples and left false for random flips,
	// which a parser may legitimately tolerate as long as queries stay safe.
	wantReject bool
	// capBytes overrides the default allocation budget for this sample.
	capBytes int64
}

// truncateBytes shortens the slice to a deterministic length in [1, len].
func truncateBytes(rng *splitMix64, b []byte) []byte {
	n := 1 + int(rng.next()%uint64(len(b)))
	return b[:n]
}

// setU16 writes a (possibly oversized) uint16 at off.
func setU16(b []byte, off int, v uint16) {
	if off+2 <= len(b) {
		putU16(b[off:off+2], v)
	}
}

// setU32 writes a (possibly oversized) uint32 at off.
func setU32(b []byte, off int, v uint32) {
	if off+4 <= len(b) {
		putU32(b[off:off+4], v)
	}
}

// clone returns an independent copy of b so mutations never alias the builder
// output reused across samples.
func clone(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

// tableRecord returns the (offset, length) recorded in the sfnt directory for
// tag, or (0,0,false) if the tag is absent.
func tableRecord(b []byte, tag string) (off, length int, ok bool) {
	if len(b) < 12 {
		return 0, 0, false
	}
	n := int(u16(b[4:]))
	for i := 0; i < n; i++ {
		r := 12 + 16*i
		if r+16 > len(b) {
			return 0, 0, false
		}
		if string(b[r:r+4]) == tag {
			return int(u32(b[r+8:])), int(u32(b[r+12:])), true
		}
	}
	return 0, 0, false
}

// setTableRecordLength overwrites the length field of tag's directory record.
func setTableRecordLength(b []byte, tag string, length uint32) bool {
	n := int(u16(b[4:]))
	for i := 0; i < n; i++ {
		r := 12 + 16*i
		if r+16 > len(b) {
			return false
		}
		if string(b[r:r+4]) == tag {
			putU32(b[r+12:r+16], length)
			return true
		}
	}
	return false
}

// setTableRecordOffset overwrites the offset field of tag's directory record.
func setTableRecordOffset(b []byte, tag string, offset uint32) bool {
	n := int(u16(b[4:]))
	for i := 0; i < n; i++ {
		r := 12 + 16*i
		if r+16 > len(b) {
			return false
		}
		if string(b[r:r+4]) == tag {
			putU32(b[r+8:r+12], offset)
			return true
		}
	}
	return false
}

// buildCorpus constructs the full deterministic sample list. Named samples
// cover specific regressions; randomized samples broaden coverage while
// remaining reproducible through fuzzSeed.
func buildCorpus(t *testing.T) []fuzzSample {
	t.Helper()
	rng := newSplitMix64(*fuzzSeed)

	baseTTF := buildMinimalTrueType(minimalFontOptions{numGlyphs: 4, addKern: true, addGPOS: true})
	compoundTTF := buildMinimalTrueType(minimalFontOptions{numGlyphs: 4, compound: true})
	gposBase := buildMinimalTrueType(minimalFontOptions{numGlyphs: 4, addGPOS: true})

	var samples []fuzzSample

	// ---- Collection family -------------------------------------------------
	addCollectionSamples(&samples, baseTTF)

	// ---- cmap family -------------------------------------------------------
	addCmapSamples(&samples, rng, baseTTF)

	// ---- name family -------------------------------------------------------
	addNameSamples(&samples, rng, baseTTF)

	// ---- GPOS family -------------------------------------------------------
	addGPOSSamples(&samples, gposBase)

	// ---- glyf / compound family --------------------------------------------
	addGlyfSamples(&samples, rng, compoundTTF)

	// ---- CFF family (built from the checked-in local OTF fixture) ----------
	addCFFSamples(&samples, rng, t)

	// ---- directory / offset / length family --------------------------------
	addDirectorySamples(&samples, rng, baseTTF)

	// ---- randomized whole-font mutations -----------------------------------
	addRandomSamples(&samples, rng, baseTTF, *fuzzIters)

	return samples
}

// addCollectionSamples exercises ParseCollection and per-member Font access.
func addCollectionSamples(samples *[]fuzzSample, ttf []byte) {
	specs := []ttcFontSpec{
		{tables: minimalTrueTypeTables(minimalFontOptions{numGlyphs: 4}), magic: 0x00010000},
		{tables: minimalTrueTypeTables(minimalFontOptions{numGlyphs: 4, addKern: true}), magic: 0x00010000},
	}
	goodTTC := buildTTC(specs)

	add := func(id, kind string, data []byte, off int, capBytes int64) {
		*samples = append(*samples, fuzzSample{
			id: id, family: famCollection, stage: stageCollection, tag: "ttcf",
			offset: off, kind: kind, data: data, collection: true, capBytes: capBytes,
			wantReject: kind != "valid",
		})
	}

	// Valid collection: must parse and expose both members.
	add("collection/valid", "valid", goodTTC, 0, 0)

	// Zero fonts in the TTC header (numFonts at offset 8).
	b := clone(goodTTC)
	setU32(b, 8, 0)
	add("collection/zero-fonts", "oversized-count", b, 8, 0)

	// Oversized font count claiming more members than the file holds.
	b = clone(goodTTC)
	setU32(b, 8, maxNumFonts+1)
	add("collection/too-many-fonts", "oversized-count", b, 8, 0)

	// Font offset table pointing beyond the file.
	b = clone(goodTTC)
	setU32(b, 12, uint32(len(goodTTC))+1024)
	add("collection/font-offset-oob", "out-of-bounds", b, 12, 0)

	// Truncated collection: cut before the second member's offset table.
	cut := len(goodTTC) / 2
	add("collection/truncated", "truncation", goodTTC[:cut], cut, 0)

	// Truncated to just the 12-byte TTC header, before any offset array.
	add("collection/header-only", "truncation", goodTTC[:12], 12, 0)

	// Bad collection magic.
	b = clone(goodTTC)
	b[0], b[1], b[2], b[3] = 'x', 'x', 'x', 'x'
	add("collection/bad-magic", "bad-magic", b, 0, 0)
}

// addCmapSamples targets cmap offsets, subtables and segment counts.
func addCmapSamples(samples *[]fuzzSample, rng *splitMix64, ttf []byte) {
	cmapOff, cmapLen, ok := tableRecord(ttf, "cmap")
	if !ok {
		return
	}
	add := func(id, kind string, mut func([]byte), off int) {
		b := clone(ttf)
		mut(b)
		*samples = append(*samples, fuzzSample{
			id: "cmap/" + id, family: famCmap, stage: stageCmap, tag: "cmap",
			offset: off, kind: kind, data: b, wantReject: true,
		})
	}

	// Oversized subtable count in the cmap header (numSubtables at +2).
	add("huge-num-subtables", "oversized-count",
		func(b []byte) { setU16(b, cmapOff+2, 0xffff) }, cmapOff+2)

	// Truncate the cmap table mid-subtable by shrinking its directory length.
	add("length-truncated", "truncation",
		func(b []byte) { setTableRecordLength(b, "cmap", uint32(cmapLen/2)) }, cmapOff)

	// cmap length shorter than its 4-byte header.
	add("length-too-short", "truncation",
		func(b []byte) { setTableRecordLength(b, "cmap", 2) }, cmapOff)

	// Oversized format-4 segment count: segCountX2 is at subtable offset 6; the
	// single subtable starts after the 4+8 byte cmap header.
	subOff := cmapOff + 12
	add("format4-huge-segcount", "oversized-count",
		func(b []byte) { setU16(b, subOff+6, 0xffff) }, subOff+6)

	// Subtable offset beyond the cmap table (encoding record at +4).
	add("subtable-offset-oob", "out-of-bounds",
		func(b []byte) { setU32(b, cmapOff+4+4, uint32(cmapLen)+64) }, cmapOff+8)

	// Truncate the whole file inside the cmap table.
	cut := cmapOff + cmapLen/2
	*samples = append(*samples, fuzzSample{
		id: "cmap/file-truncated", family: famCmap, stage: stageCmap, tag: "cmap",
		offset: cut, kind: "truncation", data: clone(ttf)[:cut],
	})

	// Randomized single-byte edits within cmap bounds. A flip may or may not
	// invalidate parsing, so these do not set wantReject; the gate still
	// guarantees no panic and query-safety.
	for i := 0; i < 3; i++ {
		b := clone(ttf)
		pos := cmapOff + int(rng.next()%uint64(cmapLen))
		b[pos] ^= byte(1 + rng.next()%255)
		*samples = append(*samples, fuzzSample{
			id: fmt.Sprintf("cmap/random-flip-%d", i), family: famCmap, stage: stageCmap,
			tag: "cmap", offset: pos, kind: "random-flip", data: b,
		})
	}
}

// addNameSamples targets the name table header and string storage.
func addNameSamples(samples *[]fuzzSample, rng *splitMix64, ttf []byte) {
	nameOff, nameLen, ok := tableRecord(ttf, "name")
	if !ok {
		return
	}
	add := func(id, kind string, mut func([]byte), off int) {
		b := clone(ttf)
		mut(b)
		*samples = append(*samples, fuzzSample{
			id: "name/" + id, family: famName, stage: stageName, tag: "name",
			offset: off, kind: kind, data: b, wantReject: true,
		})
	}

	// name table shorter than its 6-byte header.
	add("length-too-short", "truncation",
		func(b []byte) { setTableRecordLength(b, "name", 4) }, nameOff)

	// Oversized name record count (numSubtables at +2).
	add("huge-num-records", "oversized-count",
		func(b []byte) { setU16(b, nameOff+2, 0xffff) }, nameOff+2)

	// Name record claiming a string whose end lies beyond the table. The view
	// is bounded by nameLength, so inflate both the record's string length and
	// offset to guarantee the [offset, offset+length) range escapes the table.
	add("string-offset-oob", "out-of-bounds",
		func(b []byte) {
			rec := nameOff + 6                 // the single name record.
			setU16(b, rec+8, uint16(nameLen))  // nameLength larger than table.
			setU16(b, rec+10, uint16(nameLen)) // nameOffset at end of table.
		}, nameOff+6+10)

	// Truncate the file inside the name table.
	cut := nameOff + nameLen/2
	*samples = append(*samples, fuzzSample{
		id: "name/file-truncated", family: famName, stage: stageName, tag: "name",
		offset: cut, kind: "truncation", data: clone(ttf)[:cut],
	})

	// A random flip within the name table.
	b := clone(ttf)
	pos := nameOff + int(rng.next()%uint64(nameLen))
	b[pos] ^= byte(1 + rng.next()%255)
	*samples = append(*samples, fuzzSample{
		id: "name/random-flip", family: famName, stage: stageName, tag: "name",
		offset: pos, kind: "random-flip", data: b,
	})
}

// addGPOSSamples injects malformed PairPos subtables through the valid GPOS
// skeleton, exercising eager parse-time and lazy Kern-time validation.
func addGPOSSamples(samples *[]fuzzSample, ttf []byte) {
	add := func(id, kind string, sub []byte, off int) {
		b := buildMinimalTrueType(minimalFontOptions{
			numGlyphs: 4, addGPOS: true, gposSubtable: sub,
		})
		*samples = append(*samples, fuzzSample{
			id: "gpos/" + id, family: famGPOS, stage: stageGPOS, tag: "GPOS",
			offset: off, kind: kind, data: b, wantReject: kind != "unsupported",
		})
	}

	valid := buildValidPairPosSubtable()

	// Oversized pairValueCount in the PairSet (at subtable offset 20).
	bad := clone(valid)
	putU16(bad[20:22], 1000)
	add("pairset-huge-count", "oversized-count", bad, 20)

	// Coverage offset pointing outside the subtable.
	bad = clone(valid)
	putU16(bad[2:4], 0xffff)
	add("coverage-offset-oob", "out-of-bounds", bad, 2)

	// PairSet offset pointing outside the subtable.
	bad = clone(valid)
	putU16(bad[10:12], 0xfffe)
	add("pairset-offset-oob", "out-of-bounds", bad, 10)

	// Truncated subtable: cut inside the PairPos header so the eager
	// varLenView of the pairSet array runs past the subtable bounds.
	add("subtable-truncated", "truncation", valid[:8], 8)

	// Unsupported value format (valueFormat1 != X_ADVANCE). Parsing returns no
	// kern func rather than an error; the gate still asserts no panic.
	bad = clone(valid)
	putU16(bad[4:6], 0x0008)
	add("unsupported-value-format", "unsupported", bad, 4)
}

// addGlyfSamples targets simple and compound glyph parsing, including deep and
// self-referential compounds.
func addGlyfSamples(samples *[]fuzzSample, rng *splitMix64, compoundTTF []byte) {
	glyfOff, glyfLen, ok := tableRecord(compoundTTF, "glyf")
	if !ok {
		return
	}

	// A deterministic byte flip inside the glyf table. It may be tolerated at
	// parse time and only fail when the glyph is loaded, or tolerated entirely,
	// so wantReject stays false; the runner still proves no panic and
	// query-safety.
	pos := glyfOff + int(rng.next()%uint64(glyfLen))
	b := clone(compoundTTF)
	b[pos] ^= byte(1 + rng.next()%255)
	*samples = append(*samples, fuzzSample{
		id: "glyf/random-flip", family: famGlyf, stage: stageGlyf, tag: "glyf",
		offset: pos, kind: "random-flip", data: b,
	})

	// loca/glyf length mismatch: claim a loca table half its real size.
	locaOff, locaLen, ok := tableRecord(compoundTTF, "loca")
	if ok {
		b := clone(compoundTTF)
		setTableRecordLength(b, "loca", uint32(locaLen/2))
		*samples = append(*samples, fuzzSample{
			id: "loca/length-mismatch", family: famGlyf, stage: stageLoca, tag: "loca",
			offset: locaOff, kind: "truncation", data: b, wantReject: true,
		})
	}

	// Oversized glyph count in maxp (numGlyphs at +4).
	maxpOff, _, ok := tableRecord(compoundTTF, "maxp")
	if ok {
		b := clone(compoundTTF)
		setU16(b, maxpOff+4, 0xffff)
		*samples = append(*samples, fuzzSample{
			id: "maxp/huge-num-glyphs", family: famGlyf, stage: stageMaxp, tag: "maxp",
			offset: maxpOff + 4, kind: "oversized-count", data: b, wantReject: true,
		})
	}

	// Truncate the file inside the glyf table so a loca offset points past EOF.
	cut := glyfOff + glyfLen/2
	*samples = append(*samples, fuzzSample{
		id: "glyf/file-truncated", family: famGlyf, stage: stageGlyf, tag: "glyf",
		offset: cut, kind: "truncation", data: clone(compoundTTF)[:cut], wantReject: true,
	})

	// Self-referential compound: rewrite glyph 1's component glyphIndex to
	// point at glyph 1 itself. loadCompoundGlyf must terminate via its
	// recursion-depth / stack-size guards rather than loop forever.
	if locaOff, locaLen, lok := tableRecord(compoundTTF, "loca"); lok && locaLen >= 12 {
		g1Start := glyfOff + int(u32(compoundTTF[locaOff+4:]))
		// glyf header is 10 bytes; the component's glyphIndex is the uint16 at
		// compound-data offset 2.
		compIndexOff := g1Start + glyfHeaderLen + 2
		b := clone(compoundTTF)
		if compIndexOff+2 <= len(b) {
			putU16(b[compIndexOff:compIndexOff+2], 1) // component -> glyph 1.
			*samples = append(*samples, fuzzSample{
				id: "glyf/self-referential-compound", family: famGlyf, stage: stageGlyf,
				tag: "glyf", offset: compIndexOff, kind: "compound-cycle",
				data: b, wantReject: true,
			})
		}
	}
}

// addCFFSamples mutates the checked-in CFFTest.otf fixture, which lives in the
// module's local testdata and therefore requires no network or host fonts.
func addCFFSamples(samples *[]fuzzSample, rng *splitMix64, t *testing.T) {
	otf, err := os.ReadFile(filepath.FromSlash("../testdata/CFFTest.otf"))
	if err != nil {
		t.Fatalf("read CFF fixture: %v", err)
	}
	cffOff, cffLen, ok := tableRecord(otf, "CFF ")
	if !ok {
		t.Fatal("CFF table not found in fixture")
	}
	add := func(id, kind string, mut func([]byte), off int) {
		b := clone(otf)
		mut(b)
		*samples = append(*samples, fuzzSample{
			id: "cff/" + id, family: famCFF, stage: stageCFF, tag: "CFF ",
			offset: off, kind: kind, data: b, wantReject: true,
		})
	}

	// Bad CFF major version (first byte must be 1).
	add("bad-version", "bad-magic", func(b []byte) { b[cffOff] = 9 }, cffOff)

	// Truncate the CFF table via its directory length.
	add("length-truncated", "truncation",
		func(b []byte) { setTableRecordLength(b, "CFF ", uint32(cffLen/2)) }, cffOff)

	// Truncate the file inside the CFF table.
	cut := cffOff + cffLen/2
	*samples = append(*samples, fuzzSample{
		id: "cff/file-truncated", family: famCFF, stage: stageCFF, tag: "CFF ",
		offset: cut, kind: "truncation", data: clone(otf)[:cut], wantReject: true,
	})

	// Random flip inside CFF bounds.
	pos := cffOff + int(rng.next()%uint64(cffLen))
	b := clone(otf)
	b[pos] ^= byte(1 + rng.next()%255)
	*samples = append(*samples, fuzzSample{
		id: "cff/random-flip", family: famCFF, stage: stageCFF, tag: "CFF ",
		offset: pos, kind: "random-flip", data: b,
	})
}

// addDirectorySamples corrupts the top-level offset table and table records.
func addDirectorySamples(samples *[]fuzzSample, rng *splitMix64, ttf []byte) {
	add := func(id string, stage fuzzStage, kind string, mut func([]byte), off int) {
		b := clone(ttf)
		mut(b)
		*samples = append(*samples, fuzzSample{
			id: "directory/" + id, family: famDirectory, stage: stage, tag: "offset-table",
			offset: off, kind: kind, data: b, wantReject: true,
		})
	}

	// Oversized table count.
	add("huge-num-tables", stageDirectory, "oversized-count",
		func(b []byte) { setU16(b, 4, maxNumTables+1) }, 4)

	// Unrecognized sfnt magic.
	add("bad-magic", stageDirectory, "bad-magic",
		func(b []byte) { b[0], b[1], b[2], b[3] = 0xde, 0xad, 0xbe, 0xef }, 0)

	// Table offset beyond the file (head record's offset field).
	headOff, _, ok := tableRecord(ttf, "head")
	if ok {
		add("head-offset-oob", stageHead, "out-of-bounds",
			func(b []byte) { setTableRecordOffset(b, "head", uint32(len(b))+256) }, headOff)
	}

	// head table with an invalid (too-short) declared length.
	add("head-length-too-short", stageHead, "truncation",
		func(b []byte) { setTableRecordLength(b, "head", 10) }, 0)

	// hmtx length inconsistent with numGlyphs/numHMetrics.
	add("hmtx-bad-length", stageHmtx, "out-of-bounds",
		func(b []byte) { setTableRecordLength(b, "hmtx", 7) }, 0)

	// A random flip inside the offset table region. Tolerated flips are
	// acceptable, so wantReject is left false; query-safety is still enforced.
	pos := int(rng.next() % uint64(12+16*int(u16(ttf[4:]))))
	b := clone(ttf)
	b[pos] ^= byte(1 + rng.next()%255)
	*samples = append(*samples, fuzzSample{
		id: "directory/random-flip", family: famDirectory, stage: stageDirectory,
		tag: "offset-table", offset: pos, kind: "random-flip", data: b,
	})
}

// addRandomSamples applies whole-file truncation and flips to broaden
// coverage. Each edit is fully determined by the shared fixed seed.
func addRandomSamples(samples *[]fuzzSample, rng *splitMix64, ttf []byte, iters int) {
	for i := 0; i < iters; i++ {
		var (
			b    []byte
			off  int
			kind string
		)
		switch rng.next() % 3 {
		case 0:
			b = truncateBytes(rng, clone(ttf))
			off = len(b)
			kind = "truncation"
		case 1:
			b = clone(ttf)
			off = int(rng.next() % uint64(len(b)))
			b[off] ^= byte(1 + rng.next()%255)
			kind = "random-flip"
		default:
			b = clone(ttf)
			off = int(rng.next() % uint64(len(b)-4))
			setU32(b, off, uint32(rng.next()))
			kind = "random-u32"
		}
		*samples = append(*samples, fuzzSample{
			id: fmt.Sprintf("random/%02d", i), family: famRandom, stage: stageDirectory,
			tag: "*", offset: off, kind: kind, data: b,
		})
	}
}

// runSample executes one sample under a watchdog, capturing panics, timeouts,
// error strings and allocation totals. It never calls t.Fatal: every outcome
// is returned as a record so a single bad sample is reported rather than
// aborting the gate.
func runSample(s fuzzSample) fuzzRecord {
	rec := fuzzRecord{
		ID: s.id, Family: string(s.family), Stage: string(s.stage), Tag: s.tag,
		Offset: s.offset, Length: len(s.data), Kind: s.kind,
		DeclaredCap: s.capBytes, WantReject: s.wantReject,
	}

	done := make(chan sampleResult, 1)

	go func() {
		var (
			r    sampleResult
			mem0 runtime.MemStats
			mem1 runtime.MemStats
		)
		runtime.GC()
		runtime.ReadMemStats(&mem0)
		defer func() {
			runtime.ReadMemStats(&mem1)
			r.allocB = int64(mem1.TotalAlloc - mem0.TotalAlloc)
			r.allocs = mem1.Mallocs - mem0.Mallocs
			if p := recover(); p != nil {
				r.panicked = true
				r.parseErr = fmt.Sprintf("panic: %v", p)
			}
			done <- r
		}()

		if s.collection {
			runCollectionSample(s, &r)
			return
		}
		runFontSample(s, &r)
	}()

	select {
	case r := <-done:
		rec.ParseErr, rec.QueryErr, rec.API = r.parseErr, r.queryErr, r.api
		rec.Accepted, rec.Panicked, rec.TimedOut = r.accepted, r.panicked, false
		rec.AllocBytes, rec.Allocs = r.allocB, r.allocs
	case <-time.After(*fuzzTimeout):
		rec.TimedOut = true
		rec.ParseErr = "timeout"
	}
	return rec
}

// sampleResult collects everything runSample observes for one sample.
type sampleResult struct {
	parseErr string
	queryErr string
	api      string
	accepted bool
	panicked bool
	timedOut bool
	allocB   int64
	allocs   uint64
}

// runFontSample drives a single-font sample through Parse and the full query
// API when parsing succeeds.
func runFontSample(s fuzzSample, r *sampleResult) {
	f, err := Parse(s.data)
	if err != nil {
		r.parseErr = err.Error()
		return
	}
	r.accepted = true
	if out := exerciseFont(f); out != "ok" {
		r.api = out
		r.queryErr = out
		return
	}
	r.api = "ok"
}

// runCollectionSample drives a TTC sample through ParseCollection and every
// member Font, then exercises the first successfully parsed member.
func runCollectionSample(s fuzzSample, r *sampleResult) {
	c, err := ParseCollection(s.data)
	if err != nil {
		r.parseErr = err.Error()
		return
	}
	r.accepted = true
	for i := 0; i < c.NumFonts(); i++ {
		f, ferr := c.Font(i)
		if ferr != nil {
			r.api = fmt.Sprintf("Font(%d):%v", i, ferr)
			r.queryErr = r.api
			continue
		}
		if out := exerciseFont(f); out != "ok" {
			r.api = fmt.Sprintf("Font(%d):%v", i, out)
			r.queryErr = r.api
			return
		}
	}
	if r.api == "" {
		r.api = "ok"
	}
}

// exerciseFont runs the query surface required by the gate: GlyphIndex,
// LoadGlyph, Kern and Metrics, plus GlyphAdvance and Bounds. Any failure is
// returned as a non-"ok" string; a panic propagates to runSample's recover.
func exerciseFont(f *Font) string {
	var b Buffer
	ppem := fixed.Int26_6(f.UnitsPerEm())
	if ppem == 0 {
		ppem = fixed.I(16)
	}

	for _, r := range []rune{'A', 'B', 'x', '\x00'} {
		gi, err := f.GlyphIndex(&b, r)
		if err != nil {
			return fmt.Sprintf("GlyphIndex(%q):%v", r, err)
		}
		if _, err := f.LoadGlyph(&b, gi, ppem, nil); err != nil && err != ErrNotFound && err != ErrColoredGlyph {
			return fmt.Sprintf("LoadGlyph(%d):%v", gi, err)
		}
		if _, err := f.GlyphAdvance(&b, gi, ppem, font.HintingNone); err != nil && err != ErrNotFound {
			return fmt.Sprintf("GlyphAdvance(%d):%v", gi, err)
		}
		if _, err := f.Kern(&b, gi, 0, ppem, font.HintingNone); err != nil && err != ErrNotFound {
			return fmt.Sprintf("Kern(%d):%v", gi, err)
		}
	}
	// Exercise a concrete covered pair (glyph 1 + glyph 2) so PairPos GPOS
	// faults that only surface during kern lookup are reached.
	if _, err := f.Kern(&b, 1, 2, ppem, font.HintingNone); err != nil && err != ErrNotFound {
		return fmt.Sprintf("Kern(1,2):%v", err)
	}
	if _, err := f.Metrics(&b, ppem, font.HintingNone); err != nil {
		return fmt.Sprintf("Metrics:%v", err)
	}
	if _, err := f.Bounds(&b, ppem, font.HintingNone); err != nil {
		return fmt.Sprintf("Bounds:%v", err)
	}
	// Name is parsed lazily; exercise it so name-table faults are reported
	// rather than deferred forever. ErrNotFound means no matching record.
	if _, err := f.Name(&b, NameIDFamily); err != nil && err != ErrNotFound {
		return fmt.Sprintf("Name:%v", err)
	}
	return "ok"
}

// TestStructuredFuzz is the quality gate: it builds the corpus, runs every
// sample under a watchdog and asserts the no-panic / no-timeout / bounded-alloc
// invariants, then compares or writes the stable JSONL summary.
func TestStructuredFuzz(t *testing.T) {
	samples := buildCorpus(t)
	if len(samples) < 20 {
		t.Fatalf("corpus too small: %d samples", len(samples))
	}
	// Every required regression family must be present. The gate is meant to
	// retain at least three classes of regression samples; requiring the full
	// set here prevents a future edit from silently dropping a family.
	requiredFamilies := map[fuzzFamily]bool{
		famCollection: true,
		famCmap:       true,
		famName:       true,
		famGPOS:       true,
		famGlyf:       true,
		famCFF:        true,
		famDirectory:  true,
		famRandom:     true,
	}
	seenFamily := map[fuzzFamily]int{}
	for _, s := range samples {
		seenFamily[s.family]++
	}
	for fam := range requiredFamilies {
		if seenFamily[fam] == 0 {
			t.Errorf("regression family %q has no samples", fam)
		}
	}
	// Each of the three core fault kinds must be represented.
	seenKind := map[string]bool{}
	for _, s := range samples {
		seenKind[s.kind] = true
	}
	for _, kind := range []string{"truncation", "out-of-bounds", "oversized-count"} {
		if !seenKind[kind] {
			t.Errorf("required fault kind %q is not represented", kind)
		}
	}

	records := make([]fuzzRecord, 0, len(samples))
	for _, s := range samples {
		rec := runSample(s)
		records = append(records, rec)

		if rec.Panicked {
			t.Errorf("%s: panic captured: %s", rec.ID, rec.ParseErr)
		}
		if rec.TimedOut {
			t.Errorf("%s: exceeded %s deadline", rec.ID, *fuzzTimeout)
		}
		// A deterministic fault injection must be observable. For single fonts
		// Parse has to reject the sample; for collections either the collection
		// header or a member Font access must report an error. A sample that
		// parses cleanly despite wantReject means the injection did not reach
		// the intended stage.
		if rec.WantReject && rec.ParseErr == "" && rec.QueryErr == "" {
			t.Errorf("%s: fault injection was not reported (parsed and queried cleanly)", rec.ID)
		}
		cap := rec.DeclaredCap
		if cap == 0 {
			cap = *fuzzAllocCap
		}
		if rec.AllocBytes > cap {
			t.Errorf("%s: allocated %d bytes, budget %d", rec.ID, rec.AllocBytes, cap)
		}
	}

	path := filepath.FromSlash("fuzz_golden.jsonl")
	if *fuzzUpdate {
		writeGolden(t, path, records)
		t.Logf("wrote %d records to %s", len(records), path)
		return
	}
	compareGolden(t, path, records)
	emitSummary(t, records)
}

// canonicalRecord is the on-disk golden form. Volatile runtime fields
// (allocations) are intentionally excluded so the file is byte-stable across
// machines, Go versions and -race.
type canonicalRecord struct {
	ID         string `json:"id"`
	Family     string `json:"family"`
	Stage      string `json:"stage"`
	Tag        string `json:"tag"`
	Offset     int    `json:"offset"`
	Length     int    `json:"length"`
	Kind       string `json:"kind"`
	Accepted   bool   `json:"accepted"`
	WantReject bool   `json:"want_reject"`
	Panicked   bool   `json:"panicked"`
	TimedOut   bool   `json:"timed_out"`
	ParseErr   string `json:"parse_err,omitempty"`
	QueryErr   string `json:"query_err,omitempty"`
	API        string `json:"api"`
}

func toCanonical(records []fuzzRecord) []canonicalRecord {
	out := make([]canonicalRecord, len(records))
	for i, r := range records {
		out[i] = canonicalRecord{
			ID: r.ID, Family: r.Family, Stage: r.Stage, Tag: r.Tag,
			Offset: r.Offset, Length: r.Length, Kind: r.Kind, Accepted: r.Accepted,
			WantReject: r.WantReject, Panicked: r.Panicked, TimedOut: r.TimedOut,
			ParseErr: r.ParseErr, QueryErr: r.QueryErr, API: r.API,
		}
	}
	return out
}

func writeGolden(t *testing.T, path string, records []fuzzRecord) {
	t.Helper()
	canon := toCanonical(records)
	var buf []byte
	for _, r := range canon {
		line, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal golden: %v", err)
		}
		buf = append(buf, line...)
		buf = append(buf, '\n')
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write golden: %v", err)
	}
}

func compareGolden(t *testing.T, path string, records []fuzzRecord) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s: %v (run with -fuzzUpdate to create it)", path, err)
	}
	want := map[string]canonicalRecord{}
	for _, line := range splitJSONL(data) {
		var r canonicalRecord
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatalf("decode golden line %q: %v", string(line), err)
		}
		want[r.ID] = r
	}

	got := toCanonical(records)
	if len(got) != len(want) {
		t.Errorf("sample count changed: got %d, golden %d (run -fuzzUpdate if intended)",
			len(got), len(want))
	}
	for _, g := range got {
		w, ok := want[g.ID]
		if !ok {
			t.Errorf("new sample %q not in golden (run -fuzzUpdate)", g.ID)
			continue
		}
		if g != w {
			t.Errorf("sample %s drift:\n got  %+v\n want %+v", g.ID, g, w)
		}
	}
}

func splitJSONL(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i, c := range data {
		if c == '\n' {
			if i > start {
				lines = append(lines, data[start:i])
			}
			start = i + 1
		}
	}
	if start < len(data) {
		lines = append(lines, data[start:])
	}
	return lines
}

// emitSummary logs every full JSONL record (including allocation telemetry)
// and optionally mirrors it to -fuzzSummary.
func emitSummary(t *testing.T, records []fuzzRecord) {
	t.Helper()
	var buf []byte
	for _, r := range records {
		line, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal summary: %v", err)
		}
		buf = append(buf, line...)
		buf = append(buf, '\n')
		t.Logf("JSONL %s", line)
	}
	if *fuzzSummary != "" {
		if err := os.MkdirAll(filepath.Dir(*fuzzSummary), 0o755); err != nil {
			t.Fatalf("summary dir: %v", err)
		}
		if err := os.WriteFile(*fuzzSummary, buf, 0o644); err != nil {
			t.Fatalf("write summary: %v", err)
		}
	}
}
