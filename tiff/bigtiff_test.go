// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// alignUp returns the smallest multiple of a that is not less than v.
func alignUp(v, a int) int {
	return (v + a - 1) &^ (a - 1)
}

// testEntry is an IFD entry for the synthetic TIFF builder below.
type testEntry struct {
	tag  uint16
	typ  uint16
	vals []uint64
}

// encodeTestVals encodes vals as IFD values of the given data type.
func encodeTestVals(enc binary.ByteOrder, typ uint16, vals []uint64) []byte {
	l, ok := datatypeLen(typ)
	if !ok {
		panic(fmt.Sprintf("unsupported data type %d", typ))
	}
	b := make([]byte, int(l)*len(vals))
	for i, v := range vals {
		switch typ {
		case dtByte:
			b[i] = byte(v)
		case dtShort:
			enc.PutUint16(b[2*i:], uint16(v))
		case dtLong:
			enc.PutUint32(b[4*i:], uint32(v))
		case dtLong8, dtIFD8:
			enc.PutUint64(b[8*i:], v)
		default:
			panic(fmt.Sprintf("unsupported data type %d", typ))
		}
	}
	return b
}

// buildTIFF builds a classic TIFF or BigTIFF image in memory. The image data
// pix is stored immediately after the file header, so a strip or tile offset
// is 8 for a classic TIFF and 16 for a BigTIFF. Out-of-line values are stored
// after the IFD, 8-byte aligned in BigTIFF as the specification requires.
func buildTIFF(t *testing.T, big bool, enc binary.ByteOrder, entries []testEntry, pix []byte) []byte {
	t.Helper()

	ents := append([]testEntry(nil), entries...)
	sort.Slice(ents, func(i, j int) bool { return ents[i].tag < ents[j].tag })
	for i := 1; i < len(ents); i++ {
		if ents[i].tag == ents[i-1].tag {
			t.Fatalf("duplicate tag %d", ents[i].tag)
		}
	}

	headerLen, countLen, entryLen, valueOff, valueLen, nextLen := 8, 2, ifdLen, 8, 4, 4
	if big {
		headerLen, countLen, entryLen, valueOff, valueLen, nextLen = 16, 8, ifdBigLen, 12, 8, 8
	}

	dataOff := headerLen
	ifdOff := alignUp(dataOff+len(pix), 8)
	ifdSize := countLen + entryLen*len(ents) + nextLen
	pos := alignUp(ifdOff+ifdSize, 8)

	// Assign out-of-line values.
	vals := make([][]byte, len(ents))
	valOff := make([]int, len(ents))
	for i, e := range ents {
		vals[i] = encodeTestVals(enc, e.typ, e.vals)
		if len(vals[i]) > valueLen {
			pos = alignUp(pos, 8)
			valOff[i] = pos
			pos += len(vals[i])
		}
	}

	buf := make([]byte, pos)
	switch enc.Uint16([]byte{1, 0}) {
	case 0x1:
		buf[0], buf[1] = 'I', 'I'
	case 0x100:
		buf[0], buf[1] = 'M', 'M'
	default:
		t.Fatal("odd byte order")
	}
	if big {
		enc.PutUint16(buf[2:4], versionBig)
		enc.PutUint16(buf[4:6], 8) // Byte size of offsets.
		enc.PutUint16(buf[6:8], 0) // Reserved.
		enc.PutUint64(buf[8:16], uint64(ifdOff))
	} else {
		enc.PutUint16(buf[2:4], versionClassic)
		enc.PutUint32(buf[4:8], uint32(ifdOff))
	}
	copy(buf[dataOff:], pix)

	o := ifdOff
	if big {
		enc.PutUint64(buf[o:], uint64(len(ents)))
	} else {
		enc.PutUint16(buf[o:], uint16(len(ents)))
	}
	o += countLen
	for i, e := range ents {
		enc.PutUint16(buf[o:], e.tag)
		enc.PutUint16(buf[o+2:], e.typ)
		if big {
			enc.PutUint64(buf[o+4:], uint64(len(e.vals)))
		} else {
			enc.PutUint32(buf[o+4:], uint32(len(e.vals)))
		}
		if len(vals[i]) <= valueLen {
			copy(buf[o+valueOff:o+valueOff+valueLen], vals[i])
		} else {
			if big {
				enc.PutUint64(buf[o+valueOff:], uint64(valOff[i]))
			} else {
				enc.PutUint32(buf[o+valueOff:], uint32(valOff[i]))
			}
			copy(buf[valOff[i]:], vals[i])
		}
		o += entryLen
	}
	// The buffer is zero-filled, so the next-IFD offset is zero: there is no
	// next IFD.
	return buf
}

// gray8Strips returns a gray8 test image and the entries describing it as
// three strips. The strip offset data type is LONG for a classic TIFF and
// LONG8 (which is stored out-of-line) for a BigTIFF.
func gray8Strips(big bool) ([]testEntry, []byte) {
	const w, h, rowsPerStrip = 8, 8, 3
	pix := make([]byte, w*h)
	for i := range pix {
		pix[i] = byte(i * 3)
	}
	base := uint64(8)
	if big {
		base = 16
	}
	offTyp := uint16(dtLong)
	if big {
		offTyp = dtLong8
	}
	offs := []uint64{base, base + 3*w, base + 6*w}
	counts := []uint64{3 * w, 3 * w, 2 * w}
	return []testEntry{
		{tImageWidth, dtShort, []uint64{w}},
		{tImageLength, dtShort, []uint64{h}},
		{tBitsPerSample, dtShort, []uint64{8}},
		{tCompression, dtShort, []uint64{cNone}},
		{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
		{tStripOffsets, offTyp, offs},
		{tRowsPerStrip, dtShort, []uint64{rowsPerStrip}},
		{tStripByteCounts, dtLong, counts},
		{tSamplesPerPixel, dtShort, []uint64{1}},
	}, pix
}

// gray8StripsShort is gray8Strips with SHORT strip offsets and byte counts.
func gray8StripsShort(big bool) ([]testEntry, []byte) {
	ents, pix := gray8Strips(big)
	for i := range ents {
		switch ents[i].tag {
		case tStripOffsets, tStripByteCounts:
			ents[i].typ = dtShort
		}
	}
	return ents, pix
}

// rgb16Strip returns a 4x4 16-bit RGB image in a single strip. The pixel data
// is little-endian; the caller byte-swaps it for a big-endian file.
func rgb16Strip(big bool) ([]testEntry, []byte) {
	const w, h = 4, 4
	pix := make([]byte, w*h*6)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := (y*w + x) * 6
			binary.LittleEndian.PutUint16(pix[o:], uint16(x*0x1111))
			binary.LittleEndian.PutUint16(pix[o+2:], uint16(y*0x1111))
			binary.LittleEndian.PutUint16(pix[o+4:], uint16((x+y)*0x0808))
		}
	}
	base := uint64(8)
	if big {
		base = 16
	}
	offTyp := uint16(dtLong)
	if big {
		offTyp = dtLong8
	}
	return []testEntry{
		{tImageWidth, dtShort, []uint64{w}},
		{tImageLength, dtShort, []uint64{h}},
		{tBitsPerSample, dtShort, []uint64{16, 16, 16}},
		{tCompression, dtShort, []uint64{cNone}},
		{tPhotometricInterpretation, dtShort, []uint64{pRGB}},
		{tStripOffsets, offTyp, []uint64{base}},
		{tRowsPerStrip, dtShort, []uint64{h}},
		{tStripByteCounts, dtLong, []uint64{uint64(len(pix))}},
		{tSamplesPerPixel, dtShort, []uint64{3}},
	}, pix
}

// swap16 returns b with every pair of bytes swapped, converting pixel data
// between little- and big-endian.
func swap16(b []byte) []byte {
	out := append([]byte(nil), b...)
	for i := 0; i+1 < len(out); i += 2 {
		out[i], out[i+1] = out[i+1], out[i]
	}
	return out
}

// tiledGray8 returns a 32x32 gray8 image stored as four 16x16 tiles.
func tiledGray8(big bool) ([]testEntry, []byte) {
	const w, h, tile = 32, 32, 16
	pix := make([]byte, w*h)
	var offs []uint64
	base := 8
	if big {
		base = 16
	}
	for ty := 0; ty < h/tile; ty++ {
		for tx := 0; tx < w/tile; tx++ {
			offs = append(offs, uint64(base+(ty*(w/tile)+tx)*tile*tile))
			for y := 0; y < tile; y++ {
				for x := 0; x < tile; x++ {
					gx, gy := tx*tile+x, ty*tile+y
					pix[(ty*(w/tile)+tx)*tile*tile+y*tile+x] = byte(gx*3 + gy*5)
				}
			}
		}
	}
	offTyp := uint16(dtLong)
	if big {
		offTyp = dtLong8
	}
	return []testEntry{
		{tImageWidth, dtShort, []uint64{w}},
		{tImageLength, dtShort, []uint64{h}},
		{tBitsPerSample, dtShort, []uint64{8}},
		{tCompression, dtShort, []uint64{cNone}},
		{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
		{tSamplesPerPixel, dtShort, []uint64{1}},
		{tTileWidth, dtShort, []uint64{tile}},
		{tTileLength, dtShort, []uint64{tile}},
		{tTileOffsets, offTyp, offs},
		{tTileByteCounts, dtLong, []uint64{tile * tile, tile * tile, tile * tile, tile * tile}},
	}, pix
}

// TestBigTIFFSynthetic checks that a BigTIFF decodes to the same pixels as the
// equivalent classic TIFF, in both byte orders and with both strip and tile
// layouts. It also checks a few pixels directly, so that a shared bug in the
// builder cannot hide a decoding error.
func TestBigTIFFSynthetic(t *testing.T) {
	cases := []struct {
		name string
		// build returns the entries and pixel data for the given format.
		// swap reports whether pixel data must be byte-swapped to match the
		// file's byte order.
		build func(big bool) ([]testEntry, []byte)
		swap  bool
		check func(t *testing.T, img image.Image)
	}{
		{
			name:  "gray8-strips-long8-offsets",
			build: gray8Strips,
		},
		{
			name:  "gray8-strips-short-offsets",
			build: gray8StripsShort,
		},
		{
			name:  "rgb16-single-strip",
			build: rgb16Strip,
			swap:  true,
			check: func(t *testing.T, img image.Image) {
				rgba, ok := img.(*image.RGBA64)
				if !ok {
					t.Fatalf("got %T, want *image.RGBA64", img)
				}
				for y := 0; y < 4; y++ {
					for x := 0; x < 4; x++ {
						want := color.RGBA64{
							R: uint16(x * 0x1111),
							G: uint16(y * 0x1111),
							B: uint16((x + y) * 0x0808),
							A: 0xffff,
						}
						if got := rgba.RGBA64At(x, y); got != want {
							t.Fatalf("(%d, %d): got %v, want %v", x, y, got, want)
						}
					}
				}
			},
		},
		{
			name:  "tiled-gray8",
			build: tiledGray8,
			check: func(t *testing.T, img image.Image) {
				for y := 0; y < 32; y++ {
					for x := 0; x < 32; x++ {
						want := uint8(x*3 + y*5)
						r, _, _, _ := img.At(x, y).RGBA()
						if got := uint8(r >> 8); got != want {
							t.Fatalf("(%d, %d): got %d, want %d", x, y, got, want)
						}
					}
				}
			},
		},
	}

	for _, enc := range []struct {
		name string
		bo   binary.ByteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	} {
		for _, c := range cases {
			t.Run(enc.name+"/"+c.name, func(t *testing.T) {
				classicEnts, classicPix := c.build(false)
				bigEnts, bigPix := c.build(true)
				if c.swap {
					if enc.bo == binary.BigEndian {
						classicPix = swap16(classicPix)
						bigPix = swap16(bigPix)
					}
				}
				classic := buildTIFF(t, false, enc.bo, classicEnts, classicPix)
				big := buildTIFF(t, true, enc.bo, bigEnts, bigPix)

				want, err := Decode(bytes.NewReader(classic))
				if err != nil {
					t.Fatalf("classic Decode: %v", err)
				}
				got, err := Decode(bytes.NewReader(big))
				if err != nil {
					t.Fatalf("BigTIFF Decode: %v", err)
				}
				compare(t, want, got)

				wantCfg, err := DecodeConfig(bytes.NewReader(classic))
				if err != nil {
					t.Fatalf("classic DecodeConfig: %v", err)
				}
				gotCfg, err := DecodeConfig(bytes.NewReader(big))
				if err != nil {
					t.Fatalf("BigTIFF DecodeConfig: %v", err)
				}
				compareConfig(t, wantCfg, gotCfg)

				if c.check != nil {
					c.check(t, got)
				}
			})
		}
	}
}

// compareConfig checks that two configurations describe the same image.
func compareConfig(t *testing.T, want, got image.Config) {
	t.Helper()
	if want.Width != got.Width || want.Height != got.Height {
		t.Fatalf("config size: got %dx%d, want %dx%d", got.Width, got.Height, want.Width, want.Height)
	}
	if reflect.TypeOf(want.ColorModel) != reflect.TypeOf(got.ColorModel) {
		t.Fatalf("color model type: got %T, want %T", got.ColorModel, want.ColorModel)
	}
	if wp, ok := want.ColorModel.(color.Palette); ok {
		gp := got.ColorModel.(color.Palette)
		if !reflect.DeepEqual(wp, gp) {
			t.Fatalf("palette: got %v, want %v", gp, wp)
		}
	}
}

// classicToBigTIFF rewrites the classic TIFF file src as a BigTIFF. If
// long8Offsets is true, strip or tile offsets are written as LONG8 values.
//
// The rewrite moves every referenced block of pixel data and every
// out-of-line IFD value to a new, 8-byte-aligned location and fixes up the
// offsets that refer to them. This exercises BigTIFF decoding against every
// strip, tile and compression layout for which classic TIFF test data exists.
func classicToBigTIFF(t *testing.T, src []byte, long8Offsets bool) []byte {
	t.Helper()
	if len(src) < 8 {
		t.Fatal("classic TIFF is too short")
	}
	var enc binary.ByteOrder
	switch string(src[0:2]) {
	case "II":
		enc = binary.LittleEndian
	case "MM":
		enc = binary.BigEndian
	default:
		t.Fatalf("bad byte order %q", src[0:2])
	}
	if v := enc.Uint16(src[2:4]); v != versionClassic {
		t.Fatalf("version: got %d, want %d", v, versionClassic)
	}
	ifdOff := uint64(enc.Uint32(src[4:8]))
	if ifdOff+2 > uint64(len(src)) {
		t.Fatalf("IFD offset %d out of range", ifdOff)
	}
	n := int(enc.Uint16(src[ifdOff:]))
	if ifdOff+2+uint64(12*n) > uint64(len(src)) {
		t.Fatalf("IFD of %d entries out of range", n)
	}

	type entry struct {
		tag      uint16
		typ      uint16
		count    uint64
		raw      []byte
		isOffset bool
	}
	entries := make([]entry, n)
	for i := 0; i < n; i++ {
		p := src[ifdOff+2+uint64(12*i):]
		e := entry{
			tag:   enc.Uint16(p[0:2]),
			typ:   enc.Uint16(p[2:4]),
			count: uint64(enc.Uint32(p[4:8])),
		}
		l, ok := datatypeLen(e.typ)
		if !ok || e.typ > dtRational {
			t.Fatalf("tag %d: unsupported data type %d", e.tag, e.typ)
		}
		size := e.count * uint64(l)
		if size > uint64(len(src)) {
			t.Fatalf("tag %d: value size %d out of range", e.tag, size)
		}
		if size <= 4 {
			e.raw = append([]byte(nil), p[8:8+size]...)
		} else {
			off := uint64(enc.Uint32(p[8:12]))
			if off+size > uint64(len(src)) {
				t.Fatalf("tag %d: value offset %d out of range", e.tag, off)
			}
			e.raw = append([]byte(nil), src[off:off+size]...)
		}
		entries[i] = e
	}

	// Locate the strip or tile offsets and their byte counts, and copy the
	// referenced pixel data into new segments.
	offTag, countTag := uint16(0), uint16(0)
	for _, e := range entries {
		switch e.tag {
		case tStripOffsets:
			offTag, countTag = tStripOffsets, tStripByteCounts
		case tTileOffsets:
			offTag, countTag = tTileOffsets, tTileByteCounts
		}
	}
	type segment struct {
		old, size, new int
	}
	var segs []segment
	if offTag != 0 {
		var offEnt, countEnt *entry
		for i := range entries {
			switch entries[i].tag {
			case offTag:
				offEnt = &entries[i]
			case countTag:
				countEnt = &entries[i]
			}
		}
		if offEnt == nil || countEnt == nil {
			t.Fatalf("tag %d without tag %d", offTag, countTag)
		}
		offs := decodeTestVals(t, enc, offEnt.typ, offEnt.count, offEnt.raw)
		sizes := decodeTestVals(t, enc, countEnt.typ, countEnt.count, countEnt.raw)
		if len(offs) != len(sizes) {
			t.Fatalf("tag %d has %d values, tag %d has %d", offTag, len(offs), countTag, len(sizes))
		}
		for i := range offs {
			if offs[i]+sizes[i] > uint64(len(src)) {
				t.Fatalf("pixel data %d (offset %d, size %d) out of range", i, offs[i], sizes[i])
			}
			segs = append(segs, segment{old: int(offs[i]), size: int(sizes[i])})
		}
		for i := range entries {
			if entries[i].tag == offTag {
				entries[i].isOffset = true
			}
		}
	}

	// Decide the output type and size of every entry's value.
	outTyp := make([]uint16, n)
	valSize := make([]int, n)
	for i, e := range entries {
		outTyp[i] = e.typ
		switch {
		case e.isOffset:
			if long8Offsets {
				outTyp[i] = dtLong8
			}
			l, _ := datatypeLen(outTyp[i])
			valSize[i] = int(l) * len(segs)
		default:
			valSize[i] = len(e.raw)
		}
	}

	// Lay out the IFD, then out-of-line values, then pixel data.
	const headerLen = 16
	ifdSize := 8 + ifdBigLen*n + 8
	pos := headerLen + ifdSize
	valOff := make([]int, n)
	for i := range entries {
		if valSize[i] > 8 {
			pos = alignUp(pos, 8)
			valOff[i] = pos
			pos += valSize[i]
		}
	}
	for i := range segs {
		pos = alignUp(pos, 8)
		segs[i].new = pos
		pos += segs[i].size
	}

	// Encode the final values, now that pixel data offsets are known.
	newOffs := make([]uint64, len(segs))
	for i := range segs {
		newOffs[i] = uint64(segs[i].new)
	}
	outRaw := make([][]byte, n)
	for i, e := range entries {
		if e.isOffset {
			outRaw[i] = encodeTestVals(enc, outTyp[i], newOffs)
			if len(outRaw[i]) != valSize[i] {
				t.Fatalf("tag %d: value size %d, want %d", e.tag, len(outRaw[i]), valSize[i])
			}
		} else {
			outRaw[i] = e.raw
		}
	}

	buf := make([]byte, pos)
	switch enc.Uint16([]byte{1, 0}) {
	case 0x1:
		buf[0], buf[1] = 'I', 'I'
	case 0x100:
		buf[0], buf[1] = 'M', 'M'
	}
	enc.PutUint16(buf[2:4], versionBig)
	enc.PutUint16(buf[4:6], 8) // Byte size of offsets.
	enc.PutUint16(buf[6:8], 0) // Reserved.
	enc.PutUint64(buf[8:16], headerLen)

	o := headerLen
	enc.PutUint64(buf[o:], uint64(n))
	o += 8
	for i, e := range entries {
		enc.PutUint16(buf[o:], e.tag)
		enc.PutUint16(buf[o+2:], outTyp[i])
		enc.PutUint64(buf[o+4:], uint64(len(outRaw[i]))/uint64(mustDatatypeLen(t, outTyp[i])))
		if len(outRaw[i]) <= 8 {
			copy(buf[o+12:o+20], outRaw[i])
		} else {
			enc.PutUint64(buf[o+12:], uint64(valOff[i]))
			copy(buf[valOff[i]:], outRaw[i])
		}
		o += ifdBigLen
	}
	enc.PutUint64(buf[o:], 0) // No next IFD.
	for _, s := range segs {
		copy(buf[s.new:], src[s.old:s.old+s.size])
	}
	return buf
}

// decodeTestVals decodes n IFD values of the given data type.
func decodeTestVals(t *testing.T, enc binary.ByteOrder, typ uint16, n uint64, raw []byte) []uint64 {
	t.Helper()
	l, ok := datatypeLen(typ)
	if !ok {
		t.Fatalf("unsupported data type %d", typ)
	}
	if uint64(len(raw)) < n*uint64(l) {
		t.Fatalf("value data is %d bytes, want %d", len(raw), n*uint64(l))
	}
	vals := make([]uint64, n)
	for i := range vals {
		switch typ {
		case dtByte:
			vals[i] = uint64(raw[i])
		case dtShort:
			vals[i] = uint64(enc.Uint16(raw[2*i:]))
		case dtLong:
			vals[i] = uint64(enc.Uint32(raw[4*i:]))
		case dtLong8, dtIFD8:
			vals[i] = enc.Uint64(raw[8*i:])
		default:
			t.Fatalf("unsupported data type %d", typ)
		}
	}
	return vals
}

func mustDatatypeLen(t *testing.T, typ uint16) uint32 {
	t.Helper()
	l, ok := datatypeLen(typ)
	if !ok {
		t.Fatalf("unsupported data type %d", typ)
	}
	return l
}

// TestBigTIFFFromClassic converts every classic TIFF test file to BigTIFF and
// checks that the configuration and pixels are unchanged.
func TestBigTIFFFromClassic(t *testing.T) {
	names, err := filepath.Glob(testdataDir + "*.tiff")
	if err != nil {
		t.Fatal(err)
	}
	tested := 0
	for _, name := range names {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		want, err := Decode(bytes.NewReader(src))
		if err != nil {
			// Some test files are deliberately invalid or zero-sized.
			continue
		}
		wantCfg, err := DecodeConfig(bytes.NewReader(src))
		if err != nil {
			t.Fatalf("%s: classic DecodeConfig: %v", name, err)
		}
		tested++
		for _, long8Offsets := range []bool{false, true} {
			name := filepath.Base(name)
			t.Run(name+"/long8="+strconv.FormatBool(long8Offsets), func(t *testing.T) {
				big := classicToBigTIFF(t, src, long8Offsets)
				gotCfg, err := DecodeConfig(bytes.NewReader(big))
				if err != nil {
					t.Fatalf("BigTIFF DecodeConfig: %v", err)
				}
				compareConfig(t, wantCfg, gotCfg)
				got, err := Decode(bytes.NewReader(big))
				if err != nil {
					t.Fatalf("BigTIFF Decode: %v", err)
				}
				compare(t, want, got)
			})
		}
	}
	if tested == 0 {
		t.Fatal("no classic TIFF test files could be decoded")
	}
}

// TestBigTIFFCompressions checks that the existing compression schemes decode
// identically from a BigTIFF. TestBigTIFFFromClassic already covers this via
// the test data, but this test makes the set of exercised schemes explicit.
func TestBigTIFFCompressions(t *testing.T) {
	names := []string{
		"bw-uncompressed.tiff",                // cNone
		"bw-deflate.tiff",                     // cDeflate
		"bw-packbits.tiff",                    // cPackBits
		"blue-purple-pink.lzwcompressed.tiff", // cLZW
		"bw-gopher_ccittGroup3.tiff",          // cG3
		"bw-gopher_ccittGroup4.tiff",          // cG4
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			src, err := os.ReadFile(testdataDir + name)
			if err != nil {
				t.Fatal(err)
			}
			want, err := Decode(bytes.NewReader(src))
			if err != nil {
				t.Fatalf("classic Decode: %v", err)
			}
			got, err := Decode(bytes.NewReader(classicToBigTIFF(t, src, true)))
			if err != nil {
				t.Fatalf("BigTIFF Decode: %v", err)
			}
			compare(t, want, got)
		})
	}
}

// countingReaderAt limits the total number of bytes read through it, so that
// a decoder which scans to a far-away offset fails the test instead of
// consuming an unbounded amount of data.
type countingReaderAt struct {
	r     io.ReaderAt
	limit int64
	read  int64
}

func (c *countingReaderAt) Read(p []byte) (int, error) {
	return 0, fmt.Errorf("unexpected sequential read")
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	c.read += int64(len(p))
	if c.read > c.limit {
		return 0, fmt.Errorf("read %d bytes, exceeding the %d byte limit", c.read, c.limit)
	}
	return c.r.ReadAt(p, off)
}

// TestBigTIFFSparseFile checks that pixel data located beyond 4 GiB can be
// decoded from a sparse file without reading (or allocating) the intervening
// bytes. The file's logical size is larger than 4 GiB, which a 32-bit offset
// cannot address.
func TestBigTIFFSparseFile(t *testing.T) {
	const (
		w, h = 8, 8
	)
	// Larger than a 32-bit offset, 8-byte aligned.
	const sparseOff int64 = 1<<32 + 0x1000
	pix := make([]byte, w*h)
	for i := range pix {
		pix[i] = byte(i*7 + 1)
	}

	for _, enc := range []struct {
		name string
		bo   binary.ByteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	} {
		t.Run(enc.name, func(t *testing.T) {
			ents := []testEntry{
				{tImageWidth, dtShort, []uint64{w}},
				{tImageLength, dtShort, []uint64{h}},
				{tBitsPerSample, dtShort, []uint64{8}},
				{tCompression, dtShort, []uint64{cNone}},
				{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
				{tStripOffsets, dtLong8, []uint64{uint64(sparseOff)}},
				{tRowsPerStrip, dtShort, []uint64{h}},
				{tStripByteCounts, dtLong8, []uint64{uint64(len(pix))}},
				{tSamplesPerPixel, dtShort, []uint64{1}},
			}
			meta := buildTIFF(t, true, enc.bo, ents, nil)
			if end := int(enc.bo.Uint64(meta[8:16])) + 8 + ifdBigLen*len(ents) + 8; end > len(meta) {
				t.Fatalf("metadata is %d bytes, want at least %d", len(meta), end)
			}

			path := filepath.Join(t.TempDir(), "sparse.tif")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			if _, err := f.WriteAt(meta, 0); err != nil {
				t.Fatal(err)
			}
			if _, err := f.WriteAt(pix, sparseOff); err != nil {
				t.Skipf("filesystem does not support a sparse file at offset %d: %v", sparseOff, err)
			}
			if err := f.Truncate(sparseOff + int64(len(pix))); err != nil {
				t.Skipf("filesystem does not support a sparse file of size %d: %v", sparseOff+int64(len(pix)), err)
			}
			fi, err := f.Stat()
			if err != nil {
				t.Fatal(err)
			}
			if fi.Size() <= math.MaxUint32 {
				t.Fatalf("file size is %d, want more than %d", fi.Size(), uint64(math.MaxUint32))
			}

			// Limit reads so that scanning through the sparse gap fails.
			cr := &countingReaderAt{r: f, limit: 1 << 20}
			cfg, err := DecodeConfig(cr)
			if err != nil {
				t.Fatalf("DecodeConfig: %v", err)
			}
			if cfg.Width != w || cfg.Height != h {
				t.Fatalf("config size: got %dx%d, want %dx%d", cfg.Width, cfg.Height, w, h)
			}
			img, err := Decode(cr)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if got := img.Bounds(); got != image.Rect(0, 0, w, h) {
				t.Fatalf("bounds: got %v, want %v", got, image.Rect(0, 0, w, h))
			}
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					r, _, _, _ := img.At(x, y).RGBA()
					if got, want := uint8(r>>8), pix[y*w+x]; got != want {
						t.Fatalf("(%d, %d): got %d, want %d", x, y, got, want)
					}
				}
			}
		})
	}
}

// findIFDEntry returns the offset of the IFD entry with the given tag in a
// classic TIFF or BigTIFF file.
func findIFDEntry(t *testing.T, data []byte, enc binary.ByteOrder, tag uint16) int {
	t.Helper()
	switch enc.Uint16(data[2:4]) {
	case versionBig:
		ifdOff := int(enc.Uint64(data[8:16]))
		n := int(enc.Uint64(data[ifdOff:]))
		for i := 0; i < n; i++ {
			o := ifdOff + 8 + i*ifdBigLen
			if o+ifdBigLen > len(data) {
				break
			}
			if enc.Uint16(data[o:]) == tag {
				return o
			}
		}
	default:
		ifdOff := int(enc.Uint32(data[4:8]))
		n := int(enc.Uint16(data[ifdOff:]))
		for i := 0; i < n; i++ {
			o := ifdOff + 2 + i*ifdLen
			if o+ifdLen > len(data) {
				break
			}
			if enc.Uint16(data[o:]) == tag {
				return o
			}
		}
	}
	t.Fatalf("tag %d not found", tag)
	return 0
}

// gray8StripsColorMap is gray8Strips with an additional ColorMap entry. Its
// value (three color planes of two colors) is stored out of line in both
// classic TIFF and BigTIFF, and, unlike the strip offset tables, it is parsed
// by newDecoder. This makes it convenient for testing the parsing of
// out-of-line IFD values.
func gray8StripsColorMap(big bool) ([]testEntry, []byte) {
	ents, pix := gray8Strips(big)
	ents = append(ents, testEntry{tColorMap, dtShort, []uint64{
		0x0000, 0xffff, // Red.
		0x0000, 0xffff, // Green.
		0x0000, 0xffff, // Blue.
	}})
	return ents, pix
}

// TestBigTIFFInvalid checks that malformed BigTIFF files are rejected.
func TestBigTIFFInvalid(t *testing.T) {
	enc := binary.LittleEndian
	ents, pix := gray8StripsColorMap(true)
	valid := buildTIFF(t, true, enc, ents, pix)
	ifdOff := int(enc.Uint64(valid[8:16]))
	// The ColorMap value is stored out of line and is parsed while reading the
	// IFD, so mutating it affects DecodeConfig as well as Decode.
	mapEntry := findIFDEntry(t, valid, enc, tColorMap)

	mutate := func(t *testing.T, data []byte, f func(b []byte)) []byte {
		t.Helper()
		b := append([]byte(nil), data...)
		f(b)
		return b
	}

	tests := []struct {
		name string
		data func(t *testing.T) []byte
	}{
		{"bad-offset-size", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint16(b[4:6], 4) })
		}},
		{"non-zero-reserved", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint16(b[6:8], 1) })
		}},
		{"bad-version", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint16(b[2:4], 44) })
		}},
		{"truncated-header", func(t *testing.T) []byte {
			return valid[:12]
		}},
		{"truncated-ifd", func(t *testing.T) []byte {
			return valid[:ifdOff+10]
		}},
		{"ifd-offset-zero", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], 0) })
		}},
		{"ifd-offset-in-header", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], 8) })
		}},
		{"ifd-offset-too-large", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], 1<<63) })
		}},
		{"ifd-offset-max", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], math.MaxInt64) })
		}},
		{"ifd-offset-near-max", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], math.MaxInt64-8) })
		}},
		{"ifd-offset-past-eof", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[8:16], 1<<40) })
		}},
		{"too-many-ifd-entries", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[ifdOff:], maxIFDEntries+1) })
		}},
		{"entry-count-overflow", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[mapEntry+4:], 1<<62) })
		}},
		{"misaligned-value-offset", func(t *testing.T) []byte {
			if got := enc.Uint64(valid[mapEntry+12:]); got%8 != 0 {
				t.Fatalf("test setup: ColorMap value offset %d is not 8-byte aligned", got)
			}
			// A 4-byte-aligned, but not 8-byte-aligned, offset.
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[mapEntry+12:], enc.Uint64(b[mapEntry+12:])+4) })
		}},
		{"value-offset-too-large", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint64(b[mapEntry+12:], math.MaxUint64&^7) })
		}},
		{"bad-datatype", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) { enc.PutUint16(b[mapEntry+2:], 6) })
		}},
		{"unsorted-tags", func(t *testing.T) []byte {
			return mutate(t, valid, func(b []byte) {
				first := ifdOff + 8
				enc.PutUint16(b[first:], tSamplesPerPixel)
			})
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := tc.data(t)
			if _, err := Decode(bytes.NewReader(data)); err == nil {
				t.Error("Decode: got nil error, want non-nil")
			}
			if _, err := DecodeConfig(bytes.NewReader(data)); err == nil {
				t.Error("DecodeConfig: got nil error, want non-nil")
			}
		})
	}
}

// TestBigTIFFInvalidOffsets checks that invalid strip and tile offsets and
// byte counts are rejected by Decode. DecodeConfig does not examine these
// values, since it only reads the image configuration.
func TestBigTIFFInvalidOffsets(t *testing.T) {
	enc := binary.LittleEndian
	ents, pix := gray8Strips(true)
	for i := range ents {
		switch ents[i].tag {
		case tStripOffsets:
			ents[i].vals = ents[i].vals[:1]
		case tStripByteCounts:
			ents[i].vals = []uint64{uint64(len(pix))}
		}
	}
	data := buildTIFF(t, true, enc, ents, pix)
	offEntry := findIFDEntry(t, data, enc, tStripOffsets)
	countEntry := findIFDEntry(t, data, enc, tStripByteCounts)

	tests := []struct {
		name   string
		mutate func(b []byte)
	}{
		{"offset-too-large", func(b []byte) { enc.PutUint64(b[offEntry+12:], math.MaxUint64&^7) }},
		{"offset-past-eof", func(b []byte) { enc.PutUint64(b[offEntry+12:], 1<<40) }},
		{"byte-count-too-large", func(b []byte) { enc.PutUint64(b[countEntry+12:], 1<<40) }},
		{"truncated-pixel-data", func(b []byte) { enc.PutUint64(b[offEntry+12:], uint64(len(b)-4)) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := append([]byte(nil), data...)
			tc.mutate(b)
			if _, err := Decode(bytes.NewReader(b)); err == nil {
				t.Error("Decode: got nil error, want non-nil")
			}
		})
	}
}

// TestBigTIFFLong8InClassic checks that the BigTIFF-only data types are
// rejected in a classic TIFF.
func TestBigTIFFLong8InClassic(t *testing.T) {
	enc := binary.LittleEndian
	classicEnts, classicPix := gray8StripsColorMap(false)
	data := buildTIFF(t, false, enc, classicEnts, classicPix)
	mapEntry := findIFDEntry(t, data, enc, tColorMap)
	for _, typ := range []uint16{dtLong8, dtSLong8, dtIFD8} {
		b := append([]byte(nil), data...)
		enc.PutUint16(b[mapEntry+2:], typ)
		if _, err := Decode(bytes.NewReader(b)); err == nil {
			t.Errorf("data type %d: Decode: got nil error, want non-nil", typ)
		}
		if _, err := DecodeConfig(bytes.NewReader(b)); err == nil {
			t.Errorf("data type %d: DecodeConfig: got nil error, want non-nil", typ)
		}
	}
}

// TestBigTIFFBufferErrors checks that invalid out-of-line value offsets fail
// cleanly when the input does not implement io.ReaderAt, which forces the
// decoder to buffer the input.
func TestBigTIFFBufferErrors(t *testing.T) {
	enc := binary.BigEndian
	ents, pix := gray8StripsColorMap(true)
	data := buildTIFF(t, true, enc, ents, pix)
	mapEntry := findIFDEntry(t, data, enc, tColorMap)
	for _, off64 := range []uint64{1 << 40, math.MaxUint64 &^ 7} {
		b := append([]byte(nil), data...)
		enc.PutUint64(b[mapEntry+12:], off64)
		r := ioReader{bytes.NewReader(b)}
		if _, err := Decode(r); err == nil {
			t.Errorf("offset %#x: Decode: got nil error, want non-nil", off64)
		}
		r = ioReader{bytes.NewReader(b)}
		if _, err := DecodeConfig(r); err == nil {
			t.Errorf("offset %#x: DecodeConfig: got nil error, want non-nil", off64)
		}
	}
}

// TestBigTIFFBufferPath checks that BigTIFF decoding works when the input is
// an io.Reader but not an io.ReaderAt, which forces the decoder to buffer the
// input.
func TestBigTIFFBufferPath(t *testing.T) {
	ents, pix := gray8Strips(true)
	for _, enc := range []struct {
		name string
		bo   binary.ByteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	} {
		data := buildTIFF(t, true, enc.bo, ents, pix)
		want, err := Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: Decode: %v", enc.name, err)
		}
		got, err := Decode(ioReader{bytes.NewReader(data)})
		if err != nil {
			t.Fatalf("%s: Decode (buffered): %v", enc.name, err)
		}
		compare(t, want, got)
	}
}

// TestBigTIFFValueOffsetAlignment checks that out-of-line values must begin at
// an 8-byte-aligned address, as the BigTIFF specification requires. The offset
// itself is 8-byte aligned in the builder, so moving it by 4 yields an offset
// that is aligned to the width of a LONG but not to 8 bytes.
func TestBigTIFFValueOffsetAlignment(t *testing.T) {
	enc := binary.LittleEndian
	ents, pix := gray8StripsColorMap(true)
	data := buildTIFF(t, true, enc, ents, pix)
	mapEntry := findIFDEntry(t, data, enc, tColorMap)

	valOff := int(enc.Uint64(data[mapEntry+12:]))
	if valOff%8 != 0 {
		t.Fatalf("test setup: ColorMap value offset %d is not 8-byte aligned", valOff)
	}
	// Place the value at a 4-byte-aligned, but not 8-byte-aligned, offset.
	const valLen = 3 * 2 * 2 // Three planes of two SHORT values.
	newOff := alignUp(len(data), 8) + 4
	data = append(data, make([]byte, newOff+valLen-len(data))...)
	copy(data[newOff:newOff+valLen], data[valOff:valOff+valLen])
	enc.PutUint64(data[mapEntry+12:], uint64(newOff))
	if _, err := Decode(bytes.NewReader(data)); err == nil {
		t.Error("Decode with a 4-byte-aligned value: got nil error, want non-nil")
	}
	if _, err := DecodeConfig(bytes.NewReader(data)); err == nil {
		t.Error("DecodeConfig with a 4-byte-aligned value: got nil error, want non-nil")
	}
}

// TestBigTIFFZeroSizedImages checks that zero-sized BigTIFF images are
// rejected, matching classic TIFF behavior.
func TestBigTIFFZeroSizedImages(t *testing.T) {
	for _, size := range []struct {
		w, h int
	}{{0, 0}, {1, 0}, {0, 1}} {
		for _, enc := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
			ents := []testEntry{
				{tImageWidth, dtShort, []uint64{uint64(size.w)}},
				{tImageLength, dtShort, []uint64{uint64(size.h)}},
				{tBitsPerSample, dtShort, []uint64{8}},
				{tCompression, dtShort, []uint64{cNone}},
				{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
				{tStripOffsets, dtLong8, []uint64{16}},
				{tRowsPerStrip, dtShort, []uint64{uint64(max(size.h, 1))}},
				{tStripByteCounts, dtLong8, []uint64{0}},
			}
			data := buildTIFF(t, true, enc, ents, nil)
			if _, err := Decode(bytes.NewReader(data)); err == nil {
				t.Errorf("Decode(%dx%d, %T): got nil error, want non-nil", size.w, size.h, enc)
			}
		}
	}
}

// TestBigTIFFImageDecode checks that image.Decode and image.DecodeConfig
// recognize BigTIFF through the registered magic.
func TestBigTIFFImageDecode(t *testing.T) {
	for _, enc := range []struct {
		name string
		bo   binary.ByteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	} {
		ents, pix := gray8Strips(true)
		data := buildTIFF(t, true, enc.bo, ents, pix)
		cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: image.DecodeConfig: %v", enc.name, err)
		}
		if format != "tiff" {
			t.Fatalf("%s: format: got %q, want %q", enc.name, format, "tiff")
		}
		if cfg.Width != 8 || cfg.Height != 8 {
			t.Fatalf("%s: config size: got %dx%d, want 8x8", enc.name, cfg.Width, cfg.Height)
		}
		img, format, err := image.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("%s: image.Decode: %v", enc.name, err)
		}
		if format != "tiff" {
			t.Fatalf("%s: format: got %q, want %q", enc.name, format, "tiff")
		}
		if got, want := img.Bounds().Size(), image.Pt(8, 8); got != want {
			t.Fatalf("%s: bounds: got %v, want %v", enc.name, got, want)
		}
	}
}

// TestBigTIFFIFDEntryCount checks that unusual IFD entry counts are handled
// without panicking or overflowing. A count of exactly maxIFDEntries passes
// the count check; the read then fails because the file is much smaller.
func TestBigTIFFIFDEntryCount(t *testing.T) {
	enc := binary.LittleEndian
	ents, pix := gray8Strips(true)
	data := buildTIFF(t, true, enc, ents, pix)
	ifdOff := int(enc.Uint64(data[8:16]))
	for _, tc := range []struct {
		n          uint64
		wantErr    bool
		wantErrMsg string
	}{
		{n: 0, wantErr: true}, // No entries: the image tags are missing.
		{n: 1, wantErr: true},
		{n: maxIFDEntries, wantErr: true},
		{n: maxIFDEntries + 1, wantErr: true, wantErrMsg: "too many IFD entries"},
		{n: 1 << 63, wantErr: true, wantErrMsg: "too many IFD entries"},
		{n: math.MaxUint64, wantErr: true, wantErrMsg: "too many IFD entries"},
	} {
		b := append([]byte(nil), data...)
		enc.PutUint64(b[ifdOff:], tc.n)
		for _, decode := range []func() error{
			func() error {
				_, err := Decode(bytes.NewReader(b))
				return err
			},
			func() error {
				_, err := DecodeConfig(bytes.NewReader(b))
				return err
			},
		} {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("count %d: panic: %v", tc.n, r)
					}
				}()
				err := decode()
				if tc.wantErr && err == nil {
					t.Errorf("count %d: got nil error, want non-nil", tc.n)
				}
				if !tc.wantErr && err != nil {
					t.Errorf("count %d: got %v, want nil", tc.n, err)
				}
				if tc.wantErrMsg != "" && err != nil && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Errorf("count %d: got %v, want an error containing %q", tc.n, err, tc.wantErrMsg)
				}
			}()
		}
	}
}

// TestBigTIFFLongValues checks the handling of 64-bit values that do not fit
// in an int, which differ between 32- and 64-bit platforms. The decoder must
// reject values it cannot represent rather than overflow or silently truncate,
// and must reject dimensions whose total size overflows.
func TestBigTIFFLongValues(t *testing.T) {
	enc := binary.LittleEndian
	const (
		errNone   = ""
		errValue  = "IFD value too large"
		errTooBig = "image too large"
	)
	tests := []struct {
		name   string
		w, h   uint64
		want32 string // Expected error substring on 32-bit platforms, or errNone.
		want64 string // Expected error substring on 64-bit platforms, or errNone.
	}{
		{"small", 8, 1, errNone, errNone},
		{"int32-width", 1 << 31, 1, errValue, errNone},
		{"uint32-width", math.MaxUint32, 1, errValue, errNone},
		{"huge-square", 1 << 32, 1 << 32, errValue, errTooBig},
		{"max-uint64", math.MaxUint64, 1, errValue, errValue},
		{"huge-height", 8, 1 << 40, errValue, errNone},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ents := []testEntry{
				{tImageWidth, dtLong8, []uint64{tc.w}},
				{tImageLength, dtLong8, []uint64{tc.h}},
				{tBitsPerSample, dtShort, []uint64{8}},
				{tCompression, dtShort, []uint64{cNone}},
				{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
				{tStripOffsets, dtLong8, []uint64{16}},
				{tStripByteCounts, dtLong8, []uint64{0}},
			}
			data := buildTIFF(t, true, enc, ents, nil)
			want := tc.want64
			if strconv.IntSize == 32 {
				want = tc.want32
			}
			cfg, err := DecodeConfig(bytes.NewReader(data))
			if want == errNone {
				if err != nil {
					t.Fatalf("DecodeConfig: %v", err)
				}
				if uint64(cfg.Width) != tc.w || uint64(cfg.Height) != tc.h {
					t.Fatalf("config size: got %dx%d, want %dx%d", cfg.Width, cfg.Height, tc.w, tc.h)
				}
				return
			}
			if !errFormatContains(err, want) {
				t.Fatalf("DecodeConfig: got %v, want an error containing %q", err, want)
			}
		})
	}
}

// TestBigTIFFManyEntries checks that a large IFD with many entries, which a
// uint16 entry count cannot express, decodes correctly.
func TestBigTIFFManyEntries(t *testing.T) {
	enc := binary.LittleEndian
	const n = 60000 // Fewer than maxIFDEntries, all tags distinct and ascending.
	ents := []testEntry{
		{tImageWidth, dtShort, []uint64{8}},
		{tImageLength, dtShort, []uint64{8}},
		{tBitsPerSample, dtShort, []uint64{8}},
		{tCompression, dtShort, []uint64{cNone}},
		{tPhotometricInterpretation, dtShort, []uint64{pBlackIsZero}},
		{tStripOffsets, dtLong8, []uint64{16}},
		{tStripByteCounts, dtLong8, []uint64{64}},
	}
	for i := 0; i < n; i++ {
		ents = append(ents, testEntry{uint16(5000 + i), dtShort, []uint64{0}})
	}
	pix := make([]byte, 64)
	data := buildTIFF(t, true, enc, ents, pix)
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got, want := img.Bounds().Size(), image.Pt(8, 8); got != want {
		t.Fatalf("bounds: got %v, want %v", got, want)
	}
}

// TestDatatypeLen checks the data type length table.
func TestDatatypeLen(t *testing.T) {
	for _, tc := range []struct {
		typ uint16
		len uint32
		ok  bool
	}{
		{dtByte, 1, true},
		{dtASCII, 1, true},
		{dtShort, 2, true},
		{dtLong, 4, true},
		{dtRational, 8, true},
		{dtLong8, 8, true},
		{dtSLong8, 8, true},
		{dtIFD8, 8, true},
		{0, 0, false},
		{6, 0, false},
		{19, 0, false},
	} {
		l, ok := datatypeLen(tc.typ)
		if l != tc.len || ok != tc.ok {
			t.Errorf("datatypeLen(%d) = (%d, %t), want (%d, %t)", tc.typ, l, ok, tc.len, tc.ok)
		}
	}
}

// errFormatContains reports whether err is a FormatError or UnsupportedError
// whose message contains substr. It is used to assert that malformed input is
// diagnosed rather than merely failing for an unrelated reason.
func errFormatContains(err error, substr string) bool {
	if err == nil {
		return false
	}
	switch err.(type) {
	case FormatError, UnsupportedError:
	default:
		return false
	}
	return strings.Contains(err.Error(), substr)
}

// TestBigTIFFErrorMessages checks that the new error checks report useful
// errors, rather than relying on a later read failure.
func TestBigTIFFErrorMessages(t *testing.T) {
	enc := binary.LittleEndian
	ents, pix := gray8StripsColorMap(true)
	valid := buildTIFF(t, true, enc, ents, pix)
	mapEntry := findIFDEntry(t, valid, enc, tColorMap)

	tests := []struct {
		name    string
		mutate  func(b []byte)
		wantErr string
	}{
		{"offset-size", func(b []byte) { enc.PutUint16(b[4:6], 16) }, "offset size"},
		{"reserved", func(b []byte) { enc.PutUint16(b[6:8], 1) }, "reserved"},
		{"misaligned", func(b []byte) {
			enc.PutUint64(b[mapEntry+12:], enc.Uint64(b[mapEntry+12:])+4)
		}, "misaligned"},
		{"value-offset-too-large", func(b []byte) {
			enc.PutUint64(b[mapEntry+12:], math.MaxUint64&^7)
		}, "too large"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := append([]byte(nil), valid...)
			tc.mutate(b)
			_, err := Decode(bytes.NewReader(b))
			if !errFormatContains(err, tc.wantErr) {
				t.Errorf("Decode: got %v, want a format error containing %q", err, tc.wantErr)
			}
		})
	}
}
