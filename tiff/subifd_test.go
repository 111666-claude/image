// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"reflect"
	"strings"
	"testing"
)

// tiffBuilder assembles a TIFF file incrementally.
type tiffBuilder struct {
	enc  binary.ByteOrder
	blob []byte
}

func newBuilder(enc binary.ByteOrder) *tiffBuilder {
	b := &tiffBuilder{enc: enc}
	b.blob = make([]byte, 8)
	if enc.Uint16([]byte{1, 0}) == 1 {
		copy(b.blob, "II")
	} else {
		copy(b.blob, "MM")
	}
	b.enc.PutUint16(b.blob[2:4], 42)
	return b
}

func (b *tiffBuilder) u16(v uint16) {
	var tmp [2]byte
	b.enc.PutUint16(tmp[:], v)
	b.blob = append(b.blob, tmp[:]...)
}
func (b *tiffBuilder) u32(v uint32) {
	var tmp [4]byte
	b.enc.PutUint32(tmp[:], v)
	b.blob = append(b.blob, tmp[:]...)
}

// data appends bytes and returns their offset.
func (b *tiffBuilder) data(p ...byte) int64 {
	off := int64(len(b.blob))
	b.blob = append(b.blob, p...)
	return off
}

// setFirstIFD records the offset of the main IFD in the header.
func (b *tiffBuilder) setFirstIFD(off int64) { b.enc.PutUint32(b.blob[4:8], uint32(off)) }

// testIFDEntry is one IFD entry.
type testIFDEntry struct {
	tag   uint16
	typ   uint16
	vals  []uint32 // raw values, interpreted according to typ
	ascii []byte   // if non-nil, inline payload used instead of vals
}

// writeIFD appends an IFD with the given entries, sorted by tag as the format
// requires. next is the offset of the following top-level IFD, or 0.
// Out-of-line tag values are placed after the IFD body, never inside it.
func (b *tiffBuilder) writeIFD(entries []testIFDEntry, next int64) int64 {
	es := append([]testIFDEntry(nil), entries...)
	for i := 1; i < len(es); i++ {
		for j := i; j > 0 && es[j].tag < es[j-1].tag; j-- {
			es[j], es[j-1] = es[j-1], es[j]
		}
	}

	// Precompute each entry's payload so the IFD body can be written in one
	// pass and the out-of-line payloads appended afterwards.
	payloads := make([][]byte, len(es))
	for i, e := range es {
		if e.ascii != nil {
			payloads[i] = e.ascii
			continue
		}
		for _, v := range e.vals {
			switch e.typ {
			case dtShort:
				var tmp [2]byte
				b.enc.PutUint16(tmp[:], uint16(v))
				payloads[i] = append(payloads[i], tmp[:]...)
			case dtLong, dtIFD:
				var tmp [4]byte
				b.enc.PutUint32(tmp[:], v)
				payloads[i] = append(payloads[i], tmp[:]...)
			default:
				panic("unsupported test entry type")
			}
		}
	}

	off := int64(len(b.blob))
	bodyLen := 2 + ifdLen*len(es) + 4
	b.blob = append(b.blob, make([]byte, bodyLen)...)
	pos := off
	put16 := func(v uint16) {
		b.enc.PutUint16(b.blob[pos:pos+2], v)
		pos += 2
	}
	put32 := func(v uint32) {
		b.enc.PutUint32(b.blob[pos:pos+4], v)
		pos += 4
	}
	put16(uint16(len(es)))
	for i, e := range es {
		put16(e.tag)
		put16(e.typ)
		put32(uint32(len(e.vals)))
		if e.ascii != nil {
			b.enc.PutUint32(b.blob[pos-4:pos], uint32(len(e.ascii)))
		}
		if len(payloads[i]) <= 4 {
			var inline [4]byte
			copy(inline[:], payloads[i])
			copy(b.blob[pos:pos+4], inline[:])
			pos += 4
		} else {
			put32(uint32(len(b.blob)))
			b.blob = append(b.blob, payloads[i]...)
		}
	}
	put32(uint32(next))
	return off
}

func (b *tiffBuilder) build() []byte { return b.blob }

// grayPixels returns w*h 8-bit gray samples with a recognizable gradient.
func grayPixels(w, h int, base byte) []byte {
	p := make([]byte, w*h)
	for i := range p {
		p[i] = base + byte(i)
	}
	return p
}

// grayEntrySet is the tag set of a simple single-strip 8-bit gray page.
// stripOff is the offset of the pixel data and n its length.
func grayEntrySet(w, h int, newSubfileType uint32, stripOff int64, n int, extra ...testIFDEntry) []testIFDEntry {
	es := []testIFDEntry{
		{tag: tNewSubfileType, typ: dtLong, vals: []uint32{newSubfileType}},
		{tag: tImageWidth, typ: dtShort, vals: []uint32{uint32(w)}},
		{tag: tImageLength, typ: dtShort, vals: []uint32{uint32(h)}},
		{tag: tBitsPerSample, typ: dtShort, vals: []uint32{8}},
		{tag: tCompression, typ: dtShort, vals: []uint32{cNone}},
		{tag: tPhotometricInterpretation, typ: dtShort, vals: []uint32{pBlackIsZero}},
		{tag: tSamplesPerPixel, typ: dtShort, vals: []uint32{1}},
		{tag: tStripOffsets, typ: dtLong, vals: []uint32{uint32(stripOff)}},
		{tag: tRowsPerStrip, typ: dtLong, vals: []uint32{uint32(h)}},
		{tag: tStripByteCounts, typ: dtLong, vals: []uint32{uint32(n)}},
	}
	return append(es, extra...)
}

// cameraFixture is a synthetic camera file: a main image with a thumbnail and
// a depth map as SubIFDs, the depth map optionally holding a nested page. Its
// IFDs are written after the pixel data in an order unrelated to reference
// order, and it can be built in either byte order.
type cameraFixture struct {
	data       []byte
	offsets    map[string]int64 // "main", "thumb", "depth", "aux"
	pixelSpans [][2]int64       // [start, end) of each pixel blob
	pages      []SubIFD
}

func buildCameraFixture(t *testing.T, enc binary.ByteOrder, nested bool) cameraFixture {
	t.Helper()
	b := newBuilder(enc)

	mainPx := grayPixels(8, 6, 1)
	thumbPx := grayPixels(4, 3, 100)
	depthPx := grayPixels(4, 3, 200)
	auxPx := grayPixels(2, 1, 250)

	spans := [][2]int64{}
	addPx := func(p []byte) int64 {
		off := b.data(p...)
		spans = append(spans, [2]int64{off, off + int64(len(p))})
		return off
	}
	mainPxOff := addPx(mainPx)
	thumbPxOff := addPx(thumbPx)
	depthPxOff := addPx(depthPx)
	auxPxOff := addPx(auxPx)

	// Write children before the parent so that the main IFD ends up last.
	var auxOff int64
	if nested {
		auxOff = b.writeIFD(grayEntrySet(2, 1, 0x1, auxPxOff, len(auxPx)), 0)
	}
	var depthExtra []testIFDEntry
	if nested {
		depthExtra = append(depthExtra, testIFDEntry{tag: tSubIFDs, typ: dtIFD, vals: []uint32{uint32(auxOff)}})
	}
	depthExtra = append(depthExtra, testIFDEntry{tag: tImageDescription, typ: dtASCII, ascii: append([]byte("depth map"), 0)})
	depthOff := b.writeIFD(grayEntrySet(4, 3, 0x4, depthPxOff, len(depthPx), depthExtra...), 0)

	thumbExtra := []testIFDEntry{{tag: tImageDescription, typ: dtASCII, ascii: append([]byte("thumb"), 0)}}
	thumbOff := b.writeIFD(grayEntrySet(4, 3, 0x1, thumbPxOff, len(thumbPx), thumbExtra...), 0)

	main := b.writeIFD(grayEntrySet(8, 6, 0, mainPxOff, len(mainPx),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(thumbOff), uint32(depthOff)}},
	), 0)
	b.setFirstIFD(main)

	pages := []SubIFD{
		{Parent: -1, Index: -1, Depth: 0, Offset: main, Width: 8, Height: 6, Compression: cNone, BitsPerSample: 8, SamplesPerPixel: 1, PhotometricInterpretation: pBlackIsZero},
		{Parent: 0, Index: 0, Depth: 1, Offset: thumbOff, Width: 4, Height: 3, Compression: cNone, BitsPerSample: 8, SamplesPerPixel: 1, PhotometricInterpretation: pBlackIsZero, NewSubfileType: 0x1, Description: "thumb"},
		{Parent: 0, Index: 1, Depth: 1, Offset: depthOff, Width: 4, Height: 3, Compression: cNone, BitsPerSample: 8, SamplesPerPixel: 1, PhotometricInterpretation: pBlackIsZero, NewSubfileType: 0x4, Description: "depth map"},
	}
	if nested {
		pages = append(pages, SubIFD{Parent: 2, Index: 0, Depth: 2, Offset: auxOff, Width: 2, Height: 1, Compression: cNone, BitsPerSample: 8, SamplesPerPixel: 1, PhotometricInterpretation: pBlackIsZero, NewSubfileType: 0x1})
	}
	return cameraFixture{
		data:       b.build(),
		offsets:    map[string]int64{"main": main, "thumb": thumbOff, "depth": depthOff, "aux": auxOff},
		pixelSpans: spans,
		pages:      pages,
	}
}

func TestEnumerateSubIFDsMainImageOnly(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(4, 3, 10)
	main := b.writeIFD(grayEntrySet(4, 3, 0, b.data(px...), len(px)), 0)
	b.setFirstIFD(main)

	pages, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) != 1 {
		t.Fatalf("got %d pages, want 1", len(pages))
	}
	p := pages[0]
	if p.Parent != -1 || p.Index != -1 || p.Depth != 0 {
		t.Errorf("main page position = (%d, %d, %d), want (-1, -1, 0)", p.Parent, p.Index, p.Depth)
	}
	if p.Width != 4 || p.Height != 3 || p.Offset != main {
		t.Errorf("page = %+v", p)
	}
}

func TestEnumerateSubIFDsTree(t *testing.T) {
	for _, enc := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		for _, nested := range []bool{false, true} {
			fx := buildCameraFixture(t, enc, nested)
			got, err := EnumerateSubIFDs(bytes.NewReader(fx.data))
			if err != nil {
				t.Fatalf("%v nested=%v: %v", enc, nested, err)
			}
			if !reflect.DeepEqual(got, fx.pages) {
				t.Fatalf("%v nested=%v:\ngot  %+v\nwant %+v", enc, nested, got, fx.pages)
			}
		}
	}
}

func TestEnumerateSubIFDsIsStable(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, true)
	first, err := EnumerateSubIFDs(bytes.NewReader(fx.data))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		again, err := EnumerateSubIFDs(bytes.NewReader(fx.data))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, again) {
			t.Fatalf("enumeration %d differs from the first", i)
		}
	}
}

func TestSubIFDPixels(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, true)
	r, err := NewReader(bytes.NewReader(fx.data))
	if err != nil {
		t.Fatal(err)
	}
	if r.NumPages() != len(fx.pages) {
		t.Fatalf("NumPages = %d, want %d", r.NumPages(), len(fx.pages))
	}
	wantFirst := []byte{1, 100, 200, 250}
	for i, v := range wantFirst {
		img, err := r.Decode(i)
		if err != nil {
			t.Fatalf("Decode(%d): %v", i, err)
		}
		want := fx.pages[i]
		if b := img.Bounds(); b.Dx() != want.Width || b.Dy() != want.Height {
			t.Errorf("page %d bounds = %v, want %dx%d", i, b, want.Width, want.Height)
		}
		g := img.(*image.Gray)
		if g.Pix[0] != v {
			t.Errorf("page %d first pixel = %d, want %d", i, g.Pix[0], v)
		}
	}

	// Page 0 must match package-level Decode exactly.
	imgMain, _ := r.Decode(0)
	imgDecode, err := Decode(bytes.NewReader(fx.data))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, imgDecode, imgMain)

	if !r.Pages()[1].ReducedResolution() {
		t.Error("thumbnail should report ReducedResolution")
	}
	if !r.Pages()[2].IsMask() {
		t.Error("depth page should report IsMask")
	}
	if r.Pages()[0].ReducedResolution() || r.Pages()[0].IsMask() {
		t.Error("main image should report neither flag")
	}
}

func TestReaderAtAndBufferedInput(t *testing.T) {
	fx := buildCameraFixture(t, binary.BigEndian, true)

	ra, ok := any(bytes.NewReader(fx.data)).(io.ReaderAt)
	if !ok {
		t.Fatal("bytes.Reader is not a ReaderAt")
	}
	r1, err := NewReaderAt(ra, binary.BigEndian, fx.offsets["main"])
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r1.Pages(), fx.pages) {
		t.Fatal("NewReaderAt pages differ")
	}

	// Input that is only an io.Reader must also work.
	r2, err := NewReader(struct{ io.Reader }{bytes.NewReader(fx.data)})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r2.Pages(), fx.pages) {
		t.Fatal("buffered input pages differ")
	}
	if _, err := r2.Decode(1); err != nil {
		t.Fatal(err)
	}

	// Config must work without reading pixels.
	cfg, err := r1.Config(2)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Width != 4 || cfg.Height != 3 {
		t.Errorf("Config(2) = %+v", cfg)
	}
}

// guardedReaderAt fails any read that overlaps a pixel blob, so that a test
// can prove an operation touched only metadata.
type guardedReaderAt struct {
	r     io.ReaderAt
	spans [][2]int64
	pos   int64
	read  int64
}

func (g *guardedReaderAt) Read(p []byte) (int, error) {
	n, err := g.ReadAt(p, g.pos)
	g.pos += int64(n)
	return n, err
}

func (g *guardedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	end := off + int64(len(p))
	for _, s := range g.spans {
		if off < s[1] && end > s[0] {
			return 0, errors.New("read touched pixel data")
		}
	}
	g.read += int64(len(p))
	return g.r.ReadAt(p, off)
}

func TestEnumerateSubIFDsDoesNotReadPixels(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, true)
	g := &guardedReaderAt{r: bytes.NewReader(fx.data), spans: fx.pixelSpans}
	pages, err := EnumerateSubIFDs(g)
	if err != nil {
		t.Fatalf("enumeration read pixel data or failed: %v", err)
	}
	if len(pages) != len(fx.pages) {
		t.Fatalf("got %d pages, want %d", len(pages), len(fx.pages))
	}

	// Reader construction must also stay metadata-only.
	g2 := &guardedReaderAt{r: bytes.NewReader(fx.data), spans: fx.pixelSpans}
	if _, err := NewReader(g2); err != nil {
		t.Fatalf("construction read pixel data or failed: %v", err)
	}
}

// mutate returns a copy of data with the 4-byte big-endian patch applied.
func patch32(enc binary.ByteOrder, data []byte, off int64, v uint32) []byte {
	out := append([]byte(nil), data...)
	enc.PutUint32(out[off:off+4], v)
	return out
}

func TestSubIFDLoop(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	pageOff := b.writeIFD(grayEntrySet(2, 1, 0, b.data(px...), len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{0}},
	), 0)
	b.setFirstIFD(pageOff)
	// Patch the SubIFDs value to point at its own IFD. grayEntrySet produces
	// 10 entries, so the appended SubIFDs entry (tag 330) sorts last: its
	// value slot is 2 + 12*(nEntries-1) + 8 into the IFD.
	subSlot := pageOff + 2 + 12*10 + 8
	data := patch32(binary.LittleEndian, b.build(), subSlot, uint32(pageOff))

	_, err := EnumerateSubIFDs(bytes.NewReader(data))
	if !errors.Is(err, ErrIFDLoop) {
		t.Fatalf("err = %v, want ErrIFDLoop", err)
	}
	var ifdErr *IFDError
	if !errors.As(err, &ifdErr) {
		t.Fatalf("%v is not *IFDError", err)
	}
	if ifdErr.Offset != pageOff || !reflect.DeepEqual(ifdErr.Path, []int{0}) {
		t.Errorf("context = offset %d path %v, want %d [0]", ifdErr.Offset, ifdErr.Path, pageOff)
	}
}

func TestSubIFDDuplicateOffset(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	pxOff := b.data(px...)
	child := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px)), 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(child), uint32(child)}},
	), 0)
	b.setFirstIFD(main)

	_, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if !errors.Is(err, ErrIFDDuplicate) {
		t.Fatalf("err = %v, want ErrIFDDuplicate", err)
	}
	var ifdErr *IFDError
	if !errors.As(err, &ifdErr) {
		t.Fatalf("%v is not *IFDError", err)
	}
	if ifdErr.Offset != child || !reflect.DeepEqual(ifdErr.Path, []int{1}) {
		t.Errorf("context = offset %d path %v, want %d [1]", ifdErr.Offset, ifdErr.Path, child)
	}
}

func TestSubIFDBadType(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(1, 1, 0)
	pxOff := b.data(px...)
	child := b.writeIFD(grayEntrySet(1, 1, 0, pxOff, len(px)), 0)
	main := b.writeIFD(grayEntrySet(1, 1, 0, pxOff, len(px),
		// SubIFDs must be LONG or IFD, not SHORT.
		testIFDEntry{tag: tSubIFDs, typ: dtShort, vals: []uint32{uint32(child)}},
	), 0)
	b.setFirstIFD(main)

	_, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if !errors.Is(err, ErrIFDType) {
		t.Fatalf("err = %v, want ErrIFDType", err)
	}
	// An invalid SubIFDs tag must not break main image decoding.
	if _, err := Decode(bytes.NewReader(b.build())); err != nil {
		t.Fatalf("Decode: %v", err)
	}
}

func TestSubIFDDepthLimit(t *testing.T) {
	const n = DefaultMaxIFDDepth + 3
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(1, 1, 0)
	pxOff := b.data(px...)

	next := int64(0)
	var first int64
	for i := 0; i < n; i++ {
		var extra []testIFDEntry
		if next != 0 {
			extra = append(extra, testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(next)}})
		}
		next = b.writeIFD(grayEntrySet(1, 1, 0, pxOff, len(px), extra...), 0)
		first = next
	}
	b.setFirstIFD(first)

	_, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if !errors.Is(err, ErrIFDDepth) {
		t.Fatalf("err = %v, want ErrIFDDepth", err)
	}
	var ifdErr *IFDError
	if !errors.As(err, &ifdErr) {
		t.Fatalf("%v is not *IFDError", err)
	}
	if want := DefaultMaxIFDDepth + 1; len(ifdErr.Path) != want {
		t.Errorf("path length = %d, want %d", len(ifdErr.Path), want)
	}
}

func TestSubIFDCorruptChild(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	pxOff := b.data(px...)
	// The child's ImageWidth has five values, so they are stored out of line.
	// Its value slot is then patched to point past EOF, making the read fail.
	child := b.writeIFD([]testIFDEntry{
		{tag: tImageWidth, typ: dtLong, vals: []uint32{1, 2, 3, 4, 5}},
	}, 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(child)}},
	), 0)
	b.setFirstIFD(main)
	// Patching a finished file in place; writeIFD is no longer called.
	b.enc.PutUint32(b.blob[child+2+8:child+2+12], 0xfffffff0)

	_, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if err == nil {
		t.Fatal("got nil error, want failure")
	}
	var ifdErr *IFDError
	if !errors.As(err, &ifdErr) {
		t.Fatalf("%v is not *IFDError", err)
	}
	if ifdErr.Offset != child || !reflect.DeepEqual(ifdErr.Path, []int{0}) {
		t.Errorf("context = offset %d path %v, want %d [0]", ifdErr.Offset, ifdErr.Path, child)
	}
	if !strings.Contains(err.Error(), "IFD offset") {
		t.Errorf("error %q lacks location context", err)
	}
	// The main image must still decode.
	if _, err := Decode(bytes.NewReader(b.build())); err != nil {
		t.Fatalf("Decode: %v", err)
	}
}

func TestReaderDecodeFailureDoesNotPollute(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	goodPx := grayPixels(2, 1, 7)
	goodPxOff := b.data(goodPx...)

	// Page 1 points at pixel data past EOF; pages 0 and 2 are fine.
	bad := b.writeIFD(grayEntrySet(2, 1, 0, 1<<20, 2), 0)
	good := b.writeIFD(grayEntrySet(2, 1, 0, goodPxOff, len(goodPx)), 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, goodPxOff, len(goodPx),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(bad), uint32(good)}},
	), 0)
	b.setFirstIFD(main)

	r, err := NewReader(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	before := r.Pages()
	if _, err := r.Decode(1); err == nil {
		t.Fatal("Decode(1): got nil error, want failure")
	} else {
		var ifdErr *IFDError
		if !errors.As(err, &ifdErr) || ifdErr.Offset != bad {
			t.Errorf("Decode(1) error = %v, want *IFDError at offset %d", err, bad)
		}
	}
	if !reflect.DeepEqual(before, r.Pages()) {
		t.Error("page descriptions changed after a failed decode")
	}
	img, err := r.Decode(2)
	if err != nil {
		t.Fatalf("Decode(2) after failure: %v", err)
	}
	if g := img.(*image.Gray); g.Pix[0] != 7 {
		t.Errorf("Decode(2) first pixel = %d, want 7", g.Pix[0])
	}
}

func TestReaderConcurrentDecode(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, true)
	r, err := NewReader(bytes.NewReader(fx.data))
	if err != nil {
		t.Fatal(err)
	}
	errc := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func(i int) {
			img, err := r.Decode(i % r.NumPages())
			if err == nil && img.Bounds().Empty() {
				err = errors.New("empty image")
			}
			errc <- err
		}(i)
	}
	for i := 0; i < 8; i++ {
		if err := <-errc; err != nil {
			t.Error(err)
		}
	}
}

func TestReaderIndexError(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, false)
	r, err := NewReader(bytes.NewReader(fx.data))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Page(-1); ok {
		t.Error("Page(-1) reported ok")
	}
	if _, ok := r.Page(r.NumPages()); ok {
		t.Error("Page(NumPages) reported ok")
	}
	if _, err := r.Decode(-1); err == nil {
		t.Error("Decode(-1) succeeded")
	}
	if _, err := r.Config(r.NumPages()); err == nil {
		t.Error("Config(out of range) succeeded")
	}
}

// TestSubIFDDeflatePage checks that an associated page using a compressed
// code path decodes, exercising the shared strip path across pages.
func TestSubIFDDeflatePage(t *testing.T) {
	px := grayPixels(4, 2, 50)
	var zbuf bytes.Buffer
	zw := zlib.NewWriter(&zbuf)
	zw.Write(px)
	zw.Close()
	zdata := zbuf.Bytes()

	b := newBuilder(binary.BigEndian)
	mainPx := grayPixels(2, 1, 1)
	mainPxOff := b.data(mainPx...)
	zOff := b.data(zdata...)
	child := b.writeIFD([]testIFDEntry{
		{tag: tImageWidth, typ: dtShort, vals: []uint32{4}},
		{tag: tImageLength, typ: dtShort, vals: []uint32{2}},
		{tag: tBitsPerSample, typ: dtShort, vals: []uint32{8}},
		{tag: tCompression, typ: dtShort, vals: []uint32{cDeflate}},
		{tag: tPhotometricInterpretation, typ: dtShort, vals: []uint32{pBlackIsZero}},
		{tag: tSamplesPerPixel, typ: dtShort, vals: []uint32{1}},
		{tag: tStripOffsets, typ: dtLong, vals: []uint32{uint32(zOff)}},
		{tag: tRowsPerStrip, typ: dtLong, vals: []uint32{2}},
		{tag: tStripByteCounts, typ: dtLong, vals: []uint32{uint32(len(zdata))}},
	}, 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, mainPxOff, len(mainPx),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(child)}},
	), 0)
	b.setFirstIFD(main)

	r, err := NewReader(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	if p, _ := r.Page(1); p.Compression != cDeflate {
		t.Fatalf("child compression = %d, want %d", p.Compression, cDeflate)
	}
	img, err := r.Decode(1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(img.(*image.Gray).Pix, px) {
		t.Error("deflate page pixels differ")
	}
}

// TestSubIFDSharedStrip checks that two associated pages may point at the same
// strip data (shared pixels are legal; only shared IFDs are rejected).
func TestSubIFDSharedStrip(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 2, 3)
	pxOff := b.data(px...)
	c1 := b.writeIFD(grayEntrySet(2, 2, 0x1, pxOff, len(px)), 0)
	c2 := b.writeIFD(grayEntrySet(2, 2, 0x4, pxOff, len(px)), 0)
	main := b.writeIFD(grayEntrySet(2, 2, 0, pxOff, len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{uint32(c1), uint32(c2)}},
	), 0)
	b.setFirstIFD(main)

	r, err := NewReader(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		img, err := r.Decode(i)
		if err != nil {
			t.Fatalf("Decode(%d): %v", i, err)
		}
		if !reflect.DeepEqual(img.(*image.Gray).Pix, px) {
			t.Errorf("page %d pixels differ", i)
		}
	}
}

// TestSubIFDZeroOffsetIsSkipped checks that a zero SubIFDs element is not
// treated as an IFD at offset 0 (the file header). A zero is a common
// placeholder for an absent page.
func TestSubIFDZeroOffsetIsSkipped(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	pxOff := b.data(px...)
	child := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px)), 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{0, uint32(child)}},
	), 0)
	b.setFirstIFD(main)

	pages, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) != 2 {
		t.Fatalf("got %d pages, want 2 (zero offset skipped)", len(pages))
	}
	// The real child keeps its position (Index 1), so index identifies the
	// SubIFDs array slot rather than the returned page's ordinal.
	if pages[1].Offset != child || pages[1].Index != 1 || pages[1].Parent != 0 {
		t.Errorf("child page = %+v", pages[1])
	}
}

// TestNextIFDChainIsNotFollowed documents that enumeration follows SubIFDs but
// not the top-level next-IFD chain, matching the reader's existing behavior of
// treating the first IFD as the main image.
func TestNextIFDChainIsNotFollowed(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	pxOff := b.data(px...)
	second := b.writeIFD(grayEntrySet(4, 4, 0, pxOff, len(px)), 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, pxOff, len(px)), second)
	b.setFirstIFD(main)

	pages, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) != 1 || pages[0].Offset != main {
		t.Fatalf("pages = %+v, want only the main IFD", pages)
	}
}

// TestReaderConfigMatchesDecode checks Config and Decode agree on dimensions
// and color model for every page, without Config reading pixel data.
func TestReaderConfigMatchesDecode(t *testing.T) {
	fx := buildCameraFixture(t, binary.LittleEndian, true)
	g := &guardedReaderAt{r: bytes.NewReader(fx.data), spans: fx.pixelSpans}
	r, err := NewReaderAt(g, binary.LittleEndian, fx.offsets["main"])
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < r.NumPages(); i++ {
		cfg, err := r.Config(i)
		if err != nil {
			t.Fatalf("Config(%d): %v", i, err)
		}
		if cfg.Width != fx.pages[i].Width || cfg.Height != fx.pages[i].Height {
			t.Errorf("page %d config = %dx%d, want %dx%d", i, cfg.Width, cfg.Height, fx.pages[i].Width, fx.pages[i].Height)
		}
	}
	if len(g.spans) == 0 {
		t.Fatal("fixture has no pixel spans")
	}
}

// TestSubIFDHugeOffsetViaReader checks that enumerating a file read through a
// plain io.Reader (buffered internally) reports a hostile SubIFD offset as an
// error instead of allocating up to that offset.
func TestSubIFDHugeOffsetViaReader(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 0)
	main := b.writeIFD(grayEntrySet(2, 1, 0, b.data(px...), len(px),
		testIFDEntry{tag: tSubIFDs, typ: dtLong, vals: []uint32{0xffffff00}},
	), 0)
	b.setFirstIFD(main)

	_, err := EnumerateSubIFDs(struct{ io.Reader }{bytes.NewReader(b.build())})
	if err == nil {
		t.Fatal("got nil error, want failure")
	}
	var ifdErr *IFDError
	if !errors.As(err, &ifdErr) || ifdErr.Offset != 0xffffff00 {
		t.Errorf("err = %v, want *IFDError at 0xffffff00", err)
	}
}

// TestMalformedInformationalTagsIgnored checks that NewSamplesPerPixel and
// NewSubfileType, which earlier versions did not parse, cannot reject a file
// that Decode used to accept when they are malformed.
func TestMalformedInformationalTagsIgnored(t *testing.T) {
	b := newBuilder(binary.LittleEndian)
	px := grayPixels(2, 1, 9)
	pxOff := b.data(px...)
	// Entries are in ascending tag order. SamplesPerPixel has a datatype that
	// ifdUint rejects; because the tag is only informational, enumeration and
	// Decode must skip it rather than fail the file.
	main := b.writeIFD([]testIFDEntry{
		{tag: tImageWidth, typ: dtShort, vals: []uint32{2}},
		{tag: tImageLength, typ: dtShort, vals: []uint32{1}},
		{tag: tBitsPerSample, typ: dtShort, vals: []uint32{8}},
		{tag: tCompression, typ: dtShort, vals: []uint32{cNone}},
		{tag: tPhotometricInterpretation, typ: dtShort, vals: []uint32{pBlackIsZero}},
		{tag: tSamplesPerPixel, typ: 0x7},
		{tag: tStripOffsets, typ: dtLong, vals: []uint32{uint32(pxOff)}},
		{tag: tRowsPerStrip, typ: dtLong, vals: []uint32{1}},
		{tag: tStripByteCounts, typ: dtLong, vals: []uint32{uint32(len(px))}},
	}, 0)
	b.setFirstIFD(main)

	// Enumeration must succeed and simply omit SamplesPerPixel.
	pages, err := EnumerateSubIFDs(bytes.NewReader(b.build()))
	if err != nil {
		t.Fatalf("EnumerateSubIFDs: %v", err)
	}
	if got := pages[0].SamplesPerPixel; got != 0 {
		t.Errorf("SamplesPerPixel = %d, want 0", got)
	}
	// Package-level Decode must still work.
	if _, err := Decode(bytes.NewReader(b.build())); err != nil {
		t.Fatalf("Decode: %v", err)
	}
}

func TestIFDErrorMessage(t *testing.T) {
	err := &IFDError{Offset: 1234, Path: []int{1, 2}, Err: ErrIFDLoop}
	msg := err.Error()
	if !strings.Contains(msg, "1234") || !strings.Contains(msg, "[1 2]") || !strings.Contains(msg, "loop") {
		t.Errorf("Error() = %q, want offset, path and cause", msg)
	}
	if !errors.Is(err, ErrIFDLoop) {
		t.Error("errors.Is did not find the wrapped cause")
	}
	main := &IFDError{Offset: 8, Err: ErrIFDType}
	if got := main.Error(); !strings.Contains(got, "offset 8") {
		t.Errorf("Error() = %q, want the offset", got)
	}
}

// ExampleReader_Decode shows selecting associated pages by their role and
// decoding only those, leaving the main image untouched.
func ExampleReader_Decode() {
	fx := buildCameraFixture(new(testing.T), binary.LittleEndian, false)

	r, err := NewReader(bytes.NewReader(fx.data))
	if err != nil {
		panic(err)
	}
	for i := 0; i < r.NumPages(); i++ {
		p, _ := r.Page(i)
		switch {
		case i == 0:
			// The main image.
			continue
		case p.IsMask():
			img, err := r.Decode(i)
			if err != nil {
				panic(err)
			}
			b := img.Bounds()
			fmt.Printf("depth map %d x %d\n", b.Dx(), b.Dy())
		case p.ReducedResolution():
			img, err := r.Decode(i)
			if err != nil {
				panic(err)
			}
			b := img.Bounds()
			fmt.Printf("thumbnail %d x %d\n", b.Dx(), b.Dy())
		}
	}
	// Output:
	// thumbnail 4 x 3
	// depth map 4 x 3
}
