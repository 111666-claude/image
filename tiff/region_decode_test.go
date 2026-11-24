// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"image"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
)

// genPattern fills pix (laid out as the decoder's target image type) with a
// deterministic, non-uniform pattern so that predictor and channel ordering
// bugs are visible. w is the image width; the layout matches t.mode/t.bpp.
func genPattern(mode imageMode, bpp, w, h int) []byte {
	spp := samplesPerPixel(mode)
	if mode == mRGB {
		spp = 3
	}
	bps := bpp / 8
	if bps == 0 {
		bps = 1
	}
	stride := w * spp * bps
	pix := make([]byte, stride*h)
	put := func(x, y, s int, v uint16) {
		off := y*stride + (x*spp+s)*bps
		if bps == 2 {
			binary.LittleEndian.PutUint16(pix[off:off+2], v)
		} else {
			pix[off] = byte(v)
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for s := 0; s < spp; s++ {
				// Distinct formula per channel; include transparent samples so
				// alpha handling is exercised.
				v := uint16((x*37 + y*17 + s*71) & 0xff)
				if bps == 2 {
					v = uint16(x*257 + y*131 + s*4099)
				}
				if s == spp-1 && (mode == mRGBA || mode == mNRGBA) {
					// Vary alpha, including fully transparent pixels.
					if bps == 2 {
						v = uint16((x + y) & 1 * 0xffff)
					} else {
						v = uint16((x+y)&1) * 0xff
					}
				}
				put(x, y, s, v)
			}
		}
	}
	return pix
}

// decodeFull is a thin wrapper around Decode over an in-memory byte slice.
func decodeFull(t *testing.T, data []byte) image.Image {
	t.Helper()
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	return img
}

// compareRegion asserts that every pixel of sub within want equals the
// corresponding pixel of full.
func compareRegion(t *testing.T, full, sub image.Image, want image.Rectangle) {
	t.Helper()
	if sub.Bounds() != want {
		t.Fatalf("bounds = %v, want %v", sub.Bounds(), want)
	}
	for y := want.Min.Y; y < want.Max.Y; y++ {
		for x := want.Min.X; x < want.Max.X; x++ {
			r0, g0, b0, a0 := full.At(x, y).RGBA()
			r1, g1, b1, a1 := sub.At(x, y).RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				t.Fatalf("pixel (%d,%d): full %v region %v", x, y, full.At(x, y), sub.At(x, y))
			}
		}
	}
}

// regionMatrix returns a set of rectangles exercising origin, tangency,
// cross-block spans and out-of-bounds clamping.
func regionMatrix(w, h int) []image.Rectangle {
	return []image.Rectangle{
		image.Rect(0, 0, w, h),
		image.Rect(-8, -8, w+8, h+8),
		image.Rect(1, 1, minInt(7, w), minInt(7, h)),
		image.Rect(w/2-1, h/2-1, w/2+2, h/2+2), // straddles a block corner
		image.Rect(0, 0, w, 1),
		image.Rect(0, h-1, w, h),
		image.Rect(w-1, 0, w, h),
		image.Rect(0, 0, 1, 1),
		image.Rect(w-1, h-1, w, h),
		image.Rect(-10, h/2, 3, h/2+3),
		image.Rect(w-3, -10, w+10, 3),
		image.Rect(w+1, h+1, w+10, h+10), // fully outside
	}
}

// TestDecodeRegionMatrix builds synthetic strip and tile images across the
// compression / predictor / planar matrix and checks DecodeRegion against a
// full Decode for many rectangles.
func TestDecodeRegionMatrix(t *testing.T) {
	const w, h = 33, 29 // deliberately non-multiples of the 16x16 tile
	modes := []struct {
		mode imageMode
		bpp  int
	}{
		{mGray, 8},
		{mGray, 16},
		{mGrayInvert, 8},
		{mRGB, 8},
		{mRGB, 16},
		{mRGBA, 8},
		{mRGBA, 16},
		{mNRGBA, 8},
		{mNRGBA, 16},
	}
	compressions := []uint{cNone, cDeflate, cPackBits}
	planars := []uint{pcChunky, pcPlanar}

	for _, md := range modes {
		for _, comp := range compressions {
			for _, planar := range planars {
				// Planar only applies to multi-sample RGB images.
				if planar == pcPlanar && md.mode != mRGB && md.mode != mRGBA && md.mode != mNRGBA {
					continue
				}
				for _, predictor := range []uint{prNone, prHorizontal} {
					if predictor == prHorizontal && comp == cNone {
						continue // predictor only meaningful with compression here
					}
					for _, tiled := range []bool{false, true} {
						name := modeName(md.mode, md.bpp) +
							"/" + compName(comp) +
							"/pred" + strconv.Itoa(int(predictor)) +
							"/" + planarName(planar) +
							"/" + layoutName(tiled)
						t.Run(name, func(t *testing.T) {
							st := &syntheticTIFF{
								width: w, height: h,
								mode: md.mode, bpp: md.bpp,
								compression: comp, predictor: predictor, planar: planar,
								pix: genPattern(md.mode, md.bpp, w, h),
							}
							if tiled {
								st.tileWidth, st.tileLength = 16, 16
							} else {
								st.rowsPerStrip = 8
							}
							data := buildTIFF(st)
							full := decodeFull(t, data)

							ra := bytes.NewReader(data)
							for _, rect := range regionMatrix(w, h) {
								want := rect.Intersect(image.Rect(0, 0, w, h))
								sub, err := DecodeRegion(ra, rect)
								if want.Empty() {
									if err == nil {
										t.Fatalf("rect %v: want error for empty intersection", rect)
									}
									continue
								}
								if err != nil {
									t.Fatalf("rect %v: %v", rect, err)
								}
								compareRegion(t, full, sub, want)
							}
						})
					}
				}
			}
		}
	}
}

// TestDecodeRegionReusableReaderAt calls DecodeRegion repeatedly on the same
// ReaderAt and confirms the results are stable and identical to Decode.
func TestDecodeRegionReusableReaderAt(t *testing.T) {
	st := &syntheticTIFF{
		width: 40, height: 30, mode: mRGBA, bpp: 8,
		compression: cDeflate, predictor: prHorizontal, planar: pcChunky,
		tileWidth: 16, tileLength: 16,
		pix: genPattern(mRGBA, 8, 40, 30),
	}
	data := buildTIFF(st)
	full := decodeFull(t, data)
	ra := bytes.NewReader(data)
	rect := image.Rect(5, 6, 33, 24)
	want := rect
	for k := 0; k < 5; k++ {
		sub, err := DecodeRegion(ra, rect)
		if err != nil {
			t.Fatalf("call %d: %v", k, err)
		}
		compareRegion(t, full, sub, want)
	}
}

// TestDecodeRegionErrors covers the documented failure modes and asserts that
// errors carry strip/tile context.
func TestDecodeRegionErrors(t *testing.T) {
	st := &syntheticTIFF{
		width: 24, height: 24, mode: mRGB, bpp: 8,
		compression: cNone, planar: pcChunky,
		tileWidth: 16, tileLength: 16,
		pix: genPattern(mRGB, 8, 24, 24),
	}
	data := buildTIFF(st)

	t.Run("empty rect", func(t *testing.T) {
		if _, err := DecodeRegion(bytes.NewReader(data), image.Rect(0, 0, 0, 0)); err == nil {
			t.Fatal("want error for empty rectangle")
		}
	})
	t.Run("negative width", func(t *testing.T) {
		// image.Rect canonicalizes, so build the inverted rectangle directly.
		bad := image.Rectangle{Min: image.Pt(10, 10), Max: image.Pt(2, 20)}
		if _, err := DecodeRegion(bytes.NewReader(data), bad); err == nil {
			t.Fatal("want error for inverted rectangle")
		}
	})
	t.Run("no overlap", func(t *testing.T) {
		for _, r := range []image.Rectangle{
			image.Rect(24, 0, 30, 5),
			image.Rect(-5, -5, 0, 0),
			image.Rect(0, 24, 24, 30),
		} {
			if _, err := DecodeRegion(bytes.NewReader(data), r); err == nil {
				t.Fatalf("rect %v: want non-overlap error", r)
			}
		}
	})
	t.Run("resource cap", func(t *testing.T) {
		// A header claiming enormous dimensions with a trivial payload must be
		// rejected before any pixel allocation.
		huge := makeHugeHeaderTIFF()
		_, err := DecodeRegion(bytes.NewReader(huge), image.Rect(0, 0, 40000, 40000))
		if err == nil {
			t.Fatal("want error for oversized region")
		}
	})
	t.Run("bad offset has tile context", func(t *testing.T) {
		corrupt := corruptLastTileOffset(data)
		// Targeting the corrupt tile yields a contextual error.
		_, err := DecodeRegion(bytes.NewReader(corrupt), image.Rect(16, 16, 24, 24))
		if err == nil || !strings.Contains(err.Error(), "tile") {
			t.Fatalf("want tile-context error, got %v", err)
		}
		// A region in a healthy tile must not touch the corrupt one.
		img, err := DecodeRegion(bytes.NewReader(corrupt), image.Rect(0, 0, 8, 8))
		if err != nil {
			t.Fatalf("healthy tile region failed: %v", err)
		}
		if img.Bounds() != image.Rect(0, 0, 8, 8) {
			t.Fatalf("bounds = %v", img.Bounds())
		}
	})
	t.Run("short read has strip context", func(t *testing.T) {
		strip := (&syntheticTIFF{
			width: 24, height: 24, mode: mGray, bpp: 8,
			compression: cNone, planar: pcChunky, rowsPerStrip: 8,
			pix: genPattern(mGray, 8, 24, 24),
		})
		sd := buildTIFF(strip)
		corrupt := shrinkLastStripByteCount(sd)
		_, err := DecodeRegion(bytes.NewReader(corrupt), image.Rect(0, 16, 24, 24))
		if err == nil || !strings.Contains(err.Error(), "strip") {
			t.Fatalf("want strip-context error, got %v", err)
		}
	})
	t.Run("duplicate tile offsets", func(t *testing.T) {
		// Reusing one tile's offset for every tile is legal; a region decode
		// must succeed and match the full decode of the same data.
		dup := duplicateTileOffsets(data)
		full := decodeFull(t, dup)
		sub, err := DecodeRegion(bytes.NewReader(dup), image.Rect(0, 0, 24, 24))
		if err != nil {
			t.Fatalf("duplicate offsets: %v", err)
		}
		compareRegion(t, full, sub, image.Rect(0, 0, 24, 24))
	})
}

// TestDecodeRegionFixtures runs DecodeRegion against the checked-in TIFF
// fixtures (strip, tile, LZW, CCITT, palette, 16-bit and alpha-bearing) and
// cross-checks every pixel against the full Decode output.
func TestDecodeRegionFixtures(t *testing.T) {
	fixtures := []string{
		"video-001.tiff",
		"video-001-uncompressed.tiff",
		"video-001-strip-64.tiff",
		"video-001-tile-64x64.tiff",
		"video-001-16bit.tiff",
		"video-001-gray.tiff",
		"video-001-gray-16bit.tiff",
		"video-001-paletted.tiff",
		"blue-purple-pink.lzwcompressed.tiff",
		"bw-uncompressed.tiff",
		"bw-deflate.tiff",
		"bw-packbits.tiff",
		"bw-gopher_ccittGroup3.tiff",
		"bw-gopher_ccittGroup4.tiff",
	}
	for _, name := range fixtures {
		t.Run(name, func(t *testing.T) {
			data := readTestData(t, name)
			full := decodeFull(t, data)
			b := full.Bounds()
			ra := bytes.NewReader(data)
			for _, rect := range regionMatrix(b.Dx(), b.Dy()) {
				want := rect.Intersect(b)
				sub, err := DecodeRegion(ra, rect)
				if want.Empty() {
					if err == nil {
						t.Fatalf("rect %v: want error", rect)
					}
					continue
				}
				if err != nil {
					t.Fatalf("rect %v: %v", rect, err)
				}
				compareRegion(t, full, sub, want)
			}
		})
	}
}

// TestDecodeRegionNoUnrelatedReads confirms blocks outside the requested
// rectangle are never read from the underlying ReaderAt.
func TestDecodeRegionNoUnrelatedReads(t *testing.T) {
	st := &syntheticTIFF{
		width: 48, height: 48, mode: mRGB, bpp: 8,
		compression: cDeflate, planar: pcChunky,
		tileWidth: 16, tileLength: 16,
		pix: genPattern(mRGB, 8, 48, 48),
	}
	data := buildTIFF(st)

	fullRA := &countingReaderAt{b: data}
	if _, err := DecodeRegion(fullRA, image.Rect(0, 0, 48, 48)); err != nil {
		t.Fatal(err)
	}
	subRA := &countingReaderAt{b: data}
	if _, err := DecodeRegion(subRA, image.Rect(2, 2, 10, 10)); err != nil {
		t.Fatal(err)
	}
	if subRA.bytes >= fullRA.bytes {
		t.Fatalf("region read %d bytes, full read %d bytes; expected strictly less",
			subRA.bytes, fullRA.bytes)
	}
}

type countingReaderAt struct {
	b     []byte
	bytes int64
}

func (c *countingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	c.bytes += int64(len(p))
	return bytes.NewReader(c.b).ReadAt(p, off)
}

// --- fixture mutation helpers -------------------------------------------------

func readTestData(t *testing.T, name string) []byte {
	t.Helper()
	f, err := os.Open(testdataDir + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// makeHugeHeaderTIFF builds an uncompressed gray TIFF whose IFD claims
// 100000x100000 pixels but carries only a tiny payload. It is assembled
// directly (rather than via buildTIFF) so that no pixel-sized allocation is
// needed to produce it.
func makeHugeHeaderTIFF() []byte {
	enc := binary.LittleEndian
	var b bytes.Buffer
	b.WriteString(leHeader)
	// A single byte of "pixel" data; the IFD will claim far more.
	pixelOff := uint32(b.Len())
	b.WriteByte(0)

	ifdOff := uint32(b.Len())
	type e struct {
		tag uint16
		typ uint16
		cnt uint32
		val uint32
	}
	entries := []e{
		{tImageWidth, dtLong, 1, 100000},
		{tImageLength, dtLong, 1, 100000},
		{tBitsPerSample, dtShort, 1, uint32(8) | uint32(8)<<16},
		{tCompression, dtShort, 1, uint32(cNone) | uint32(cNone)<<16},
		{tPhotometricInterpretation, dtShort, 1, uint32(pBlackIsZero) | uint32(pBlackIsZero)<<16},
		{tStripOffsets, dtLong, 1, pixelOff},
		{tRowsPerStrip, dtLong, 1, 100000},
		{tStripByteCounts, dtLong, 1, 1},
	}
	var hdr [2]byte
	enc.PutUint16(hdr[:], uint16(len(entries)))
	b.Write(hdr[:])
	var ent [ifdLen]byte
	for _, x := range entries {
		enc.PutUint16(ent[0:2], x.tag)
		enc.PutUint16(ent[2:4], x.typ)
		enc.PutUint32(ent[4:8], x.cnt)
		enc.PutUint32(ent[8:12], x.val)
		b.Write(ent[:])
	}
	b.Write(make([]byte, 4)) // next IFD = 0
	out := b.Bytes()
	enc.PutUint32(out[4:8], ifdOff)
	return out
}

// patchIFDLongArray finds tag in the IFD and returns the file offset of its
// long-valued array.
func patchIFDLongArray(b []byte, tag uint16) int {
	enc := binary.LittleEndian
	ifdOff := int(enc.Uint32(b[4:8]))
	n := int(enc.Uint16(b[ifdOff : ifdOff+2]))
	for k := 0; k < n; k++ {
		e := ifdOff + 2 + 12*k
		if enc.Uint16(b[e:e+2]) == tag {
			return int(enc.Uint32(b[e+8 : e+12]))
		}
	}
	return -1
}

// ifdTagCount returns the value count of the given tag's IFD entry.
func ifdTagCount(b []byte, tag uint16) uint32 {
	enc := binary.LittleEndian
	ifdOff := int(enc.Uint32(b[4:8]))
	n := int(enc.Uint16(b[ifdOff : ifdOff+2]))
	for k := 0; k < n; k++ {
		e := ifdOff + 2 + 12*k
		if enc.Uint16(b[e:e+2]) == tag {
			return enc.Uint32(b[e+4 : e+8])
		}
	}
	return 0
}

func corruptLastTileOffset(b []byte) []byte {
	out := append([]byte(nil), b...)
	arr := patchIFDLongArray(out, tTileOffsets)
	if arr < 0 {
		panic("no tile offsets")
	}
	// Point the last tile offset far past EOF.
	last := int(ifdTagCount(out, tTileOffsets)) - 1
	binary.LittleEndian.PutUint32(out[arr+last*4:], uint32(len(out)+1_000_000))
	return out
}

func duplicateTileOffsets(b []byte) []byte {
	out := append([]byte(nil), b...)
	offArr := patchIFDLongArray(out, tTileOffsets)
	cntArr := patchIFDLongArray(out, tTileByteCounts)
	if offArr < 0 {
		panic("no tile offsets")
	}
	count := ifdTagCount(out, tTileOffsets)
	first := binary.LittleEndian.Uint32(out[offArr : offArr+4])
	firstCnt := binary.LittleEndian.Uint32(out[cntArr : cntArr+4])
	for k := uint32(1); k < count; k++ {
		binary.LittleEndian.PutUint32(out[offArr+int(k)*4:], first)
		binary.LittleEndian.PutUint32(out[cntArr+int(k)*4:], firstCnt)
	}
	return out
}

func shrinkLastStripByteCount(b []byte) []byte {
	out := append([]byte(nil), b...)
	arr := patchIFDLongArray(out, tStripByteCounts)
	if arr < 0 {
		panic("no strip byte counts")
	}
	count := ifdTagCount(out, tStripByteCounts)
	// Make the last strip claim only one byte.
	binary.LittleEndian.PutUint32(out[arr+int(count-1)*4:], 1)
	return out
}

// --- small naming/format helpers ----------------------------------------------

func modeName(m imageMode, bpp int) string {
	return map[imageMode]string{
		mGray: "gray", mGrayInvert: "grayinv", mRGB: "rgb",
		mRGBA: "rgba", mNRGBA: "nrgba", mPaletted: "paletted",
	}[m] + strconv.Itoa(bpp)
}

func compName(c uint) string {
	switch c {
	case cNone:
		return "none"
	case cDeflate:
		return "deflate"
	case cPackBits:
		return "packbits"
	default:
		return "comp" + strconv.Itoa(int(c))
	}
}

func planarName(p uint) string {
	if p == pcPlanar {
		return "planar"
	}
	return "chunky"
}

func layoutName(tiled bool) string {
	if tiled {
		return "tile"
	}
	return "strip"
}
