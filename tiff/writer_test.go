// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"testing"
)

var roundtripTests = []struct {
	filename string
	opts     *Options
}{
	{"video-001.tiff", nil},
	{"video-001-16bit.tiff", nil},
	{"video-001-gray.tiff", nil},
	{"video-001-gray-16bit.tiff", nil},
	{"video-001-paletted.tiff", nil},
	{"bw-packbits.tiff", nil},
	{"video-001.tiff", &Options{Predictor: true}},
	{"video-001.tiff", &Options{Compression: Deflate}},
	{"video-001.tiff", &Options{Predictor: true, Compression: Deflate}},
}

func openImage(filename string) (image.Image, error) {
	f, err := os.Open(testdataDir + filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

func TestRoundtrip(t *testing.T) {
	for _, rt := range roundtripTests {
		img, err := openImage(rt.filename)
		if err != nil {
			t.Fatal(err)
		}

		out := new(bytes.Buffer)
		err = Encode(out, img, rt.opts)
		if err != nil {
			t.Fatal(err)
		}

		img2, err := Decode(&buffer{buf: out.Bytes()})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, img, img2)
	}
}

// TestRoundtrip2 tests that encoding and decoding an image whose
// origin is not (0, 0) gives the same thing.
func TestRoundtrip2(t *testing.T) {
	m0 := image.NewRGBA(image.Rect(3, 4, 9, 8))
	for i := range m0.Pix {
		m0.Pix[i] = byte(i)
	}
	out := new(bytes.Buffer)
	if err := Encode(out, m0, nil); err != nil {
		t.Fatal(err)
	}
	m1, err := Decode(&buffer{buf: out.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, m0, m1)
}

func TestUnsupported(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 1, 1))
	out := new(bytes.Buffer)
	err := Encode(out, img, &Options{Compression: LZW})
	if err == nil {
		t.Error("tiff.Encode(LZW): no error returned, expected an error")
	}
}

// tiledRoundtripTests exercises tiled encoding across single-tile,
// multi-tile, exact-multiple and padded-edge dimensions.
var tiledRoundtripTests = []struct {
	name       string
	w, h       int
	tileW, tH  int
	rgba       bool
	nonZeroOrg bool
}{
	{"gray-padded-edges", 33, 17, 16, 16, false, false},
	{"gray-single-tile-smaller-than-image-tile", 10, 10, 16, 16, false, false},
	{"gray-single-exact-tile", 16, 16, 16, 16, false, false},
	{"gray-exact-grid", 48, 32, 16, 16, false, false},
	{"rgba-padded-edges", 30, 20, 16, 16, true, false},
	{"rgba-rectangular-tiles", 64, 48, 32, 16, true, false},
	{"rgba-exact-grid", 32, 32, 16, 16, true, false},
	{"rgba-non-zero-origin", 22, 18, 16, 16, true, true},
}

func newTiledTestImage(t *testing.T, w, h int, rgba, nonZeroOrg bool) image.Image {
	t.Helper()
	if rgba {
		base := image.NewRGBA(image.Rect(0, 0, w+8, h+8))
		for i := range base.Pix {
			base.Pix[i] = byte(i*7 + 3)
		}
		if nonZeroOrg {
			return base.SubImage(image.Rect(8, 8, 8+w, 8+h)).(*image.RGBA)
		}
		return base.SubImage(image.Rect(0, 0, w, h)).(*image.RGBA)
	}
	base := image.NewGray(image.Rect(0, 0, w+8, h+8))
	for i := range base.Pix {
		base.Pix[i] = byte(i*5 + 1)
	}
	return base.SubImage(image.Rect(0, 0, w, h)).(*image.Gray)
}

func TestTiledEncode(t *testing.T) {
	for _, tc := range tiledRoundtripTests {
		t.Run(tc.name, func(t *testing.T) {
			img := newTiledTestImage(t, tc.w, tc.h, tc.rgba, tc.nonZeroOrg)
			bpp := 1
			if tc.rgba {
				bpp = 4
			}

			out := new(bytes.Buffer)
			opt := &Options{Tiled: true, TileWidth: tc.tileW, TileLength: tc.tH}
			if err := Encode(out, img, opt); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			data := out.Bytes()

			// The output must decode through the existing decoder and the
			// pixels must survive the round trip, including padded edges.
			img2, err := Decode(&buffer{buf: data})
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			compare(t, img, img2)

			d, err := newDecoder(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("newDecoder: %v", err)
			}
			if got := d.firstVal(tTileWidth); got != uint(tc.tileW) {
				t.Errorf("TileWidth: got %d, want %d", got, tc.tileW)
			}
			if got := d.firstVal(tTileLength); got != uint(tc.tH) {
				t.Errorf("TileLength: got %d, want %d", got, tc.tH)
			}
			// Tiled images must not carry any strip tags.
			if got := d.firstVal(tRowsPerStrip); got != 0 {
				t.Errorf("RowsPerStrip present: got %d, want 0", got)
			}
			if _, ok := d.ifd[tStripOffsets]; ok {
				t.Error("unexpected StripOffsets tag in tiled output")
			}
			if _, ok := d.ifd[tStripByteCounts]; ok {
				t.Error("unexpected StripByteCounts tag in tiled output")
			}
			if got := d.firstVal(tPhotometricInterpretation); tc.rgba && got != pRGB || !tc.rgba && got != pBlackIsZero {
				t.Errorf("PhotometricInterpretation: got %d", got)
			}

			wantTilesAcross := (tc.w + tc.tileW - 1) / tc.tileW
			wantTilesDown := (tc.h + tc.tH - 1) / tc.tH
			wantNumTiles := wantTilesAcross * wantTilesDown
			offs, err := d.parseIFDOffsets(tTileOffsets, wantNumTiles)
			if err != nil {
				t.Fatalf("parse TileOffsets: %v", err)
			}
			counts, err := d.parseIFDOffsets(tTileByteCounts, wantNumTiles)
			if err != nil {
				t.Fatalf("parse TileByteCounts: %v", err)
			}
			if len(offs) != wantNumTiles || len(counts) != wantNumTiles {
				t.Fatalf("tile count: got %d offsets / %d counts, want %d", len(offs), len(counts), wantNumTiles)
			}

			tileBytes := uint(tc.tileW * tc.tH * bpp)
			for i := range offs {
				if counts[i] != tileBytes {
					t.Errorf("tile %d ByteCount: got %d, want %d", i, counts[i], tileBytes)
				}
				// Classic TIFF offsets are 32-bit and tile data must be
				// word-aligned.
				if offs[i]%2 != 0 {
					t.Errorf("tile %d offset %d is not word-aligned", i, offs[i])
				}
				if want := uint(8 + i*int(tileBytes)); offs[i] != want {
					t.Errorf("tile %d offset: got %d, want %d", i, offs[i], want)
				}
				if int64(offs[i])+int64(counts[i]) > int64(len(data)) {
					t.Errorf("tile %d extends past end of file", i)
				}
			}

			// Check the raw IFD on disk, bypassing the decoder's maps:
			// tags must be strictly ascending and tile arrays must use
			// the LONG datatype.
			assertTiledIFD(t, data, wantNumTiles, tc.rgba)
		})
	}
}

// assertTiledIFD parses the IFD directly from the encoded bytes and checks
// that the tags are sorted in ascending order and that the tile-related
// entries use the datatypes mandated by the TIFF specification.
func assertTiledIFD(t *testing.T, data []byte, numTiles int, rgba bool) {
	t.Helper()
	ifdOffset := binary.LittleEndian.Uint32(data[4:8])
	n := int(binary.LittleEndian.Uint16(data[ifdOffset : ifdOffset+2]))
	prevTag := uint16(0)
	seen := map[uint16]bool{}
	for i := 0; i < n; i++ {
		e := data[int(ifdOffset)+2+i*ifdLen:]
		tag := binary.LittleEndian.Uint16(e[0:2])
		datatype := binary.LittleEndian.Uint16(e[2:4])
		count := binary.LittleEndian.Uint32(e[4:8])
		if i > 0 && tag <= prevTag {
			t.Fatalf("IFD tags not ascending: %d follows %d", tag, prevTag)
		}
		prevTag = tag
		seen[tag] = true
		switch tag {
		case tTileOffsets, tTileByteCounts:
			if datatype != dtLong {
				t.Errorf("tag %d: got datatype %d, want LONG", tag, datatype)
			}
			if int(count) != numTiles {
				t.Errorf("tag %d: got count %d, want %d", tag, count, numTiles)
			}
		case tTileWidth, tTileLength:
			if datatype != dtShort {
				t.Errorf("tag %d: got datatype %d, want SHORT", tag, datatype)
			}
		}
	}
	for _, tag := range []uint16{tTileWidth, tTileLength, tTileOffsets, tTileByteCounts} {
		if !seen[tag] {
			t.Errorf("required tile tag %d missing from IFD", tag)
		}
	}
	for _, tag := range []uint16{tStripOffsets, tStripByteCounts, tRowsPerStrip} {
		if seen[tag] {
			t.Errorf("strip tag %d must not appear in a tiled IFD", tag)
		}
	}
	if rgba && !seen[tExtraSamples] {
		t.Error("RGBA tiled image is missing the ExtraSamples tag")
	}
}

// TestTiledFixtureInterop decodes the shipped tiled fixture, re-encodes it
// through the new tiled encoder and verifies pixel-identical round trip.
func TestTiledFixtureInterop(t *testing.T) {
	img, err := openImage("video-001-tile-64x64.tiff")
	if err != nil {
		t.Fatal(err)
	}
	rgba, ok := img.(*image.RGBA)
	if !ok {
		t.Fatalf("fixture decoded as %T, want *image.RGBA", img)
	}

	out := new(bytes.Buffer)
	if err := Encode(out, rgba, &Options{Tiled: true, TileWidth: 64, TileLength: 64}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	img2, err := Decode(&buffer{buf: out.Bytes()})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	compare(t, img, img2)

	d, err := newDecoder(bytes.NewReader(out.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if got := d.firstVal(tTileWidth); got != 64 {
		t.Errorf("TileWidth: got %d, want 64", got)
	}
	if got := d.firstVal(tTileLength); got != 64 {
		t.Errorf("TileLength: got %d, want 64", got)
	}
}

func TestTiledEdgePadding(t *testing.T) {
	const (
		w, h      = 20, 20
		tileW, tH = 16, 16
		tileBytes = tileW * tH
	)
	img := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8(x + 4*y)})
		}
	}

	out := new(bytes.Buffer)
	if err := Encode(out, img, &Options{Tiled: true, TileWidth: tileW, TileLength: tH}); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	data := out.Bytes()

	d, err := newDecoder(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("newDecoder: %v", err)
	}
	offs, err := d.parseIFDOffsets(tTileOffsets, 4)
	if err != nil || len(offs) != 4 {
		t.Fatalf("TileOffsets: %v (len=%d)", err, len(offs))
	}

	// Tile (1, 0) covers columns 16..31; only columns 16..19 hold image
	// data, the remaining 12 columns of every row must be zero padding.
	tile := data[offs[1] : int(offs[1])+tileBytes]
	for ty := 0; ty < tH; ty++ {
		for tx := 0; tx < tileW; tx++ {
			v := tile[ty*tileW+tx]
			switch {
			case tx < 4 && ty < h:
				if want := uint8(16 + tx + 4*ty); v != want {
					t.Errorf("tile(1,0) pixel (%d,%d): got %d, want %d", tx, ty, v, want)
				}
			case v != 0:
				t.Errorf("tile(1,0) padding pixel (%d,%d): got %d, want 0", tx, ty, v)
			}
		}
	}

	// The bottom-right tile only covers a 4x4 block of real pixels.
	tile = data[offs[3] : int(offs[3])+tileBytes]
	for ty := 0; ty < tH; ty++ {
		for tx := 0; tx < tileW; tx++ {
			v := tile[ty*tileW+tx]
			switch {
			case tx < 4 && ty < 4:
				if want := uint8(16 + tx + 4*(16+ty)); v != want {
					t.Errorf("tile(1,1) pixel (%d,%d): got %d, want %d", tx, ty, v, want)
				}
			case v != 0:
				t.Errorf("tile(1,1) padding pixel (%d,%d): got %d, want 0", tx, ty, v)
			}
		}
	}
}

// TestTiledEncodeErrors verifies that invalid parameter and unsupported
// combinations are rejected before any byte is written.
func TestTiledEncodeErrors(t *testing.T) {
	gray := image.NewGray(image.Rect(0, 0, 24, 24))
	rgba := image.NewRGBA(image.Rect(0, 0, 24, 24))

	tests := []struct {
		name string
		img  image.Image
		opt  *Options
	}{
		{"zero tile width", gray, &Options{Tiled: true, TileWidth: 0, TileLength: 16}},
		{"negative tile length", gray, &Options{Tiled: true, TileWidth: 16, TileLength: -16}},
		{"width not multiple of 16", gray, &Options{Tiled: true, TileWidth: 15, TileLength: 16}},
		{"length not multiple of 16", gray, &Options{Tiled: true, TileWidth: 16, TileLength: 17}},
		{"deflate compression", rgba, &Options{Tiled: true, TileWidth: 16, TileLength: 16, Compression: Deflate}},
		{"lzw compression", gray, &Options{Tiled: true, TileWidth: 16, TileLength: 16, Compression: LZW}},
		{"unsupported gray16 image", image.NewGray16(image.Rect(0, 0, 24, 24)), &Options{Tiled: true, TileWidth: 16, TileLength: 16}},
		{"unsupported nrgba image", image.NewNRGBA(image.Rect(0, 0, 24, 24)), &Options{Tiled: true, TileWidth: 16, TileLength: 16}},
		{"unsupported paletted image", image.NewPaletted(image.Rect(0, 0, 24, 24), color.Palette{}), &Options{Tiled: true, TileWidth: 16, TileLength: 16}},
		{"tile exceeds image size", gray, &Options{Tiled: true, TileWidth: 2048, TileLength: 2048}},
		{"tile dimensions overflow", gray, &Options{Tiled: true, TileWidth: 1 << 30, TileLength: 1 << 30}},
		// Hand-built images with oversized Rects: validation must reject
		// these before touching Pix, so no backing allocation is needed.
		{"too many tiles", &image.Gray{Pix: make([]byte, 1), Stride: 1, Rect: image.Rect(0, 0, 400000, 400000)}, &Options{Tiled: true, TileWidth: 16, TileLength: 16}},
		{"file offset overflow", &image.Gray{Pix: make([]byte, 1), Stride: 1, Rect: image.Rect(0, 0, 70000, 70000)}, &Options{Tiled: true, TileWidth: 16, TileLength: 16}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := new(bytes.Buffer)
			err := Encode(out, tc.img, tc.opt)
			if err == nil {
				t.Fatal("Encode: got nil error, want non-nil")
			}
			if out.Len() != 0 {
				t.Fatalf("Encode wrote %d bytes before returning an error", out.Len())
			}
		})
	}
}

// TestTiledDisabledUnchanged verifies that the default strip-based encoding
// is byte-identical to the previous behavior when tiled mode is not enabled.
func TestTiledDisabledUnchanged(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 33, 17))
	for i := range img.Pix {
		img.Pix[i] = byte(i * 3)
	}

	ref := new(bytes.Buffer)
	if err := Encode(ref, img, nil); err != nil {
		t.Fatal(err)
	}
	out := new(bytes.Buffer)
	if err := Encode(out, img, &Options{Tiled: false, TileWidth: 16, TileLength: 16}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(ref.Bytes(), out.Bytes()) {
		t.Fatal("enabling Tiled:false changed the strip-based output")
	}

	// The reference file must be a strip file with no tile tags.
	d, err := newDecoder(bytes.NewReader(ref.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if d.firstVal(tTileWidth) != 0 {
		t.Error("default output unexpectedly contains TileWidth")
	}
	if _, ok := d.ifd[tTileOffsets]; ok {
		t.Error("default output unexpectedly contains TileOffsets")
	}
}

func benchmarkEncode(b *testing.B, name string, pixelSize int) {
	b.Helper()
	img, err := openImage(name)
	if err != nil {
		b.Fatal(err)
	}
	s := img.Bounds().Size()
	b.SetBytes(int64(s.X * s.Y * pixelSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}

func BenchmarkEncode(b *testing.B)         { benchmarkEncode(b, "video-001.tiff", 4) }
func BenchmarkEncodePaletted(b *testing.B) { benchmarkEncode(b, "video-001-paletted.tiff", 1) }
func BenchmarkEncodeGray(b *testing.B)     { benchmarkEncode(b, "video-001-gray.tiff", 1) }
func BenchmarkEncodeGray16(b *testing.B)   { benchmarkEncode(b, "video-001-gray-16bit.tiff", 2) }
func BenchmarkEncodeRGBA(b *testing.B)     { benchmarkEncode(b, "video-001.tiff", 4) }
func BenchmarkEncodeRGBA64(b *testing.B)   { benchmarkEncode(b, "video-001-16bit.tiff", 8) }
