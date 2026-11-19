// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/lzw"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"sort"
	"strings"
	"testing"
)

// cmykBuildConfig describes a synthetic CMYK TIFF to build with buildCMYKTIFF.
type cmykBuildConfig struct {
	enc           byteOrder
	width         int
	height        int
	rowsPerStrip  int
	compression   uint16
	predictor     bool
	planarConfig  uint16 // 0 omits the PlanarConfiguration tag
	bitsPerSample []uint16
	tiled         bool
	tileW         int
	tileH         int
	// offsetOverride, when non-zero, replaces the single strip/tile offset.
	offsetOverride uint32
	// countOverride, when non-zero, replaces the single strip/tile byte count.
	countOverride uint32
}

// horizontalPredictCMYK applies the TIFF horizontal differencing predictor to
// one block of chunky pixel data, with samples channels per pixel and width
// pixels per row. It is the inverse of the decoder's prHorizontal handling.
func horizontalPredictCMYK(buf []byte, samples, width int) []byte {
	out := append([]byte(nil), buf...)
	rowBytes := samples * width
	for y := 0; y < len(out)/rowBytes; y++ {
		row := out[y*rowBytes : (y+1)*rowBytes]
		for i := samples; i < len(row); i++ {
			row[i] = row[i] - row[i-samples]
		}
	}
	return out
}

// packBitsEncode PackBits-encodes src. It is the inverse of the package's
// unpackBits decoder.
func packBitsEncode(src []byte) []byte {
	var dst []byte
	for i := 0; i < len(src); {
		run := 1
		for i+run < len(src) && src[i+run] == src[i] && run < 128 {
			run++
		}
		if run >= 2 {
			dst = append(dst, byte(-(run - 1)), src[i])
			i += run
			continue
		}
		// Gather a literal run, stopping at the start of a repeated pair.
		start := i
		for i < len(src) && i-start < 128 {
			if i+1 < len(src) && src[i] == src[i+1] {
				break
			}
			i++
		}
		dst = append(dst, byte(i-start-1))
		dst = append(dst, src[start:i]...)
	}
	return dst
}

func compressCMYKBlock(t *testing.T, block []byte, compression uint16) []byte {
	t.Helper()
	switch compression {
	case cNone:
		return block
	case cLZW:
		var buf bytes.Buffer
		w := lzw.NewWriter(&buf, lzw.MSB, 8)
		if _, err := w.Write(block); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	case cDeflate:
		var buf bytes.Buffer
		zw := zlib.NewWriter(&buf)
		if _, err := zw.Write(block); err != nil {
			t.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	case cPackBits:
		return packBitsEncode(block)
	default:
		t.Fatalf("unsupported test compression %d", compression)
		return nil
	}
}

// rawEntry is one IFD entry for the test-only assembleTIFF helper.
type rawEntry struct {
	tag  uint16
	kind uint16 // dtShort or dtLong
	vals []uint32
}

// assembleTIFF writes a minimal, spec-correct TIFF: header, pixel blocks, IFD
// and a pointer area for entries wider than four bytes. It does not rely on
// the package's other test helpers, whose offset bookkeeping assumes at most
// one indirect entry.
func assembleTIFF(enc byteOrder, blocks [][]byte, entries []rawEntry) []byte {
	sort.Slice(entries, func(i, j int) bool { return entries[i].tag < entries[j].tag })

	var b []byte
	if enc == binary.LittleEndian {
		b = append(b, 'I', 'I', 0x2a, 0x00)
	} else {
		b = append(b, 'M', 'M', 0x00, 0x2a)
	}
	b = enc.AppendUint32(b, 0) // IFD offset, patched at the end.
	for _, blk := range blocks {
		b = append(b, blk...)
	}
	ifdOff := len(b)
	pointerStart := ifdOff + 2 + ifdLen*len(entries) + 4

	b = enc.AppendUint16(b, uint16(len(entries)))
	var pointerArea []byte
	for _, e := range entries {
		b = enc.AppendUint16(b, e.tag)
		b = enc.AppendUint16(b, e.kind)
		b = enc.AppendUint32(b, uint32(len(e.vals)))
		dataLen := int(lengths[e.kind]) * len(e.vals)
		if dataLen <= 4 {
			var inline [4]byte
			for i, v := range e.vals {
				if e.kind == dtShort {
					enc.PutUint16(inline[2*i:], uint16(v))
				} else {
					enc.PutUint32(inline[4*i:], v)
				}
			}
			b = append(b, inline[:]...)
			continue
		}
		b = enc.AppendUint32(b, uint32(pointerStart+len(pointerArea)))
		for _, v := range e.vals {
			if e.kind == dtShort {
				pointerArea = enc.AppendUint16(pointerArea, uint16(v))
			} else {
				pointerArea = enc.AppendUint32(pointerArea, v)
			}
		}
	}
	b = enc.AppendUint32(b, 0) // No next IFD.
	b = append(b, pointerArea...)
	enc.PutUint32(b[4:8], uint32(ifdOff))
	return b
}

// buildCMYKTIFF synthesizes an 8-bit chunky CMYK TIFF from pix, which must be
// laid out as C0 M0 Y0 K0 C1 M1 Y1 K1 ..., row by row.
func buildCMYKTIFF(t *testing.T, cfg cmykBuildConfig, pix []byte) []byte {
	t.Helper()

	var blocks [][]byte
	var offsets, counts []uint32
	emit := func(block []byte) {
		off := uint32(8)
		for _, prev := range blocks {
			off += uint32(len(prev))
		}
		blocks = append(blocks, compressCMYKBlock(t, block, cfg.compression))
		offsets = append(offsets, off)
		counts = append(counts, uint32(len(blocks[len(blocks)-1])))
	}

	rowBytes := cfg.width * 4
	if cfg.tiled {
		tileBytes := cfg.tileW * cfg.tileH * 4
		for ty := 0; ty < cfg.height; ty += cfg.tileH {
			for tx := 0; tx < cfg.width; tx += cfg.tileW {
				// Tiles are always full tileW x tileH; pixels outside the
				// image bounds are padding (zero-filled).
				block := make([]byte, tileBytes)
				for y := ty; y < min(ty+cfg.tileH, cfg.height); y++ {
					for x := tx; x < min(tx+cfg.tileW, cfg.width); x++ {
						src := pix[(y*cfg.width+x)*4 : (y*cfg.width+x+1)*4]
						dst := block[((y-ty)*cfg.tileW+(x-tx))*4:]
						copy(dst, src)
					}
				}
				emit(block)
			}
		}
	} else {
		rps := cfg.rowsPerStrip
		if rps <= 0 {
			rps = cfg.height
		}
		for y0 := 0; y0 < cfg.height; y0 += rps {
			y1 := min(y0+rps, cfg.height)
			block := make([]byte, 0, (y1-y0)*rowBytes)
			for y := y0; y < y1; y++ {
				block = append(block, pix[y*rowBytes:(y+1)*rowBytes]...)
			}
			if cfg.predictor {
				block = horizontalPredictCMYK(block, 4, cfg.width)
			}
			emit(block)
		}
	}

	if cfg.offsetOverride != 0 {
		offsets = []uint32{cfg.offsetOverride}
	}
	if cfg.countOverride != 0 {
		counts = []uint32{cfg.countOverride}
	}

	bps := make([]uint32, len(cfg.bitsPerSample))
	for i, b := range cfg.bitsPerSample {
		bps[i] = uint32(b)
	}
	entries := []rawEntry{
		{tImageWidth, dtLong, []uint32{uint32(cfg.width)}},
		{tImageLength, dtLong, []uint32{uint32(cfg.height)}},
		{tBitsPerSample, dtShort, bps},
		{tCompression, dtShort, []uint32{uint32(cfg.compression)}},
		{tPhotometricInterpretation, dtShort, []uint32{pCMYK}},
		{tSamplesPerPixel, dtShort, []uint32{uint32(len(cfg.bitsPerSample))}},
	}
	if cfg.planarConfig != 0 {
		entries = append(entries, rawEntry{tPlanarConfiguration, dtShort, []uint32{uint32(cfg.planarConfig)}})
	}
	if cfg.predictor {
		entries = append(entries, rawEntry{tPredictor, dtShort, []uint32{prHorizontal}})
	}
	if cfg.tiled {
		entries = append(entries,
			rawEntry{tTileWidth, dtLong, []uint32{uint32(cfg.tileW)}},
			rawEntry{tTileLength, dtLong, []uint32{uint32(cfg.tileH)}},
			rawEntry{tTileOffsets, dtLong, offsets},
			rawEntry{tTileByteCounts, dtLong, counts},
		)
	} else {
		rps := cfg.rowsPerStrip
		if rps <= 0 {
			rps = cfg.height
		}
		entries = append(entries,
			rawEntry{tRowsPerStrip, dtShort, []uint32{uint32(rps)}},
			rawEntry{tStripOffsets, dtLong, offsets},
			rawEntry{tStripByteCounts, dtLong, counts},
		)
	}
	return assembleTIFF(cfg.enc, blocks, entries)
}

// checkCMYKPixels asserts that img is a *image.CMYK with the expected bounds
// and raw C/M/Y/K channel values.
func checkCMYKPixels(t *testing.T, img image.Image, w, h int, pix []byte) {
	t.Helper()
	c, ok := img.(*image.CMYK)
	if !ok {
		t.Fatalf("got %T, want *image.CMYK", img)
	}
	if got, want := c.Bounds(), image.Rect(0, 0, w, h); got != want {
		t.Fatalf("bounds: got %v, want %v", got, want)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			got := c.CMYKAt(x, y)
			want := color.CMYK{
				C: pix[(y*w+x)*4+0],
				M: pix[(y*w+x)*4+1],
				Y: pix[(y*w+x)*4+2],
				K: pix[(y*w+x)*4+3],
			}
			if got != want {
				t.Fatalf("pixel at (%d, %d): got %v, want %v", x, y, got, want)
			}
		}
	}
}

// TestDecodeCMYK decodes a hand-built 2x2 CMYK color block through every
// supported compression path and both byte orders, checking the raw channel
// values, the concrete image type and the bounds.
func TestDecodeCMYK(t *testing.T) {
	// Every channel of every pixel carries a distinct value so that a
	// swapped or dropped channel is caught.
	pix := []byte{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
		130, 140, 150, 160,
	}
	bps := []uint16{8, 8, 8, 8}

	type encDef struct {
		name        string
		compression uint16
		predictor   bool
	}
	encodings := []encDef{
		{"none", cNone, false},
		{"lzw", cLZW, false},
		{"lzw-predictor", cLZW, true},
		{"deflate", cDeflate, false},
		{"packbits", cPackBits, false},
	}
	orders := []struct {
		name string
		enc  byteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	}

	for _, order := range orders {
		for _, e := range encodings {
			t.Run(order.name+"/"+e.name, func(t *testing.T) {
				cfg := cmykBuildConfig{
					enc:           order.enc,
					width:         2,
					height:        2,
					rowsPerStrip:  2,
					compression:   e.compression,
					predictor:     e.predictor,
					planarConfig:  pcChunky,
					bitsPerSample: bps,
				}
				data := buildCMYKTIFF(t, cfg, pix)

				img, err := Decode(bytes.NewReader(data))
				if err != nil {
					t.Fatalf("Decode: %v", err)
				}
				checkCMYKPixels(t, img, 2, 2, pix)

				// Also exercise the buffered io.Reader code path.
				img, err = Decode(ioReader{bytes.NewReader(data)})
				if err != nil {
					t.Fatalf("Decode via buffer: %v", err)
				}
				checkCMYKPixels(t, img, 2, 2, pix)

				cfg2, err := DecodeConfig(bytes.NewReader(data))
				if err != nil {
					t.Fatalf("DecodeConfig: %v", err)
				}
				if cfg2.Width != 2 || cfg2.Height != 2 {
					t.Fatalf("config size: got %dx%d", cfg2.Width, cfg2.Height)
				}
				if cfg2.ColorModel != color.CMYKModel {
					t.Fatalf("config color model: got %v, want color.CMYKModel", cfg2.ColorModel)
				}
			})
		}
	}
}

// TestDecodeCMYKStrips checks a non-square, odd-width image split over several
// strips, including a final strip holding only one row.
func TestDecodeCMYKStrips(t *testing.T) {
	const w, h, rps = 3, 3, 2
	pix := make([]byte, w*h*4)
	for i := range pix {
		pix[i] = byte(i*7 + 3)
	}
	for _, compression := range []uint16{cNone, cLZW, cDeflate, cPackBits} {
		t.Run(compressionName(compression), func(t *testing.T) {
			cfg := cmykBuildConfig{
				enc:           binary.LittleEndian,
				width:         w,
				height:        h,
				rowsPerStrip:  rps,
				compression:   compression,
				bitsPerSample: []uint16{8, 8, 8, 8},
			}
			data := buildCMYKTIFF(t, cfg, pix)
			img, err := Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			checkCMYKPixels(t, img, w, h, pix)
		})
	}
}

// TestDecodeCMYKTruncatedLastStrip builds a multi-strip image and truncates
// the final strip (which holds a single row here), so that the short-block
// handling on the last strip must report missing pixel data.
func TestDecodeCMYKTruncatedLastStrip(t *testing.T) {
	const w, h, rps = 3, 3, 2
	pix := make([]byte, w*h*4)
	cfg := cmykBuildConfig{
		enc:           binary.LittleEndian,
		width:         w,
		height:        h,
		rowsPerStrip:  rps,
		compression:   cNone,
		bitsPerSample: []uint16{8, 8, 8, 8},
	}
	data := buildCMYKTIFF(t, cfg, pix)

	// Locate the StripByteCounts entry (tag 279, two LONG values) and shrink
	// the second count from 12 (one row) to 8.
	tagPos := bytes.Index(data, []byte{0x17, 0x01}) // 279, little-endian
	if tagPos < 0 {
		t.Fatal("StripByteCounts entry not found")
	}
	pointer := binary.LittleEndian.Uint32(data[tagPos+8 : tagPos+12])
	binary.LittleEndian.PutUint32(data[pointer+4:pointer+8], 8)

	_, err := Decode(bytes.NewReader(data))
	if err == nil || !strings.Contains(err.Error(), "not enough pixel data") {
		t.Fatalf("Decode: got %v, want error containing %q", err, "not enough pixel data")
	}
}

// TestDecodeCMYKTile checks a tiled CMYK image whose single tile is padded
// beyond the image bounds, so decoding must not write or read the padding.
func TestDecodeCMYKTile(t *testing.T) {
	const w, h, tile = 15, 15, 16
	pix := make([]byte, w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for k := 0; k < 4; k++ {
				pix[(y*w+x)*4+k] = byte(x*17 + y*3 + k)
			}
		}
	}
	for _, compression := range []uint16{cNone, cDeflate} {
		t.Run(compressionName(compression), func(t *testing.T) {
			cfg := cmykBuildConfig{
				enc:           binary.LittleEndian,
				width:         w,
				height:        h,
				compression:   compression,
				tiled:         true,
				tileW:         tile,
				tileH:         tile,
				bitsPerSample: []uint16{8, 8, 8, 8},
			}
			data := buildCMYKTIFF(t, cfg, pix)
			img, err := Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			checkCMYKPixels(t, img, w, h, pix)
		})
	}
}

// TestDecodeCMYKErrors checks that malformed CMYK inputs fail with explicit
// errors instead of panicking or producing a partial image.
func TestDecodeCMYKErrors(t *testing.T) {
	pix := []byte{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
		130, 140, 150, 160,
	}
	base := func() cmykBuildConfig {
		return cmykBuildConfig{
			enc:           binary.LittleEndian,
			width:         2,
			height:        2,
			rowsPerStrip:  2,
			compression:   cNone,
			planarConfig:  pcChunky,
			bitsPerSample: []uint16{8, 8, 8, 8},
		}
	}

	tests := []struct {
		name     string
		mutate   func(*cmykBuildConfig)
		wantSub  string
		wantKind string // "format", "unsupported" or "any"
	}{{
		name:     "MissingChannel",
		mutate:   func(c *cmykBuildConfig) { c.bitsPerSample = []uint16{8, 8, 8} },
		wantSub:  "wrong number of samples for CMYK",
		wantKind: "format",
	}, {
		name:     "MixedBitsPerSample",
		mutate:   func(c *cmykBuildConfig) { c.bitsPerSample = []uint16{8, 8, 8, 16} },
		wantSub:  "wrong number of samples for 8bit CMYK",
		wantKind: "format",
	}, {
		name:     "SixteenBitsPerSample",
		mutate:   func(c *cmykBuildConfig) { c.bitsPerSample = []uint16{16, 16, 16, 16} },
		wantSub:  "BitsPerSample of 16 for CMYK",
		wantKind: "unsupported",
	}, {
		name:     "FourBitsPerSample",
		mutate:   func(c *cmykBuildConfig) { c.bitsPerSample = []uint16{4, 4, 4, 4} },
		wantSub:  "BitsPerSample of 4",
		wantKind: "unsupported",
	}, {
		name:     "PlanarConfiguration",
		mutate:   func(c *cmykBuildConfig) { c.planarConfig = pcPlanar },
		wantSub:  "planar configuration",
		wantKind: "unsupported",
	}, {
		name:     "InvalidPlanarConfiguration",
		mutate:   func(c *cmykBuildConfig) { c.planarConfig = 3 },
		wantSub:  "planar configuration",
		wantKind: "unsupported",
	}, {
		name:     "TruncatedStripData",
		mutate:   func(c *cmykBuildConfig) { c.countOverride = 7 },
		wantSub:  "not enough pixel data",
		wantKind: "format",
	}, {
		name:     "StripOffsetOutOfRange",
		mutate:   func(c *cmykBuildConfig) { c.offsetOverride = 1_000_000 },
		wantKind: "any",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base()
			tt.mutate(&cfg)
			data := buildCMYKTIFF(t, cfg, pix)
			_, err := Decode(bytes.NewReader(data))
			if err == nil {
				t.Fatal("Decode: got nil error, want non-nil")
			}
			if tt.wantSub != "" && !strings.Contains(err.Error(), tt.wantSub) {
				t.Fatalf("error %q does not contain %q", err.Error(), tt.wantSub)
			}
			switch tt.wantKind {
			case "format":
				var fe FormatError
				if !errors.As(err, &fe) {
					t.Fatalf("got %T, want tiff.FormatError", err)
				}
			case "unsupported":
				var ue UnsupportedError
				if !errors.As(err, &ue) {
					t.Fatalf("got %T, want tiff.UnsupportedError", err)
				}
			}
		})
	}
}

// TestDecodeCMYKTooLarge checks the image-size safety bound for CMYK images.
func TestDecodeCMYKTooLarge(t *testing.T) {
	data := assembleTIFF(binary.BigEndian, nil, []rawEntry{
		{tImageWidth, dtLong, []uint32{0x7f000000}},
		{tImageLength, dtLong, []uint32{0x7f000000}},
		{tBitsPerSample, dtShort, []uint32{8, 8, 8, 8}},
		{tPhotometricInterpretation, dtShort, []uint32{pCMYK}},
		{tStripOffsets, dtLong, []uint32{8}},
		{tStripByteCounts, dtLong, []uint32{0}},
	})
	if _, err := DecodeConfig(bytes.NewReader(data)); err == nil ||
		!strings.Contains(err.Error(), "image too large") {
		t.Fatalf("DecodeConfig: got %v, want error containing %q", err, "image too large")
	}
	if _, err := Decode(bytes.NewReader(data)); err == nil ||
		!strings.Contains(err.Error(), "image too large") {
		t.Fatalf("Decode: got %v, want error containing %q", err, "image too large")
	}
}

func compressionName(c uint16) string {
	switch c {
	case cNone:
		return "none"
	case cLZW:
		return "lzw"
	case cDeflate:
		return "deflate"
	case cPackBits:
		return "packbits"
	}
	return "unknown"
}
