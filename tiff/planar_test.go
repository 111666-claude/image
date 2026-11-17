// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/lzw"
	"encoding/binary"
	"image"
	"image/color"
	"io"
	"testing"
)

// newPlanarTIFF returns a valid little- or big-endian TIFF header.
// The newTIFF helper in reader_test.go only writes the magic number in
// big-endian order, so it cannot be used for little-endian fixtures.
func newPlanarTIFF(enc byteOrder) []byte {
	b := make([]byte, 8)
	switch enc.Uint16([]byte{1, 0}) {
	case 0x1:
		b[0], b[1] = 'I', 'I'
		enc.PutUint16(b[2:4], 42)
	case 0x100:
		b[0], b[1] = 'M', 'M'
		enc.PutUint16(b[2:4], 42)
	default:
		panic("odd byte order")
	}
	// The IFD offset is filled in by appendIFD.
	return b
}

// planarFixture describes a planar-separate (PlanarConfiguration = 2) TIFF
// image to synthesize in memory.
type planarFixture struct {
	width        int
	height       int
	rowsPerStrip int
	bpp          uint // 8 or 16
	order        byteOrder
	lzw          bool

	// planarConfig selects the value of the PlanarConfiguration tag.
	// Defaults to pcPlanar when zero.
	planarConfig uint16
	// predictor is the value of the Predictor tag; zero omits the tag.
	predictor uint16
	// photometric defaults to pRGB when zero.
	photometric uint16
	// extraSamples adds a fourth component plane when true.
	extraSamples bool

	// offsets and counts override the generated StripOffsets and
	// StripByteCounts arrays when non-nil.
	offsets []uint32
	counts  []uint32
}

// stripeComponent returns the component value of the fixed three-color
// stripe pattern at column x for component c (0=R, 1=G, 2=B).
// The first third of the columns is red, the second green and the last blue.
func stripeComponent(x, width, c int, bpp uint) uint16 {
	stripe := x * 3 / width
	if stripe != c {
		return 0
	}
	if bpp == 16 {
		switch c {
		case 0:
			return 0x1234
		case 1:
			return 0x5678
		}
		return 0x9abc
	}
	switch c {
	case 0:
		return 0xff
	case 1:
		return 0x80
	}
	return 0x01
}

// buildPlanarTIFF synthesizes a planar-separate TIFF described by f and
// returns its bytes together with the per-strip offsets and byte counts.
func buildPlanarTIFF(t *testing.T, f planarFixture) (data []byte, offsets, counts []uint32) {
	t.Helper()

	if f.planarConfig == 0 {
		f.planarConfig = pcPlanar
	}
	if f.photometric == 0 {
		f.photometric = pRGB
	}
	if f.rowsPerStrip == 0 {
		f.rowsPerStrip = f.height
	}

	numSamples := 3
	if f.extraSamples {
		numSamples = 4
	}
	stripsDown := (f.height + f.rowsPerStrip - 1) / f.rowsPerStrip

	data = newPlanarTIFF(f.order)

	// Append one data chunk per plane and strip, in plane-major order, as
	// required by the TIFF specification for planar-separate images.
	for plane := 0; plane < numSamples; plane++ {
		for j := 0; j < stripsDown; j++ {
			ymin := j * f.rowsPerStrip
			ymax := ymin + f.rowsPerStrip
			if ymax > f.height {
				ymax = f.height
			}
			var raw []byte
			for y := ymin; y < ymax; y++ {
				for x := 0; x < f.width; x++ {
					v := stripeComponent(x, f.width, plane, f.bpp)
					if f.bpp == 16 {
						raw = f.order.AppendUint16(raw, v)
					} else {
						raw = append(raw, byte(v))
					}
				}
			}

			chunk := raw
			if f.lzw {
				var buf bytes.Buffer
				w := lzw.NewWriter(&buf, lzw.MSB, 8)
				if _, err := w.Write(raw); err != nil {
					t.Fatalf("lzw write: %v", err)
				}
				if err := w.Close(); err != nil {
					t.Fatalf("lzw close: %v", err)
				}
				chunk = buf.Bytes()
			}

			offsets = append(offsets, uint32(len(data)))
			counts = append(counts, uint32(len(chunk)))
			data = append(data, chunk...)
		}
	}
	if f.offsets != nil {
		offsets = f.offsets
	}
	if f.counts != nil {
		counts = f.counts
	}

	bps := make([]uint16, numSamples)
	for i := range bps {
		bps[i] = uint16(f.bpp)
	}

	compression := uint16(cNone)
	if f.lzw {
		compression = cLZW
	}

	entries := map[uint16]any{
		tImageWidth:                uint32(f.width),
		tImageLength:               uint32(f.height),
		tBitsPerSample:             bps,
		tCompression:               compression,
		tPhotometricInterpretation: f.photometric,
		tStripOffsets:              offsets,
		tSamplesPerPixel:           uint16(numSamples),
		tRowsPerStrip:              uint32(f.rowsPerStrip),
		tStripByteCounts:           counts,
		tPlanarConfiguration:       f.planarConfig,
	}
	if f.predictor != 0 {
		entries[tPredictor] = f.predictor
	}
	if f.extraSamples {
		entries[tExtraSamples] = uint16(2)
	}
	data = appendIFD(data, f.order, entries)
	return data, offsets, counts
}

// wantStripeImage builds the interleaved image equivalent to the stripe
// pattern produced by buildPlanarTIFF.
func wantStripeImage(f planarFixture) image.Image {
	rect := image.Rect(0, 0, f.width, f.height)
	if f.bpp == 16 {
		img := image.NewRGBA64(rect)
		for y := 0; y < f.height; y++ {
			for x := 0; x < f.width; x++ {
				img.SetRGBA64(x, y, color.RGBA64{
					R: stripeComponent(x, f.width, 0, f.bpp),
					G: stripeComponent(x, f.width, 1, f.bpp),
					B: stripeComponent(x, f.width, 2, f.bpp),
					A: 0xffff,
				})
			}
		}
		return img
	}
	img := image.NewRGBA(rect)
	for y := 0; y < f.height; y++ {
		for x := 0; x < f.width; x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(stripeComponent(x, f.width, 0, f.bpp)),
				G: uint8(stripeComponent(x, f.width, 1, f.bpp)),
				B: uint8(stripeComponent(x, f.width, 2, f.bpp)),
				A: 0xff,
			})
		}
	}
	return img
}

func TestDecodePlanarStrips(t *testing.T) {
	const (
		width        = 9
		height       = 7
		rowsPerStrip = 3 // 3+3+1 rows: the final strip per plane is short.
	)

	tests := []struct {
		name string
		f    planarFixture
	}{
		{"8bit-uncompressed-le", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 8, order: binary.LittleEndian,
		}},
		{"8bit-uncompressed-be", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 8, order: binary.BigEndian,
		}},
		{"16bit-uncompressed-le", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 16, order: binary.LittleEndian,
		}},
		{"16bit-uncompressed-be", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 16, order: binary.BigEndian,
		}},
		{"8bit-lzw-le", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 8, order: binary.LittleEndian, lzw: true,
		}},
		{"16bit-lzw-be", planarFixture{
			width: width, height: height, rowsPerStrip: rowsPerStrip, bpp: 16, order: binary.BigEndian, lzw: true,
		}},
		{"8bit-single-strip", planarFixture{
			width: width, height: height, bpp: 8, order: binary.LittleEndian,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, _, _ := buildPlanarTIFF(t, tt.f)

			// Exercise both the io.ReaderAt fast path and the buffering
			// path used when only an io.Reader is available.
			for _, buffered := range []bool{false, true} {
				var r io.Reader = bytes.NewReader(data)
				if buffered {
					r = ioReader{r}
				}
				cfg, err := DecodeConfig(r)
				if err != nil {
					t.Fatalf("DecodeConfig: %v", err)
				}
				if cfg.Width != width || cfg.Height != height {
					t.Fatalf("DecodeConfig size: got %dx%d, want %dx%d", cfg.Width, cfg.Height, width, height)
				}

				if buffered {
					r = ioReader{bytes.NewReader(data)}
				} else {
					r = bytes.NewReader(data)
				}
				img, err := Decode(r)
				if err != nil {
					t.Fatalf("Decode: %v", err)
				}

				if tt.f.bpp == 16 {
					if _, ok := img.(*image.RGBA64); !ok {
						t.Fatalf("got %T, want *image.RGBA64", img)
					}
				} else {
					if _, ok := img.(*image.RGBA); !ok {
						t.Fatalf("got %T, want *image.RGBA", img)
					}
				}
				compare(t, wantStripeImage(tt.f), img)
			}
		})
	}
}

// TestDecodePlanarStrips16bitPrecision checks every 16-bit sample byte,
// including a file whose byte order differs from the native order.
func TestDecodePlanarStrips16bitPrecision(t *testing.T) {
	f := planarFixture{
		width: 9, height: 7, rowsPerStrip: 3, bpp: 16,
		order: binary.LittleEndian,
	}
	data, _, _ := buildPlanarTIFF(t, f)
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	rgba := img.(*image.RGBA64)
	for y := 0; y < f.height; y++ {
		for x := 0; x < f.width; x++ {
			want := color.RGBA64{
				R: stripeComponent(x, f.width, 0, 16),
				G: stripeComponent(x, f.width, 1, 16),
				B: stripeComponent(x, f.width, 2, 16),
				A: 0xffff,
			}
			if got := rgba.RGBA64At(x, y); got != want {
				t.Fatalf("pixel (%d,%d): got %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestDecodePlanarStripsErrors(t *testing.T) {
	base := func() planarFixture {
		return planarFixture{
			width: 9, height: 7, rowsPerStrip: 3, bpp: 8,
			order: binary.BigEndian,
		}
	}

	t.Run("invalid planar configuration", func(t *testing.T) {
		f := base()
		f.planarConfig = 3
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("deflate compression unsupported", func(t *testing.T) {
		f := base()
		_, offsets, counts := buildPlanarTIFF(t, f)
		// Rebuild the file with a deflate compression value while reusing
		// the uncompressed strip payloads.
		data := planarTIFFWithEntries(t, f, map[uint16]any{tCompression: uint16(cDeflate)}, offsets, counts)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("horizontal predictor unsupported", func(t *testing.T) {
		f := base()
		f.predictor = prHorizontal
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("tiles unsupported", func(t *testing.T) {
		f := base()
		_, offsets, counts := buildPlanarTIFF(t, f)
		data := planarTIFFWithTiles(t, f, offsets, counts)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("extra samples unsupported", func(t *testing.T) {
		f := base()
		f.extraSamples = true
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("gray plane unsupported", func(t *testing.T) {
		f := base()
		f.photometric = pBlackIsZero
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("wrong number of planar strips", func(t *testing.T) {
		f := base()
		_, offsets, counts := buildPlanarTIFF(t, f)
		// Nine strips are required (3 planes x 3 strips); provide eight.
		f.offsets = offsets[:len(offsets)-1]
		f.counts = counts[:len(counts)-1]
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("overlapping strip offsets", func(t *testing.T) {
		f := base()
		_, offsets, _ := buildPlanarTIFF(t, f)
		// Point the first strip of the green plane at the red plane data.
		f.offsets = append([]uint32(nil), offsets...)
		f.offsets[3] = f.offsets[0]
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("truncated uncompressed strip", func(t *testing.T) {
		f := base()
		_, _, counts := buildPlanarTIFF(t, f)
		// Claim one fewer byte than the last strip contains.
		f.counts = append([]uint32(nil), counts...)
		f.counts[len(f.counts)-1]--
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("truncated lzw strip", func(t *testing.T) {
		f := base()
		f.lzw = true
		_, _, counts := buildPlanarTIFF(t, f)
		// Cut the compressed last strip short.
		f.counts = append([]uint32(nil), counts...)
		f.counts[len(f.counts)-1] /= 2
		if f.counts[len(f.counts)-1] == 0 {
			f.counts[len(f.counts)-1] = 1
		}
		data, _, _ := buildPlanarTIFF(t, f)
		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})

	t.Run("physical file truncation", func(t *testing.T) {
		f := base()
		data, offsets, _ := buildPlanarTIFF(t, f)
		// Cut the file so the final strip's data is partially missing.
		end := int64(offsets[len(offsets)-1]) + 1
		if int(end) >= len(data) {
			t.Fatalf("test setup: truncation point %d beyond file length %d", end, len(data))
		}
		if _, err := Decode(ioReader{bytes.NewReader(data[:end])}); err == nil {
			t.Fatal("got nil error, want non-nil")
		}
	})
}

// planarTIFFWithEntries rebuilds a planar TIFF, overriding the given IFD
// entries while reusing previously generated strip data ranges.
func planarTIFFWithEntries(t *testing.T, f planarFixture, override map[uint16]any, offsets, counts []uint32) []byte {
	t.Helper()
	numSamples := 3
	bps := make([]uint16, numSamples)
	for i := range bps {
		bps[i] = uint16(f.bpp)
	}
	compression := uint16(cNone)
	if f.lzw {
		compression = cLZW
	}
	entries := map[uint16]any{
		tImageWidth:                uint32(f.width),
		tImageLength:               uint32(f.height),
		tBitsPerSample:             bps,
		tCompression:               compression,
		tPhotometricInterpretation: uint16(pRGB),
		tStripOffsets:              offsets,
		tSamplesPerPixel:           uint16(numSamples),
		tRowsPerStrip:              uint32(f.rowsPerStrip),
		tStripByteCounts:           counts,
		tPlanarConfiguration:       uint16(pcPlanar),
	}
	for k, v := range override {
		entries[k] = v
	}

	// Recover the original pixel data: everything between the 8-byte header
	// and the IFD pointed at by the header.
	orig, _, _ := buildPlanarTIFF(t, f)
	ifdOff := f.order.Uint32(orig[4:8])
	data := append([]byte(nil), orig[:ifdOff]...)
	return appendIFD(data, f.order, entries)
}

// planarTIFFWithTiles synthesizes a planar-separate TIFF that also carries
// tile tags, which the decoder must reject.
func planarTIFFWithTiles(t *testing.T, f planarFixture, offsets, counts []uint32) []byte {
	t.Helper()
	data := newPlanarTIFF(f.order)
	// Reuse the strip payloads as the tile payloads.
	orig, _, _ := buildPlanarTIFF(t, f)
	ifdOff := f.order.Uint32(orig[4:8])
	data = append(data, orig[8:ifdOff]...)

	entries := map[uint16]any{
		tImageWidth:                uint32(f.width),
		tImageLength:               uint32(f.height),
		tBitsPerSample:             []uint16{uint16(f.bpp), uint16(f.bpp), uint16(f.bpp)},
		tCompression:               uint16(cNone),
		tPhotometricInterpretation: uint16(pRGB),
		tTileWidth:                 uint32(8),
		tTileLength:                uint32(8),
		tTileOffsets:               offsets,
		tTileByteCounts:            counts,
		tSamplesPerPixel:           uint16(3),
		tPlanarConfiguration:       uint16(pcPlanar),
	}
	return appendIFD(data, f.order, entries)
}
