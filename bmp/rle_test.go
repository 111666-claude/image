// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bmp

import (
	"bytes"
	"image"
	"image/color"
	"io"
	"math/rand/v2"
	"testing"
)

// rleTestPalette returns a palette of n distinct colors and the equivalent
// color table entries for a constructed BMP file. BMP color tables store
// colors as B, G, R, so the entries are written in that order.
func rleTestPalette(n int) (color.Palette, []colorTableEntry) {
	pal := make(color.Palette, n)
	entries := make([]colorTableEntry, n)
	for i := range pal {
		c := color.RGBA{uint8(i * 17), uint8(i * 29), uint8(255 - i*13), 0xFF}
		pal[i] = c
		entries[i] = colorTableEntry{c.B, c.G, c.R}
	}
	return pal, entries
}

// rleBMP builds a BMP file with the given properties. A negative height
// produces a top-down file. The pixel data follows the color table directly,
// as decodeConfig requires.
func rleBMP(width, height int32, bpp uint16, compression uint32, pal []colorTableEntry, data []byte) []byte {
	return bmpBuilder{
		width:        width,
		height:       height,
		planes:       1,
		bitsPerPixel: bpp,
		compression:  compression,
		imageSize:    uint32(len(data)),
		colorsUsed:   uint32(len(pal)),
		colorTable:   pal,
		data:         data,
	}.Bytes()
}

// wantImage builds the image that a decoded bitmap is expected to produce.
// pix is row-major and top-down.
func wantImage(w, h int, pix []byte, pal color.Palette) image.Image {
	m := image.NewPaletted(image.Rect(0, 0, w, h), pal)
	copy(m.Pix, pix)
	return m
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// TestDecodeRLE tests that decoding constructed BI_RLE8 and BI_RLE4 bitmaps
// produces the expected pixels, for both bottom-up and top-down files.
func TestDecodeRLE(t *testing.T) {
	pal8, entries8 := rleTestPalette(8)
	pal16, entries16 := rleTestPalette(16)

	// RLE8, 4x3, bottom-up. The first stored row is the bottom display row.
	//   4,1                  display row 2: indexes 1, 1, 1, 1
	//   0,0                  end of line
	//   1,3  2,2  1,3        display row 1: indexes 3, 2, 2, 3
	//   0,0                  end of line
	//   0,3  0,1,2  0        display row 0: absolute 0, 1, 2, then padding;
	//                        the fourth pixel is never written and stays 0
	//   0,1                  end of bitmap
	//   0xAA                 trailing byte, ignored
	rle8BottomUp := []byte{
		4, 1,
		0, 0,
		1, 3, 2, 2, 1, 3,
		0, 0,
		0, 3, 0, 1, 2, 0,
		0, 1,
		0xAA,
	}

	// RLE8, 4x3, bottom-up, with a DELTA that skips display row 1 entirely.
	// For a bottom-up file, dy moves the cursor toward the top of the image,
	// i.e. the next stored row.
	rle8Delta := []byte{
		4, 2, // display row 2: indexes 2
		0, 2, 0, 1, // DELTA: move to the start of display row 1
		0, 0, // end of line: move to display row 0
		4, 1, // display row 0: indexes 1
		0, 1,
	}

	// RLE8, 4x3, top-down (negative height): rows are stored in display order.
	rle8TopDown := []byte{
		1, 3, 2, 2, 1, 3,
		0, 0,
		0, 3, 0, 1, 2, 0,
		0, 1,
	}

	// RLE8, 4x3, top-down, with a DELTA that skips the middle row. For a
	// top-down file, dy moves down in display order.
	rle8DeltaTopDown := []byte{
		4, 1,
		0, 0, // end of line: move to display row 1
		0, 2, 0, 1, // DELTA: skip display row 1 and move to row 2
		4, 2,
		0, 1,
	}

	// RLE8, 4x1: a DELTA skips the first two pixels horizontally.
	rle8DeltaX := []byte{
		0, 2, 2, 0, // move to x=2
		2, 3,
		0, 1,
	}

	// RLE8, 4x2: an early end-of-bitmap leaves the top display row unwritten.
	rle8EarlyEOB := []byte{
		4, 5, // bottom display row
		0, 1,
	}

	// RLE4, 4x2, bottom-up. Encoded runs alternate the high and low nibbles.
	rle4 := []byte{
		2, 0x56, 2, 0x78, // display row 1: indexes 5, 6, 7, 8
		0, 0,
		2, 0x12, 2, 0x34, // display row 0: indexes 1, 2, 3, 4
		0, 1,
	}

	// RLE4 absolute mode, 5 pixels: 3 bytes of data plus one pad byte.
	rle4AbsoluteOdd := []byte{
		0, 5, 0x12, 0x34, 0x50, 0xEE,
		0, 1,
	}

	// RLE4 absolute mode, 4 pixels: 2 bytes of data, no pad byte.
	rle4AbsoluteNoPad := []byte{
		0, 4, 0x12, 0x34,
		0, 1,
	}

	// RLE4 absolute mode, 6 pixels: 3 bytes of data plus one pad byte.
	rle4AbsoluteSix := []byte{
		0, 6, 0x12, 0x34, 0x56, 0xEE,
		0, 1,
	}

	// RLE4, 4x3, top-down, with a DELTA that skips the middle row.
	rle4DeltaTopDown := []byte{
		2, 0x12, 2, 0x34,
		0, 0,
		0, 2, 0, 1,
		2, 0x56, 2, 0x78,
		0, 1,
	}

	for _, tc := range []struct {
		name          string
		width, height int32
		bpp           uint16
		compression   uint32
		pal           color.Palette
		entries       []colorTableEntry
		data          []byte
		want          []byte // row-major, top-down palette indexes
	}{{
		name: "rle8 bottom-up", width: 4, height: 3, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8BottomUp,
		want: []byte{
			// row 0
			0, 1, 2, 0,
			// row 1
			3, 2, 2, 3,
			// row 2
			1, 1, 1, 1,
		},
	}, {
		name: "rle8 bottom-up delta", width: 4, height: 3, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8Delta,
		want: []byte{
			1, 1, 1, 1,
			0, 0, 0, 0,
			2, 2, 2, 2,
		},
	}, {
		name: "rle8 top-down", width: 4, height: -3, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8TopDown,
		want: []byte{
			3, 2, 2, 3,
			0, 1, 2, 0,
			0, 0, 0, 0,
		},
	}, {
		name: "rle8 top-down delta", width: 4, height: -3, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8DeltaTopDown,
		want: []byte{
			1, 1, 1, 1,
			0, 0, 0, 0,
			2, 2, 2, 2,
		},
	}, {
		name: "rle8 delta x", width: 4, height: 1, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8DeltaX,
		want: []byte{0, 0, 3, 3},
	}, {
		name: "rle8 early end of bitmap", width: 4, height: 2, bpp: 8, compression: biRLE8,
		pal: pal8, entries: entries8, data: rle8EarlyEOB,
		want: []byte{
			0, 0, 0, 0,
			5, 5, 5, 5,
		},
	}, {
		name: "rle4 bottom-up", width: 4, height: 2, bpp: 4, compression: biRLE4,
		pal: pal16, entries: entries16, data: rle4,
		want: []byte{
			1, 2, 3, 4,
			5, 6, 7, 8,
		},
	}, {
		name: "rle4 absolute odd", width: 5, height: 1, bpp: 4, compression: biRLE4,
		pal: pal16, entries: entries16, data: rle4AbsoluteOdd,
		want: []byte{1, 2, 3, 4, 5},
	}, {
		name: "rle4 absolute no padding", width: 4, height: 1, bpp: 4, compression: biRLE4,
		pal: pal16, entries: entries16, data: rle4AbsoluteNoPad,
		want: []byte{1, 2, 3, 4},
	}, {
		name: "rle4 absolute six", width: 6, height: 1, bpp: 4, compression: biRLE4,
		pal: pal16, entries: entries16, data: rle4AbsoluteSix,
		want: []byte{1, 2, 3, 4, 5, 6},
	}, {
		name: "rle4 top-down delta", width: 4, height: -3, bpp: 4, compression: biRLE4,
		pal: pal16, entries: entries16, data: rle4DeltaTopDown,
		want: []byte{
			1, 2, 3, 4,
			0, 0, 0, 0,
			5, 6, 7, 8,
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rleBMP(tc.width, tc.height, tc.bpp, tc.compression, tc.entries, tc.data)
			img, err := Decode(bytes.NewReader(b))
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			want := wantImage(int(tc.width), abs(int(tc.height)), tc.want, tc.pal)
			if err := compare(want, img); err != nil {
				t.Errorf("%v", err)
			}
			p, ok := img.(*image.Paletted)
			if !ok {
				t.Fatalf("Decode returned %T, want *image.Paletted", img)
			}
			if len(p.Palette) != len(tc.pal) {
				t.Errorf("palette length: got %d, want %d", len(p.Palette), len(tc.pal))
			}

			config, err := DecodeConfig(bytes.NewReader(b))
			if err != nil {
				t.Fatalf("DecodeConfig: %v", err)
			}
			if config.Width != int(tc.width) || config.Height != abs(int(tc.height)) {
				t.Errorf("DecodeConfig: got %dx%d, want %dx%d",
					config.Width, config.Height, tc.width, abs(int(tc.height)))
			}
			if err := compare(want, &image.Paletted{
				Pix:     p.Pix,
				Stride:  p.Stride,
				Rect:    p.Rect,
				Palette: config.ColorModel.(color.Palette),
			}); err != nil {
				t.Errorf("DecodeConfig palette mismatch: %v", err)
			}
		})
	}
}

// TestDecodeRLEMatchesUncompressed compares RLE bitmaps against equivalent
// uncompressed bitmaps encoding the same hard-coded pixel grid.
func TestDecodeRLEMatchesUncompressed(t *testing.T) {
	for _, tc := range []struct {
		name          string
		bpp           uint16
		width, height int32
		compression   uint32
		rleData       []byte
		grid          []byte // row-major, top-down palette indexes
	}{{
		name: "rle8", bpp: 8, width: 4, height: 3, compression: biRLE8,
		rleData: []byte{
			4, 1,
			0, 0,
			1, 3, 2, 2, 1, 3,
			0, 0,
			0, 3, 0, 1, 2, 0,
			0, 1,
		},
		grid: []byte{
			0, 1, 2, 0,
			3, 2, 2, 3,
			1, 1, 1, 1,
		},
	}, {
		name: "rle4", bpp: 4, width: 4, height: 2, compression: biRLE4,
		rleData: []byte{
			2, 0x56, 2, 0x78,
			0, 0,
			2, 0x12, 2, 0x34,
			0, 1,
		},
		grid: []byte{
			1, 2, 3, 4,
			5, 6, 7, 8,
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			_, entries := rleTestPalette(1 << tc.bpp)

			imgRLE, err := Decode(bytes.NewReader(
				rleBMP(tc.width, tc.height, tc.bpp, tc.compression, entries, tc.rleData)))
			if err != nil {
				t.Fatalf("Decode RLE: %v", err)
			}

			// Encode the same grid as an uncompressed, bottom-up, 4-byte
			// aligned bitmap.
			w := int(tc.width)
			h := int(tc.height)
			rowBytes := w
			if tc.bpp == 4 {
				rowBytes = (w + 1) / 2
			}
			rowBytes = (rowBytes + 3) &^ 3
			raw := make([]byte, rowBytes*h)
			for y := 0; y < h; y++ {
				row := tc.grid[y*w : (y+1)*w]
				dst := raw[(h-1-y)*rowBytes:]
				if tc.bpp == 8 {
					copy(dst, row)
					continue
				}
				for x := 0; x < w; x += 2 {
					var lo byte
					if x+1 < w {
						lo = row[x+1]
					}
					dst[x/2] = row[x]<<4 | lo
				}
			}
			imgRaw, err := Decode(bytes.NewReader(
				rleBMP(tc.width, tc.height, tc.bpp, biRGB, entries, raw)))
			if err != nil {
				t.Fatalf("Decode uncompressed: %v", err)
			}

			if err := compare(imgRLE, imgRaw); err != nil {
				t.Errorf("RLE and uncompressed images differ: %v", err)
			}
		})
	}
}

// TestDecodeRLEErrors tests that malformed RLE input is rejected with the
// expected error, without panicking or writing out of bounds.
func TestDecodeRLEErrors(t *testing.T) {
	_, entries4 := rleTestPalette(4)
	_, entries16 := rleTestPalette(16)

	for _, tc := range []struct {
		name          string
		width, height int32
		bpp           uint16
		compression   uint32
		entries       []colorTableEntry
		data          []byte
		wantErr       error
	}{{
		name: "encoded run past row", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{2, 1, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "absolute run past row", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 3, 1, 2, 3, 0, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "delta x past row", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 2, 2, 0, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "delta y past image", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 2, 0, 1, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "delta y past image top-down", width: 1, height: -1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 2, 0, 1, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "end of line past image", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{1, 1, 0, 0, 0, 0, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "encoded run overflow", width: 4, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{4, 1, 4, 1, 0, 1},
		wantErr: errInvalidRLEData,
	}, {
		name: "invalid palette index", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{1, 4, 0, 1},
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "invalid absolute palette index", width: 3, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 3, 0, 1, 4, 0, 0, 1},
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "invalid rle4 palette index", width: 1, height: 1, bpp: 4, compression: biRLE4,
		entries: entries4, data: []byte{1, 0xF0, 0, 1},
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "truncated color byte", width: 2, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{2},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated absolute pixels", width: 3, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 3, 1, 2},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated absolute padding", width: 3, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 3, 1, 2, 3},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated absolute rle4 padding", width: 5, height: 1, bpp: 4, compression: biRLE4,
		entries: entries16, data: []byte{0, 5, 0x12, 0x34, 0x50},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "missing end of bitmap", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{1, 1},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "empty data", width: 1, height: 1, bpp: 8, compression: biRLE8,
		entries: entries4, data: nil,
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated delta dx", width: 4, height: 4, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 2},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated delta dy", width: 4, height: 4, bpp: 8, compression: biRLE8,
		entries: entries4, data: []byte{0, 2, 0},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "truncated rle4 absolute pixels", width: 3, height: 1, bpp: 4, compression: biRLE4,
		entries: entries16, data: []byte{0, 3, 0x12},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		name: "invalid absolute rle4 high nibble", width: 3, height: 1, bpp: 4, compression: biRLE4,
		entries: entries4, data: []byte{0, 3, 0xF1, 0x00, 0x00},
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "invalid absolute rle4 low nibble", width: 3, height: 1, bpp: 4, compression: biRLE4,
		entries: entries4, data: []byte{0, 3, 0x1F, 0x00, 0x00},
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "rle8 with 4 bpp", width: 4, height: 1, bpp: 4, compression: biRLE8,
		entries: entries16, data: []byte{0, 1},
		wantErr: ErrUnsupported,
	}, {
		name: "rle4 with 8 bpp", width: 4, height: 1, bpp: 8, compression: biRLE4,
		entries: entries4, data: []byte{0, 1},
		wantErr: ErrUnsupported,
	}, {
		name: "rle8 with 24 bpp", width: 4, height: 1, bpp: 24, compression: biRLE8,
		entries: nil, data: []byte{0, 1},
		wantErr: ErrUnsupported,
	}, {
		name: "unknown compression", width: 1, height: 1, bpp: 8, compression: 4,
		entries: entries4, data: []byte{0, 1},
		wantErr: ErrUnsupported,
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rleBMP(tc.width, tc.height, tc.bpp, tc.compression, tc.entries, tc.data)
			_, err := Decode(bytes.NewReader(b))
			if err != tc.wantErr {
				t.Errorf("Decode error %v; want %v", err, tc.wantErr)
			}

			// A compression/bit-depth mismatch is detected from the headers, so
			// DecodeConfig must reject it too. Other errors are in the RLE
			// stream, which DecodeConfig does not read.
			_, cfgErr := DecodeConfig(bytes.NewReader(b))
			if tc.wantErr == ErrUnsupported {
				if cfgErr != ErrUnsupported {
					t.Errorf("DecodeConfig error %v; want ErrUnsupported", cfgErr)
				}
			} else if cfgErr != nil {
				t.Errorf("DecodeConfig: %v", cfgErr)
			}
		})
	}
}

// TestDecodeRLEZeroSized checks that a 0x0 RLE image decodes without reading
// any pixel data.
func TestDecodeRLEZeroSized(t *testing.T) {
	_, entries := rleTestPalette(4)
	b := rleBMP(0, 0, 8, biRLE8, entries, nil)
	img, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got := img.Bounds().Size(); got.X != 0 || got.Y != 0 {
		t.Errorf("bounds: got %v, want 0x0", got)
	}
}

// encodeRLE8Random appends RLE8 packets covering row, using a mix of encoded
// and absolute runs. An encoded run can only repeat one palette index, so it
// is limited to the length of the run of equal indexes starting at row[0].
// row must be non-empty.
func encodeRLE8Random(row []byte, rnd *rand.Rand) []byte {
	var b []byte
	for len(row) > 0 {
		// A run of equal indexes can be encoded directly.
		run := 1
		for run < len(row) && run < 255 && row[run] == row[0] {
			run++
		}
		if run == 1 && len(row) >= 3 && rnd.IntN(2) == 0 {
			n := 3 + rnd.IntN(min(len(row)-2, 253))
			b = append(b, 0, byte(n))
			b = append(b, row[:n]...)
			if n&1 != 0 {
				b = append(b, 0) // pad to a 16-bit boundary
			}
			row = row[n:]
			continue
		}
		b = append(b, byte(run), row[0])
		row = row[run:]
	}
	return b
}

// encodeRLE4Random appends RLE4 packets covering row, using a mix of encoded
// and absolute runs. An encoded RLE4 run alternates two palette indexes taken
// from the high and low nibbles of the run byte, so it is limited to the
// length of the alternating pattern at the start of row.
func encodeRLE4Random(row []byte, rnd *rand.Rand) []byte {
	var b []byte
	for len(row) > 0 {
		// The alternating pattern (row[0], row[1]) can be encoded directly.
		run := 1
		for run < len(row) && run < 255 && row[run] == row[run%2] {
			run++
		}
		if run < 3 && len(row) >= 3 && rnd.IntN(2) == 0 {
			n := 3 + rnd.IntN(min(len(row)-2, 253))
			b = append(b, 0, byte(n))
			for i := 0; i < n; i += 2 {
				lo := byte(0)
				if i+1 < n {
					lo = row[i+1]
				}
				b = append(b, row[i]<<4|lo)
			}
			if ((n+1)/2)&1 != 0 {
				b = append(b, 0) // pad to a 16-bit boundary
			}
			row = row[n:]
			continue
		}
		if run == 1 {
			// A single pixel still needs a run byte; pad the low nibble.
			b = append(b, 1, row[0]<<4)
			row = row[1:]
			continue
		}
		var index byte = row[0] << 4
		if run > 1 {
			index |= row[1]
		}
		b = append(b, byte(run), index)
		row = row[run:]
	}
	return b
}

// TestDecodeRLERandom generates random pixel grids, encodes them as RLE, and
// checks that decoding reproduces the grid exactly. It exercises encoded runs,
// absolute runs, both nibble orders of RLE4, and absolute-run padding, for
// bottom-up and top-down files.
func TestDecodeRLERandom(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 2))
	pal, entries := rleTestPalette(16)

	for _, bpp := range []uint16{8, 4} {
		compression := uint32(biRLE8)
		if bpp == 4 {
			compression = biRLE4
		}
		for iter := 0; iter < 200; iter++ {
			w := 1 + rnd.IntN(40)
			h := 1 + rnd.IntN(20)
			topDown := rnd.IntN(2) == 0

			grid := make([]byte, w*h)
			for i := range grid {
				grid[i] = byte(rnd.IntN(16))
			}

			var data []byte
			encodeRow := encodeRLE8Random
			if bpp == 4 {
				encodeRow = encodeRLE4Random
			}
			// emitRow encodes grid row y. canSkipRow says whether a vertical
			// DELTA may be used to skip it: a DELTA past the last row would be
			// out of bounds.
			emitRow := func(y int, canSkipRow bool) {
				row := grid[y*w : (y+1)*w]
				// A row that is entirely palette index 0 can be skipped with a
				// DELTA instead of being drawn.
				allZero := true
				for _, v := range row {
					if v != 0 {
						allZero = false
						break
					}
				}
				if allZero && canSkipRow && rnd.IntN(2) == 0 {
					data = append(data, 0, 2, 0, 1) // DELTA: next row
					return
				}
				// Skip a run of leading zeros horizontally with a DELTA.
				k := 0
				for k < len(row) && row[k] == 0 && rnd.IntN(8) != 0 {
					k++
				}
				if k > 0 {
					data = append(data, 0, 2, byte(k), 0) // DELTA: k pixels right
					row = row[k:]
				}
				if len(row) > 0 {
					data = append(data, encodeRow(row, rnd)...)
				}
				data = append(data, 0, 0) // end of line
			}
			if topDown {
				for y := 0; y < h; y++ {
					emitRow(y, y < h-1)
				}
			} else {
				for y := h - 1; y >= 0; y-- {
					emitRow(y, y > 0)
				}
			}
			data = append(data, 0, 1) // end of bitmap

			height := int32(h)
			if topDown {
				height = -height
			}
			b := rleBMP(int32(w), height, bpp, compression, entries, data)
			img, err := Decode(bytes.NewReader(b))
			if err != nil {
				t.Fatalf("bpp=%d topDown=%v %dx%d: Decode: %v", bpp, topDown, w, h, err)
			}
			if err := compare(wantImage(w, h, grid, pal), img); err != nil {
				t.Fatalf("bpp=%d topDown=%v %dx%d: %v", bpp, topDown, w, h, err)
			}
		}
	}
}

// FuzzDecodeRLE feeds arbitrary bytes to the RLE decoder. It checks that
// malformed input is rejected with an error rather than panicking or writing
// out of bounds.
func FuzzDecodeRLE(f *testing.F) {
	_, entries8 := rleTestPalette(4)
	_, entries16 := rleTestPalette(16)
	f.Add(uint8(8), uint8(1), []byte{4, 1, 0, 0, 0, 3, 0, 1, 2, 0, 0, 1})
	f.Add(uint8(4), uint8(2), []byte{4, 0x12, 0, 0, 0, 3, 0x34, 0x50, 0, 0, 1})
	f.Add(uint8(8), uint8(1), []byte{0, 2, 1, 0, 0, 1})
	f.Add(uint8(8), uint8(1), []byte{})

	f.Fuzz(func(t *testing.T, bpp uint8, compression uint8, data []byte) {
		// Keep the image small so that fuzzing stays fast.
		const w, h = 6, 5
		bpp = 8
		comp := uint32(biRLE8)
		entries := entries8
		if compression&1 != 0 {
			bpp = 4
			comp = biRLE4
			entries = entries16
		}
		height := int32(h)
		if compression&2 != 0 {
			height = -height
		}
		b := rleBMP(w, height, uint16(bpp), comp, entries, data)
		img, err := Decode(bytes.NewReader(b))
		if err != nil {
			return
		}
		// A successfully decoded image must have the requested bounds and a
		// palette index in range for every pixel.
		p := img.(*image.Paletted)
		if p.Bounds().Size() != image.Pt(w, h) {
			t.Fatalf("bounds: got %v, want %dx%d", p.Bounds(), w, h)
		}
		for _, v := range p.Pix {
			if int(v) >= len(p.Palette) {
				t.Fatalf("palette index %d out of range [0, %d)", v, len(p.Palette))
			}
		}
	})
}
