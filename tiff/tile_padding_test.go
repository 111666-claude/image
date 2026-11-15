// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// tilePaddingVariant describes one tiled TIFF pixel format exercised by
// TestTiledPadding. encode writes the samples of pixel (x, y) into dst and
// want returns the expected pre-multiplied 16-bit RGBA values.
type tilePaddingVariant struct {
	name        string
	photometric uint16
	bps         []uint16
	extra       uint16 // 0 means no ExtraSamples tag
	palette     []uint16
	encode      func(x, y int, dst []byte)
	want        func(x, y int) (r, g, b, a uint32)
}

func tilePaddingTestVariants() []tilePaddingVariant {
	v16 := func(x, y int) uint16 { return uint16((x*1000 + y*7) & 0xffff) }
	v8 := func(x, y int) byte { return byte((x*17 + y*31) & 0xff) }

	pal256 := make([]uint16, 3*256)
	for i := 0; i < 256; i++ {
		pal256[i] = uint16(i) * 257
		pal256[256+i] = uint16(255-i) * 257
		pal256[512+i] = uint16((i*7)&0xff) * 257
	}
	pal2 := []uint16{
		0x1111, 0x4444, // R
		0x2222, 0x5555, // G
		0x3333, 0x6666, // B
	}

	return []tilePaddingVariant{{
		name:        "gray8-blackiszero",
		photometric: pBlackIsZero,
		bps:         []uint16{8},
		encode: func(x, y int, dst []byte) {
			dst[0] = v8(x, y)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			g := uint32(v8(x, y)) * 257
			return g, g, g, 0xffff
		},
	}, {
		name:        "gray8-whiteiszero",
		photometric: pWhiteIsZero,
		bps:         []uint16{8},
		encode: func(x, y int, dst []byte) {
			dst[0] = v8(x, y)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			g := uint32(0xff-v8(x, y)) * 257
			return g, g, g, 0xffff
		},
	}, {
		name:        "gray16-blackiszero",
		photometric: pBlackIsZero,
		bps:         []uint16{16},
		encode: func(x, y int, dst []byte) {
			binary.LittleEndian.PutUint16(dst, v16(x, y))
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			g := uint32(v16(x, y))
			return g, g, g, 0xffff
		},
	}, {
		name:        "gray16-whiteiszero",
		photometric: pWhiteIsZero,
		bps:         []uint16{16},
		encode: func(x, y int, dst []byte) {
			binary.LittleEndian.PutUint16(dst, v16(x, y))
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			g := uint32(0xffff - v16(x, y))
			return g, g, g, 0xffff
		},
	}, {
		name:        "bilevel-blackiszero",
		photometric: pBlackIsZero,
		bps:         []uint16{1},
		encode: func(x, y int, dst []byte) {
			if (x+y)%2 == 1 {
				dst[0] = 1
			}
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			if (x+y)%2 == 1 {
				return 0xffff, 0xffff, 0xffff, 0xffff
			}
			return 0, 0, 0, 0xffff
		},
	}, {
		name:        "paletted-1bpp",
		photometric: pPaletted,
		bps:         []uint16{1},
		palette:     pal2,
		encode: func(x, y int, dst []byte) {
			if (x+y)%2 == 1 {
				dst[0] = 1
			}
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			i := (x + y) % 2
			return uint32(pal2[i]), uint32(pal2[2+i]), uint32(pal2[4+i]), 0xffff
		},
	}, {
		name:        "paletted-8bpp",
		photometric: pPaletted,
		bps:         []uint16{8},
		palette:     pal256,
		encode: func(x, y int, dst []byte) {
			dst[0] = byte(x*13 + y)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			i := int(byte(x*13 + y))
			return uint32(pal256[i]), uint32(pal256[256+i]), uint32(pal256[512+i]), 0xffff
		},
	}, {
		name:        "rgb8",
		photometric: pRGB,
		bps:         []uint16{8, 8, 8},
		encode: func(x, y int, dst []byte) {
			dst[0] = byte(x * 19)
			dst[1] = byte(y * 23)
			dst[2] = byte((x + y) * 11)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(byte(x*19)) * 257, uint32(byte(y*23)) * 257, uint32(byte((x+y)*11)) * 257, 0xffff
		},
	}, {
		name:        "rgb16",
		photometric: pRGB,
		bps:         []uint16{16, 16, 16},
		encode: func(x, y int, dst []byte) {
			binary.LittleEndian.PutUint16(dst[0:], uint16(x*1000))
			binary.LittleEndian.PutUint16(dst[2:], uint16(y*1000))
			binary.LittleEndian.PutUint16(dst[4:], v16(x, y))
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(uint16(x * 1000)), uint32(uint16(y * 1000)), uint32(v16(x, y)), 0xffff
		},
	}, {
		name:        "rgba8-associated",
		photometric: pRGB,
		bps:         []uint16{8, 8, 8, 8},
		extra:       1,
		encode: func(x, y int, dst []byte) {
			dst[0] = byte(x * 19)
			dst[1] = byte(y * 23)
			dst[2] = byte((x + y) * 11)
			dst[3] = 0xff
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(byte(x*19)) * 257, uint32(byte(y*23)) * 257, uint32(byte((x+y)*11)) * 257, 0xffff
		},
	}, {
		name:        "nrgba8-unassociated",
		photometric: pRGB,
		bps:         []uint16{8, 8, 8, 8},
		extra:       2,
		encode: func(x, y int, dst []byte) {
			dst[0] = byte(x * 19)
			dst[1] = byte(y * 23)
			dst[2] = byte((x + y) * 11)
			dst[3] = 0xff
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(byte(x*19)) * 257, uint32(byte(y*23)) * 257, uint32(byte((x+y)*11)) * 257, 0xffff
		},
	}, {
		name:        "rgba16-associated",
		photometric: pRGB,
		bps:         []uint16{16, 16, 16, 16},
		extra:       1,
		encode: func(x, y int, dst []byte) {
			binary.LittleEndian.PutUint16(dst[0:], uint16(x*1000))
			binary.LittleEndian.PutUint16(dst[2:], uint16(y*1000))
			binary.LittleEndian.PutUint16(dst[4:], v16(x, y))
			binary.LittleEndian.PutUint16(dst[6:], 0xffff)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(uint16(x * 1000)), uint32(uint16(y * 1000)), uint32(v16(x, y)), 0xffff
		},
	}, {
		name:        "nrgba16-unassociated",
		photometric: pRGB,
		bps:         []uint16{16, 16, 16, 16},
		extra:       2,
		encode: func(x, y int, dst []byte) {
			binary.LittleEndian.PutUint16(dst[0:], uint16(x*1000))
			binary.LittleEndian.PutUint16(dst[2:], uint16(y*1000))
			binary.LittleEndian.PutUint16(dst[4:], v16(x, y))
			binary.LittleEndian.PutUint16(dst[6:], 0xffff)
		},
		want: func(x, y int) (uint32, uint32, uint32, uint32) {
			return uint32(uint16(x * 1000)), uint32(uint16(y * 1000)), uint32(v16(x, y)), 0xffff
		},
	},
	}
}

// buildTiledTIFF constructs an uncompressed tiled TIFF of the given image and
// tile dimensions. All tile padding (horizontal and vertical) is filled with
// 0xff, so any padding mistakenly decoded as pixels shows up as white.
func buildTiledTIFF(t *testing.T, v tilePaddingVariant, imgW, imgH, tileW, tileH int) []byte {
	t.Helper()
	spp := len(v.bps)
	bpp := int(v.bps[0])
	enc := binary.LittleEndian

	var tileRowBytes int
	if bpp == 1 {
		tileRowBytes = (tileW + 7) / 8
	} else {
		tileRowBytes = tileW * spp * (bpp / 8)
	}
	tileSize := tileRowBytes * tileH

	blocksAcross := (imgW + tileW - 1) / tileW
	blocksDown := (imgH + tileH - 1) / tileH

	// Header.
	data := []byte{'I', 'I', 0x2a, 0x00, 0, 0, 0, 0}

	var tileOffsets, tileByteCounts []uint32
	px := make([]byte, spp*2)
	for j := 0; j < blocksDown; j++ {
		for i := 0; i < blocksAcross; i++ {
			tile := make([]byte, tileSize)
			for k := range tile {
				tile[k] = 0xff
			}
			x0, y0 := i*tileW, j*tileH
			for ty := 0; ty < tileH && y0+ty < imgH; ty++ {
				for tx := 0; tx < tileW && x0+tx < imgW; tx++ {
					x, y := x0+tx, y0+ty
					for k := range px {
						px[k] = 0
					}
					v.encode(x, y, px)
					if bpp == 1 {
						bit := byte(0)
						if px[0]&1 != 0 {
							bit = 0x80 >> (tx % 8)
						}
						if px[0]&1 != 0 {
							tile[ty*tileRowBytes+tx/8] |= bit
						} else {
							tile[ty*tileRowBytes+tx/8] &^= 0x80 >> (tx % 8)
						}
					} else {
						n := spp * (bpp / 8)
						copy(tile[ty*tileRowBytes+tx*n:], px[:n])
					}
				}
			}
			tileOffsets = append(tileOffsets, uint32(len(data)))
			tileByteCounts = append(tileByteCounts, uint32(len(tile)))
			data = append(data, tile...)
		}
	}

	entries := map[uint16]interface{}{
		tImageWidth:                uint32(imgW),
		tImageLength:               uint32(imgH),
		tCompression:               uint16(cNone),
		tPhotometricInterpretation: uint16(v.photometric),
		tSamplesPerPixel:           uint16(spp),
		tTileWidth:                 uint32(tileW),
		tTileLength:                uint32(tileH),
		tTileOffsets:               tileOffsets,
		tTileByteCounts:            tileByteCounts,
	}
	if len(v.bps) == 1 {
		entries[tBitsPerSample] = v.bps[0]
	} else {
		entries[tBitsPerSample] = v.bps
	}
	if v.extra != 0 {
		entries[tExtraSamples] = uint16(v.extra)
	}
	if v.palette != nil {
		entries[tColorMap] = v.palette
	}
	return appendIFD(data, enc, entries)
}

// TestTiledPadding verifies that the horizontal and vertical padding inside
// TIFF tiles does not shift or overwrite actual pixels, across every pixel
// format the decoder supports. The 13 variants cover grayscale (both
// photometric interpretations), palette-color, 16-bit color and alpha images.
func TestTiledPadding(t *testing.T) {
	// Each case uses a tile larger than the image so every tile row carries
	// horizontal padding. The second geometry uses a 2x2 tile grid so that
	// interior tile edges are exercised as well, and an odd pixel width makes
	// the sub-byte (1 bpp) rows end mid-byte.
	cases := []struct {
		name         string
		imgW, imgH   int
		tileW, tileH int
	}{
		{"single-padded-tile", 13, 13, 16, 16},
		{"multi-tile-grid", 30, 30, 16, 16},
		{"odd-width-subbyte", 11, 9, 16, 16},
		// A 7-pixel row occupies one byte while the tile row is two bytes,
		// so flushing the bit reader alone cannot reach the next row start.
		{"narrow-subbyte", 7, 7, 16, 16},
		// 2x2 tile grid where the right-column tiles hold only 5 pixels per
		// row, exercising interior tile edges with sub-byte row padding.
		{"multi-tile-subbyte", 21, 20, 16, 16},
	}
	for _, v := range tilePaddingTestVariants() {
		t.Run(v.name, func(t *testing.T) {
			for _, c := range cases {
				t.Run(c.name, func(t *testing.T) {
					raw := buildTiledTIFF(t, v, c.imgW, c.imgH, c.tileW, c.tileH)
					img, err := Decode(bytes.NewReader(raw))
					if err != nil {
						t.Fatalf("Decode: %v", err)
					}
					b := img.Bounds()
					if b.Dx() != c.imgW || b.Dy() != c.imgH {
						t.Fatalf("bounds = %s, want %dx%d", b, c.imgW, c.imgH)
					}
					for y := 0; y < c.imgH; y++ {
						for x := 0; x < c.imgW; x++ {
							r, g, bl, a := img.At(x, y).RGBA()
							wr, wg, wb, wa := v.want(x, y)
							if r != wr || g != wg || bl != wb || a != wa {
								t.Fatalf("pixel (%d,%d): got (%d,%d,%d,%d), want (%d,%d,%d,%d)",
									x, y, r, g, bl, a, wr, wg, wb, wa)
							}
						}
					}
				})
			}
		})
	}
}
