// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"strings"
	"testing"
)

// jpegTestImage builds a deterministic RGB image with a gradient pattern so
// that every pixel and every edge is distinguishable.
func jpegTestImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8((x * 13) % 256)
			img.Pix[i+1] = uint8((y * 29) % 256)
			img.Pix[i+2] = uint8((x + y*7) % 256)
			img.Pix[i+3] = 0xff
		}
	}
	return img
}

// encodeJPEGStrips splits src into horizontal strips of rowsPerStrip rows,
// JPEG-encodes each strip and returns both the encoded strips and the image
// reconstructed by decoding those same strips through the standard library's
// JPEG decoder, which is the exact pixel data the TIFF decoder is expected to
// produce.
func encodeJPEGStrips(t *testing.T, src *image.RGBA, rowsPerStrip int) (strips [][]byte, expected *image.RGBA) {
	t.Helper()
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	expected = image.NewRGBA(src.Bounds())
	for y := 0; y < h; y += rowsPerStrip {
		ymax := min(y+rowsPerStrip, h)
		var buf bytes.Buffer
		if err := jpeg.Encode(&buf, src.SubImage(image.Rect(0, y, w, ymax)), &jpeg.Options{Quality: 100}); err != nil {
			t.Fatalf("jpeg.Encode: %v", err)
		}
		strips = append(strips, buf.Bytes())
		appendExpected(t, expected, y, buf.Bytes())
	}
	return strips, expected
}

// asRGBJPEG injects an Adobe APP14 marker with transform 0 immediately after
// the SOI marker. A baseline JPEG with that marker is interpreted by the
// standard library's JPEG decoder as RGB-instead-of-YCbCr, which returns an
// *image.RGBA, exercising the fast copy path of the TIFF decoder.
func asRGBJPEG(t *testing.T, jfif []byte) []byte {
	t.Helper()
	var marker bytes.Buffer
	marker.Write([]byte{0xff, 0xee}) // APP14
	marker.Write([]byte{0x00, 0x0e}) // length: 12 payload bytes + 2 length bytes
	marker.WriteString("Adobe")
	marker.Write([]byte{0x00, 0x64}) // version 100
	marker.Write([]byte{0x00, 0x00}) // flags0
	marker.Write([]byte{0x00, 0x00}) // flags1
	marker.WriteByte(0)              // transform 0: unknown, in practice RGB

	out := make([]byte, 0, len(jfif)+len(marker.Bytes()))
	out = append(out, jfif[:2]...)
	out = append(out, marker.Bytes()...)
	out = append(out, jfif[2:]...)
	return out
}

// appendExpected decodes one JPEG strip and records its pixels into expected
// at row offset y, using the same At/RGBA conversion as the TIFF decoder.
func appendExpected(t *testing.T, expected *image.RGBA, y int, jfif []byte) {
	t.Helper()
	strip, err := jpeg.Decode(bytes.NewReader(jfif))
	if err != nil {
		t.Fatalf("jpeg.Decode: %v", err)
	}
	sb := strip.Bounds()
	for yy := 0; yy < sb.Dy(); yy++ {
		for xx := 0; xx < sb.Dx(); xx++ {
			i := expected.PixOffset(xx, y+yy)
			r, g, b, a := strip.At(sb.Min.X+xx, sb.Min.Y+yy).RGBA()
			expected.Pix[i+0] = uint8(r >> 8)
			expected.Pix[i+1] = uint8(g >> 8)
			expected.Pix[i+2] = uint8(b >> 8)
			expected.Pix[i+3] = uint8(a >> 8)
		}
	}
}

// buildJPEGTIFF assembles a little-endian JPEG-in-TIFF with the given strips.
// extra entries override or extend the standard IFD.
func buildJPEGTIFF(t *testing.T, photometric uint16, w, h, rowsPerStrip int, strips [][]byte, extra map[uint16]any) []byte {
	t.Helper()
	enc := binary.LittleEndian

	offsets := make([]uint32, len(strips))
	counts := make([]uint32, len(strips))
	// Placeholder 8-byte header; strip data follows immediately.
	data := make([]byte, 8)
	for i, s := range strips {
		offsets[i] = uint32(len(data))
		counts[i] = uint32(len(s))
		data = append(data, s...)
	}

	entries := map[uint16]any{
		tImageWidth:                uint32(w),
		tImageLength:               uint32(h),
		tBitsPerSample:             []uint16{8, 8, 8},
		tCompression:               uint16(cJPEG),
		tPhotometricInterpretation: uint16(photometric),
		tRowsPerStrip:              uint16(rowsPerStrip),
		tStripOffsets:              offsets,
		tStripByteCounts:           counts,
	}
	for tag, v := range extra {
		entries[tag] = v
	}
	data = appendIFD(data, enc, entries)

	// Write a valid little-endian TIFF header ("II", magic 42).
	// appendIFD stored the IFD offset in bytes 4:8.
	copy(data[0:4], leHeader)
	return data
}

const (
	jpegTIFFWidth        = 16
	jpegTIFFHeight       = 7 // two strips: 4 rows and a short final strip of 3 rows.
	jpegTIFFRowsPerStrip = 4
)

// TestDecodeJPEGStrips decodes a fixed two-strip JPEG TIFF and checks every
// pixel, including the image edges and the boundary between the two strips.
func TestDecodeJPEGStrips(t *testing.T) {
	src := jpegTestImage(jpegTIFFWidth, jpegTIFFHeight)
	strips, expected := encodeJPEGStrips(t, src, jpegTIFFRowsPerStrip)
	if len(strips) != 2 {
		t.Fatalf("got %d strips, want 2", len(strips))
	}

	for _, photometric := range []struct {
		name string
		tag  uint16
	}{
		{"RGB", pRGB},
		{"YCbCr", pYCbCr},
	} {
		t.Run(photometric.name, func(t *testing.T) {
			data := buildJPEGTIFF(t, photometric.tag, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)

			// Exercise both the direct io.ReaderAt path and the buffering
			// reader path.
			for _, path := range []struct {
				name string
				r    func([]byte) io.Reader
			}{
				{"ReaderAt", func(b []byte) io.Reader { return bytes.NewReader(b) }},
				{"buffered", func(b []byte) io.Reader { return ioReader{bytes.NewReader(b)} }},
			} {
				t.Run(path.name, func(t *testing.T) {
					img, err := Decode(path.r(data))
					if err != nil {
						t.Fatalf("Decode: %v", err)
					}
					compare(t, expected, img)

					got := img.(*image.RGBA)
					// Explicit edge and strip-boundary checks.
					corners := []image.Point{
						{0, 0}, {jpegTIFFWidth - 1, 0},
						{0, jpegTIFFHeight - 1}, {jpegTIFFWidth - 1, jpegTIFFHeight - 1},
					}
					for _, p := range corners {
						if got.RGBAAt(p.X, p.Y) != expected.RGBAAt(p.X, p.Y) {
							t.Errorf("edge pixel %v: got %v, want %v", p, got.RGBAAt(p.X, p.Y), expected.RGBAAt(p.X, p.Y))
						}
					}
					// The last row of the first strip and the first row of the
					// short final strip must land on their correct rows.
					for _, y := range []int{jpegTIFFRowsPerStrip - 1, jpegTIFFRowsPerStrip} {
						if got.RGBAAt(0, y) != expected.RGBAAt(0, y) {
							t.Errorf("boundary pixel (0, %d): got %v, want %v", y, got.RGBAAt(0, y), expected.RGBAAt(0, y))
						}
					}
					if got.Bounds() != image.Rect(0, 0, jpegTIFFWidth, jpegTIFFHeight) {
						t.Errorf("bounds: got %v", got.Bounds())
					}
				})
			}
		})
	}
}

// TestDecodeJPEGRGBStrips covers the case where each JPEG stream is itself
// encoded as RGB (Adobe APP14 transform 0), so the JPEG decoder hands back an
// *image.RGBA that the TIFF decoder copies row-by-row rather than converting.
func TestDecodeJPEGRGBStrips(t *testing.T) {
	src := jpegTestImage(jpegTIFFWidth, jpegTIFFHeight)
	base, _ := encodeJPEGStrips(t, src, jpegTIFFRowsPerStrip)

	strips := make([][]byte, len(base))
	expected := image.NewRGBA(src.Bounds())
	for i, s := range base {
		strips[i] = asRGBJPEG(t, s)
		// Sanity check: the marker really does select the RGB output type.
		dec, err := jpeg.Decode(bytes.NewReader(strips[i]))
		if err != nil {
			t.Fatalf("jpeg.Decode: %v", err)
		}
		if _, ok := dec.(*image.RGBA); !ok {
			t.Fatalf("strip %d: got %T, want *image.RGBA", i, dec)
		}
		y := i * jpegTIFFRowsPerStrip
		appendExpected(t, expected, y, strips[i])
	}

	data := buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
	img, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	compare(t, expected, img)
}

// TestDecodeJPEGConfig checks that DecodeConfig only reads the configuration
// and never touches the JPEG strips: deliberately empty strip data must not
// stop it from returning the dimensions and color model.
func TestDecodeJPEGConfig(t *testing.T) {
	src := jpegTestImage(jpegTIFFWidth, jpegTIFFHeight)
	strips, _ := encodeJPEGStrips(t, src, jpegTIFFRowsPerStrip)
	// Point both strips at empty data at the header offset.
	data := buildJPEGTIFF(t, pYCbCr, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, map[uint16]any{
		tStripOffsets:    []uint32{8, 8},
		tStripByteCounts: []uint32{0, 0},
	})

	cfg, err := DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("DecodeConfig: %v", err)
	}
	if cfg.Width != jpegTIFFWidth || cfg.Height != jpegTIFFHeight {
		t.Errorf("config size: got %dx%d, want %dx%d", cfg.Width, cfg.Height, jpegTIFFWidth, jpegTIFFHeight)
	}
	// YCbCr JPEG is converted to RGB, so the exposed model is RGBA.
	if cfg.ColorModel != color.RGBAModel {
		t.Errorf("color model: got %v, want RGBAModel", cfg.ColorModel)
	}
}

// TestDecodeJPEGErrors covers the explicit failure modes for JPEG strips.
func TestDecodeJPEGErrors(t *testing.T) {
	src := jpegTestImage(jpegTIFFWidth, jpegTIFFHeight)
	goodStrips, _ := encodeJPEGStrips(t, src, jpegTIFFRowsPerStrip)

	tests := []struct {
		name            string
		configOnly      bool
		buildData       func() []byte
		wantErr         string
		wantUnsupported bool
	}{{
		name: "old-style JPEG compression",
		buildData: func() []byte {
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, map[uint16]any{
				tCompression: uint16(cJPEGOld),
			})
		},
		wantErr:         "old-style JPEG",
		wantUnsupported: true,
	}, {
		name: "planar configuration",
		buildData: func() []byte {
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, map[uint16]any{
				tPlanarConfiguration: uint16(pcPlanar),
			})
		},
		wantErr:         "planar",
		wantUnsupported: true,
	}, {
		name: "16-bit samples",
		buildData: func() []byte {
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, map[uint16]any{
				tBitsPerSample: []uint16{16, 16, 16},
			})
		},
		wantErr:         "BitsPerSample",
		wantUnsupported: true,
	}, {
		name: "gray photometric interpretation",
		buildData: func() []byte {
			return buildJPEGTIFF(t, pBlackIsZero, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, nil)
		},
		// A non-RGB/YCbCr photometric with three components is rejected while
		// validating the photometric interpretation, before any strip is read.
		wantErr:         "extra samples",
		wantUnsupported: true,
	}, {
		name: "empty first strip is not a complete JPEG stream",
		buildData: func() []byte {
			strips := [][]byte{{}, goodStrips[1]}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 0: JPEG stream is missing the SOI marker",
	}, {
		name: "strip holding only SOI/EOI is not a complete JPEG stream",
		buildData: func() []byte {
			strips := [][]byte{{0xff, 0xd8, 0xff, 0xd9}, goodStrips[1]}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 0: corrupt JPEG data",
	}, {
		name: "strip missing SOI marker",
		buildData: func() []byte {
			bad := bytes.Clone(goodStrips[0])
			bad[0], bad[1] = 0x00, 0x00
			strips := [][]byte{bad, goodStrips[1]}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 0: JPEG stream is missing the SOI marker",
	}, {
		name: "second strip missing EOI marker",
		buildData: func() []byte {
			strips := [][]byte{goodStrips[0], goodStrips[1][:len(goodStrips[1])-2]}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 1: JPEG stream is missing the EOI marker",
	}, {
		name: "corrupt second strip",
		buildData: func() []byte {
			bad := append(append([]byte{0xff, 0xd8}, bytes.Repeat([]byte{0x00, 0x7f}, 32)...), 0xff, 0xd9)
			strips := [][]byte{goodStrips[0], bad}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 1: corrupt JPEG data",
	}, {
		name: "second strip size mismatch",
		buildData: func() []byte {
			// A full-height (16x4) JPEG where a 16x3 strip is expected.
			tall := jpegTestImage(jpegTIFFWidth, jpegTIFFRowsPerStrip)
			var buf bytes.Buffer
			if err := jpeg.Encode(&buf, tall, &jpeg.Options{Quality: 100}); err != nil {
				t.Fatal(err)
			}
			strips := [][]byte{goodStrips[0], buf.Bytes()}
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, strips, nil)
		},
		wantErr: "strip 1: JPEG image size",
	}, {
		name: "second strip offset out of bounds",
		buildData: func() []byte {
			data := buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, nil)
			// Rebuild with the second offset pointing past the end of the file.
			return buildJPEGTIFF(t, pRGB, jpegTIFFWidth, jpegTIFFHeight, jpegTIFFRowsPerStrip, goodStrips, map[uint16]any{
				tStripOffsets: []uint32{8, uint32(len(data) + 64)},
			})
		},
		wantErr: "strip 1: cannot read JPEG data",
	}, {
		name:       "image too large",
		configOnly: true,
		buildData: func() []byte {
			return buildJPEGTIFF(t, pRGB, 1, 1, 1, goodStrips[:1], map[uint16]any{
				tImageWidth:  uint32(0x7f000000),
				tImageLength: uint32(0x7f000000),
			})
		},
		wantErr: "image too large",
	}}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := tc.buildData()

			check := func(err error) {
				t.Helper()
				if err == nil {
					t.Fatalf("got nil error, want error containing %q", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("got error %q, want one containing %q", err.Error(), tc.wantErr)
				}
				if tc.wantUnsupported {
					var u UnsupportedError
					if !errors.As(err, &u) {
						t.Fatalf("got %T, want UnsupportedError", err)
					}
				}
			}

			if tc.configOnly {
				_, err := DecodeConfig(bytes.NewReader(data))
				check(err)
			}
			// All failures must surface through both reader implementations.
			_, err := Decode(bytes.NewReader(data))
			check(err)
			_, err = Decode(ioReader{bytes.NewReader(data)})
			check(err)
		})
	}
}
