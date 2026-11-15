// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bmp implements a BMP image decoder and encoder.
//
// The BMP specification is at http://www.digicamsoft.com/bmp/bmp.html.
package bmp // import "golang.org/x/image/bmp"

import (
	"errors"
	"image"
	"image/color"
	"io"

	"golang.org/x/image/internal/safemath"
)

// ErrUnsupported means that the input BMP image uses a valid but unsupported
// feature.
var ErrUnsupported = errors.New("bmp: unsupported BMP image")

var (
	errInvalidPaletteIndex = errors.New("bmp: invalid palette index")
	errInvalidRLEData      = errors.New("bmp: invalid RLE data")
)

// BMP compression methods, as stored in the DIB header's compression field.
const (
	biRGB       = 0
	biRLE8      = 1
	biRLE4      = 2
	biBitFields = 3
)

func readUint16(b []byte) uint16 {
	return uint16(b[0]) | uint16(b[1])<<8
}

func readUint32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// decodePaletted reads a 1, 2, 4 or 8 bit-per-pixel BMP image from r.
// If topDown is false, the image rows will be read bottom-up.
func decodePaletted(r io.Reader, c image.Config, topDown bool, bpp int) (image.Image, error) {
	palette := c.ColorModel.(color.Palette)
	paletted := image.NewPaletted(image.Rect(0, 0, c.Width, c.Height), palette)
	if c.Width == 0 || c.Height == 0 {
		return paletted, nil
	}
	y0, y1, yDelta := c.Height-1, -1, -1
	if topDown {
		y0, y1, yDelta = 0, c.Height, +1
	}

	pixelsPerByte := 8 / bpp
	// Pad up to ensure each row is 4-bytes aligned.
	bytesPerRow := ((c.Width+pixelsPerByte-1)/pixelsPerByte + 3) &^ 3
	b := make([]byte, bytesPerRow)

	for y := y0; y != y1; y += yDelta {
		p := paletted.Pix[y*paletted.Stride : y*paletted.Stride+c.Width]
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		byteIndex, bitIndex, mask := 0, 8, byte((1<<bpp)-1)
		for pixIndex := 0; pixIndex < c.Width; pixIndex++ {
			bitIndex -= bpp
			paletteIndex := (b[byteIndex]) >> bitIndex & mask
			if int(paletteIndex) >= len(palette) {
				return nil, errInvalidPaletteIndex
			}
			p[pixIndex] = paletteIndex
			if bitIndex == 0 {
				byteIndex++
				bitIndex = 8
			}
		}
	}
	return paletted, nil
}

// decodeRLE reads a BI_RLE8 (bpp == 8) or BI_RLE4 (bpp == 4) BMP image from r.
// If topDown is false, the image rows will be read bottom-up.
//
// RLE data is a stream of packets rather than rows: decoding keeps a cursor
// that end-of-line, end-of-bitmap and DELTA packets move. Pixels that no
// packet writes keep their zero value, i.e. palette index 0.
//
// For BI_RLE8, an encoded run repeats a single palette index. For BI_RLE4, an
// encoded run alternates the run byte's high and low nibbles, starting with
// the high nibble. Absolute-mode data is zero padded to a 16-bit boundary.
//
// An end-of-bitmap packet may legally appear before every row has been
// written, in which case the remaining pixels are palette index 0. Bytes
// following end-of-bitmap are ignored, as they can be padding.
func decodeRLE(r io.Reader, c image.Config, topDown bool, bpp int) (image.Image, error) {
	palette := c.ColorModel.(color.Palette)
	paletted := image.NewPaletted(image.Rect(0, 0, c.Width, c.Height), palette)
	if c.Width == 0 || c.Height == 0 {
		return paletted, nil
	}

	// Reading byte-by-byte with io.ReadFull keeps truncation reporting
	// consistent with the non-RLE decoders (io.ErrUnexpectedEOF).
	var scratch [1]byte
	readByte := func() (byte, error) {
		if _, err := io.ReadFull(r, scratch[:]); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return 0, err
		}
		return scratch[0], nil
	}

	// The cursor is in image coordinates: row 0 is the top row. A bottom-up
	// image therefore starts at its lower-left corner and each end-of-line
	// moves the cursor up one row (y decreases).
	x, y := 0, 0
	yStep := 1
	if !topDown {
		y = c.Height - 1
		yStep = -1
	}

	writePixel := func(paletteIndex byte) error {
		if int(paletteIndex) >= len(palette) {
			return errInvalidPaletteIndex
		}
		paletted.Pix[y*paletted.Stride+x] = paletteIndex
		return nil
	}

	// Every packet starts with two bytes: a nonzero first byte means encoded
	// mode and the second byte is the color data; a zero first byte means the
	// second byte selects an escape or an absolute run.
	for {
		count, err := readByte()
		if err != nil {
			return nil, err
		}
		code, err := readByte()
		if err != nil {
			return nil, err
		}

		// Only end-of-bitmap may be seen once the cursor has been moved past
		// the last row (a trailing end-of-line before end-of-bitmap, for
		// example). Any packet that would draw or move from there is invalid.
		if !(count == 0 && code == 1) &&
			(y < 0 || y >= c.Height || x < 0 || x > c.Width) {
			return nil, errInvalidRLEData
		}

		if count != 0 {
			// Encoded mode: repeat the color data count times, within this row.
			if x+int(count) > c.Width {
				return nil, errInvalidRLEData
			}
			for i := 0; i < int(count); i++ {
				paletteIndex := code
				if bpp == 4 {
					// The first pixel uses the high nibble, the second the
					// low nibble, alternating.
					paletteIndex = code >> 4
					if i&1 != 0 {
						paletteIndex = code & 0x0F
					}
				}
				if err := writePixel(paletteIndex); err != nil {
					return nil, err
				}
				x++
			}
			continue
		}

		switch code {
		case 0: // End of line: move to the start of the next row.
			x = 0
			y += yStep

		case 1: // End of bitmap: stop. Trailing bytes are padding.
			return paletted, nil

		case 2: // DELTA: move the cursor two-dimensionally without drawing.
			dx, err := readByte()
			if err != nil {
				return nil, err
			}
			dy, err := readByte()
			if err != nil {
				return nil, err
			}
			newX := x + int(dx)
			newY := y + int(dy)*yStep
			if newX > c.Width || newY < 0 || newY >= c.Height {
				return nil, errInvalidRLEData
			}
			x, y = newX, newY

		default: // Absolute mode: code literal pixels, then word padding.
			n := int(code)
			if x+n > c.Width {
				return nil, errInvalidRLEData
			}
			if bpp == 8 {
				for i := 0; i < n; i++ {
					paletteIndex, err := readByte()
					if err != nil {
						return nil, err
					}
					if err := writePixel(paletteIndex); err != nil {
						return nil, err
					}
					x++
				}
				// An odd number of bytes is padded to a 16-bit boundary.
				if n&1 != 0 {
					if _, err := readByte(); err != nil {
						return nil, err
					}
				}
			} else {
				// Each byte holds two pixels: the high nibble is written
				// first and the low nibble second.
				for i := 0; i < n; i += 2 {
					b, err := readByte()
					if err != nil {
						return nil, err
					}
					if err := writePixel(b >> 4); err != nil {
						return nil, err
					}
					x++
					if i+1 < n {
						if err := writePixel(b & 0x0F); err != nil {
							return nil, err
						}
						x++
					}
				}
				// A run whose packed byte count is odd is padded to a 16-bit
				// boundary.
				if ((n+1)/2)&1 != 0 {
					if _, err := readByte(); err != nil {
						return nil, err
					}
				}
			}
		}
	}
}

// decodeRGB reads a 24 bit-per-pixel BMP image from r.
// If topDown is false, the image rows will be read bottom-up.
func decodeRGB(r io.Reader, c image.Config, topDown bool) (image.Image, error) {
	rgba := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
	if c.Width == 0 || c.Height == 0 {
		return rgba, nil
	}
	// There are 3 bytes per pixel, and each row is 4-byte aligned.
	b := make([]byte, (3*c.Width+3)&^3)
	y0, y1, yDelta := c.Height-1, -1, -1
	if topDown {
		y0, y1, yDelta = 0, c.Height, +1
	}
	for y := y0; y != y1; y += yDelta {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		p := rgba.Pix[y*rgba.Stride : y*rgba.Stride+c.Width*4]
		for i, j := 0, 0; i < len(p); i, j = i+4, j+3 {
			// BMP images are stored in BGR order rather than RGB order.
			p[i+0] = b[j+2]
			p[i+1] = b[j+1]
			p[i+2] = b[j+0]
			p[i+3] = 0xFF
		}
	}
	return rgba, nil
}

// decodeNRGBA reads a 32 bit-per-pixel BMP image from r.
// If topDown is false, the image rows will be read bottom-up.
func decodeNRGBA(r io.Reader, c image.Config, topDown, allowAlpha bool) (image.Image, error) {
	rgba := image.NewNRGBA(image.Rect(0, 0, c.Width, c.Height))
	if c.Width == 0 || c.Height == 0 {
		return rgba, nil
	}
	y0, y1, yDelta := c.Height-1, -1, -1
	if topDown {
		y0, y1, yDelta = 0, c.Height, +1
	}
	for y := y0; y != y1; y += yDelta {
		p := rgba.Pix[y*rgba.Stride : y*rgba.Stride+c.Width*4]
		if _, err := io.ReadFull(r, p); err != nil {
			return nil, err
		}
		for i := 0; i < len(p); i += 4 {
			// BMP images are stored in BGRA order rather than RGBA order.
			p[i+0], p[i+2] = p[i+2], p[i+0]
			if !allowAlpha {
				p[i+3] = 0xFF
			}
		}
	}
	return rgba, nil
}

// Decode reads a BMP image from r and returns it as an image.Image.
// Limitation: The file must be 1, 2, 4, 8, 24 or 32 bits per pixel and
// uncompressed, or 4 or 8 bits per pixel using BI_RLE4 or BI_RLE8 compression.
func Decode(r io.Reader) (image.Image, error) {
	c, bpp, compression, topDown, allowAlpha, err := decodeConfig(r)
	if err != nil {
		return nil, err
	}
	if compression == biRLE4 || compression == biRLE8 {
		return decodeRLE(r, c, topDown, bpp)
	}
	switch bpp {
	case 1, 2, 4, 8:
		return decodePaletted(r, c, topDown, bpp)
	case 24:
		return decodeRGB(r, c, topDown)
	case 32:
		return decodeNRGBA(r, c, topDown, allowAlpha)
	}
	panic("unreachable")
}

// DecodeConfig returns the color model and dimensions of a BMP image without
// decoding the entire image.
// Limitation: The file must be 1, 2, 4, 8, 24 or 32 bits per pixel and
// uncompressed, or 4 or 8 bits per pixel using BI_RLE4 or BI_RLE8 compression.
func DecodeConfig(r io.Reader) (image.Config, error) {
	config, _, _, _, _, err := decodeConfig(r)
	return config, err
}

func decodeConfig(r io.Reader) (config image.Config, bitsPerPixel int, compression uint32, topDown bool, allowAlpha bool, err error) {
	// We only support those BMP images with one of the following DIB headers:
	// - BITMAPINFOHEADER (40 bytes)
	// - BITMAPV4HEADER (108 bytes)
	// - BITMAPV5HEADER (124 bytes)
	const (
		fileHeaderLen   = 14
		infoHeaderLen   = 40
		v4InfoHeaderLen = 108
		v5InfoHeaderLen = 124
	)
	var b [1024]byte
	if _, err := io.ReadFull(r, b[:fileHeaderLen+4]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, 0, 0, false, false, err
	}
	if string(b[:2]) != "BM" {
		return image.Config{}, 0, 0, false, false, errors.New("bmp: invalid format")
	}
	offset := readUint32(b[10:14])
	infoLen := readUint32(b[14:18])
	if infoLen != infoHeaderLen && infoLen != v4InfoHeaderLen && infoLen != v5InfoHeaderLen {
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	if _, err := io.ReadFull(r, b[fileHeaderLen+4:fileHeaderLen+infoLen]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, 0, 0, false, false, err
	}
	width := int(int32(readUint32(b[18:22])))
	height := int(int32(readUint32(b[22:26])))
	if height < 0 {
		height, topDown = -height, true
	}
	if width < 0 || height < 0 {
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	if (width == 0) != (height == 0) {
		// We'll take 0x0, but Nx0 or 0xN is suspicious.
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	// Check that the image fits in memory.
	// This conservatively assumes 4 bytes per pixel,
	// rather than using the actual pixel size.
	if _, ok := safemath.Mul3(width, height, 4); !ok {
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	// We only support 1 plane. The compression must be either uncompressed or
	// a paletted run-length encoding whose bit depth matches.
	planes, bpp, compression := readUint16(b[26:28]), readUint16(b[28:30]), readUint32(b[30:34])
	// if compression is set to BI_BITFIELDS, but the bitmask is set to the default bitmask
	// that would be used if compression was set to 0, we can continue as if compression was 0
	if compression == biBitFields && infoLen > infoHeaderLen &&
		readUint32(b[54:58]) == 0xff0000 && readUint32(b[58:62]) == 0xff00 &&
		readUint32(b[62:66]) == 0xff && readUint32(b[66:70]) == 0xff000000 {
		compression = biRGB
	}
	if planes != 1 {
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	switch compression {
	case biRGB:
	case biRLE8:
		if bpp != 8 {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}
	case biRLE4:
		if bpp != 4 {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}
	default:
		return image.Config{}, 0, 0, false, false, ErrUnsupported
	}
	switch bpp {
	case 1, 2, 4, 8:
		colorUsed := readUint32(b[46:50])

		if colorUsed == 0 {
			colorUsed = 1 << bpp
		} else if colorUsed > (1 << bpp) {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}

		if offset != fileHeaderLen+infoLen+colorUsed*4 {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}
		_, err = io.ReadFull(r, b[:colorUsed*4])
		if err != nil {
			return image.Config{}, 0, 0, false, false, err
		}
		pcm := make(color.Palette, colorUsed)
		for i := range pcm {
			// BMP images are stored in BGR order rather than RGB order.
			// Every 4th byte is padding.
			pcm[i] = color.RGBA{b[4*i+2], b[4*i+1], b[4*i+0], 0xFF}
		}
		return image.Config{ColorModel: pcm, Width: width, Height: height}, int(bpp), compression, topDown, false, nil
	case 24:
		if offset != fileHeaderLen+infoLen {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}
		return image.Config{ColorModel: color.RGBAModel, Width: width, Height: height}, 24, compression, topDown, false, nil
	case 32:
		if offset != fileHeaderLen+infoLen {
			return image.Config{}, 0, 0, false, false, ErrUnsupported
		}
		// 32 bits per pixel is possibly RGBX (X is padding) or RGBA (A is
		// alpha transparency). However, for BMP images, "Alpha is a
		// poorly-documented and inconsistently-used feature" says
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.cc;l=621
		//
		// That goes on to say "BITMAPV3HEADER+ have an alpha bitmask in the
		// info header... so we respect it at all times... [For earlier
		// (smaller) headers we] ignore alpha in Windows V3 BMPs except inside
		// ICO files".
		//
		// "Ignore" means to always set alpha to 0xFF (fully opaque):
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.h;l=272
		//
		// Confusingly, "Windows V3" does not correspond to BITMAPV3HEADER, but
		// instead corresponds to the earlier (smaller) BITMAPINFOHEADER:
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.cc;l=258
		//
		// This Go package does not support ICO files and the (infoLen >
		// infoHeaderLen) condition distinguishes BITMAPINFOHEADER (40 bytes)
		// vs later (larger) headers.
		allowAlpha = infoLen > infoHeaderLen
		return image.Config{ColorModel: color.RGBAModel, Width: width, Height: height}, 32, compression, topDown, allowAlpha, nil
	}
	return image.Config{}, 0, 0, false, false, ErrUnsupported
}

func init() {
	image.RegisterFormat("bmp", "BM????\x00\x00\x00\x00", Decode, DecodeConfig)
}
