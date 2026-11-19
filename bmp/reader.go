// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bmp implements a BMP image decoder and encoder.
//
// The BMP specification is at http://www.digicamsoft.com/bmp/bmp.html.
package bmp // import "golang.org/x/image/bmp"

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"

	"golang.org/x/image/internal/safemath"
)

// ErrUnsupported means that the input BMP image uses a valid but unsupported
// feature.
var ErrUnsupported = errors.New("bmp: unsupported BMP image")

var errInvalidPaletteIndex = errors.New("bmp: invalid palette index")

// errPNGSizeMismatch means that the dimensions declared in the BMP headers do
// not match the dimensions of the embedded PNG image.
var errPNGSizeMismatch = errors.New("bmp: PNG dimensions do not match BMP header")

// BMP compression values.
const (
	biRGB       = 0
	biBitfields = 3
	biPNG       = 5
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
// Limitation: The file must be 8, 24 or 32 bits per pixel, or use BI_PNG
// compression (an embedded PNG image).
func Decode(r io.Reader) (image.Image, error) {
	c, bpp, topDown, allowAlpha, pngPayload, err := decodeConfig(r)
	if err != nil {
		return nil, err
	}
	if pngPayload != nil {
		return png.Decode(bytes.NewReader(pngPayload))
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
// Limitation: The file must be 8, 24 or 32 bits per pixel, or use BI_PNG
// compression (an embedded PNG image).
func DecodeConfig(r io.Reader) (image.Config, error) {
	config, _, _, _, _, err := decodeConfig(r)
	return config, err
}

func decodeConfig(r io.Reader) (config image.Config, bitsPerPixel int, topDown bool, allowAlpha bool, pngPayload []byte, err error) {
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
		return image.Config{}, 0, false, false, nil, err
	}
	if string(b[:2]) != "BM" {
		return image.Config{}, 0, false, false, nil, errors.New("bmp: invalid format")
	}
	offset := readUint32(b[10:14])
	infoLen := readUint32(b[14:18])
	if infoLen != infoHeaderLen && infoLen != v4InfoHeaderLen && infoLen != v5InfoHeaderLen {
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	if _, err := io.ReadFull(r, b[fileHeaderLen+4:fileHeaderLen+infoLen]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, 0, false, false, nil, err
	}
	width := int(int32(readUint32(b[18:22])))
	height := int(int32(readUint32(b[22:26])))
	if height < 0 {
		height, topDown = -height, true
	}
	if width < 0 || height < 0 {
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	if (width == 0) != (height == 0) {
		// We'll take 0x0, but Nx0 or 0xN is suspicious.
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	// Check that the image fits in memory.
	// This conservatively assumes 4 bytes per pixel,
	// rather than using the actual pixel size.
	if _, ok := safemath.Mul3(width, height, 4); !ok {
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	// We only support 1 plane and 8, 24 or 32 bits per pixel and no
	// compression, or BI_PNG compression.
	planes, bpp, compression := readUint16(b[26:28]), readUint16(b[28:30]), readUint32(b[30:34])
	// if compression is set to BI_BITFIELDS, but the bitmask is set to the default bitmask
	// that would be used if compression was set to 0, we can continue as if compression was 0
	if compression == biBitfields && infoLen > infoHeaderLen &&
		readUint32(b[54:58]) == 0xff0000 && readUint32(b[58:62]) == 0xff00 &&
		readUint32(b[62:66]) == 0xff && readUint32(b[66:70]) == 0xff000000 {
		compression = biRGB
	}
	if planes != 1 {
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	if compression == biPNG {
		if topDown {
			// Compressed BMP images must be stored bottom-up: a negative
			// height is only meaningful for uncompressed images.
			return image.Config{}, 0, false, false, nil, ErrUnsupported
		}
		c, payload, err := decodePNGConfig(r, b[:fileHeaderLen+infoLen], offset, width, height)
		if err != nil {
			return image.Config{}, 0, false, false, nil, err
		}
		return c, 0, false, false, payload, nil
	}
	if compression != biRGB {
		return image.Config{}, 0, false, false, nil, ErrUnsupported
	}
	switch bpp {
	case 1, 2, 4, 8:
		colorUsed := readUint32(b[46:50])

		if colorUsed == 0 {
			colorUsed = 1 << bpp
		} else if colorUsed > (1 << bpp) {
			return image.Config{}, 0, false, false, nil, ErrUnsupported
		}

		if offset != fileHeaderLen+infoLen+colorUsed*4 {
			return image.Config{}, 0, false, false, nil, ErrUnsupported
		}
		_, err = io.ReadFull(r, b[:colorUsed*4])
		if err != nil {
			return image.Config{}, 0, false, false, nil, err
		}
		pcm := make(color.Palette, colorUsed)
		for i := range pcm {
			// BMP images are stored in BGR order rather than RGB order.
			// Every 4th byte is padding.
			pcm[i] = color.RGBA{b[4*i+2], b[4*i+1], b[4*i+0], 0xFF}
		}
		return image.Config{ColorModel: pcm, Width: width, Height: height}, int(bpp), topDown, false, nil, nil
	case 24:
		if offset != fileHeaderLen+infoLen {
			return image.Config{}, 0, false, false, nil, ErrUnsupported
		}
		return image.Config{ColorModel: color.RGBAModel, Width: width, Height: height}, 24, topDown, false, nil, nil
	case 32:
		if offset != fileHeaderLen+infoLen {
			return image.Config{}, 0, false, false, nil, ErrUnsupported
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
		return image.Config{ColorModel: color.RGBAModel, Width: width, Height: height}, 32, topDown, allowAlpha, nil, nil
	}
	return image.Config{}, 0, false, false, nil, ErrUnsupported
}

// decodePNGConfig reads the embedded PNG payload of a BI_PNG-compressed BMP
// image. The BMP headers (file header and info header) have already been
// consumed from r and are passed as header; r is positioned at the first byte
// after them. It returns the image configuration (taken from the PNG, after
// cross-checking it against the BMP header) and the PNG payload itself, so
// that Decode can decode the pixels without re-reading the stream. Only the
// PNG header is parsed here; the pixel data is not decoded.
func decodePNGConfig(r io.Reader, header []byte, offset uint32, width, height int) (image.Config, []byte, error) {
	fileSize := readUint32(header[2:6])
	imageSize := readUint32(header[34:38])
	headerLen := uint32(len(header))
	if offset < headerLen {
		return image.Config{}, nil, errors.New("bmp: PNG data offset overlaps BMP headers")
	}
	if fileSize != 0 && offset > fileSize {
		return image.Config{}, nil, errors.New("bmp: PNG data offset out of range")
	}
	// Skip any gap between the BMP headers and the PNG payload.
	if _, err := io.CopyN(io.Discard, r, int64(offset-headerLen)); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return image.Config{}, nil, err
	}
	// Bound the payload by the declared image size, or failing that, by the
	// declared file size. Any trailing data beyond the payload is ignored.
	limit := int64(-1)
	if imageSize != 0 {
		limit = int64(imageSize)
	} else if fileSize != 0 {
		limit = int64(fileSize - offset)
	}
	var (
		payload []byte
		err     error
	)
	if limit < 0 {
		payload, err = io.ReadAll(r)
	} else {
		payload, err = io.ReadAll(io.LimitReader(r, limit))
	}
	if err != nil {
		return image.Config{}, nil, err
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(payload))
	if err != nil {
		return image.Config{}, nil, err
	}
	// Check that the image fits in memory, conservatively assuming 4 bytes
	// per pixel.
	if _, ok := safemath.Mul3(cfg.Width, cfg.Height, 4); !ok {
		return image.Config{}, nil, ErrUnsupported
	}
	// A BMP header that declares a non-zero size must agree with the PNG.
	if (width != 0 || height != 0) && (width != cfg.Width || height != cfg.Height) {
		return image.Config{}, nil, errPNGSizeMismatch
	}
	return image.Config{ColorModel: cfg.ColorModel, Width: cfg.Width, Height: cfg.Height}, payload, nil
}

func init() {
	image.RegisterFormat("bmp", "BM????\x00\x00\x00\x00", Decode, DecodeConfig)
}
