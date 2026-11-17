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
	"math/bits"

	"golang.org/x/image/internal/safemath"
)

// ErrUnsupported means that the input BMP image uses a valid but unsupported
// feature.
var ErrUnsupported = errors.New("bmp: unsupported BMP image")

var (
	errInvalidPaletteIndex = errors.New("bmp: invalid palette index")
	errInvalidBitFields    = errors.New("bmp: invalid bit field masks")
)

// BMP compression methods, as stored in the DIB header's compression field.
const (
	biRGB       = 0
	biBitFields = 3
)

// DIB header sizes. The first 40 bytes are common to all of them.
const (
	fileHeaderLen   = 14
	infoHeaderLen   = 40
	v2InfoHeaderLen = 52
	v3InfoHeaderLen = 56
	v4InfoHeaderLen = 108
	v5InfoHeaderLen = 124
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

// channel describes one color channel of a BI_BITFIELDS image. A zero mask
// means that the channel is absent.
type channel struct {
	mask  uint32
	shift uint
	n     uint    // Number of bits in the channel.
	lut   []uint8 // Maps the channel value to 8 bits; nil if n >= 8.
}

func newChannel(mask uint32) channel {
	if mask == 0 {
		return channel{}
	}
	n := uint(bits.OnesCount32(mask))
	ch := channel{
		mask:  mask,
		shift: uint(bits.TrailingZeros32(mask)),
		n:     n,
	}
	if n < 8 {
		// Scale from [0, (1<<n)-1] to [0, 255], rounding to nearest. This is
		// the same expansion as Chromium's BMP decoder, e.g. the 5-bit values
		// map to 0, 8, 16, 25, 33, ... 247, 255.
		max := uint32(1<<n) - 1
		ch.lut = make([]uint8, 1<<n)
		for i := range ch.lut {
			ch.lut[i] = uint8((uint32(i)*0xff + max/2) / max)
		}
	}
	return ch
}

// scale extracts the channel from pixel and scales it to the 8-bit range used
// by image.RGBA and image.NRGBA. An absent channel reads as fully opaque,
// which only happens for alpha: the red, green and blue masks are required.
func (ch channel) scale(pixel uint32) uint8 {
	if ch.mask == 0 {
		return 0xff
	}
	x := (pixel & ch.mask) >> ch.shift
	if ch.lut != nil {
		return ch.lut[x]
	}
	if ch.n > 8 {
		// Use the most significant 8 bits. This maps the maximum channel
		// value to 255, as an exact rescale would.
		x >>= ch.n - 8
	}
	return uint8(x)
}

// bitFields describes the channel layout of a BI_BITFIELDS image. The alpha
// channel has a zero mask if the image is opaque.
type bitFields struct {
	red, green, blue, alpha channel
}

// validate checks that each non-zero mask is a contiguous run of bits that
// fits in a pixel, that the red, green and blue masks are all present, and
// that the masks do not overlap. An absent alpha mask means fully opaque.
func (bf *bitFields) validate(bpp uint16) error {
	pixelMask := uint32(0xffff)
	if bpp == 32 {
		pixelMask = 0xffffffff
	}
	if bf.red.mask == 0 || bf.green.mask == 0 || bf.blue.mask == 0 {
		return errInvalidBitFields
	}
	used := uint32(0)
	for _, m := range [...]uint32{bf.red.mask, bf.green.mask, bf.blue.mask, bf.alpha.mask} {
		if m == 0 {
			continue
		}
		if m&^pixelMask != 0 {
			return errInvalidBitFields
		}
		// A mask is contiguous if, after shifting off its trailing zero bits,
		// it consists solely of 1 bits.
		t := m >> bits.TrailingZeros32(m)
		if t&(t+1) != 0 {
			return errInvalidBitFields
		}
		if used&m != 0 {
			return errInvalidBitFields
		}
		used |= m
	}
	return nil
}

// decodeBitFields reads a 16 or 32 bit-per-pixel BI_BITFIELDS BMP image from r.
// If topDown is false, the image rows will be read bottom-up.
func decodeBitFields(r io.Reader, c image.Config, topDown bool, bpp int, bf *bitFields) (image.Image, error) {
	// Pad up to ensure each row is 4-bytes aligned.
	bytesPerRow := (bpp/8*c.Width + 3) &^ 3
	// BMP images store non-premultiplied alpha, so use NRGBA when there is an
	// alpha channel.
	var img image.Image
	var pix []byte
	var stride int
	if bf.alpha.mask != 0 {
		nrgba := image.NewNRGBA(image.Rect(0, 0, c.Width, c.Height))
		img, pix, stride = nrgba, nrgba.Pix, nrgba.Stride
	} else {
		rgba := image.NewRGBA(image.Rect(0, 0, c.Width, c.Height))
		img, pix, stride = rgba, rgba.Pix, rgba.Stride
	}
	if c.Width == 0 || c.Height == 0 {
		return img, nil
	}
	b := make([]byte, bytesPerRow)
	y0, y1, yDelta := c.Height-1, -1, -1
	if topDown {
		y0, y1, yDelta = 0, c.Height, +1
	}
	for y := y0; y != y1; y += yDelta {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		p := pix[y*stride : y*stride+c.Width*4]
		for x := 0; x < c.Width; x++ {
			var v uint32
			if bpp == 16 {
				v = uint32(b[2*x]) | uint32(b[2*x+1])<<8
			} else {
				v = readUint32(b[4*x:])
			}
			p[4*x+0] = bf.red.scale(v)
			p[4*x+1] = bf.green.scale(v)
			p[4*x+2] = bf.blue.scale(v)
			p[4*x+3] = bf.alpha.scale(v)
		}
	}
	return img, nil
}

// Decode reads a BMP image from r and returns it as an image.Image.
// Limitation: The file must be 8, 16, 24 or 32 bits per pixel.
func Decode(r io.Reader) (image.Image, error) {
	info, err := decodeConfig(r)
	if err != nil {
		return nil, err
	}
	if info.bitFields != nil {
		return decodeBitFields(r, info.config, info.topDown, info.bpp, info.bitFields)
	}
	switch info.bpp {
	case 1, 2, 4, 8:
		return decodePaletted(r, info.config, info.topDown, info.bpp)
	case 24:
		return decodeRGB(r, info.config, info.topDown)
	case 32:
		return decodeNRGBA(r, info.config, info.topDown, info.allowAlpha)
	}
	panic("unreachable")
}

// DecodeConfig returns the color model and dimensions of a BMP image without
// decoding the entire image.
// Limitation: The file must be 8, 16, 24 or 32 bits per pixel.
func DecodeConfig(r io.Reader) (image.Config, error) {
	info, err := decodeConfig(r)
	return info.config, err
}

// dibInfo is the result of parsing the file header, DIB header and, for
// BI_BITFIELDS images, the channel masks.
type dibInfo struct {
	config     image.Config
	bpp        int
	topDown    bool
	allowAlpha bool       // 32 bpp BI_RGB: whether alpha should be respected.
	bitFields  *bitFields // Non-nil for BI_BITFIELDS images.
}

func decodeConfig(r io.Reader) (info dibInfo, err error) {
	// We only support those BMP images with one of the following DIB headers:
	// - BITMAPINFOHEADER (40 bytes)
	// - BITMAPV2INFOHEADER (52 bytes): BITMAPINFOHEADER plus RGB masks
	// - BITMAPV3INFOHEADER (56 bytes): V2 plus an alpha mask
	// - BITMAPV4HEADER (108 bytes)
	// - BITMAPV5HEADER (124 bytes)
	//
	// The V2 and V3 headers are only valid with BI_BITFIELDS compression.
	var b [1024]byte
	if _, err := io.ReadFull(r, b[:fileHeaderLen+4]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return dibInfo{}, err
	}
	if string(b[:2]) != "BM" {
		return dibInfo{}, errors.New("bmp: invalid format")
	}
	offset := readUint32(b[10:14])
	infoLen := readUint32(b[14:18])
	switch infoLen {
	case infoHeaderLen, v2InfoHeaderLen, v3InfoHeaderLen, v4InfoHeaderLen, v5InfoHeaderLen:
	default:
		return dibInfo{}, ErrUnsupported
	}
	if _, err := io.ReadFull(r, b[fileHeaderLen+4:fileHeaderLen+infoLen]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return dibInfo{}, err
	}
	// The V2 and V3 headers are only defined for BI_BITFIELDS images.
	if (infoLen == v2InfoHeaderLen || infoLen == v3InfoHeaderLen) && readUint32(b[30:34]) != biBitFields {
		return dibInfo{}, ErrUnsupported
	}
	width := int(int32(readUint32(b[18:22])))
	height := int(int32(readUint32(b[22:26])))
	topDown := false
	if height < 0 {
		height, topDown = -height, true
	}
	if width < 0 || height < 0 {
		return dibInfo{}, ErrUnsupported
	}
	if (width == 0) != (height == 0) {
		// We'll take 0x0, but Nx0 or 0xN is suspicious.
		return dibInfo{}, ErrUnsupported
	}
	// Check that the image fits in memory.
	// This conservatively assumes 4 bytes per pixel,
	// rather than using the actual pixel size.
	if _, ok := safemath.Mul3(width, height, 4); !ok {
		return dibInfo{}, ErrUnsupported
	}
	// We only support 1 plane.
	planes, bpp, compression := readUint16(b[26:28]), readUint16(b[28:30]), readUint32(b[30:34])
	if planes != 1 {
		return dibInfo{}, ErrUnsupported
	}
	switch compression {
	case biBitFields:
		switch bpp {
		case 16, 32:
		default:
			return dibInfo{}, ErrUnsupported
		}
		bf, err := readBitFields(r, b[:], offset, infoLen, bpp)
		if err != nil {
			return dibInfo{}, err
		}
		model := color.RGBAModel
		if bf.alpha.mask != 0 {
			model = color.NRGBAModel
		}
		return dibInfo{
			config:    image.Config{ColorModel: model, Width: width, Height: height},
			bpp:       int(bpp),
			topDown:   topDown,
			bitFields: bf,
		}, nil
	case biRGB:
	default:
		return dibInfo{}, ErrUnsupported
	}
	switch bpp {
	case 1, 2, 4, 8:
		colorUsed := readUint32(b[46:50])

		if colorUsed == 0 {
			colorUsed = 1 << bpp
		} else if colorUsed > (1 << bpp) {
			return dibInfo{}, ErrUnsupported
		}

		if offset != fileHeaderLen+infoLen+colorUsed*4 {
			return dibInfo{}, ErrUnsupported
		}
		_, err = io.ReadFull(r, b[:colorUsed*4])
		if err != nil {
			return dibInfo{}, err
		}
		pcm := make(color.Palette, colorUsed)
		for i := range pcm {
			// BMP images are stored in BGR order rather than RGB order.
			// Every 4th byte is padding.
			pcm[i] = color.RGBA{b[4*i+2], b[4*i+1], b[4*i+0], 0xFF}
		}
		return dibInfo{
			config:  image.Config{ColorModel: pcm, Width: width, Height: height},
			bpp:     int(bpp),
			topDown: topDown,
		}, nil
	case 24:
		if offset != fileHeaderLen+infoLen {
			return dibInfo{}, ErrUnsupported
		}
		return dibInfo{
			config:  image.Config{ColorModel: color.RGBAModel, Width: width, Height: height},
			bpp:     24,
			topDown: topDown,
		}, nil
	case 32:
		if offset != fileHeaderLen+infoLen {
			return dibInfo{}, ErrUnsupported
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
		return dibInfo{
			config:     image.Config{ColorModel: color.RGBAModel, Width: width, Height: height},
			bpp:        32,
			topDown:    topDown,
			allowAlpha: infoLen > infoHeaderLen,
		}, nil
	}
	return dibInfo{}, ErrUnsupported
}

// readBitFields reads the channel masks of a BI_BITFIELDS image. b holds the
// file header and the whole DIB header. offset is the file header's pixel data
// offset.
//
// For headers larger than a BITMAPINFOHEADER, the masks are part of the header:
// BITMAPV2INFOHEADER has red, green and blue masks and later headers add an
// alpha mask. For a BITMAPINFOHEADER, the masks follow the header, so they are
// read from r, which is then positioned at the pixel data. The number of masks
// in that case is inferred from the pixel data offset: three masks, or four
// when an alpha mask is present.
func readBitFields(r io.Reader, b []byte, offset, infoLen uint32, bpp uint16) (*bitFields, error) {
	// The masks always start at file offset 54: 14 bytes of file header plus
	// 40 bytes that are common to every DIB header.
	const maskOffset = fileHeaderLen + infoHeaderLen
	// Masks inside a larger DIB header are only present for BI_BITFIELDS, so
	// b already holds them. For a BITMAPINFOHEADER they follow the header, so
	// they are read here and the pixel data offset tells us how many.
	n := 0 // Number of mask bytes.
	switch infoLen {
	case infoHeaderLen:
		if offset != maskOffset+12 && offset != maskOffset+16 {
			return nil, ErrUnsupported
		}
		n = int(offset - maskOffset)
	case v2InfoHeaderLen:
		if offset != fileHeaderLen+infoLen {
			return nil, ErrUnsupported
		}
		n = 12 // Red, green and blue; there is no alpha mask.
	default: // V3, V4 and V5 headers.
		if offset != fileHeaderLen+infoLen {
			return nil, ErrUnsupported
		}
		n = 16 // Red, green, blue and alpha.
	}
	if len(b) < maskOffset+n {
		return nil, errInvalidBitFields
	}
	if infoLen == infoHeaderLen {
		if _, err := io.ReadFull(r, b[maskOffset:maskOffset+n]); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return nil, err
		}
	}
	red := readUint32(b[maskOffset+0:])
	green := readUint32(b[maskOffset+4:])
	blue := readUint32(b[maskOffset+8:])
	var alpha uint32
	if n == 16 {
		alpha = readUint32(b[maskOffset+12:])
	}
	bf := &bitFields{
		red:   newChannel(red),
		green: newChannel(green),
		blue:  newChannel(blue),
		alpha: newChannel(alpha),
	}
	if err := bf.validate(bpp); err != nil {
		return nil, err
	}
	return bf, nil
}

func init() {
	image.RegisterFormat("bmp", "BM????\x00\x00\x00\x00", Decode, DecodeConfig)
}
