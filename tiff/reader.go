// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tiff implements a TIFF image decoder and encoder.
//
// The TIFF specification is at http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
package tiff // import "golang.org/x/image/tiff"

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"

	"golang.org/x/image/ccitt"
	"golang.org/x/image/tiff/lzw"
)

// A FormatError reports that the input is not a valid TIFF image.
type FormatError string

func (e FormatError) Error() string {
	return "tiff: invalid format: " + string(e)
}

// An UnsupportedError reports that the input uses a valid but
// unimplemented feature.
type UnsupportedError string

func (e UnsupportedError) Error() string {
	return "tiff: unsupported feature: " + string(e)
}

var (
	errNoPixels          = FormatError("not enough pixel data")
	errInvalidColorIndex = FormatError("invalid color index")
	errNoSuchPage        = errors.New("no such page")
)

const maxChunkSize = 10 << 20 // 10M

// safeReadAt is a verbatim copy of internal/saferio.ReadDataAt from the
// standard library, which is used to read data from a reader using a length
// provided by untrusted data, without allocating the entire slice ahead of time
// if it is large (>maxChunkSize). This allows us to avoid allocating giant
// slices before learning that we can't actually read that much data from the
// reader.
func safeReadAt(r io.ReaderAt, n uint64, off int64) ([]byte, error) {
	if int64(n) < 0 || n != uint64(int(n)) {
		// n is too large to fit in int, so we can't allocate
		// a buffer large enough. Treat this as a read failure.
		return nil, io.ErrUnexpectedEOF
	}

	if n < maxChunkSize {
		buf := make([]byte, n)
		_, err := r.ReadAt(buf, off)
		if err != nil {
			// io.SectionReader can return EOF for n == 0,
			// but for our purposes that is a success.
			if err != io.EOF || n > 0 {
				return nil, err
			}
		}
		return buf, nil
	}

	var buf []byte
	buf1 := make([]byte, maxChunkSize)
	for n > 0 {
		next := n
		if next > maxChunkSize {
			next = maxChunkSize
		}
		_, err := r.ReadAt(buf1[:next], off)
		if err != nil {
			return nil, err
		}
		buf = append(buf, buf1[:next]...)
		n -= next
		off += int64(next)
	}
	return buf, nil
}

type decoder struct {
	r         io.ReaderAt
	byteOrder binary.ByteOrder
	config    image.Config
	mode      imageMode
	bpp       uint
	features  map[int][]uint
	palette   []color.Color

	// Floating-point sample decoding (SampleFormat 3).
	isFloat     bool
	page        int // Zero-based page index, used for error context.
	samples     int // Samples per pixel.
	floatBytes  int // Bytes per floating-point sample.
	floatOpts   FloatOptions
	outBits     int    // Output bit depth: 8 or 16.
	outMax      uint32 // 0xff or 0xffff.
	grayAlpha   bool   // Gray plus alpha, stored in an NRGBA or RGBA image.
	whiteIsZero bool   // PhotometricInterpretation is pWhiteIsZero.

	buf   []byte
	off   int    // Current offset in buf.
	v     uint32 // Buffer value for reading with arbitrary bit depths.
	nbits uint   // Remaining number of bits in v.
}

// firstVal returns the first uint of the features entry with the given tag,
// or 0 if the tag does not exist.
func (d *decoder) firstVal(tag int) uint {
	f := d.features[tag]
	if len(f) == 0 {
		return 0
	}
	return f[0]
}

// ifdUint decodes the IFD entry in p, which must be of the Byte, Short
// or Long type, and returns the decoded uint values.
func (d *decoder) ifdUint(p []byte) (u []uint, err error) {
	var raw []byte
	if len(p) < ifdLen {
		return nil, FormatError("bad IFD entry")
	}

	datatype := d.byteOrder.Uint16(p[2:4])
	if dt := int(datatype); dt <= 0 || dt >= len(lengths) {
		return nil, UnsupportedError("IFD entry datatype")
	}

	count := d.byteOrder.Uint32(p[4:8])
	if count > math.MaxInt32/lengths[datatype] {
		return nil, FormatError("IFD data too large")
	}
	if datalen := lengths[datatype] * count; datalen > 4 {
		// The IFD contains a pointer to the real value.
		raw, err = safeReadAt(d.r, uint64(datalen), int64(d.byteOrder.Uint32(p[8:12])))
	} else {
		raw = p[8 : 8+datalen]
	}
	if err != nil {
		return nil, err
	}

	u = make([]uint, count)
	switch datatype {
	case dtByte:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(raw[i])
		}
	case dtShort:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(d.byteOrder.Uint16(raw[2*i : 2*(i+1)]))
		}
	case dtLong:
		for i := uint32(0); i < count; i++ {
			u[i] = uint(d.byteOrder.Uint32(raw[4*i : 4*(i+1)]))
		}
	default:
		return nil, UnsupportedError("data type")
	}
	return u, nil
}

// parseIFD decides whether the IFD entry in p is "interesting" and
// stows away the data in the decoder. It returns the tag number of the
// entry and an error, if any.
func (d *decoder) parseIFD(p []byte) (int, error) {
	tag := d.byteOrder.Uint16(p[0:2])
	switch tag {
	case tBitsPerSample,
		tExtraSamples,
		tPhotometricInterpretation,
		tCompression,
		tPredictor,
		tStripOffsets,
		tStripByteCounts,
		tRowsPerStrip,
		tSamplesPerPixel,
		tPlanarConfiguration,
		tTileWidth,
		tTileLength,
		tTileOffsets,
		tTileByteCounts,
		tImageLength,
		tImageWidth,
		tFillOrder,
		tT4Options,
		tT6Options:
		val, err := d.ifdUint(p)
		if err != nil {
			return 0, err
		}
		d.features[int(tag)] = val
	case tColorMap:
		val, err := d.ifdUint(p)
		if err != nil {
			return 0, err
		}
		numcolors := len(val) / 3
		if len(val)%3 != 0 || numcolors <= 0 || numcolors > 256 {
			return 0, FormatError("bad ColorMap length")
		}
		d.palette = make([]color.Color, numcolors)
		for i := 0; i < numcolors; i++ {
			d.palette[i] = color.RGBA64{
				uint16(val[i]),
				uint16(val[i+numcolors]),
				uint16(val[i+2*numcolors]),
				0xffff,
			}
		}
	case tSampleFormat:
		// Page 27 of the spec: If the SampleFormat is present and
		// the value is not 1 [= unsigned integer data], a Baseline
		// TIFF reader that cannot handle the SampleFormat value
		// must terminate the import process gracefully.
		//
		// This implementation additionally supports IEEE floating
		// point data (value 3). Validation that the bit depths and
		// channel counts are consistent with floating point happens
		// in newDecoder, once all of the IFD has been read.
		val, err := d.ifdUint(p)
		if err != nil {
			return 0, err
		}
		for _, v := range val {
			if v != sfUnsignedInt && v != sfFloat {
				return 0, UnsupportedError("sample format")
			}
		}
		d.features[int(tag)] = val
	}
	return int(tag), nil
}

// readBits reads n bits from the internal buffer starting at the current offset.
func (d *decoder) readBits(n uint) (v uint32, ok bool) {
	for d.nbits < n {
		d.v <<= 8
		if d.off >= len(d.buf) {
			return 0, false
		}
		d.v |= uint32(d.buf[d.off])
		d.off++
		d.nbits += 8
	}
	d.nbits -= n
	rv := d.v >> d.nbits
	d.v &^= rv << d.nbits
	return rv, true
}

// flushBits discards the unread bits in the buffer used by readBits.
// It is used at the end of a line.
func (d *decoder) flushBits() {
	d.v = 0
	d.nbits = 0
}

// minInt returns the smaller of x or y.
func minInt(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// decode decodes the raw data of an image.
// It reads from d.buf and writes the strip or tile into dst.
func (d *decoder) decode(dst image.Image, xmin, ymin, xmax, ymax int) error {
	d.off = 0

	// Apply horizontal predictor if necessary.
	// In this case, p contains the color difference to the preceding pixel.
	// See page 64-65 of the spec.
	if d.firstVal(tPredictor) == prHorizontal {
		switch d.bpp {
		case 16:
			var off int
			n := 2 * len(d.features[tBitsPerSample]) // bytes per sample times samples per pixel
			for y := ymin; y < ymax; y++ {
				off += n
				for x := 0; x < (xmax-xmin-1)*n; x += 2 {
					if off+2 > len(d.buf) {
						return errNoPixels
					}
					v0 := d.byteOrder.Uint16(d.buf[off-n : off-n+2])
					v1 := d.byteOrder.Uint16(d.buf[off : off+2])
					d.byteOrder.PutUint16(d.buf[off:off+2], v1+v0)
					off += 2
				}
			}
		case 8:
			var off int
			n := 1 * len(d.features[tBitsPerSample]) // bytes per sample times samples per pixel
			for y := ymin; y < ymax; y++ {
				off += n
				for x := 0; x < (xmax-xmin-1)*n; x++ {
					if off >= len(d.buf) {
						return errNoPixels
					}
					d.buf[off] += d.buf[off-n]
					off++
				}
			}
		case 1:
			return UnsupportedError("horizontal predictor with 1 BitsPerSample")
		}
	}

	rMaxX := minInt(xmax, dst.Bounds().Max.X)
	rMaxY := minInt(ymax, dst.Bounds().Max.Y)
	switch d.mode {
	case mGray, mGrayInvert:
		if d.bpp == 16 {
			img := dst.(*image.Gray16)
			for y := ymin; y < rMaxY; y++ {
				for x := xmin; x < rMaxX; x++ {
					if d.off+2 > len(d.buf) {
						return errNoPixels
					}
					v := d.byteOrder.Uint16(d.buf[d.off : d.off+2])
					d.off += 2
					if d.mode == mGrayInvert {
						v = 0xffff - v
					}
					img.SetGray16(x, y, color.Gray16{v})
				}
				if rMaxX == img.Bounds().Max.X {
					d.off += 2 * (xmax - img.Bounds().Max.X)
				}
			}
		} else {
			img := dst.(*image.Gray)
			max := uint32((1 << d.bpp) - 1)
			for y := ymin; y < rMaxY; y++ {
				for x := xmin; x < rMaxX; x++ {
					v, ok := d.readBits(d.bpp)
					if !ok {
						return errNoPixels
					}
					v = v * 0xff / max
					if d.mode == mGrayInvert {
						v = 0xff - v
					}
					img.SetGray(x, y, color.Gray{uint8(v)})
				}
				d.flushBits()
			}
		}
	case mPaletted:
		img := dst.(*image.Paletted)
		pLen := len(d.palette)
		for y := ymin; y < rMaxY; y++ {
			for x := xmin; x < rMaxX; x++ {
				v, ok := d.readBits(d.bpp)
				if !ok {
					return errNoPixels
				}
				idx := uint8(v)
				if int(idx) >= pLen {
					return errInvalidColorIndex
				}
				img.SetColorIndex(x, y, idx)
			}
			d.flushBits()
		}
	case mRGB:
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			for y := ymin; y < rMaxY; y++ {
				for x := xmin; x < rMaxX; x++ {
					if d.off+6 > len(d.buf) {
						return errNoPixels
					}
					r := d.byteOrder.Uint16(d.buf[d.off+0 : d.off+2])
					g := d.byteOrder.Uint16(d.buf[d.off+2 : d.off+4])
					b := d.byteOrder.Uint16(d.buf[d.off+4 : d.off+6])
					d.off += 6
					img.SetRGBA64(x, y, color.RGBA64{r, g, b, 0xffff})
				}
			}
		} else {
			img := dst.(*image.RGBA)
			for y := ymin; y < rMaxY; y++ {
				min := img.PixOffset(xmin, y)
				max := img.PixOffset(rMaxX, y)
				off := (y - ymin) * (xmax - xmin) * 3
				for i := min; i < max; i += 4 {
					if off+3 > len(d.buf) {
						return errNoPixels
					}
					img.Pix[i+0] = d.buf[off+0]
					img.Pix[i+1] = d.buf[off+1]
					img.Pix[i+2] = d.buf[off+2]
					img.Pix[i+3] = 0xff
					off += 3
				}
			}
		}
	case mNRGBA:
		if d.bpp == 16 {
			img := dst.(*image.NRGBA64)
			for y := ymin; y < rMaxY; y++ {
				for x := xmin; x < rMaxX; x++ {
					if d.off+8 > len(d.buf) {
						return errNoPixels
					}
					r := d.byteOrder.Uint16(d.buf[d.off+0 : d.off+2])
					g := d.byteOrder.Uint16(d.buf[d.off+2 : d.off+4])
					b := d.byteOrder.Uint16(d.buf[d.off+4 : d.off+6])
					a := d.byteOrder.Uint16(d.buf[d.off+6 : d.off+8])
					d.off += 8
					img.SetNRGBA64(x, y, color.NRGBA64{r, g, b, a})
				}
			}
		} else {
			img := dst.(*image.NRGBA)
			for y := ymin; y < rMaxY; y++ {
				min := img.PixOffset(xmin, y)
				max := img.PixOffset(rMaxX, y)
				i0, i1 := (y-ymin)*(xmax-xmin)*4, (y-ymin+1)*(xmax-xmin)*4
				if i1 > len(d.buf) {
					return errNoPixels
				}
				copy(img.Pix[min:max], d.buf[i0:i1])
			}
		}
	case mRGBA:
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			for y := ymin; y < rMaxY; y++ {
				for x := xmin; x < rMaxX; x++ {
					if d.off+8 > len(d.buf) {
						return errNoPixels
					}
					r := d.byteOrder.Uint16(d.buf[d.off+0 : d.off+2])
					g := d.byteOrder.Uint16(d.buf[d.off+2 : d.off+4])
					b := d.byteOrder.Uint16(d.buf[d.off+4 : d.off+6])
					a := d.byteOrder.Uint16(d.buf[d.off+6 : d.off+8])
					d.off += 8
					img.SetRGBA64(x, y, color.RGBA64{r, g, b, a})
				}
			}
		} else {
			img := dst.(*image.RGBA)
			for y := ymin; y < rMaxY; y++ {
				min := img.PixOffset(xmin, y)
				max := img.PixOffset(rMaxX, y)
				i0, i1 := (y-ymin)*(xmax-xmin)*4, (y-ymin+1)*(xmax-xmin)*4
				if i1 > len(d.buf) {
					return errNoPixels
				}
				copy(img.Pix[min:max], d.buf[i0:i1])
			}
		}
	}

	return nil
}

func newDecoder(r io.Reader) (*decoder, error) {
	return newDecoderPage(r, 0, nil)
}

// newDecoderPage decodes the IFD of the page with the given zero-based index.
// opt may be nil.
func newDecoderPage(r io.Reader, page int, opt *DecodeOptions) (*decoder, error) {
	d := &decoder{
		r:        newReaderAt(r),
		features: make(map[int][]uint),
		page:     page,
	}

	p := make([]byte, 8)
	if _, err := d.r.ReadAt(p, 0); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	switch string(p[0:4]) {
	case leHeader:
		d.byteOrder = binary.LittleEndian
	case beHeader:
		d.byteOrder = binary.BigEndian
	default:
		return nil, FormatError("malformed header")
	}

	ifdOffset := int64(d.byteOrder.Uint32(p[4:8]))
	for i := 0; i < page; i++ {
		if ifdOffset == 0 {
			return nil, makePageError(page, -1, errNoSuchPage)
		}
		next, err := d.nextIFDOffset(ifdOffset)
		if err != nil {
			return nil, makePageError(page, -1, err)
		}
		ifdOffset = next
	}
	if ifdOffset == 0 {
		return nil, makePageError(page, -1, errNoSuchPage)
	}

	// The first two bytes contain the number of entries (12 bytes each).
	if _, err := d.r.ReadAt(p[0:2], ifdOffset); err != nil {
		return nil, err
	}
	numItems := int(d.byteOrder.Uint16(p[0:2]))

	// All IFD entries are read in one chunk.
	var err error
	p, err = safeReadAt(d.r, uint64(ifdLen*numItems), ifdOffset+2)
	if err != nil {
		return nil, err
	}

	prevTag := -1
	for i := 0; i < len(p); i += ifdLen {
		tag, err := d.parseIFD(p[i : i+ifdLen])
		if err != nil {
			// Attach page context to failures on floating-point pages, where
			// the per-page contract matters, while leaving integer error
			// values untouched.
			if d.ifdIsFloat(p) {
				return nil, makePageError(page, -1, err)
			}
			return nil, err
		}
		if tag <= prevTag {
			return nil, FormatError("tags are not sorted in ascending order")
		}
		prevTag = tag
	}

	if err := d.setup(opt); err != nil {
		if d.isFloat {
			return nil, d.pageError(-1, err)
		}
		return nil, err
	}
	return d, nil
}

// nextIFDOffset returns the offset of the IFD following the one at off.
func (d *decoder) nextIFDOffset(off int64) (int64, error) {
	var p [2]byte
	if _, err := d.r.ReadAt(p[:], off); err != nil {
		return 0, err
	}
	n := int64(d.byteOrder.Uint16(p[:]))
	var q [4]byte
	if _, err := d.r.ReadAt(q[:], off+2+ifdLen*n); err != nil {
		return 0, err
	}
	return int64(d.byteOrder.Uint32(q[:])), nil
}

// setup validates the IFD's features and determines the image mode.
func (d *decoder) setup(opt *DecodeOptions) error {
	d.config.Width = int(d.firstVal(tImageWidth))
	d.config.Height = int(d.firstVal(tImageLength))

	if _, ok := d.features[tBitsPerSample]; !ok {
		// Default is 1 per specification.
		d.features[tBitsPerSample] = []uint{1}
	}
	d.bpp = d.firstVal(tBitsPerSample)

	// SampleFormat 3 is IEEE floating point. Every sample must be declared
	// floating point: mixed integer and float samples in one page are not
	// supported.
	if sf := d.features[tSampleFormat]; len(sf) != 0 {
		for _, v := range sf {
			if v == sfFloat {
				d.isFloat = true
			}
		}
		if d.isFloat {
			for _, v := range sf {
				if v != sfFloat {
					return UnsupportedError("mixed sample formats")
				}
			}
		}
	}
	if d.isFloat {
		var f FloatOptions
		if opt != nil && opt.Float != nil {
			f = *opt.Float
		}
		fo, err := normalizeFloatOptions(&f)
		if err != nil {
			return err
		}
		d.floatOpts = fo
		d.outBits = fo.BitDepth
		if d.outBits == 16 {
			d.outMax = 0xffff
		} else {
			d.outMax = 0xff
		}
		switch d.bpp {
		case 16, 32, 64:
			d.floatBytes = int(d.bpp) / 8
		default:
			return UnsupportedError(fmt.Sprintf("floating point BitsPerSample of %v", d.bpp))
		}
		return d.setupFloat()
	}

	switch d.bpp {
	case 0:
		return FormatError("BitsPerSample must not be 0")
	case 1, 8, 16:
		// Nothing to do, these are accepted by this implementation.
	default:
		return UnsupportedError(fmt.Sprintf("BitsPerSample of %v", d.bpp))
	}
	return d.setupInteger()
}

// setupFloat determines the image mode for a floating-point page.
func (d *decoder) setupFloat() error {
	if p := d.firstVal(tPredictor); p != 0 && p != prNone {
		// TIFF Technical Note 3 defines a floating-point predictor, but it
		// differs from the integer horizontal predictor. Decoding it as if it
		// were the integer predictor would silently produce wrong pixels.
		return UnsupportedError("predictor with floating point samples")
	}
	for _, b := range d.features[tBitsPerSample] {
		if b != d.bpp {
			return FormatError("inconsistent BitsPerSample values")
		}
	}
	nsamples := len(d.features[tBitsPerSample])
	// The data layout assumed below is chunky: one sample of each channel,
	// interleaved per pixel. PlanarConfiguration 2 (separate planes) would
	// need a different layout, so reject it rather than mis-decode.
	if pc := d.firstVal(tPlanarConfiguration); pc != 0 && pc != 1 {
		return UnsupportedError("planar configuration with floating point samples")
	}
	// If SamplesPerPixel is present it must agree with the sample count
	// implied by BitsPerSample.
	if spp := d.firstVal(tSamplesPerPixel); spp != 0 && int(spp) != nsamples {
		return FormatError("inconsistent SamplesPerPixel and BitsPerSample counts")
	}
	d.samples = nsamples
	d.grayAlpha = false

	colorModel := func(out16 bool) color.Model {
		if out16 {
			return color.RGBA64Model
		}
		return color.RGBAModel
	}
	alphaModel := func(out16 bool) color.Model {
		if out16 {
			return color.NRGBA64Model
		}
		return color.NRGBAModel
	}

	switch d.firstVal(tPhotometricInterpretation) {
	case pRGB:
		switch nsamples {
		case 3:
			d.mode = mRGB
			d.config.ColorModel = colorModel(d.outBits == 16)
		case 4:
			switch d.firstVal(tExtraSamples) {
			case esAssociated:
				d.mode = mRGBA
				d.config.ColorModel = colorModel(d.outBits == 16)
			case esUnassociated:
				d.mode = mNRGBA
				d.config.ColorModel = alphaModel(d.outBits == 16)
			default:
				return FormatError("wrong number of samples for RGB")
			}
		default:
			return FormatError("wrong number of samples for RGB")
		}
	case pBlackIsZero, pWhiteIsZero:
		if d.firstVal(tPhotometricInterpretation) == pWhiteIsZero {
			d.mode = mGrayInvert
			d.whiteIsZero = true
		} else {
			d.mode = mGray
		}
		if d.outBits == 16 {
			d.config.ColorModel = color.Gray16Model
		} else {
			d.config.ColorModel = color.GrayModel
		}
		switch nsamples {
		case 1:
			// Plain grayscale.
		case 2:
			switch d.firstVal(tExtraSamples) {
			case esAssociated:
				d.mode = mRGBA
				d.config.ColorModel = colorModel(d.outBits == 16)
			case esUnassociated:
				d.mode = mNRGBA
				d.config.ColorModel = alphaModel(d.outBits == 16)
			default:
				return FormatError("wrong number of samples for gray")
			}
			d.grayAlpha = true
		default:
			return UnsupportedError("extra samples")
		}
	default:
		return UnsupportedError("color model")
	}
	return nil
}

// setupInteger determines the image mode for an integer page.
func (d *decoder) setupInteger() error {
	// Determine the image mode.
	switch d.firstVal(tPhotometricInterpretation) {
	case pRGB:
		if d.bpp == 16 {
			for _, b := range d.features[tBitsPerSample] {
				if b != 16 {
					return FormatError("wrong number of samples for 16bit RGB")
				}
			}
		} else {
			for _, b := range d.features[tBitsPerSample] {
				if b != 8 {
					return FormatError("wrong number of samples for 8bit RGB")
				}
			}
		}
		// RGB images normally have 3 samples per pixel.
		// If there are more, ExtraSamples (p. 31-32 of the spec)
		// gives their meaning (usually an alpha channel).
		//
		// This implementation does not support extra samples
		// of an unspecified type.
		switch len(d.features[tBitsPerSample]) {
		case 3:
			d.mode = mRGB
			if d.bpp == 16 {
				d.config.ColorModel = color.RGBA64Model
			} else {
				d.config.ColorModel = color.RGBAModel
			}
		case 4:
			switch d.firstVal(tExtraSamples) {
			case 1:
				d.mode = mRGBA
				if d.bpp == 16 {
					d.config.ColorModel = color.RGBA64Model
				} else {
					d.config.ColorModel = color.RGBAModel
				}
			case 2:
				d.mode = mNRGBA
				if d.bpp == 16 {
					d.config.ColorModel = color.NRGBA64Model
				} else {
					d.config.ColorModel = color.NRGBAModel
				}
			default:
				return FormatError("wrong number of samples for RGB")
			}
		default:
			return FormatError("wrong number of samples for RGB")
		}
	case pPaletted:
		d.mode = mPaletted
		d.config.ColorModel = color.Palette(d.palette)
	case pWhiteIsZero:
		d.mode = mGrayInvert
		if d.bpp == 16 {
			d.config.ColorModel = color.Gray16Model
		} else {
			d.config.ColorModel = color.GrayModel
		}
	case pBlackIsZero:
		d.mode = mGray
		if d.bpp == 16 {
			d.config.ColorModel = color.Gray16Model
		} else {
			d.config.ColorModel = color.GrayModel
		}
	default:
		return UnsupportedError("color model")
	}
	if d.firstVal(tPhotometricInterpretation) != pRGB {
		if len(d.features[tBitsPerSample]) != 1 {
			return UnsupportedError("extra samples")
		}
	}

	return nil
}

// DecodeConfig returns the color model and dimensions of a TIFF image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	return DecodeConfigWithOptions(r, nil)
}

// DecodeConfigWithOptions is like DecodeConfig but takes the same options as
// Decode. The options affect the reported color model for floating-point
// pages: BitDepth selects an 8-bit or 16-bit model, and gray plus alpha pages
// report an RGBA-like model.
func DecodeConfigWithOptions(r io.Reader, opt *DecodeOptions) (image.Config, error) {
	d, err := newDecoderPage(r, 0, opt)
	if err != nil {
		return image.Config{}, err
	}
	return d.config, nil
}

func ccittFillOrder(tiffFillOrder uint) ccitt.Order {
	if tiffFillOrder == 2 {
		return ccitt.LSB
	}
	return ccitt.MSB
}

// Decode reads a TIFF image from r and returns it as an image.Image.
// The type of Image returned depends on the contents of the TIFF.
func Decode(r io.Reader) (img image.Image, err error) {
	return DecodeWithOptions(r, nil)
}

// DecodeWithOptions is like Decode but takes options that control how
// IEEE floating-point samples are converted to integer image types.
//
// Integer pages decode identically whether or not options are passed, apart
// from the extra validation of the Float options. For a floating-point page
// the returned image is *image.Gray (or *image.Gray16), *image.RGBA (or
// *image.RGBA64), or *image.NRGBA (or *image.NRGBA64), as selected by
// FloatOptions.BitDepth and the ExtraSamples tag.
func DecodeWithOptions(r io.Reader, opt *DecodeOptions) (img image.Image, err error) {
	return decodePage(r, 0, opt)
}

// DecodePageWithOptions is like DecodeWithOptions but decodes the page with
// the given zero-based index. Errors include a *PageError identifying the
// page (and, where applicable, the sample channel).
func DecodePageWithOptions(r io.Reader, page int, opt *DecodeOptions) (image.Image, error) {
	if page < 0 {
		return nil, makePageError(page, -1, errors.New("negative page index"))
	}
	return decodePage(r, page, opt)
}

// NumPages returns the number of pages (IFDs) in the TIFF image.
func NumPages(r io.Reader) (int, error) {
	d := &decoder{
		r:        newReaderAt(r),
		features: make(map[int][]uint),
	}
	p := make([]byte, 8)
	if _, err := d.r.ReadAt(p, 0); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return 0, err
	}
	switch string(p[0:4]) {
	case leHeader:
		d.byteOrder = binary.LittleEndian
	case beHeader:
		d.byteOrder = binary.BigEndian
	default:
		return 0, FormatError("malformed header")
	}
	off := int64(d.byteOrder.Uint32(p[4:8]))
	n := 0
	for off != 0 {
		next, err := d.nextIFDOffset(off)
		if err != nil {
			return 0, err
		}
		off = next
		n++
		if n > 1<<16 {
			return 0, FormatError("too many pages")
		}
	}
	return n, nil
}

func decodePage(r io.Reader, page int, opt *DecodeOptions) (img image.Image, err error) {
	d, err := newDecoderPage(r, page, opt)
	if err != nil {
		return
	}

	blockPadding := false
	blockWidth := d.config.Width
	blockHeight := d.config.Height
	blocksAcross := 1
	blocksDown := 1

	if d.config.Width == 0 {
		blocksAcross = 0
	}
	if d.config.Height == 0 {
		blocksDown = 0
	}

	var blockOffsets, blockCounts []uint

	if int(d.firstVal(tTileWidth)) != 0 {
		blockPadding = true

		blockWidth = int(d.firstVal(tTileWidth))
		blockHeight = int(d.firstVal(tTileLength))

		// The specification says that tile widths and lengths must be a multiple of 16.
		// We currently permit invalid sizes, but reject anything too small to limit the
		// amount of work a malicious input can force us to perform.
		if blockWidth < 8 || blockHeight < 8 {
			return nil, FormatError("tile size is too small")
		}

		if blockWidth != 0 {
			blocksAcross = (d.config.Width + blockWidth - 1) / blockWidth
		}
		if blockHeight != 0 {
			blocksDown = (d.config.Height + blockHeight - 1) / blockHeight
		}

		blockCounts = d.features[tTileByteCounts]
		blockOffsets = d.features[tTileOffsets]

	} else {
		if int(d.firstVal(tRowsPerStrip)) != 0 {
			blockHeight = int(d.firstVal(tRowsPerStrip))
		}

		if blockHeight != 0 {
			blocksDown = (d.config.Height + blockHeight - 1) / blockHeight
		}

		blockOffsets = d.features[tStripOffsets]
		blockCounts = d.features[tStripByteCounts]
	}

	// Check if we have the right number of strips/tiles, offsets and counts.
	if n := blocksAcross * blocksDown; len(blockOffsets) < n || len(blockCounts) < n {
		return nil, FormatError("inconsistent header")
	}

	imgRect := image.Rect(0, 0, d.config.Width, d.config.Height)
	if d.isFloat {
		switch d.mode {
		case mGray, mGrayInvert:
			if d.outBits == 16 {
				img = image.NewGray16(imgRect)
			} else {
				img = image.NewGray(imgRect)
			}
		case mNRGBA:
			if d.outBits == 16 {
				img = image.NewNRGBA64(imgRect)
			} else {
				img = image.NewNRGBA(imgRect)
			}
		case mRGB, mRGBA:
			if d.outBits == 16 {
				img = image.NewRGBA64(imgRect)
			} else {
				img = image.NewRGBA(imgRect)
			}
		}
	} else {
		switch d.mode {
		case mGray, mGrayInvert:
			if d.bpp == 16 {
				img = image.NewGray16(imgRect)
			} else {
				img = image.NewGray(imgRect)
			}
		case mPaletted:
			img = image.NewPaletted(imgRect, d.palette)
		case mNRGBA:
			if d.bpp == 16 {
				img = image.NewNRGBA64(imgRect)
			} else {
				img = image.NewNRGBA(imgRect)
			}
		case mRGB, mRGBA:
			if d.bpp == 16 {
				img = image.NewRGBA64(imgRect)
			} else {
				img = image.NewRGBA(imgRect)
			}
		}
	}

	if blocksAcross == 0 || blocksDown == 0 {
		return
	}
	// Maximum data per pixel is 8 bytes (RGBA64), or the actual sample size
	// for floating point pages (4 samples at up to 64 bits each). Bounding
	// the decompression limit by the block's true geometry only limits how
	// much a malicious compressed stream can expand to.
	bytesPerPixel := int64(8)
	if d.isFloat {
		var ok bool
		if bytesPerPixel, ok = checkedMul(int64(d.samples), int64(d.floatBytes)); !ok {
			return nil, d.pageError(-1, FormatError("sample count overflow"))
		}
	}
	blockMaxDataSize, ok := checkedMul(int64(blockWidth), int64(blockHeight))
	if ok {
		blockMaxDataSize, ok = checkedMul(blockMaxDataSize, bytesPerPixel)
	}
	if !ok || int64(int(blockMaxDataSize)) != blockMaxDataSize {
		return nil, d.pageError(-1, FormatError("block size overflow"))
	}
	for i := 0; i < blocksAcross; i++ {
		blkW := blockWidth
		if !blockPadding && i == blocksAcross-1 && d.config.Width%blockWidth != 0 {
			blkW = d.config.Width % blockWidth
		}
		for j := 0; j < blocksDown; j++ {
			blkH := blockHeight
			if !blockPadding && j == blocksDown-1 && d.config.Height%blockHeight != 0 {
				blkH = d.config.Height % blockHeight
			}
			offset := int64(blockOffsets[j*blocksAcross+i])
			n := int64(blockCounts[j*blocksAcross+i])
			switch d.firstVal(tCompression) {

			// According to the spec, Compression does not have a default value,
			// but some tools interpret a missing Compression value as none, so we do
			// the same.
			case cNone, 0:
				if b, ok := d.r.(*buffer); ok {
					d.buf, err = b.Slice(int(offset), int(n))
				} else {
					d.buf, err = safeReadAt(d.r, uint64(n), offset)
				}
			case cG3:
				inv := d.firstVal(tPhotometricInterpretation) == pWhiteIsZero
				order := ccittFillOrder(d.firstVal(tFillOrder))
				r := ccitt.NewReader(io.NewSectionReader(d.r, offset, n), order, ccitt.Group3, blkW, blkH, &ccitt.Options{Invert: inv, Align: false})
				d.buf, err = readBuf(r, d.buf, blockMaxDataSize)
			case cG4:
				inv := d.firstVal(tPhotometricInterpretation) == pWhiteIsZero
				order := ccittFillOrder(d.firstVal(tFillOrder))
				r := ccitt.NewReader(io.NewSectionReader(d.r, offset, n), order, ccitt.Group4, blkW, blkH, &ccitt.Options{Invert: inv, Align: false})
				d.buf, err = readBuf(r, d.buf, blockMaxDataSize)
			case cLZW:
				r := lzw.NewReader(io.NewSectionReader(d.r, offset, n), lzw.MSB, 8)
				d.buf, err = readBuf(r, d.buf, blockMaxDataSize)
				r.Close()
			case cDeflate, cDeflateOld:
				var r io.ReadCloser
				r, err = zlib.NewReader(io.NewSectionReader(d.r, offset, n))
				if err == nil {
					d.buf, err = readBuf(r, d.buf, blockMaxDataSize)
					r.Close()
				}
			case cPackBits:
				// unpackBits does not take the block geometry, so bound its
				// output the same way as the other decompressors.
				var b []byte
				b, err = unpackBits(io.NewSectionReader(d.r, offset, n))
				if err == nil {
					if int64(len(b)) > blockMaxDataSize {
						err = errNoPixels
					} else {
						d.buf = b
					}
				}
			default:
				err = UnsupportedError(fmt.Sprintf("compression value %d", d.firstVal(tCompression)))
			}
			// Only attach page/channel context on the floating-point path, so
			// that integer decoding errors and their values are unchanged.
			if d.isFloat {
				err = makePageError(d.page, -1, err)
			}
			if err != nil {
				return nil, err
			}

			xmin := i * blockWidth
			ymin := j * blockHeight
			xmax := xmin + blkW
			ymax := ymin + blkH
			if d.isFloat {
				err = d.decodeFloat(img, xmin, ymin, xmax, ymax)
			} else {
				err = d.decode(img, xmin, ymin, xmax, ymax)
			}
			if err != nil {
				return nil, err
			}
		}
	}
	return
}

func readBuf(r io.Reader, buf []byte, lim int64) ([]byte, error) {
	b := bytes.NewBuffer(buf[:0])
	_, err := b.ReadFrom(io.LimitReader(r, lim))
	return b.Bytes(), err
}

func init() {
	image.RegisterFormat("tiff", leHeader, Decode, DecodeConfig)
	image.RegisterFormat("tiff", beHeader, Decode, DecodeConfig)
}

// ifdIsFloat reports whether the raw IFD entries p contain a SampleFormat tag
// that declares IEEE floating point samples. It runs before the IFD has been
// fully parsed, solely to decide whether to attach page context to an error.
func (d *decoder) ifdIsFloat(p []byte) bool {
	for i := 0; i+ifdLen <= len(p); i += ifdLen {
		if d.byteOrder.Uint16(p[i:]) != tSampleFormat {
			continue
		}
		typ := d.byteOrder.Uint16(p[i+2:])
		count := d.byteOrder.Uint32(p[i+4:])
		if typ != dtShort || count == 0 || count > 4/2 {
			// Either not inline, or not the layout a float page uses. Such a
			// page will fail to parse for other reasons; context does not
			// matter.
			continue
		}
		for k := uint32(0); k < count; k++ {
			if d.byteOrder.Uint16(p[i+8+2*int(k):]) == sfFloat {
				return true
			}
		}
	}
	return false
}
