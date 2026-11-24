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
		val, err := d.ifdUint(p)
		if err != nil {
			return 0, err
		}
		for _, v := range val {
			if v != 1 {
				return 0, UnsupportedError("sample format")
			}
		}
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

// flushRow discards the bits of the current packed row and aligns the bit
// stream to the start of the next row.
func (d *decoder) flushRow() {
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

// maxInt returns the larger of x or y.
func maxInt(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

// decode decodes the raw data of one strip or tile from d.buf and writes the
// pixels that fall inside dst.Bounds() into dst. The block covers the
// rectangle with top-left corner (xmin, ymin) and size (xmax-xmin) by
// (ymax-ymin). Tile blocks may extend past the right and bottom edges of the
// image; the padding pixels are consumed but never written.
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

	// Only the intersection of the block and the destination is materialized.
	// Pixels outside of it are still consumed from d.buf, but never written.
	b := dst.Bounds()
	rMinX := maxInt(xmin, b.Min.X)
	rMinY := maxInt(ymin, b.Min.Y)
	rMaxX := minInt(xmax, b.Max.X)
	rMaxY := minInt(ymax, b.Max.Y)
	if rMinX >= rMaxX || rMinY >= rMaxY {
		return nil
	}

	switch d.mode {
	case mGray, mGrayInvert:
		if d.bpp == 16 {
			img := dst.(*image.Gray16)
			rowBytes := (xmax - xmin) * 2
			for y := rMinY; y < rMaxY; y++ {
				d.off = (y-ymin)*rowBytes + (rMinX-xmin)*2
				for x := rMinX; x < rMaxX; x++ {
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
			}
		} else {
			img := dst.(*image.Gray)
			max := uint32((1 << d.bpp) - 1)
			for y := ymin; y < rMaxY; y++ {
				// Packed rows are not necessarily byte-aligned (CCITT data is
				// not), so the bit stream is consumed sequentially.
				for x := xmin; x < rMinX; x++ {
					if _, ok := d.readBits(d.bpp); !ok {
						return errNoPixels
					}
				}
				for x := rMinX; x < rMaxX; x++ {
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
				// Consume the trailing padding pixels of the block row.
				for x := rMaxX; x < xmax; x++ {
					if _, ok := d.readBits(d.bpp); !ok {
						return errNoPixels
					}
				}
				d.flushRow()
			}
		}
	case mPaletted:
		img := dst.(*image.Paletted)
		pLen := len(d.palette)
		for y := ymin; y < rMaxY; y++ {
			for x := xmin; x < rMinX; x++ {
				if _, ok := d.readBits(d.bpp); !ok {
					return errNoPixels
				}
			}
			for x := rMinX; x < rMaxX; x++ {
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
			for x := rMaxX; x < xmax; x++ {
				if _, ok := d.readBits(d.bpp); !ok {
					return errNoPixels
				}
			}
			d.flushRow()
		}
	case mRGB:
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			rowBytes := (xmax - xmin) * 6
			for y := rMinY; y < rMaxY; y++ {
				d.off = (y-ymin)*rowBytes + (rMinX-xmin)*6
				for x := rMinX; x < rMaxX; x++ {
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
			rowBytes := (xmax - xmin) * 3
			for y := rMinY; y < rMaxY; y++ {
				min := img.PixOffset(rMinX, y)
				max := img.PixOffset(rMaxX, y)
				off := (y-ymin)*rowBytes + (rMinX-xmin)*3
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
			rowBytes := (xmax - xmin) * 8
			for y := rMinY; y < rMaxY; y++ {
				d.off = (y-ymin)*rowBytes + (rMinX-xmin)*8
				for x := rMinX; x < rMaxX; x++ {
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
			rowBytes := (xmax - xmin) * 4
			for y := rMinY; y < rMaxY; y++ {
				min := img.PixOffset(rMinX, y)
				max := img.PixOffset(rMaxX, y)
				i0 := (y-ymin)*rowBytes + (rMinX-xmin)*4
				i1 := i0 + (rMaxX-rMinX)*4
				if i1 > len(d.buf) {
					return errNoPixels
				}
				copy(img.Pix[min:max], d.buf[i0:i1])
			}
		}
	case mRGBA:
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			rowBytes := (xmax - xmin) * 8
			for y := rMinY; y < rMaxY; y++ {
				d.off = (y-ymin)*rowBytes + (rMinX-xmin)*8
				for x := rMinX; x < rMaxX; x++ {
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
			rowBytes := (xmax - xmin) * 4
			for y := rMinY; y < rMaxY; y++ {
				min := img.PixOffset(rMinX, y)
				max := img.PixOffset(rMaxX, y)
				i0 := (y-ymin)*rowBytes + (rMinX-xmin)*4
				i1 := i0 + (rMaxX-rMinX)*4
				if i1 > len(d.buf) {
					return errNoPixels
				}
				copy(img.Pix[min:max], d.buf[i0:i1])
			}
		}
	}

	return nil
}

func newDecoder(r io.ReaderAt) (*decoder, error) {
	d := &decoder{
		r:        r,
		features: make(map[int][]uint),
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
			return nil, err
		}
		if tag <= prevTag {
			return nil, FormatError("tags are not sorted in ascending order")
		}
		prevTag = tag
	}

	d.config.Width = int(d.firstVal(tImageWidth))
	d.config.Height = int(d.firstVal(tImageLength))

	if _, ok := d.features[tBitsPerSample]; !ok {
		// Default is 1 per specification.
		d.features[tBitsPerSample] = []uint{1}
	}
	d.bpp = d.firstVal(tBitsPerSample)
	switch d.bpp {
	case 0:
		return nil, FormatError("BitsPerSample must not be 0")
	case 1, 8, 16:
		// Nothing to do, these are accepted by this implementation.
	default:
		return nil, UnsupportedError(fmt.Sprintf("BitsPerSample of %v", d.bpp))
	}

	// Determine the image mode.
	switch d.firstVal(tPhotometricInterpretation) {
	case pRGB:
		if d.bpp == 16 {
			for _, b := range d.features[tBitsPerSample] {
				if b != 16 {
					return nil, FormatError("wrong number of samples for 16bit RGB")
				}
			}
		} else {
			for _, b := range d.features[tBitsPerSample] {
				if b != 8 {
					return nil, FormatError("wrong number of samples for 8bit RGB")
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
				return nil, FormatError("wrong number of samples for RGB")
			}
		default:
			return nil, FormatError("wrong number of samples for RGB")
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
		return nil, UnsupportedError("color model")
	}
	if d.firstVal(tPhotometricInterpretation) != pRGB {
		if len(d.features[tBitsPerSample]) != 1 {
			return nil, UnsupportedError("extra samples")
		}
	}

	// The PlanarConfiguration tag defaults to chunky (1) when absent.
	switch pc := d.firstVal(tPlanarConfiguration); pc {
	case 0, pcChunky:
	case pcPlanar:
		// Planar (separate sample planes) data is only meaningful for
		// multi-sample RGB images.
		if len(d.features[tBitsPerSample]) < 2 {
			return nil, FormatError("planar configuration with a single sample")
		}
		if d.mode != mRGB && d.mode != mRGBA && d.mode != mNRGBA {
			return nil, UnsupportedError("planar configuration for color model")
		}
	default:
		return nil, UnsupportedError("planar configuration")
	}

	return d, nil
}

// DecodeConfig returns the color model and dimensions of a TIFF image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	d, err := newDecoder(newReaderAt(r))
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

// blockLayout describes how the pixel data is partitioned into strips or
// tiles.
type blockLayout struct {
	tile         bool
	blockWidth   int
	blockHeight  int
	blocksAcross int
	blocksDown   int
	// blockPadding reports whether the blocks on the right and bottom edges
	// are padded to the full block size. This is true for tiled images.
	blockPadding bool
	blockOffsets []uint
	blockCounts  []uint
}

// layout computes the strip or tile layout of the image and validates that the
// IFD declares enough offsets and byte counts for every block.
func (d *decoder) layout() (blockLayout, error) {
	l := blockLayout{
		blockWidth:   d.config.Width,
		blockHeight:  d.config.Height,
		blocksAcross: 1,
		blocksDown:   1,
	}
	if d.config.Width == 0 {
		l.blocksAcross = 0
	}
	if d.config.Height == 0 {
		l.blocksDown = 0
	}
	// An image with no pixels has no blocks to validate or read; return before
	// checking tile dimensions so that degenerate (e.g. zero-length) tiles are
	// not rejected when there is nothing for them to hold.
	if l.blocksAcross == 0 || l.blocksDown == 0 {
		return l, nil
	}

	if int(d.firstVal(tTileWidth)) != 0 {
		l.tile = true
		l.blockPadding = true

		l.blockWidth = int(d.firstVal(tTileWidth))
		l.blockHeight = int(d.firstVal(tTileLength))

		// The specification says that tile widths and lengths must be a multiple of 16.
		// We currently permit invalid sizes, but reject anything too small to limit the
		// amount of work a malicious input can force us to perform.
		if l.blockWidth < 8 || l.blockHeight < 8 {
			return l, FormatError("tile size is too small")
		}

		if l.blockWidth != 0 {
			l.blocksAcross = (d.config.Width + l.blockWidth - 1) / l.blockWidth
		}
		if l.blockHeight != 0 {
			l.blocksDown = (d.config.Height + l.blockHeight - 1) / l.blockHeight
		}

		l.blockCounts = d.features[tTileByteCounts]
		l.blockOffsets = d.features[tTileOffsets]
	} else {
		if int(d.firstVal(tRowsPerStrip)) != 0 {
			l.blockHeight = int(d.firstVal(tRowsPerStrip))
		}

		if l.blockHeight != 0 {
			l.blocksDown = (d.config.Height + l.blockHeight - 1) / l.blockHeight
		}

		l.blockOffsets = d.features[tStripOffsets]
		l.blockCounts = d.features[tStripByteCounts]
	}

	// Chunky data holds one block per spatial position. Planar data stores the
	// sample planes one after another, so every position has one block per
	// sample.
	n := l.blocksAcross * l.blocksDown
	if d.firstVal(tPlanarConfiguration) == pcPlanar {
		n *= len(d.features[tBitsPerSample])
	}
	if len(l.blockOffsets) < n || len(l.blockCounts) < n {
		return l, FormatError("inconsistent header")
	}
	return l, nil
}

// newImage allocates an image of the decoder's color model with the given
// bounds. No pixel data beyond rect is allocated.
func (d *decoder) newImage(rect image.Rectangle) image.Image {
	switch d.mode {
	case mGray, mGrayInvert:
		if d.bpp == 16 {
			return image.NewGray16(rect)
		}
		return image.NewGray(rect)
	case mPaletted:
		return image.NewPaletted(rect, d.palette)
	case mNRGBA:
		if d.bpp == 16 {
			return image.NewNRGBA64(rect)
		}
		return image.NewNRGBA(rect)
	case mRGB, mRGBA:
		if d.bpp == 16 {
			return image.NewRGBA64(rect)
		}
		return image.NewRGBA(rect)
	}
	return nil
}

// bytesPerPixel reports how many bytes of output memory one pixel occupies.
func (d *decoder) bytesPerPixel() int {
	bps := int(d.bpp) / 8
	if bps == 0 {
		bps = 1
	}
	spp := len(d.features[tBitsPerSample])
	// mRGB has no alpha sample in the file but is materialized as RGBA.
	if d.mode == mRGB {
		spp++
	}
	return bps * spp
}

// blockErr wraps err with the identity of the strip or tile (and, for planar
// data, the sample plane) that triggered it.
func (d *decoder) blockErr(l blockLayout, idx, plane int, err error) error {
	kind := "strip"
	if l.tile {
		kind = "tile"
	}
	if d.firstVal(tPlanarConfiguration) == pcPlanar {
		return fmt.Errorf("tiff: %s %d (sample plane %d): %w", kind, idx, plane, err)
	}
	return fmt.Errorf("tiff: %s %d: %w", kind, idx, err)
}

// readBlock reads and decompresses the block with the given layout index,
// whose decoded dimensions are blkW by blkH. The returned slice is owned by
// the decoder and is valid until the next readBlock call.
func (d *decoder) readBlock(l blockLayout, idx, blkW, blkH int, blockMaxDataSize int64) ([]byte, error) {
	offset := int64(l.blockOffsets[idx])
	n := int64(l.blockCounts[idx])

	var (
		buf []byte
		err error
	)
	switch d.firstVal(tCompression) {

	// According to the spec, Compression does not have a default value,
	// but some tools interpret a missing Compression value as none, so we do
	// the same.
	case cNone, 0:
		if b, ok := d.r.(*buffer); ok {
			buf, err = b.Slice(int(offset), int(n))
		} else {
			buf, err = safeReadAt(d.r, uint64(n), offset)
		}
	case cG3:
		inv := d.firstVal(tPhotometricInterpretation) == pWhiteIsZero
		order := ccittFillOrder(d.firstVal(tFillOrder))
		r := ccitt.NewReader(io.NewSectionReader(d.r, offset, n), order, ccitt.Group3, blkW, blkH, &ccitt.Options{Invert: inv, Align: false})
		buf, err = readBuf(r, d.buf, blockMaxDataSize)
	case cG4:
		inv := d.firstVal(tPhotometricInterpretation) == pWhiteIsZero
		order := ccittFillOrder(d.firstVal(tFillOrder))
		r := ccitt.NewReader(io.NewSectionReader(d.r, offset, n), order, ccitt.Group4, blkW, blkH, &ccitt.Options{Invert: inv, Align: false})
		buf, err = readBuf(r, d.buf, blockMaxDataSize)
	case cLZW:
		r := lzw.NewReader(io.NewSectionReader(d.r, offset, n), lzw.MSB, 8)
		buf, err = readBuf(r, d.buf, blockMaxDataSize)
		r.Close()
	case cDeflate, cDeflateOld:
		var r io.ReadCloser
		r, err = zlib.NewReader(io.NewSectionReader(d.r, offset, n))
		if err != nil {
			return nil, err
		}
		buf, err = readBuf(r, d.buf, blockMaxDataSize)
		r.Close()
	case cPackBits:
		buf, err = unpackBits(io.NewSectionReader(d.r, offset, n))
	default:
		err = UnsupportedError(fmt.Sprintf("compression value %d", d.firstVal(tCompression)))
	}
	return buf, err
}

// blockRect returns the image-space rectangle of the block at grid position
// (i, j). Edge blocks are shrunk unless the layout uses padded tiles.
func (d *decoder) blockRect(l blockLayout, i, j int) image.Rectangle {
	blkW := l.blockWidth
	if !l.blockPadding && i == l.blocksAcross-1 && d.config.Width%l.blockWidth != 0 {
		blkW = d.config.Width % l.blockWidth
	}
	blkH := l.blockHeight
	if !l.blockPadding && j == l.blocksDown-1 && d.config.Height%l.blockHeight != 0 {
		blkH = d.config.Height % l.blockHeight
	}
	xmin := i * l.blockWidth
	ymin := j * l.blockHeight
	return image.Rect(xmin, ymin, xmin+blkW, ymin+blkH)
}

// decodeChunky decodes an image whose samples are stored interleaved. Only
// blocks that overlap dst.Bounds() are read and decoded, so a sub-rectangle
// decode never touches the pixel data of unrelated blocks.
func (d *decoder) decodeChunky(dst image.Image, l blockLayout) error {
	if l.blocksAcross == 0 || l.blocksDown == 0 {
		return nil
	}
	// Maximum data per pixel is 8 bytes (RGBA64).
	blockMaxDataSize := int64(l.blockWidth) * int64(l.blockHeight) * 8
	for i := 0; i < l.blocksAcross; i++ {
		for j := 0; j < l.blocksDown; j++ {
			blk := d.blockRect(l, i, j)
			if !blk.Overlaps(dst.Bounds()) {
				continue
			}
			idx := j*l.blocksAcross + i
			buf, err := d.readBlock(l, idx, blk.Dx(), blk.Dy(), blockMaxDataSize)
			if err != nil {
				return d.blockErr(l, idx, 0, err)
			}
			d.buf = buf
			if err := d.decode(dst, blk.Min.X, blk.Min.Y, blk.Max.X, blk.Max.Y); err != nil {
				return d.blockErr(l, idx, 0, err)
			}
		}
	}
	return nil
}

// pixOffseter is satisfied by every dense image type produced by newImage.
type pixOffseter interface {
	PixOffset(x, y int) int
}

// undoPlanarPredictor reverses the horizontal differencing predictor on one
// block of planar data, where every pixel of the block holds a single sample
// of d.bpp bits.
func (d *decoder) undoPlanarPredictor(blkW, blkH int) error {
	bps := int(d.bpp) / 8 // bytes per sample: 1 or 2
	for y := 0; y < blkH; y++ {
		row := y * blkW * bps
		for x := 1; x < blkW; x++ {
			off := row + x*bps
			if off+bps > len(d.buf) {
				return errNoPixels
			}
			if bps == 2 {
				prev := d.byteOrder.Uint16(d.buf[off-2 : off])
				cur := d.byteOrder.Uint16(d.buf[off : off+2])
				d.byteOrder.PutUint16(d.buf[off:off+2], prev+cur)
			} else {
				d.buf[off] += d.buf[off-1]
			}
		}
	}
	return nil
}

// decodePlanar decodes an image whose samples are stored in separate planes.
// Only blocks that overlap dst.Bounds() are read.
func (d *decoder) decodePlanar(dst image.Image, l blockLayout) error {
	if l.blocksAcross == 0 || l.blocksDown == 0 {
		return nil
	}
	spp := len(d.features[tBitsPerSample])
	spatial := l.blocksAcross * l.blocksDown
	// Maximum data per pixel is 8 bytes (RGBA64).
	blockMaxDataSize := int64(l.blockWidth) * int64(l.blockHeight) * 8
	bps := int(d.bpp) / 8 // bytes per sample: 1 or 2
	bounds := dst.Bounds()

	po, ok := dst.(pixOffseter)
	if !ok {
		return UnsupportedError("color model")
	}
	var pix []byte
	switch p := dst.(type) {
	case *image.RGBA:
		pix = p.Pix
	case *image.NRGBA:
		pix = p.Pix
	case *image.RGBA64:
		pix = p.Pix
	case *image.NRGBA64:
		pix = p.Pix
	default:
		return UnsupportedError("color model")
	}

	// RGB data has no alpha plane: fill the alpha channel with full opacity.
	if d.mode == mRGB {
		alpha := uint8(0xff)
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				pix[po.PixOffset(x, y)+spp*bps] = alpha
				if bps == 2 {
					pix[po.PixOffset(x, y)+spp*bps+1] = alpha
				}
			}
		}
	}

	for plane := 0; plane < spp; plane++ {
		for i := 0; i < l.blocksAcross; i++ {
			for j := 0; j < l.blocksDown; j++ {
				blk := d.blockRect(l, i, j)
				if !blk.Overlaps(bounds) {
					continue
				}
				spatialIdx := j*l.blocksAcross + i
				idx := plane*spatial + spatialIdx
				buf, err := d.readBlock(l, idx, blk.Dx(), blk.Dy(), blockMaxDataSize)
				if err != nil {
					return d.blockErr(l, spatialIdx, plane, err)
				}
				d.buf = buf
				if d.firstVal(tPredictor) == prHorizontal {
					if err := d.undoPlanarPredictor(blk.Dx(), blk.Dy()); err != nil {
						return d.blockErr(l, spatialIdx, plane, err)
					}
				}
				for y := maxInt(blk.Min.Y, bounds.Min.Y); y < minInt(blk.Max.Y, bounds.Max.Y); y++ {
					for x := maxInt(blk.Min.X, bounds.Min.X); x < minInt(blk.Max.X, bounds.Max.X); x++ {
						src := ((y-blk.Min.Y)*blk.Dx() + (x - blk.Min.X)) * bps
						if src+bps > len(d.buf) {
							return d.blockErr(l, spatialIdx, plane, errNoPixels)
						}
						dstOff := po.PixOffset(x, y) + plane*bps
						pix[dstOff] = d.buf[src]
						if bps == 2 {
							pix[dstOff+1] = d.buf[src+1]
						}
					}
				}
			}
		}
	}
	return nil
}

// Decode reads a TIFF image from r and returns it as an image.Image.
// The type of Image returned depends on the contents of the TIFF.
func Decode(r io.Reader) (img image.Image, err error) {
	d, err := newDecoder(newReaderAt(r))
	if err != nil {
		return nil, err
	}
	l, err := d.layout()
	if err != nil {
		return nil, err
	}

	imgRect := image.Rect(0, 0, d.config.Width, d.config.Height)
	img = d.newImage(imgRect)
	if l.blocksAcross == 0 || l.blocksDown == 0 {
		return img, nil
	}
	if d.firstVal(tPlanarConfiguration) == pcPlanar {
		err = d.decodePlanar(img, l)
	} else {
		err = d.decodeChunky(img, l)
	}
	return img, err
}

// maxRegionBytes caps the pixel memory that a single DecodeRegion call may
// allocate. The dimensions of the rectangle and of the image both originate
// from untrusted metadata, so the check runs before any pixel allocation.
const maxRegionBytes = 256 << 20 // 256 MiB

// DecodeRegion reads the pixels inside rect from a TIFF image in r and returns
// them as an image.Image. Only the strips or tiles intersecting rect are read
// and decoded, and the returned image's Bounds equal the intersection of rect
// with the image bounds; no memory is allocated for pixels outside of it.
//
// Because the data of each strip or tile is accessed at arbitrary file
// offsets, r must be an io.ReaderAt; the same reader can be used for repeated
// DecodeRegion calls. A rectangle with negative width or height, an empty
// rectangle, or one that does not overlap the image results in a FormatError.
func DecodeRegion(r io.ReaderAt, rect image.Rectangle) (image.Image, error) {
	if rect.Empty() {
		return nil, FormatError("empty source rectangle")
	}
	d, err := newDecoder(r)
	if err != nil {
		return nil, err
	}
	l, err := d.layout()
	if err != nil {
		return nil, err
	}

	imgBounds := image.Rect(0, 0, d.config.Width, d.config.Height)
	out := rect.Intersect(imgBounds)
	if out.Empty() {
		return nil, FormatError("source rectangle does not overlap the image")
	}

	// Enforce the resource limit before allocating the destination image.
	if int64(out.Dx()) > maxRegionBytes/int64(d.bytesPerPixel()) ||
		int64(out.Dy()) > maxRegionBytes/int64(d.bytesPerPixel()) ||
		int64(out.Dx())*int64(out.Dy())*int64(d.bytesPerPixel()) > maxRegionBytes {
		return nil, FormatError("decoded region is too large")
	}

	img := d.newImage(out)
	if d.firstVal(tPlanarConfiguration) == pcPlanar {
		err = d.decodePlanar(img, l)
	} else {
		err = d.decodeChunky(img, l)
	}
	return img, err
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
