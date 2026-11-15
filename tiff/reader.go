// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tiff implements a TIFF image decoder and encoder.
//
// The TIFF specification is at http://partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
//
// The decoder also accepts BigTIFF files, which extend TIFF with 64-bit
// offsets and larger IFD entries so that images larger than 4 GiB can be
// addressed. BigTIFF is described at https://bigtiff.org/.
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
	"golang.org/x/image/internal/safemath"
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
	ifd       map[int][]byte
	palette   []color.Color

	// bigTiff reports whether this is a BigTIFF (version 43) file, which
	// uses 8-byte offsets and 20-byte IFD entries.
	bigTiff bool

	buf   []byte
	off   int    // Current offset in buf.
	v     uint32 // Buffer value for reading with arbitrary bit depths.
	nbits uint   // Remaining number of bits in v.
}

// entryLen returns the length in bytes of one IFD entry.
func (d *decoder) entryLen() int {
	if d.bigTiff {
		return ifdBigLen
	}
	return ifdLen
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

// firstIntVal returns the first int of the features entry with the given tag,
// or 0 if the tag does not exist.
// If the tag overflows an int, it returns an error.
func (d *decoder) firstIntVal(tag int) (int, error) {
	v := d.firstVal(tag)
	if v > math.MaxInt {
		return 0, FormatError("IFD value too large")
	}
	return int(v), nil
}

// ifdEntryData locates the data of the IFD entry in p. The entry must be a
// complete IFD entry (12 bytes for classic TIFF, 20 bytes for BigTIFF).
//
// It returns the raw bytes holding at most maxCount values, the entry's
// datatype and the number of values included in the returned slice. If the
// entry declares more than maxCount values, only the first maxCount are
// returned; this bounds the amount of work (and memory) that untrusted input
// can cause.
func (d *decoder) ifdEntryData(p []byte, maxCount int) (raw []byte, datatype uint16, count int, err error) {
	entryLen := d.entryLen()
	if len(p) < entryLen {
		return nil, 0, 0, FormatError("bad IFD entry")
	}

	datatype = d.byteOrder.Uint16(p[2:4])
	length := uint64(0)
	if l, ok := datatypeLen(datatype); ok {
		if !d.bigTiff && datatype > dtRational {
			// LONG8, SLONG8 and IFD8 are BigTIFF-only data types.
			return nil, 0, 0, UnsupportedError("IFD entry datatype")
		}
		length = uint64(l)
	} else {
		return nil, 0, 0, UnsupportedError("IFD entry datatype")
	}

	// The number of values is a uint32 in classic TIFF and a uint64 in
	// BigTIFF. Reject counts whose total size would overflow; this also keeps
	// the count within the bounds checked below.
	var n uint64
	if d.bigTiff {
		n = d.byteOrder.Uint64(p[4:12])
	} else {
		n = uint64(d.byteOrder.Uint32(p[4:8]))
	}
	if n > math.MaxInt32/length {
		return nil, 0, 0, FormatError("IFD data too large")
	}
	datalen := n * length

	// The value is stored inline if it fits in the entry's value field, and
	// is otherwise stored elsewhere and referenced by offset. BigTIFF has an
	// 8-byte value field; classic TIFF has a 4-byte one.
	inlineLen, valueOff := uint64(4), 8
	if d.bigTiff {
		inlineLen, valueOff = 8, 12
	}

	// Make sure the entry is long enough to contain the value field. This is
	// redundant with the length check at the top of the function, but makes
	// the slice expressions below obviously safe.
	if len(p) < valueOff+int(inlineLen) {
		return nil, 0, 0, FormatError("bad IFD entry")
	}

	truncatedCount := int(n)
	if truncatedCount > maxCount {
		truncatedCount = maxCount
	}
	truncatedLen := uint64(truncatedCount) * length

	if datalen > inlineLen {
		off := d.ifdValueOffset(p, valueOff)
		if d.bigTiff && off%8 != 0 {
			// The BigTIFF specification requires all values to begin at an
			// 8-byte-aligned address.
			return nil, 0, 0, FormatError("misaligned IFD value offset")
		}
		if off > math.MaxInt64 {
			return nil, 0, 0, FormatError("IFD value offset too large")
		}
		raw, err = safeReadAt(d.r, truncatedLen, int64(off))
		if err != nil {
			return nil, 0, 0, err
		}
	} else {
		raw = p[valueOff : valueOff+int(datalen)]
	}
	return raw, datatype, truncatedCount, nil
}

// ifdValueOffset returns the value-or-offset field of the IFD entry in p.
// off is the offset of that field within the entry.
func (d *decoder) ifdValueOffset(p []byte, off int) uint64 {
	if d.bigTiff {
		return d.byteOrder.Uint64(p[off : off+8])
	}
	return uint64(d.byteOrder.Uint32(p[off : off+4]))
}

// ifdUint decodes the IFD entry in p, which must be of the Byte, Short,
// Long, Long8 or IFD8 type, and returns the decoded uint values.
//
// maxCount limits the number of values.
// If the entry contains more than maxCount values, only the first maxCount are parsed.
func (d *decoder) ifdUint(p []byte, maxCount int) (u []uint, err error) {
	raw, datatype, count, err := d.ifdEntryData(p, maxCount)
	if err != nil {
		return nil, err
	}

	u = make([]uint, count)
	switch datatype {
	case dtByte:
		for i := range u {
			u[i] = uint(raw[i])
		}
	case dtShort:
		for i := range u {
			u[i] = uint(d.byteOrder.Uint16(raw[2*i : 2*(i+1)]))
		}
	case dtLong:
		for i := range u {
			u[i] = uint(d.byteOrder.Uint32(raw[4*i : 4*(i+1)]))
		}
	case dtLong8, dtIFD8:
		for i := range u {
			v := d.byteOrder.Uint64(raw[8*i : 8*(i+1)])
			if v > math.MaxUint {
				// The value cannot be represented as a uint, which is a
				// possibility on 32-bit platforms.
				return nil, FormatError("IFD value too large")
			}
			u[i] = uint(v)
		}
	default:
		return nil, UnsupportedError("data type")
	}
	return u, nil
}

// parseIFDOffsets parses an IFD entry stored in d.ifd using ifdUint.
// For IFD entries which can be large, we delay reading and parsing the entry
// until we know the image size, which lets us set reasonable bounds on the IFD entry.
//
// The offsets are returned as int64 so that files larger than 4 GiB can be
// addressed on all platforms, including those where int is 32 bits wide.
func (d *decoder) parseIFDOffsets(tag int, maxCount int) ([]int64, error) {
	p, ok := d.ifd[tag]
	if !ok {
		return nil, nil
	}
	raw, datatype, count, err := d.ifdEntryData(p, maxCount)
	if err != nil {
		return nil, err
	}

	u := make([]int64, count)
	switch datatype {
	case dtByte:
		for i := range u {
			u[i] = int64(raw[i])
		}
	case dtShort:
		for i := range u {
			u[i] = int64(d.byteOrder.Uint16(raw[2*i : 2*(i+1)]))
		}
	case dtLong:
		for i := range u {
			u[i] = int64(d.byteOrder.Uint32(raw[4*i : 4*(i+1)]))
		}
	case dtLong8, dtIFD8:
		for i := range u {
			v := d.byteOrder.Uint64(raw[8*i : 8*(i+1)])
			if v > math.MaxInt64 {
				return nil, FormatError("IFD offset too large")
			}
			u[i] = int64(v)
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
	// smallEntryMaxCount is the limit to use for parsed IFD entries that
	// don't scale with the image size.
	//
	// We could be more precise and limit the number of entries to exactly the
	// correct limit (for example, tTileWidth should always have exactly one value),
	// but 16 is small enough to prevent excessive allocation and large enough to
	// allow for any multi-value entry.
	const smallEntryMaxCount = 16

	tag := d.byteOrder.Uint16(p[0:2])
	switch tag {
	case tBitsPerSample,
		tExtraSamples,
		tPhotometricInterpretation,
		tCompression,
		tPredictor,
		tRowsPerStrip,
		tTileWidth,
		tTileLength,
		tImageLength,
		tImageWidth,
		tFillOrder,
		tT4Options,
		tT6Options:
		val, err := d.ifdUint(p, smallEntryMaxCount)
		if err != nil {
			return 0, err
		}
		d.features[int(tag)] = val
	case tStripOffsets,
		tStripByteCounts,
		tTileOffsets,
		tTileByteCounts:
		// These keys may contain many values.
		// Stash the IFD entry for later parsing.
		v := make([]byte, len(p))
		copy(v, p)
		d.ifd[int(tag)] = v
	case tColorMap:
		const maxColors = 256
		val, err := d.ifdUint(p, (3*maxColors)+1)
		if err != nil {
			return 0, err
		}
		numcolors := len(val) / 3
		if len(val)%3 != 0 || numcolors <= 0 || numcolors > maxColors {
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
		val, err := d.ifdUint(p, smallEntryMaxCount)
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

	// calcRowBytes returns the number of bytes in a row with numSamples samples per pixel
	// and bytesPerSample bytes per sample.
	calcRowBytes := func(xmin, xmax, numSamples int, bitsPerSample uint) int {
		return int((int64(xmax-xmin)*int64(numSamples)*int64(bitsPerSample) + 7) / 8)
	}
	// calcRowOff returns the offset of the next row.
	calcRowOff := func(y, ymin, rowBytes int) int {
		return (y - ymin) * rowBytes
	}

	rMaxX := minInt(xmax, dst.Bounds().Max.X)
	rMaxY := minInt(ymax, dst.Bounds().Max.Y)
	switch d.mode {
	case mGray, mGrayInvert:
		rowBytes := calcRowBytes(xmin, xmax, 1, d.bpp)
		if d.bpp == 16 {
			img := dst.(*image.Gray16)
			for y := ymin; y < rMaxY; y++ {
				d.off = calcRowOff(y, ymin, rowBytes)
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
			}
		} else {
			img := dst.(*image.Gray)
			max := uint32((1 << d.bpp) - 1)
			for y := ymin; y < rMaxY; y++ {
				d.off = calcRowOff(y, ymin, rowBytes)
				d.flushBits()
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
			}
		}
	case mPaletted:
		img := dst.(*image.Paletted)
		pLen := len(d.palette)
		rowBytes := calcRowBytes(xmin, xmax, 1, d.bpp)
		for y := ymin; y < rMaxY; y++ {
			d.off = calcRowOff(y, ymin, rowBytes)
			d.flushBits()
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
		}
	case mRGB:
		rowBytes := calcRowBytes(xmin, xmax, 3, d.bpp)
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			for y := ymin; y < rMaxY; y++ {
				d.off = calcRowOff(y, ymin, rowBytes)
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
				d.off = calcRowOff(y, ymin, rowBytes)
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
		rowBytes := calcRowBytes(xmin, xmax, 4, d.bpp)
		if d.bpp == 16 {
			img := dst.(*image.NRGBA64)
			for y := ymin; y < rMaxY; y++ {
				d.off = calcRowOff(y, ymin, rowBytes)
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
		rowBytes := calcRowBytes(xmin, xmax, 4, d.bpp)
		if d.bpp == 16 {
			img := dst.(*image.RGBA64)
			for y := ymin; y < rMaxY; y++ {
				d.off = calcRowOff(y, ymin, rowBytes)
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

// maxBytesPerPixel is the maximum possible bytes-per-pixel,
// used for conservative bounds checking.
const maxBytesPerPixel = 8

func newDecoder(r io.Reader) (*decoder, error) {
	d := &decoder{
		r:        newReaderAt(r),
		features: make(map[int][]uint),
		ifd:      make(map[int][]byte),
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
	case leBigHeader:
		d.byteOrder = binary.LittleEndian
		d.bigTiff = true
	case beBigHeader:
		d.byteOrder = binary.BigEndian
		d.bigTiff = true
	default:
		return nil, FormatError("malformed header")
	}

	var ifdOffset int64
	headerLen := int64(8)
	if d.bigTiff {
		// A BigTIFF header is 16 bytes long: bytes 4-5 contain the width of
		// offsets, bytes 6-7 are reserved, and bytes 8-15 contain the 8-byte
		// offset of the first IFD.
		headerLen = 16
		p, err := safeReadAt(d.r, 16, 0)
		if err != nil {
			return nil, err
		}
		if offsetSize := d.byteOrder.Uint16(p[4:6]); offsetSize != 8 {
			return nil, UnsupportedError(fmt.Sprintf("BigTIFF offset size %d", offsetSize))
		}
		if reserved := d.byteOrder.Uint16(p[6:8]); reserved != 0 {
			return nil, FormatError("non-zero reserved header field")
		}
		off := d.byteOrder.Uint64(p[8:16])
		if off > math.MaxInt64 {
			return nil, FormatError("IFD offset too large")
		}
		ifdOffset = int64(off)
	} else {
		ifdOffset = int64(d.byteOrder.Uint32(p[4:8]))
	}
	if ifdOffset < headerLen || ifdOffset > math.MaxInt64-8 {
		// The IFD cannot overlap the file header, and adding the size of the
		// entry count must not overflow.
		return nil, FormatError("bad IFD offset")
	}

	// The first entry (2 bytes in classic TIFF, 8 bytes in BigTIFF) contains
	// the number of entries (12 bytes each in classic TIFF, 20 bytes each in
	// BigTIFF).
	entryLen := d.entryLen()
	countLen := int64(2)
	if d.bigTiff {
		countLen = 8
	}
	p, err := safeReadAt(d.r, uint64(countLen), ifdOffset)
	if err != nil {
		return nil, err
	}
	var numItems int64
	if d.bigTiff {
		v := d.byteOrder.Uint64(p[0:8])
		if v > maxIFDEntries {
			return nil, FormatError("too many IFD entries")
		}
		numItems = int64(v)
	} else {
		numItems = int64(d.byteOrder.Uint16(p[0:2]))
	}

	// All IFD entries are read in one chunk. The bounds on numItems and
	// entryLen keep the multiplication from overflowing.
	p, err = safeReadAt(d.r, uint64(entryLen)*uint64(numItems), ifdOffset+countLen)
	if err != nil {
		return nil, err
	}

	prevTag := -1
	for i := 0; i < len(p); i += entryLen {
		tag, err := d.parseIFD(p[i : i+entryLen])
		if err != nil {
			return nil, err
		}
		if tag <= prevTag {
			return nil, FormatError("tags are not sorted in ascending order")
		}
		prevTag = tag
	}

	if d.config.Width, err = d.firstIntVal(tImageWidth); err != nil {
		return nil, err
	}
	if d.config.Height, err = d.firstIntVal(tImageLength); err != nil {
		return nil, err
	}
	if d.config.Width == 0 || d.config.Height == 0 {
		return nil, FormatError("zero-size image")
	}
	// Check that the image fits in memory.
	// This conservatively assumes 8 bytes per pixel,
	// rather than using the actual pixel size.
	if _, ok := safemath.Mul3(d.config.Width, d.config.Height, maxBytesPerPixel); !ok {
		return nil, FormatError("image too large")
	}

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

	return d, nil
}

// DecodeConfig returns the color model and dimensions of a TIFF image without
// decoding the entire image.
func DecodeConfig(r io.Reader) (image.Config, error) {
	d, err := newDecoder(r)
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
	d, err := newDecoder(r)
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

	var blockOffsets, blockCounts []int64

	if d.firstVal(tTileWidth) != 0 {
		blockPadding = true

		if blockWidth, err = d.firstIntVal(tTileWidth); err != nil {
			return nil, err
		}
		if blockHeight, err = d.firstIntVal(tTileLength); err != nil {
			return nil, err
		}

		// The specification says that tile widths and lengths must be a multiple of 16.
		// We currently permit invalid sizes, but reject anything too small to limit the
		// amount of work a malicious input can force us to perform.
		if blockWidth < 8 || blockHeight < 8 {
			return nil, FormatError("tile size is too small")
		}
		// Same conservative assumption on bytes-per-pixel as for the image dimensions.
		if _, ok := safemath.Mul3(blockWidth, blockHeight, maxBytesPerPixel); !ok {
			return nil, FormatError("tile size is too large")
		}
		if blockWidth-d.config.Width > 16 || blockHeight-d.config.Height > 16 {
			// Tiles may be padded to the nearest multiple of 16, but one of
			// the dimensions of the tile exceeds the image dimension by more
			// than padding would require.
			//
			// Typical TIFF tiles are 256x256, but 1024x1024 appears to be
			// in occasional use. If the tile is both larger than the image
			// and has more than 1024 pixels in one dimension, it's
			// probably malicious input.
			//
			// Note that this still permits very large tiles,
			// so long as the tile is smaller than the image dimensions.
			// A gigapixel image with a 2048x2048 tile size
			// (used in some GIS applications)
			// will be valid because the image dimensions are much larger
			// than the tile size.
			if blockWidth > 1024 || blockHeight > 1024 {
				return nil, FormatError("tile size exceeds image size")
			}
		}
		if blockWidth != 0 {
			blocksAcross = (d.config.Width + blockWidth - 1) / blockWidth
		}
		if blockHeight != 0 {
			blocksDown = (d.config.Height + blockHeight - 1) / blockHeight
		}

		blockOffsets, err = d.parseIFDOffsets(tTileOffsets, blocksAcross*blocksDown)
		if err != nil {
			return nil, err
		}
		blockCounts, err = d.parseIFDOffsets(tTileByteCounts, blocksAcross*blocksDown)
		if err != nil {
			return nil, err
		}
	} else {

		if v := d.firstVal(tRowsPerStrip); v > 0 && v < uint(blockHeight) {
			blockHeight = int(v)
		}

		if blockHeight != 0 {
			// This can't overflow: We require that w*h*8 not overflow and
			// blockHeight is no more than d.config.Height.
			blocksDown = (d.config.Height + blockHeight - 1) / blockHeight
		}

		blockOffsets, err = d.parseIFDOffsets(tStripOffsets, blocksDown)
		if err != nil {
			return nil, err
		}
		blockCounts, err = d.parseIFDOffsets(tStripByteCounts, blocksDown)
		if err != nil {
			return nil, err
		}
	}

	// Check if we have the right number of strips/tiles, offsets and counts.
	if n := blocksAcross * blocksDown; len(blockOffsets) < n || len(blockCounts) < n {
		return nil, FormatError("inconsistent header")
	}

	imgRect := image.Rect(0, 0, d.config.Width, d.config.Height)
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

	if blocksAcross == 0 || blocksDown == 0 {
		return
	}
	// Maximum data per pixel is 8 bytes (RGBA64).
	blockMaxDataSize := int64(blockWidth) * int64(blockHeight) * 8
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
			offset := blockOffsets[j*blocksAcross+i]
			n := blockCounts[j*blocksAcross+i]
			switch d.firstVal(tCompression) {

			// According to the spec, Compression does not have a default value,
			// but some tools interpret a missing Compression value as none, so we do
			// the same.
			case cNone, 0:
				if n > blockMaxDataSize {
					return nil, FormatError("block data size too large")
				}
				if b, ok := d.r.(*buffer); ok {
					d.buf, err = b.Slice(offset, n)
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
				if err != nil {
					return nil, err
				}
				d.buf, err = readBuf(r, d.buf, blockMaxDataSize)
				r.Close()
			case cPackBits:
				d.buf, err = unpackBits(io.NewSectionReader(d.r, offset, n), blockMaxDataSize)
			default:
				err = UnsupportedError(fmt.Sprintf("compression value %d", d.firstVal(tCompression)))
			}
			if err != nil {
				return nil, err
			}

			xmin := i * blockWidth
			ymin := j * blockHeight
			xmax := xmin + blkW
			ymax := ymin + blkH
			err = d.decode(img, xmin, ymin, xmax, ymax)
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
	image.RegisterFormat("tiff", leBigHeader, Decode, DecodeConfig)
	image.RegisterFormat("tiff", beBigHeader, Decode, DecodeConfig)
}
