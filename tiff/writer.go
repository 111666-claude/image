// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"image"
	"io"
	"math"
	"math/bits"
	"sort"
)

// The TIFF format allows to choose the order of the different elements freely.
// The basic structure of a TIFF file written by this package is:
//
//   1. Header (8 bytes).
//   2. Image data.
//   3. Image File Directory (IFD).
//   4. "Pointer area" for larger entries in the IFD.

// We only write little-endian TIFF files.
var enc = binary.LittleEndian

// An ifdEntry is a single entry in an Image File Directory.
// A value of type dtRational is composed of two 32-bit values,
// thus data contains two uints (numerator and denominator) for a single number.
type ifdEntry struct {
	tag      int
	datatype int
	data     []uint32
}

func (e ifdEntry) putData(p []byte) {
	for _, d := range e.data {
		switch e.datatype {
		case dtByte, dtASCII:
			p[0] = byte(d)
			p = p[1:]
		case dtShort:
			enc.PutUint16(p, uint16(d))
			p = p[2:]
		case dtLong, dtRational:
			enc.PutUint32(p, uint32(d))
			p = p[4:]
		}
	}
}

type byTag []ifdEntry

func (d byTag) Len() int           { return len(d) }
func (d byTag) Less(i, j int) bool { return d[i].tag < d[j].tag }
func (d byTag) Swap(i, j int)      { d[i], d[j] = d[j], d[i] }

func encodeGray(w io.Writer, pix []uint8, dx, dy, stride int, predictor bool) error {
	if !predictor {
		return writePix(w, pix, dy, dx, stride)
	}
	buf := make([]byte, dx)
	for y := 0; y < dy; y++ {
		min := y*stride + 0
		max := y*stride + dx
		off := 0
		var v0 uint8
		for i := min; i < max; i++ {
			v1 := pix[i]
			buf[off] = v1 - v0
			v0 = v1
			off++
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func encodeGray16(w io.Writer, pix []uint8, dx, dy, stride int, predictor bool) error {
	buf := make([]byte, dx*2)
	for y := 0; y < dy; y++ {
		min := y*stride + 0
		max := y*stride + dx*2
		off := 0
		var v0 uint16
		for i := min; i < max; i += 2 {
			// An image.Gray16's Pix is in big-endian order.
			v1 := uint16(pix[i])<<8 | uint16(pix[i+1])
			if predictor {
				v0, v1 = v1, v1-v0
			}
			// We only write little-endian TIFF files.
			buf[off+0] = byte(v1)
			buf[off+1] = byte(v1 >> 8)
			off += 2
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func encodeRGBA(w io.Writer, pix []uint8, dx, dy, stride int, predictor bool) error {
	if !predictor {
		return writePix(w, pix, dy, dx*4, stride)
	}
	buf := make([]byte, dx*4)
	for y := 0; y < dy; y++ {
		min := y*stride + 0
		max := y*stride + dx*4
		off := 0
		var r0, g0, b0, a0 uint8
		for i := min; i < max; i += 4 {
			r1, g1, b1, a1 := pix[i+0], pix[i+1], pix[i+2], pix[i+3]
			buf[off+0] = r1 - r0
			buf[off+1] = g1 - g0
			buf[off+2] = b1 - b0
			buf[off+3] = a1 - a0
			off += 4
			r0, g0, b0, a0 = r1, g1, b1, a1
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func encodeRGBA64(w io.Writer, pix []uint8, dx, dy, stride int, predictor bool) error {
	buf := make([]byte, dx*8)
	for y := 0; y < dy; y++ {
		min := y*stride + 0
		max := y*stride + dx*8
		off := 0
		var r0, g0, b0, a0 uint16
		for i := min; i < max; i += 8 {
			// An image.RGBA64's Pix is in big-endian order.
			r1 := uint16(pix[i+0])<<8 | uint16(pix[i+1])
			g1 := uint16(pix[i+2])<<8 | uint16(pix[i+3])
			b1 := uint16(pix[i+4])<<8 | uint16(pix[i+5])
			a1 := uint16(pix[i+6])<<8 | uint16(pix[i+7])
			if predictor {
				r0, r1 = r1, r1-r0
				g0, g1 = g1, g1-g0
				b0, b1 = b1, b1-b0
				a0, a1 = a1, a1-a0
			}
			// We only write little-endian TIFF files.
			buf[off+0] = byte(r1)
			buf[off+1] = byte(r1 >> 8)
			buf[off+2] = byte(g1)
			buf[off+3] = byte(g1 >> 8)
			buf[off+4] = byte(b1)
			buf[off+5] = byte(b1 >> 8)
			buf[off+6] = byte(a1)
			buf[off+7] = byte(a1 >> 8)
			off += 8
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

func encode(w io.Writer, m image.Image, predictor bool) error {
	bounds := m.Bounds()
	buf := make([]byte, 4*bounds.Dx())
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		off := 0
		if predictor {
			var r0, g0, b0, a0 uint8
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, a := m.At(x, y).RGBA()
				r1 := uint8(r >> 8)
				g1 := uint8(g >> 8)
				b1 := uint8(b >> 8)
				a1 := uint8(a >> 8)
				buf[off+0] = r1 - r0
				buf[off+1] = g1 - g0
				buf[off+2] = b1 - b0
				buf[off+3] = a1 - a0
				off += 4
				r0, g0, b0, a0 = r1, g1, b1, a1
			}
		} else {
			for x := bounds.Min.X; x < bounds.Max.X; x++ {
				r, g, b, a := m.At(x, y).RGBA()
				buf[off+0] = uint8(r >> 8)
				buf[off+1] = uint8(g >> 8)
				buf[off+2] = uint8(b >> 8)
				buf[off+3] = uint8(a >> 8)
				off += 4
			}
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// writePix writes the internal byte array of an image to w. It is less general
// but much faster then encode. writePix is used when pix directly
// corresponds to one of the TIFF image types.
func writePix(w io.Writer, pix []byte, nrows, length, stride int) error {
	if length == stride {
		_, err := w.Write(pix[:nrows*length])
		return err
	}
	for ; nrows > 0; nrows-- {
		if _, err := w.Write(pix[:length]); err != nil {
			return err
		}
		pix = pix[stride:]
	}
	return nil
}

func writeIFD(w io.Writer, ifdOffset int, d []ifdEntry) error {
	var buf [ifdLen]byte
	// Make space for "pointer area" containing IFD entry data
	// longer than 4 bytes.
	parea := make([]byte, 1024)
	pstart := ifdOffset + ifdLen*len(d) + 6
	var o int // Current offset in parea.

	// The IFD has to be written with the tags in ascending order.
	sort.Sort(byTag(d))

	// Write the number of entries in this IFD.
	if err := binary.Write(w, enc, uint16(len(d))); err != nil {
		return err
	}
	for _, ent := range d {
		enc.PutUint16(buf[0:2], uint16(ent.tag))
		enc.PutUint16(buf[2:4], uint16(ent.datatype))
		count := uint32(len(ent.data))
		if ent.datatype == dtRational {
			count /= 2
		}
		enc.PutUint32(buf[4:8], count)
		datalen := int(count * lengths[ent.datatype])
		if datalen <= 4 {
			ent.putData(buf[8:12])
		} else {
			if (o + datalen) > len(parea) {
				newlen := len(parea) + 1024
				for (o + datalen) > newlen {
					newlen += 1024
				}
				newarea := make([]byte, newlen)
				copy(newarea, parea)
				parea = newarea
			}
			ent.putData(parea[o : o+datalen])
			enc.PutUint32(buf[8:12], uint32(pstart+o))
			o += datalen
		}
		if _, err := w.Write(buf[:]); err != nil {
			return err
		}
	}
	// The IFD ends with the offset of the next IFD in the file,
	// or zero if it is the last one (page 14).
	if err := binary.Write(w, enc, uint32(0)); err != nil {
		return err
	}
	_, err := w.Write(parea[:o])
	return err
}

// Options are the encoding parameters.
type Options struct {
	// Compression is the type of compression used.
	Compression CompressionType
	// Predictor determines whether a differencing predictor is used;
	// if true, instead of each pixel's color, the color difference to the
	// preceding one is saved. This improves the compression for certain
	// types of images and compressors. For example, it works well for
	// photos with Deflate compression.
	Predictor bool
	// Tiled selects tiled output instead of the default stripped output.
	// When true, the image is written as an array of independently
	// addressable tiles using the TileWidth, TileLength, TileOffsets and
	// TileByteCounts tags, and the strip tags are omitted. When false,
	// the encoder writes a single strip as before.
	//
	// Tiled output currently supports 8-bit *image.Gray and *image.RGBA
	// images with Uncompressed compression. TileWidth and TileLength must
	// both be positive multiples of 16.
	Tiled bool
	// TileWidth is the width in pixels of each tile. It is only used when
	// Tiled is true and must be a positive multiple of 16.
	TileWidth int
	// TileLength is the height in pixels of each tile. It is only used when
	// Tiled is true and must be a positive multiple of 16.
	TileLength int
}

// Encode writes the image m to w. opt determines the options used for
// encoding, such as the compression type. If opt is nil, an uncompressed
// image is written.
func Encode(w io.Writer, m image.Image, opt *Options) error {
	d := m.Bounds().Size()

	if d.X == 0 || d.Y == 0 {
		return errors.New("tiff: zero-size image")
	}

	compression := uint32(cNone)
	predictor := false
	if opt != nil {
		compression = opt.Compression.specValue()
		// The predictor field is only used with LZW. See page 64 of the spec.
		predictor = opt.Predictor && compression == cLZW
		if opt.Tiled {
			return encodeTiled(w, m, opt.TileWidth, opt.TileLength, compression)
		}
	}

	_, err := io.WriteString(w, leHeader)
	if err != nil {
		return err
	}

	// Compressed data is written into a buffer first, so that we
	// know the compressed size.
	var buf bytes.Buffer
	// dst holds the destination for the pixel data of the image --
	// either w or a writer to buf.
	var dst io.Writer
	// imageLen is the length of the pixel data in bytes.
	// The offset of the IFD is imageLen + 8 header bytes.
	var imageLen int

	switch compression {
	case cNone:
		dst = w
		// Write IFD offset before outputting pixel data.
		switch m.(type) {
		case *image.Paletted:
			imageLen = d.X * d.Y * 1
		case *image.Gray:
			imageLen = d.X * d.Y * 1
		case *image.Gray16:
			imageLen = d.X * d.Y * 2
		case *image.RGBA64:
			imageLen = d.X * d.Y * 8
		case *image.NRGBA64:
			imageLen = d.X * d.Y * 8
		default:
			imageLen = d.X * d.Y * 4
		}
		err = binary.Write(w, enc, uint32(imageLen+8))
		if err != nil {
			return err
		}
	case cDeflate:
		dst = zlib.NewWriter(&buf)
	default:
		return errors.New("tiff: unsupported compression")
	}

	pr := uint32(prNone)
	photometricInterpretation := uint32(pRGB)
	samplesPerPixel := uint32(4)
	bitsPerSample := []uint32{8, 8, 8, 8}
	extraSamples := uint32(0)
	colorMap := []uint32{}

	if predictor {
		pr = prHorizontal
	}
	switch m := m.(type) {
	case *image.Paletted:
		photometricInterpretation = pPaletted
		samplesPerPixel = 1
		bitsPerSample = []uint32{8}
		colorMap = make([]uint32, 256*3)
		for i := 0; i < 256 && i < len(m.Palette); i++ {
			r, g, b, _ := m.Palette[i].RGBA()
			colorMap[i+0*256] = uint32(r)
			colorMap[i+1*256] = uint32(g)
			colorMap[i+2*256] = uint32(b)
		}
		err = encodeGray(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.Gray:
		photometricInterpretation = pBlackIsZero
		samplesPerPixel = 1
		bitsPerSample = []uint32{8}
		err = encodeGray(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.Gray16:
		photometricInterpretation = pBlackIsZero
		samplesPerPixel = 1
		bitsPerSample = []uint32{16}
		err = encodeGray16(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.NRGBA:
		extraSamples = 2 // Unassociated alpha.
		err = encodeRGBA(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.NRGBA64:
		extraSamples = 2 // Unassociated alpha.
		bitsPerSample = []uint32{16, 16, 16, 16}
		err = encodeRGBA64(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.RGBA:
		extraSamples = 1 // Associated alpha.
		err = encodeRGBA(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	case *image.RGBA64:
		extraSamples = 1 // Associated alpha.
		bitsPerSample = []uint32{16, 16, 16, 16}
		err = encodeRGBA64(dst, m.Pix, d.X, d.Y, m.Stride, predictor)
	default:
		extraSamples = 1 // Associated alpha.
		err = encode(dst, m, predictor)
	}
	if err != nil {
		return err
	}

	if compression != cNone {
		if err = dst.(io.Closer).Close(); err != nil {
			return err
		}
		imageLen = buf.Len()
		if err = binary.Write(w, enc, uint32(imageLen+8)); err != nil {
			return err
		}
		if _, err = buf.WriteTo(w); err != nil {
			return err
		}
	}

	ifd := []ifdEntry{
		{tImageWidth, dtShort, []uint32{uint32(d.X)}},
		{tImageLength, dtShort, []uint32{uint32(d.Y)}},
		{tBitsPerSample, dtShort, bitsPerSample},
		{tCompression, dtShort, []uint32{compression}},
		{tPhotometricInterpretation, dtShort, []uint32{photometricInterpretation}},
		{tStripOffsets, dtLong, []uint32{8}},
		{tSamplesPerPixel, dtShort, []uint32{samplesPerPixel}},
		{tRowsPerStrip, dtShort, []uint32{uint32(d.Y)}},
		{tStripByteCounts, dtLong, []uint32{uint32(imageLen)}},
		// There is currently no support for storing the image
		// resolution, so give a bogus value of 72x72 dpi.
		{tXResolution, dtRational, []uint32{72, 1}},
		{tYResolution, dtRational, []uint32{72, 1}},
		{tResolutionUnit, dtShort, []uint32{resPerInch}},
	}
	if pr != prNone {
		ifd = append(ifd, ifdEntry{tPredictor, dtShort, []uint32{pr}})
	}
	if len(colorMap) != 0 {
		ifd = append(ifd, ifdEntry{tColorMap, dtShort, colorMap})
	}
	if extraSamples > 0 {
		ifd = append(ifd, ifdEntry{tExtraSamples, dtShort, []uint32{extraSamples}})
	}

	return writeIFD(w, imageLen+8, ifd)
}

// mul64 returns x*y and reports whether it fits in a uint64 without
// overflow. The inputs are assumed to be non-negative.
func mul64(x, y int64) (int64, bool) {
	hi, lo := bits.Mul64(uint64(x), uint64(y))
	if hi != 0 || lo > math.MaxInt64 {
		return 0, false
	}
	return int64(lo), true
}

// ifdIntEntry returns an IFD entry holding a single integer, using the
// smallest datatype that can represent v.
func ifdIntEntry(tag int, v uint32) ifdEntry {
	if v <= math.MaxUint16 {
		return ifdEntry{tag, dtShort, []uint32{v}}
	}
	return ifdEntry{tag, dtLong, []uint32{v}}
}

// encodeTiled writes m to w as an uncompressed tiled TIFF.
//
// All parameters are validated before anything is written, so an error never
// leaves a partially written, seemingly valid file behind.
func encodeTiled(w io.Writer, m image.Image, tileWidth, tileLength int, compression uint32) error {
	// Tiles are written one at a time without a compression stage, so only
	// uncompressed data is supported.
	if compression != cNone {
		return UnsupportedError("tiled output with the selected compression")
	}
	// The TIFF specification (p. 38-39) requires tile widths and lengths to
	// be positive multiples of 16.
	if tileWidth <= 0 || tileLength <= 0 || tileWidth%16 != 0 || tileLength%16 != 0 {
		return errors.New("tiff: TileWidth and TileLength must be positive multiples of 16")
	}

	bounds := m.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width <= 0 || height <= 0 {
		return errors.New("tiff: zero-size image")
	}
	if width > math.MaxInt32 || height > math.MaxInt32 {
		return errors.New("tiff: image is too large")
	}

	// Tiled output is implemented for the 8-bit Gray and RGBA layouts.
	var (
		gray          *image.Gray
		rgba          *image.RGBA
		bpp           int64 // bytes per pixel
		bitsPerSample []uint32
		photometric   uint32
		extraSamples  uint32
	)
	switch img := m.(type) {
	case *image.Gray:
		gray = img
		bpp = 1
		bitsPerSample = []uint32{8}
		photometric = pBlackIsZero
	case *image.RGBA:
		rgba = img
		bpp = 4
		bitsPerSample = []uint32{8, 8, 8, 8}
		photometric = pRGB
		extraSamples = 1 // Associated alpha.
	default:
		return UnsupportedError("tiled output for this image type")
	}

	// Mirror the decoder's tile sanity checks so that a file we produce can
	// always be read back by Decode. A tile that is larger than the image is
	// only acceptable as the single padded tile at the edge, and the excess
	// must not be more than the 16-pixel padding.
	if (tileWidth > width && tileWidth-width > 16) || (tileLength > height && tileLength-height > 16) {
		if tileWidth > 1024 || tileLength > 1024 {
			return FormatError("tile size exceeds image size")
		}
	}

	// All geometry math below stays in int64 so that pathological tile or
	// image dimensions cannot overflow.
	tilesAcross := int64((width + tileWidth - 1) / tileWidth)
	tilesDown := int64((height + tileLength - 1) / tileLength)
	numTiles, ok := mul64(tilesAcross, tilesDown)
	if !ok || numTiles > math.MaxInt32/4 {
		return errors.New("tiff: too many tiles")
	}

	tileBytes, ok := mul64(int64(tileWidth), int64(tileLength))
	if !ok {
		return errors.New("tiff: tile size is too large")
	}
	if tileBytes, ok = mul64(tileBytes, bpp); !ok {
		return errors.New("tiff: tile size is too large")
	}
	totalBytes, ok := mul64(numTiles, tileBytes)
	if !ok {
		return errors.New("tiff: tiled image is too large")
	}

	// The IFD follows the header and the tile data. The pointer area holds
	// the TileOffsets and TileByteCounts arrays (4 bytes per tile each), the
	// two resolution rationals and, for RGBA, the BitsPerSample array.
	// Account for the whole IFD in the 32-bit file-offset budget that the
	// classic TIFF format is limited to.
	ifdOffset := totalBytes + 8
	fileEnd := ifdOffset + int64(ifdLen)*16 + 6 + 8*numTiles + 32
	if fileEnd > math.MaxUint32 || fileEnd > math.MaxInt {
		return errors.New("tiff: tiled image is too large")
	}

	// From here on the inputs are valid and every subsequent failure is an
	// I/O error rather than a malformed output.
	if _, err := io.WriteString(w, leHeader); err != nil {
		return err
	}
	if _, err := w.Write(enc.AppendUint32(nil, uint32(ifdOffset))); err != nil {
		return err
	}

	tile := make([]byte, int(tileBytes)) // zero-filled, which provides edge padding.
	offsets := make([]uint32, numTiles)
	byteCounts := make([]uint32, numTiles)
	rowBytes := int(tileWidth) * int(bpp)

	var ti int64
	for j := int64(0); j < tilesDown; j++ {
		jy, ok := mul64(j, int64(tileLength))
		if !ok || jy > math.MaxInt64-int64(bounds.Min.Y) {
			return errors.New("tiff: tile index is too large")
		}
		y0 := bounds.Min.Y + int(jy)
		for i := int64(0); i < tilesAcross; i++ {
			ix, ok := mul64(i, int64(tileWidth))
			if !ok || ix > math.MaxInt64-int64(bounds.Min.X) {
				return errors.New("tiff: tile index is too large")
			}
			x0 := bounds.Min.X + int(ix)
			clear(tile)
			if gray != nil {
				fillGrayTile(tile, gray, x0, y0, tileWidth, tileLength)
			} else {
				fillRGBATile(tile, rgba, x0, y0, tileWidth, tileLength, rowBytes)
			}
			offset, ok := mul64(ti, tileBytes)
			if !ok || offset+8 > math.MaxUint32 {
				return errors.New("tiff: tiled image is too large")
			}
			offsets[ti] = uint32(offset + 8)
			byteCounts[ti] = uint32(tileBytes)
			if _, err := w.Write(tile); err != nil {
				return err
			}
			ti++
		}
	}

	// Entries are listed in ascending tag order, as required by the TIFF
	// specification; writeIFD also sorts defensively.
	ifd := []ifdEntry{
		ifdIntEntry(tImageWidth, uint32(width)),
		ifdIntEntry(tImageLength, uint32(height)),
		{tBitsPerSample, dtShort, bitsPerSample},
		ifdIntEntry(tCompression, compression),
		ifdIntEntry(tPhotometricInterpretation, photometric),
		ifdIntEntry(tSamplesPerPixel, uint32(bpp)),
		ifdIntEntry(tTileWidth, uint32(tileWidth)),
		ifdIntEntry(tTileLength, uint32(tileLength)),
		{tTileOffsets, dtLong, offsets},
		{tTileByteCounts, dtLong, byteCounts},
		// There is currently no support for storing the image
		// resolution, so give a bogus value of 72x72 dpi.
		{tXResolution, dtRational, []uint32{72, 1}},
		{tYResolution, dtRational, []uint32{72, 1}},
		{tResolutionUnit, dtShort, []uint32{resPerInch}},
	}
	if extraSamples > 0 {
		ifd = append(ifd, ifdEntry{tExtraSamples, dtShort, []uint32{extraSamples}})
	}

	return writeIFD(w, int(ifdOffset), ifd)
}

// fillGrayTile copies the overlapping region of img into a tileWidth by
// tileLength single-sample tile whose rows are tileWidth bytes long. Pixels
// outside the image are left zero in the pre-cleared tile.
func fillGrayTile(tile []byte, img *image.Gray, x0, y0, tileWidth, tileLength int) {
	for ty := 0; ty < tileLength; ty++ {
		y := y0 + ty
		if y < img.Rect.Min.Y || y >= img.Rect.Max.Y {
			continue
		}
		for tx := 0; tx < tileWidth; tx++ {
			x := x0 + tx
			if x < img.Rect.Min.X || x >= img.Rect.Max.X {
				continue
			}
			tile[ty*tileWidth+tx] = img.GrayAt(x, y).Y
		}
	}
}

// fillRGBATile copies the overlapping region of img into a tileWidth by
// tileLength four-sample tile whose rows are rowBytes bytes long. Pixels
// outside the image are left zero in the pre-cleared tile.
func fillRGBATile(tile []byte, img *image.RGBA, x0, y0, tileWidth, tileLength, rowBytes int) {
	for ty := 0; ty < tileLength; ty++ {
		y := y0 + ty
		if y < img.Rect.Min.Y || y >= img.Rect.Max.Y {
			continue
		}
		for tx := 0; tx < tileWidth; tx++ {
			x := x0 + tx
			if x < img.Rect.Min.X || x >= img.Rect.Max.X {
				continue
			}
			c := img.RGBAAt(x, y)
			off := ty*rowBytes + tx*4
			tile[off+0] = c.R
			tile[off+1] = c.G
			tile[off+2] = c.B
			tile[off+3] = c.A
		}
	}
}
