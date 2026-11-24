// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"image"
	"sort"
)

// syntheticTIFF describes an image to be serialized by buildTIFF. The builder
// supports the strip/tile organization, compressions, predictor and planar
// configuration exercised by the DecodeRegion tests.
type syntheticTIFF struct {
	width, height int
	// mode selects the photometric interpretation and sample layout.
	mode imageMode
	bpp  int // bits per sample: 1, 8 or 16

	compression uint
	predictor   uint // prNone or prHorizontal
	planar      uint // pcChunky or pcPlanar

	tileWidth, tileLength int // zero means strips
	rowsPerStrip          int

	// pix holds the canonical, uncompressed pixel data in image space,
	// stored in the native order of the matching *image type.
	pix []byte
}

// buildTIFF serializes t into a little-endian TIFF byte stream. The returned
// data can be decoded back by Decode and DecodeRegion.
func buildTIFF(t *syntheticTIFF) []byte {
	enc := binary.LittleEndian
	var entries []ifdEntry
	add := func(tag int, data ...uint32) {
		entries = append(entries, ifdEntry{tag: tag, datatype: dtShort, data: data})
	}
	addLong := func(tag int, data ...uint32) {
		entries = append(entries, ifdEntry{tag: tag, datatype: dtLong, data: data})
	}

	add(tImageWidth, uint32(t.width))
	add(tImageLength, uint32(t.height))

	spp := samplesPerPixel(t.mode)
	bps := make([]uint32, spp)
	for i := range bps {
		bps[i] = uint32(t.bpp)
	}
	entries = append(entries, ifdEntry{tag: tBitsPerSample, datatype: dtShort, data: bps})
	add(tCompression, uint32(t.compression))
	add(tPhotometricInterpretation, photometric(t.mode))
	if t.predictor != 0 {
		add(tPredictor, uint32(t.predictor))
	}
	add(tPlanarConfiguration, uint32(t.planar))
	if t.mode == mRGBA {
		add(tExtraSamples, 1) // associated alpha
	}
	if t.mode == mNRGBA {
		add(tExtraSamples, 2) // unassociated alpha
	}

	// Compute the block grid.
	blockW, blockH, across, down, isTile := t.blockGrid()
	spatial := across * down
	count := spatial
	if t.planar == pcPlanar {
		count = spatial * spp
	}

	// Build every block's payload.
	type block struct {
		off, n uint32
	}
	blocks := make([]block, count)
	var payload bytes.Buffer
	payload.Write(make([]byte, 8)) // reserve space for the 8-byte TIFF header

	bpsBytes := t.bpp / 8
	if bpsBytes == 0 {
		bpsBytes = 1
	}

	for plane := 0; plane < spp; plane++ {
		if t.planar != pcPlanar {
			plane = -1 // marker: interleaved
		}
		for i := 0; i < across; i++ {
			for j := 0; j < down; j++ {
				blk := t.blockRect(i, j, blockW, blockH, isTile)
				raw := t.blockPixels(blk, plane, spp, bpsBytes)
				if t.predictor == prHorizontal && t.compression != cNone {
					raw = applyHorizontalPredictor(raw, blk.Dx(), blk.Dy(), spp, bpsBytes, t.planar)
				}
				encoded := encodeBlock(raw, t.compression)

				idx := j*across + i
				if t.planar == pcPlanar {
					idx = plane*spatial + idx
				}
				off := uint32(payload.Len())
				payload.Write(encoded)
				blocks[idx] = block{off, uint32(len(encoded))}
			}
		}
		if t.planar != pcPlanar {
			break
		}
	}

	offsets := make([]uint32, count)
	sizes := make([]uint32, count)
	for k, b := range blocks {
		offsets[k] = b.off
		sizes[k] = b.n
	}
	if isTile {
		add(tTileWidth, uint32(blockW))
		add(tTileLength, uint32(blockH))
		addLong(tTileOffsets, offsets...)
		addLong(tTileByteCounts, sizes...)
	} else {
		rps := t.rowsPerStrip
		if rps == 0 {
			rps = t.height
		}
		add(tRowsPerStrip, uint32(rps))
		addLong(tStripOffsets, offsets...)
		addLong(tStripByteCounts, sizes...)
	}

	// payload already reserves the 8-byte TIFF header at its start, so the
	// recorded block offsets are absolute file offsets.
	ifdOffset := uint32(payload.Len())
	sort.Sort(byTag(entries))
	var out bytes.Buffer
	writeIFDTo(&out, int(ifdOffset), entries)
	payload.Write(out.Bytes())

	data := payload.Bytes()
	copy(data[0:8], []byte(leHeader))
	// Patch the IFD pointer in the 8-byte header.
	enc.PutUint32(data[4:8], ifdOffset)
	return data
}

func (t *syntheticTIFF) blockGrid() (blockW, blockH, across, down int, isTile bool) {
	if t.tileWidth != 0 {
		blockW, blockH = t.tileWidth, t.tileLength
		isTile = true
	} else {
		blockW = t.width
		blockH = t.rowsPerStrip
		if blockH == 0 {
			blockH = t.height
		}
	}
	if blockW != 0 {
		across = (t.width + blockW - 1) / blockW
	}
	if blockH != 0 {
		down = (t.height + blockH - 1) / blockH
	}
	return
}

func (t *syntheticTIFF) blockRect(i, j, blockW, blockH int, isTile bool) image.Rectangle {
	x0 := i * blockW
	y0 := j * blockH
	x1, y1 := x0+blockW, y0+blockH
	if !isTile {
		// Strips are not padded to the block width; trim partial edges.
		if x1 > t.width {
			x1 = t.width
		}
		if y1 > t.height {
			y1 = t.height
		}
	}
	return image.Rect(x0, y0, x1, y1)
}

// blockPixels extracts the block data. Tiles are padded to the full block
// size on the right and bottom; the padding samples are emitted as zero bytes
// (the decoder never materializes them). For planar data, plane is the sample
// index (>= 0); for chunky data it is -1.
func (t *syntheticTIFF) blockPixels(blk image.Rectangle, plane, spp, bpsBytes int) []byte {
	var buf bytes.Buffer
	stride := t.pixStride()
	rowLen := blk.Dx() * spp * bpsBytes
	if t.planar == pcPlanar {
		rowLen = blk.Dx() * bpsBytes
	}
	for y := blk.Min.Y; y < blk.Max.Y; y++ {
		rowStart := buf.Len()
		if y < t.height {
			for x := blk.Min.X; x < minInt(blk.Max.X, t.width); x++ {
				if t.planar == pcPlanar {
					off := y*stride + x*spp*bpsBytes + plane*bpsBytes
					buf.Write(t.pix[off : off+bpsBytes])
				} else {
					off := y*stride + x*spp*bpsBytes
					buf.Write(t.pix[off : off+spp*bpsBytes])
				}
			}
		}
		// Pad short rows (right edge) and fully virtual rows (bottom edge).
		for buf.Len()-rowStart < rowLen {
			buf.WriteByte(0)
		}
	}
	return buf.Bytes()
}

func (t *syntheticTIFF) pixStride() int {
	spp := samplesPerPixel(t.mode)
	if t.mode == mRGB {
		spp = 3
	}
	bps := t.bpp / 8
	if bps == 0 {
		bps = 1
	}
	return t.width * spp * bps
}

// applyHorizontalPredictor replaces every sample by its difference from the
// preceding sample in the same row (per plane for planar data).
func applyHorizontalPredictor(row []byte, blkWidth, blkHeight, spp, bpsBytes int, planar uint) []byte {
	out := append([]byte(nil), row...)
	samplesPerPixel := spp
	if planar == pcPlanar {
		samplesPerPixel = 1
	}
	rowBytes := blkWidth * samplesPerPixel * bpsBytes
	for y := 0; y < blkHeight; y++ {
		base := y * rowBytes
		for x := 1; x < blkWidth; x++ {
			for s := 0; s < samplesPerPixel; s++ {
				off := base + (x*samplesPerPixel+s)*bpsBytes
				prev := base + ((x-1)*samplesPerPixel+s)*bpsBytes
				if bpsBytes == 2 {
					cur := binary.LittleEndian.Uint16(out[off : off+2])
					pv := binary.LittleEndian.Uint16(out[prev : prev+2])
					binary.LittleEndian.PutUint16(out[off:off+2], cur-pv)
				} else {
					out[off] = out[off] - out[prev]
				}
			}
		}
	}
	return out
}

func encodeBlock(raw []byte, compression uint) []byte {
	switch compression {
	case cNone, 0:
		return append([]byte(nil), raw...)
	case cDeflate, cDeflateOld:
		var b bytes.Buffer
		w := zlib.NewWriter(&b)
		w.Write(raw)
		w.Close()
		return b.Bytes()
	case cPackBits:
		return packBits(raw)
	default:
		panic("buildTIFF: unsupported compression for synthetic data")
	}
}

// packBits encodes data with the PackBits scheme.
func packBits(data []byte) []byte {
	var out bytes.Buffer
	for i := 0; i < len(data); {
		// Find a run of identical bytes.
		run := 1
		for i+run < len(data) && data[i+run] == data[i] && run < 128 {
			run++
		}
		if run >= 3 {
			out.WriteByte(byte(1 - run))
			out.WriteByte(data[i])
			i += run
			continue
		}
		// Gather a literal run.
		start := i
		for i < len(data) {
			next := 1
			for i+next < len(data) && data[i+next] == data[i] {
				next++
			}
			if next >= 3 || i-start >= 127 {
				break
			}
			i++
		}
		lit := data[start:i]
		out.WriteByte(byte(len(lit) - 1))
		out.Write(lit)
	}
	return out.Bytes()
}

func samplesPerPixel(m imageMode) int {
	switch m {
	case mRGB:
		return 3
	case mRGBA, mNRGBA:
		return 4
	default:
		return 1
	}
}

func photometric(m imageMode) uint32 {
	switch m {
	case mRGB, mRGBA, mNRGBA:
		return pRGB
	case mGray:
		return pBlackIsZero
	case mGrayInvert:
		return pWhiteIsZero
	case mPaletted:
		return pPaletted
	default:
		return pBlackIsZero
	}
}

// writeIFDTo writes an IFD plus its pointer area at ifdOffset into w. It is a
// variant of writeIFD that works against an in-memory buffer.
func writeIFDTo(w *bytes.Buffer, ifdOffset int, entries []ifdEntry) {
	var hdr [2]byte
	binary.LittleEndian.PutUint16(hdr[:], uint16(len(entries)))
	w.Write(hdr[:])

	var parea bytes.Buffer
	pstart := ifdOffset + 2 + ifdLen*len(entries) + 4
	var buf [ifdLen]byte
	for _, e := range entries {
		enc.PutUint16(buf[0:2], uint16(e.tag))
		enc.PutUint16(buf[2:4], uint16(e.datatype))
		count := uint32(len(e.data))
		enc.PutUint32(buf[4:8], count)
		datalen := int(count * lengths[e.datatype])
		if datalen <= 4 {
			e.putData(buf[8:12])
			w.Write(buf[:])
			continue
		}
		off := parea.Len()
		tmp := make([]byte, datalen)
		e.putData(tmp)
		parea.Write(tmp)
		enc.PutUint32(buf[8:12], uint32(pstart+off))
		w.Write(buf[:])
	}
	var next [4]byte
	w.Write(next[:])
	w.Write(parea.Bytes())
}
