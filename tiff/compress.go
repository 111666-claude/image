// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bufio"
	"errors"
	"io"
)

type byteReader interface {
	io.Reader
	io.ByteReader
}

// unpackBits decodes the PackBits-compressed data in src and returns the
// uncompressed data. The output size is limited to lim bytes to prevent
// decompression bombs.
//
// The PackBits compression format is described in section 9 (p. 42)
// of the TIFF spec.
func unpackBits(r io.Reader, lim int64) ([]byte, error) {
	buf := make([]byte, 128)
	dst := make([]byte, 0, 1024)
	br, ok := r.(byteReader)
	if !ok {
		br = bufio.NewReader(r)
	}

	for {
		b, err := br.ReadByte()
		if err != nil {
			if err == io.EOF {
				return dst, nil
			}
			return nil, err
		}
		code := int(int8(b))
		switch {
		case code >= 0:
			n, err := io.ReadFull(br, buf[:code+1])
			if err != nil {
				return nil, err
			}
			dst = append(dst, buf[:n]...)
		case code == -128:
			// No-op.
		default:
			if b, err = br.ReadByte(); err != nil {
				return nil, err
			}
			for j := 0; j < 1-code; j++ {
				buf[j] = b
			}
			dst = append(dst, buf[:1-code]...)
		}
		if int64(len(dst)) > lim {
			return nil, FormatError("PackBits: decompressed data too large")
		}
	}
}

// packBitsMaxRun is the longest run that a single PackBits header can
// describe: header byte -127 encodes 128 repeated bytes.
const packBitsMaxRun = 128

// packBitsWriter implements PackBits compression (TIFF spec, section 9,
// p. 42). A PackBits stream is a sequence of header/payload pairs:
//
//   - n in [0, 127]:  the next n+1 bytes are copied literally.
//   - n in [-127, -1]: the next byte is repeated 1-n times.
//   - n == -128:       no-op.
//
// Each scanline is compressed independently: a run of equal bytes is never
// allowed to cross a row boundary, as required by the specification. The
// writer buffers all uncompressed data and emits the compressed stream on
// Close, so callers may use arbitrary Write sizes.
type packBitsWriter struct {
	w      io.Writer
	rowLen int // Length of one scanline in uncompressed bytes.
	buf    []byte
	err    error
}

// newPackBitsWriter returns a PackBits compressor writing to w. rowLen is
// the length in bytes of one image scanline; every compressed row decodes
// to exactly rowLen bytes.
func newPackBitsWriter(w io.Writer, rowLen int) *packBitsWriter {
	return &packBitsWriter{
		w:      w,
		rowLen: rowLen,
	}
}

func (p *packBitsWriter) Write(b []byte) (int, error) {
	if p.err != nil {
		return 0, p.err
	}
	p.buf = append(p.buf, b...)
	return len(b), nil
}

// Close compresses the buffered scanlines and writes them to the underlying
// writer. It returns an error if the buffered data does not contain a whole
// number of scanlines, or if the underlying writer returns one.
func (p *packBitsWriter) Close() error {
	if p.err != nil {
		return p.err
	}
	if p.rowLen <= 0 || len(p.buf)%p.rowLen != 0 {
		p.err = errors.New("tiff: PackBits input is not a whole number of scanlines")
		return p.err
	}
	out := make([]byte, 0, len(p.buf))
	for len(p.buf) > 0 {
		out = encodePackBitsRow(out, p.buf[:p.rowLen])
		p.buf = p.buf[p.rowLen:]
	}
	if _, err := p.w.Write(out); err != nil {
		p.err = err
		return err
	}
	p.buf = nil
	return nil
}

// encodePackBitsRow appends the PackBits encoding of one scanline, src, to
// dst and returns the extended buffer. Runs never extend beyond the end of
// src, so scanlines stay independent of each other.
func encodePackBitsRow(dst, src []byte) []byte {
	for i := 0; i < len(src); {
		// Look for a run of at least 2 equal bytes; shorter repeats are
		// emitted as part of a literal run.
		run := 1
		for i+run < len(src) && src[i+run] == src[i] && run < packBitsMaxRun {
			run++
		}
		if run >= 2 {
			dst = append(dst, byte(-(int8(run - 1))), src[i])
			i += run
			continue
		}

		// Gather a literal run, stopping at the start of a repeat run of
		// at least 2 equal bytes or at the 128-byte limit for one header.
		start := i
		for i < len(src) && i-start < packBitsMaxRun {
			if i+1 < len(src) && src[i] == src[i+1] {
				break
			}
			i++
		}
		lit := src[start:i]
		dst = append(dst, byte(len(lit)-1))
		dst = append(dst, lit...)
	}
	return dst
}
