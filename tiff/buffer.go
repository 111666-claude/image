// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"io"
	"math"
)

// fillChunkSize bounds how much b.fill allocates and reads at once, so that a
// hostile offset (for example a SubIFD offset near 2^32 in a tiny input) makes
// fill read to the end of the input and fail, rather than allocating the whole
// offset up front.
const fillChunkSize = 10 << 20 // 10M

// buffer buffers an io.Reader to satisfy io.ReaderAt.
type buffer struct {
	r   io.Reader
	buf []byte
}

// fill reads data from b.r until the buffer contains at least end bytes.
func (b *buffer) fill(end int) error {
	m := len(b.buf)
	for m < end {
		next := end - m
		if next > fillChunkSize {
			next = fillChunkSize
		}
		if m+next > cap(b.buf) {
			newcap := cap(b.buf) * 2
			if newcap < m+next {
				newcap = m + next
			}
			newbuf := make([]byte, m, newcap)
			copy(newbuf, b.buf[:m])
			b.buf = newbuf
		}
		b.buf = b.buf[:m+next]
		n, err := io.ReadFull(b.r, b.buf[m:m+next])
		m += n
		b.buf = b.buf[:m]
		if err != nil {
			return err
		}
	}
	return nil
}

func (b *buffer) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 || off > int64(math.MaxInt) {
		return 0, io.ErrUnexpectedEOF
	}
	o := int(off)
	end := o + len(p)
	if end < o {
		return 0, io.ErrUnexpectedEOF
	}

	err := b.fill(end)
	if o > len(b.buf) {
		return 0, io.EOF
	}
	n := copy(p, b.buf[o:])
	if err == nil && n < len(p) {
		err = io.EOF
	}
	return n, err
}

// Slice returns a slice of the underlying buffer. The slice contains
// n bytes starting at offset off.
func (b *buffer) Slice(off, n int) ([]byte, error) {
	end := off + n
	if err := b.fill(end); err != nil {
		return nil, err
	}
	return b.buf[off:end], nil
}

// newReaderAt converts an io.Reader into an io.ReaderAt.
func newReaderAt(r io.Reader) io.ReaderAt {
	if ra, ok := r.(io.ReaderAt); ok {
		return ra
	}
	return &buffer{
		r:   r,
		buf: make([]byte, 0, 1024),
	}
}
