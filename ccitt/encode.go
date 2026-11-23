// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ccitt

import (
	"context"
	"errors"
	"io"
)

var (
	errEncoderClosed    = errors.New("ccitt: encoder already closed")
	errInvalidRowLength = errors.New("ccitt: invalid row length")
)

// eolCode is the 12-bit End-of-Line code "000000000001".
var eolCode = bitString{0x001, 12}

// verticalModes maps a vertical mode adjustment (a1 - b1, in the range
// [-3, +3]) to the mode* constant selecting a modeEncodeTable entry.
var verticalModes = [7]uint32{
	modeVL3, modeVL2, modeVL1, modeV0, modeVR1, modeVR2, modeVR3,
}

// An Encoder encodes a CCITT Group3 or Group4 (fax) image, one row at a
// time, to an underlying io.Writer.
//
// Rows are supplied by WriteRow. Close writes the end-of-facsimile-block
// codes (the RTC for Group3, the EOFB for Group4) and flushes any buffered
// bits, so that the encoded stream can be decoded by NewReader with either
// the exact image height or AutoDetectHeight.
type Encoder struct {
	ctx context.Context
	bw  bitWriter

	subFormat SubFormat

	// width is the image width in pixels.
	width int

	// These fields are copied from the *Options (which may be nil).
	align  bool
	invert bool

	// curr and prev hold the current and previous rows. Each element is
	// either 0x00 (black) or 0xFF (white). They are internal copies; the
	// rows passed to WriteRow are neither modified nor retained.
	//
	// prev may be nil, when encoding the first row. The reference row above
	// the first row is implicitly all white.
	curr []byte
	prev []byte

	// started is whether the start-of-image codes (the initial EOL, for
	// Group3) have been written.
	started bool

	// closed is whether Close has been called.
	closed bool

	// err is a sticky error for the WriteRow and Close methods.
	err error
}

// NewEncoder returns an Encoder that writes a CCITT-formatted data stream to
// w, encoding rows of the given width in pixels. The order, sf and opts
// arguments have the same meaning as for NewReader.
//
// The encoder does not need to know the image height in advance: Close
// terminates the stream such that NewReader can decode it with either the
// exact height or AutoDetectHeight.
//
// The ctx is consulted by WriteRow and Close: once it is canceled, those
// methods return the context's error.
func NewEncoder(ctx context.Context, w io.Writer, order Order, sf SubFormat, width int, opts *Options) *Encoder {
	if ctx == nil {
		ctx = context.Background()
	}
	e := &Encoder{
		ctx:       ctx,
		bw:        bitWriter{w: w, order: order},
		subFormat: sf,
		align:     (opts != nil) && opts.Align,
		invert:    (opts != nil) && opts.Invert,
	}
	if width < 0 {
		e.err = errInvalidBounds
	} else if width > maxWidth {
		e.err = errUnsupportedWidth
	} else {
		e.width = width
		e.curr = make([]byte, width)
	}
	return e
}

// WriteRow encodes a single row of pixels. The row p holds (width+7)/8
// bytes, one bit per pixel, most significant bit first, in the same form
// that NewReader's Read method produces: 1 means white and 0 means black,
// unless the Invert option is set, in which case 1 means black and 0 means
// white. Bits beyond the image width are ignored.
//
// The row is copied before it is encoded; p is neither modified nor
// retained.
func (e *Encoder) WriteRow(p []byte) error {
	if err := e.ctx.Err(); err != nil {
		return err
	}
	if e.err != nil {
		return e.err
	}
	if e.closed {
		return errEncoderClosed
	}
	if len(p) != (e.width+7)/8 {
		e.err = errInvalidRowLength
		return e.err
	}
	if err := e.startEncode(); err != nil {
		e.err = err
		return e.err
	}

	e.unpackRow(p)
	var err error
	switch e.subFormat {
	case Group3:
		err = e.encodeRowGroup3()
	case Group4:
		err = e.encodeRowGroup4()
	default:
		err = errUnsupportedSubFormat
	}
	if err != nil {
		e.err = err
		return e.err
	}

	if e.prev == nil {
		e.prev = make([]byte, e.width)
	}
	e.curr, e.prev = e.prev, e.curr
	return nil
}

// Close writes the end-of-facsimile-block codes and flushes the encoded
// stream to the underlying io.Writer. For Group3, this is the RTC (Return To
// Control) of 6 consecutive EOL codes, the first of which is the EOL that
// terminates the final row (or the start-of-image EOL, for an image with no
// rows). For Group4, it is the EOFB (End Of Facsimile Block) of 2
// consecutive EOL codes.
//
// It is an error to call Close more than once, or to call WriteRow after
// Close.
func (e *Encoder) Close() error {
	if err := e.ctx.Err(); err != nil {
		return err
	}
	if e.err != nil {
		return e.err
	}
	if e.closed {
		return errEncoderClosed
	}
	e.closed = true

	if err := e.startEncode(); err != nil {
		e.err = err
		return e.err
	}

	var err error
	switch e.subFormat {
	case Group3:
		// One of the RTC's 6 consecutive EOL's has already been written:
		// either the start-of-image EOL or the EOL terminating the final
		// row. Write the 5 remaining ones.
		for i := 0; (i < 5) && (err == nil); i++ {
			err = e.bw.writeCode(eolCode)
		}

	case Group4:
		if e.align {
			err = e.bw.alignToByteBoundary()
		}
		for i := 0; (i < 2) && (err == nil); i++ {
			err = e.bw.writeCode(eolCode)
		}

	default:
		err = errUnsupportedSubFormat
	}

	if err == nil {
		err = e.bw.close()
	}
	if err != nil {
		e.err = err
	}
	return err
}

// startEncode writes the start-of-image codes, if not already written.
func (e *Encoder) startEncode() error {
	if e.started {
		return nil
	}
	switch e.subFormat {
	case Group3:
		if err := e.bw.writeCode(eolCode); err != nil {
			return err
		}
	case Group4:
		// No-op.
	default:
		return errUnsupportedSubFormat
	}
	e.started = true
	return nil
}

// unpackRow converts p (1 bit per pixel, MSB first) to e.curr (1 byte per
// pixel, 0x00 black or 0xFF white).
func (e *Encoder) unpackRow(p []byte) {
	for i := 0; i < e.width; i++ {
		white := p[i>>3]&(0x80>>uint(i&7)) != 0
		if e.invert {
			white = !white
		}
		if white {
			e.curr[i] = 0xFF
		} else {
			e.curr[i] = 0x00
		}
	}
}

// encodeRowGroup3 encodes e.curr as a Group3 row: a series of white and
// black runs, terminated by an EOL code.
func (e *Encoder) encodeRowGroup3() error {
	if e.align {
		if err := e.alignToByteBoundary(); err != nil {
			return err
		}
	}
	penColorIsWhite := true
	for wi := 0; wi < e.width; {
		run := runLength(e.curr[wi:], penColorIsWhite)
		if err := e.encodeRun(run, penColorIsWhite); err != nil {
			return err
		}
		wi += run
		penColorIsWhite = !penColorIsWhite
	}
	return e.bw.writeCode(eolCode)
}

// encodeRowGroup4 encodes e.curr as a Group4 row, relative to the reference
// row e.prev, as a series of pass, horizontal and vertical mode codes.
func (e *Encoder) encodeRowGroup4() error {
	if e.align {
		if err := e.bw.alignToByteBoundary(); err != nil {
			return err
		}
	}

	curr, prev, width := e.curr, e.prev, e.width
	wi := 0
	atStartOfRow := true
	penColorIsWhite := true

	for wi < width {
		// a1 and a2 are the next two changing elements on the coding line
		// (the current row).
		a1 := changingElement(curr, wi, penColorIsWhite)
		a2 := width
		if a1 < width {
			a2 = changingElement(curr, a1+1, !penColorIsWhite)
		}
		// b1 and b2 are the next two changing elements on the reference
		// line (the previous row), of opposite color to the pen color.
		b1 := findB(prev, width, wi, atStartOfRow, penColorIsWhite, findB1)
		b2 := findB(prev, width, wi, atStartOfRow, penColorIsWhite, findB2)

		switch d := a1 - b1; {
		case b2 < a1:
			// Pass mode.
			if err := e.bw.writeCode(modeEncodeTable[modePass]); err != nil {
				return err
			}
			wi = b2

		case (-3 <= d) && (d <= +3):
			// Vertical mode.
			if err := e.bw.writeCode(modeEncodeTable[verticalModes[d+3]]); err != nil {
				return err
			}
			wi = a1
			penColorIsWhite = !penColorIsWhite

		default:
			// Horizontal mode.
			if err := e.bw.writeCode(modeEncodeTable[modeH]); err != nil {
				return err
			}
			if err := e.encodeRun(a1-wi, penColorIsWhite); err != nil {
				return err
			}
			if err := e.encodeRun(a2-a1, !penColorIsWhite); err != nil {
				return err
			}
			wi = a2
		}
		atStartOfRow = false
	}
	return nil
}

// encodeRun writes a white or black run of the given length, as a series of
// make-up codes followed by a terminating code.
func (e *Encoder) encodeRun(run int, penColorIsWhite bool) error {
	if (run < 0) || (run > maxWidth) {
		return errRunLengthTooLong
	}
	t2, t3 := whiteEncodeTable2[:], whiteEncodeTable3[:]
	if !penColorIsWhite {
		t2, t3 = blackEncodeTable2[:], blackEncodeTable3[:]
	}
	// The largest make-up code is 2560, the final entry of each table3.
	for run >= 2560+64 {
		if err := e.bw.writeCode(t3[len(t3)-1]); err != nil {
			return err
		}
		run -= 2560
	}
	if run >= 64 {
		if err := e.bw.writeCode(t3[run/64-1]); err != nil {
			return err
		}
		run &= 63
	}
	return e.bw.writeCode(t2[run])
}

// alignToByteBoundary pads the bit stream to a byte boundary with fill bits.
// The first fill bit is a 1 (any remaining ones are 0), so that the fill
// cannot be mistaken for the leading zero bits of an EOL code by a reader
// that probes the stream before aligning to the byte boundary, such as
// NewReader with AutoDetectHeight looking for the RTC's EOLs. Decoders
// ignore the fill bits' values when aligning.
func (e *Encoder) alignToByteBoundary() error {
	if e.bw.nBits&7 != 0 {
		if err := e.bw.writeCode(bitString{1, 1}); err != nil {
			return err
		}
	}
	return e.bw.alignToByteBoundary()
}

// runLength returns the length of the maximal run of the given color at the
// start of row (1 byte per pixel, 0x00 black or 0xFF white).
func runLength(row []byte, penColorIsWhite bool) int {
	c := byte(0x00)
	if penColorIsWhite {
		c = 0xFF
	}
	n := 0
	for (n < len(row)) && (row[n] == c) {
		n++
	}
	return n
}

// changingElement returns the index of the first changing element at or
// after position i: the first pixel whose color differs from the given (pen)
// color. It returns len(row) if there is no such pixel.
func changingElement(row []byte, i int, penColorIsWhite bool) int {
	c := byte(0x00)
	if penColorIsWhite {
		c = 0xFF
	}
	for ; (i < len(row)) && (row[i] == c); i++ {
	}
	return i
}
