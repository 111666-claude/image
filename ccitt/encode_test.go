// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ccitt

import (
	"bytes"
	"context"
	"errors"
	"image"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// maskPaddingBits zeroes the bits of the final byte of a packed row that are
// beyond the image width, matching the zero padding that NewReader's Read
// method produces.
func maskPaddingBits(row []byte, width int) {
	if (width % 8) != 0 {
		row[len(row)-1] &= 0xFF << (8 - uint(width)%8)
	}
}

// roundTrip encodes the given packed rows (1 bit per pixel, MSB first) and
// checks that NewReader decodes the same bytes, both with the exact height
// and with AutoDetectHeight.
func roundTrip(t *testing.T, order Order, sf SubFormat, align, invert bool, width int, rows [][]byte) {
	t.Helper()
	height := len(rows)

	var buf bytes.Buffer
	e := NewEncoder(context.Background(), &buf, order, sf, width, &Options{
		Align:  align,
		Invert: invert,
	})
	for y, row := range rows {
		if err := e.WriteRow(row); err != nil {
			t.Fatalf("WriteRow(%d): %v", y, err)
		}
	}
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := []byte(nil)
	for _, row := range rows {
		want = append(want, row...)
	}

	for _, h := range []int{height, AutoDetectHeight} {
		r := NewReader(bytes.NewReader(buf.Bytes()), order, sf, width, h, &Options{
			Align:  align,
			Invert: invert,
		})
		got, err := io.ReadAll(r)
		if err != nil {
			t.Fatalf("height=%d: ReadAll: %v", h, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("height=%d:\ngot:  %X\nwant: %X", h, got, want)
		}
	}
}

// testRoundTrip runs roundTrip for every combination of bit order,
// sub-format, alignment and inversion.
func testRoundTrip(t *testing.T, width int, rows [][]byte) {
	t.Helper()
	for _, order := range []Order{LSB, MSB} {
		for _, sf := range []SubFormat{Group3, Group4} {
			for _, align := range []bool{false, true} {
				for _, invert := range []bool{false, true} {
					roundTrip(t, order, sf, align, invert, width, rows)
				}
			}
		}
	}
}

func TestEncodeRandomRows(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, width := range []int{1, 2, 7, 8, 9, 31, 32, 33, 153} {
		rowBytes := (width + 7) / 8
		rows := make([][]byte, 17)
		for y := range rows {
			row := make([]byte, rowBytes)
			if _, err := rng.Read(row); err != nil {
				t.Fatal(err)
			}
			maskPaddingBits(row, width)
			rows[y] = row
		}
		testRoundTrip(t, width, rows)
	}
}

func TestEncodeRandomRuns(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	// A width over 2560+64 exercises the make-up codes, including repeated
	// 2560 make-up codes for long runs.
	const width = 5000
	const rowBytes = (width + 7) / 8

	rows := make([][]byte, 9)
	for y := range rows {
		// Build a row from random runs of 0x00 and 0xFF pixels, starting
		// with a random color.
		pix := make([]byte, 0, width)
		c := byte(0x00)
		if rng.Intn(2) == 0 {
			c = 0xFF
		}
		for len(pix) < width {
			n := 1 + rng.Intn(2*width)
			if n > width-len(pix) {
				n = width - len(pix)
			}
			for i := 0; i < n; i++ {
				pix = append(pix, c)
			}
			c = ^c
		}

		// Pack to 1 bit per pixel, MSB first, 1 meaning white.
		row := make([]byte, rowBytes)
		for x, p := range pix {
			if p == 0xFF {
				row[x>>3] |= 0x80 >> uint(x&7)
			}
		}
		rows[y] = row
	}
	testRoundTrip(t, width, rows)
}

func TestEncodeSpecialRows(t *testing.T) {
	const width = 19
	const rowBytes = (width + 7) / 8

	allWhite := make([]byte, rowBytes)
	for i := range allWhite {
		allWhite[i] = 0xFF
	}
	maskPaddingBits(allWhite, width)

	allBlack := make([]byte, rowBytes)

	alternating := make([]byte, rowBytes)
	for i := range alternating {
		alternating[i] = 0xAA
	}
	maskPaddingBits(alternating, width)

	// Blank (all white) rows, all black rows, alternating pixels and
	// alternating rows.
	rows := [][]byte{
		allWhite,
		allWhite,
		allBlack,
		allWhite,
		alternating,
		allBlack,
		allBlack,
		allWhite,
	}
	testRoundTrip(t, width, rows)

	// An image with no rows at all.
	testRoundTrip(t, width, nil)
}

// packGray packs an *image.Gray into rows of 1 bit per pixel, MSB first, in
// the form that WriteRow accepts for the given invert option.
func packGray(img *image.Gray, invert bool) [][]byte {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	rows := make([][]byte, height)
	for y := 0; y < height; y++ {
		row := make([]byte, (width+7)/8)
		pix := img.Pix[(y+bounds.Min.Y-bounds.Min.Y)*img.Stride:]
		for x := 0; x < width; x++ {
			white := pix[x] >= 0x80
			if white != invert {
				row[x>>3] |= 0x80 >> uint(x&7)
			}
		}
		rows[y] = row
	}
	return rows
}

// TestEncodeGopher decodes the reference CCITT images, re-encodes them with
// the Encoder, and checks that NewReader reads back the same rows.
func TestEncodeGopher(t *testing.T) {
	for _, tt := range []struct {
		fileName string
		sf       SubFormat
	}{
		{"testdata/bw-gopher.ccitt_group3", Group3},
		{"testdata/bw-gopher.ccitt_group4", Group4},
	} {
		data, err := os.ReadFile(filepath.FromSlash(tt.fileName))
		if err != nil {
			t.Fatal(err)
		}
		img := image.NewGray(image.Rect(0, 0, 153, 55))
		if err := DecodeIntoGray(img, bytes.NewReader(data), MSB, tt.sf, nil); err != nil {
			t.Fatalf("%s: DecodeIntoGray: %v", tt.fileName, err)
		}

		for _, order := range []Order{LSB, MSB} {
			for _, align := range []bool{false, true} {
				for _, invert := range []bool{false, true} {
					opts := &Options{Align: align, Invert: invert}
					rows := packGray(img, invert)

					var buf bytes.Buffer
					e := NewEncoder(context.Background(), &buf, order, tt.sf, 153, opts)
					for y, row := range rows {
						if err := e.WriteRow(row); err != nil {
							t.Fatalf("%s: WriteRow(%d): %v", tt.fileName, y, err)
						}
					}
					if err := e.Close(); err != nil {
						t.Fatalf("%s: Close: %v", tt.fileName, err)
					}

					got, err := io.ReadAll(NewReader(bytes.NewReader(buf.Bytes()), order, tt.sf, 153, 55, opts))
					if err != nil {
						t.Fatalf("%s: ReadAll: %v", tt.fileName, err)
					}
					want := bytes.Join(rows, nil)
					if !bytes.Equal(got, want) {
						t.Fatalf("%s: order=%d align=%t invert=%t:\ngot:  %X\nwant: %X",
							tt.fileName, order, align, invert, got, want)
					}
				}
			}
		}
	}
}

// TestEncodeGoldenGroup3 checks the exact encoded bytes of a single all
// white Group3 row with Align, including the start-of-image EOL, the byte
// alignment padding and the 6-EOL RTC.
func TestEncodeGoldenGroup3(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder(context.Background(), &buf, MSB, Group3, 8, &Options{Align: true})
	if err := e.WriteRow([]byte{0xFF}); err != nil {
		t.Fatalf("WriteRow: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The bit stream is:
	//   000000000001  EOL (start of image)
	//   1000          fill bits, to a byte boundary (Align)
	//   10011         white run of 8
	//   000000000001  EOL (end of row, and first EOL of the RTC)
	//   000000000001  EOL × 5 (the rest of the RTC)
	//   000           padding, to a byte boundary
	want := []byte{
		0x00, 0x18, 0x98, 0x00, 0x80, 0x08,
		0x00, 0x80, 0x08, 0x00, 0x80, 0x08,
	}
	if got := buf.Bytes(); !bytes.Equal(got, want) {
		t.Fatalf("\ngot:  %X\nwant: %X", got, want)
	}
}

// TestEncodeGoldenGroup4 checks the exact encoded bytes of a single all
// white Group4 row, including the 2-EOL EOFB.
func TestEncodeGoldenGroup4(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder(context.Background(), &buf, MSB, Group4, 8, nil)
	if err := e.WriteRow([]byte{0xFF}); err != nil {
		t.Fatalf("WriteRow: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The bit stream is:
	//   1             vertical-0 mode (the row matches the all white
	//                 implicit reference row)
	//   000000000001  EOL × 2 (the EOFB)
	//   0000000       padding, to a byte boundary
	want := []byte{0x80, 0x08, 0x00, 0x80}
	if got := buf.Bytes(); !bytes.Equal(got, want) {
		t.Fatalf("\ngot:  %X\nwant: %X", got, want)
	}
}

// TestEncodeGoldenGroup4LSB checks that LSB order produces the bit-reversed
// bytes of the MSB encoding.
func TestEncodeGoldenGroup4LSB(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder(context.Background(), &buf, LSB, Group4, 8, nil)
	if err := e.WriteRow([]byte{0xFF}); err != nil {
		t.Fatalf("WriteRow: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := []byte{0x01, 0x10, 0x00, 0x01}
	if got := buf.Bytes(); !bytes.Equal(got, want) {
		t.Fatalf("\ngot:  %X\nwant: %X", got, want)
	}
}

func TestEncodeInvalidWidth(t *testing.T) {
	for _, tt := range []struct {
		width int
		err   error
	}{
		{-1, errInvalidBounds},
		{maxWidth + 1, errUnsupportedWidth},
	} {
		e := NewEncoder(context.Background(), io.Discard, MSB, Group3, tt.width, nil)
		if err := e.WriteRow(nil); err != tt.err {
			t.Fatalf("width=%d: WriteRow: got %v, want %v", tt.width, err, tt.err)
		}
		if err := e.Close(); err != tt.err {
			t.Fatalf("width=%d: Close: got %v, want %v", tt.width, err, tt.err)
		}
	}
}

func TestEncodeInvalidRowLength(t *testing.T) {
	e := NewEncoder(context.Background(), io.Discard, MSB, Group3, 9, nil)
	for _, n := range []int{0, 1, 3, 4} {
		if err := e.WriteRow(make([]byte, n)); err != errInvalidRowLength {
			t.Fatalf("len(row)=%d: got %v, want %v", n, err, errInvalidRowLength)
		}
	}
}

func TestEncodeUnsupportedSubFormat(t *testing.T) {
	e := NewEncoder(context.Background(), io.Discard, MSB, SubFormat(42), 8, nil)
	if err := e.WriteRow([]byte{0x00}); err != errUnsupportedSubFormat {
		t.Fatalf("WriteRow: got %v, want %v", err, errUnsupportedSubFormat)
	}
}

func TestEncodeRunLengthOverflow(t *testing.T) {
	e := &Encoder{}
	if err := e.encodeRun(maxWidth+1, true); err != errRunLengthTooLong {
		t.Fatalf("got %v, want %v", err, errRunLengthTooLong)
	}
}

func TestEncodeDoubleClose(t *testing.T) {
	e := NewEncoder(context.Background(), io.Discard, MSB, Group4, 8, nil)
	if err := e.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := e.Close(); err != errEncoderClosed {
		t.Fatalf("second Close: got %v, want %v", err, errEncoderClosed)
	}
	if err := e.WriteRow([]byte{0x00}); err != errEncoderClosed {
		t.Fatalf("WriteRow after Close: got %v, want %v", err, errEncoderClosed)
	}
}

func TestEncodeCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	e := NewEncoder(ctx, io.Discard, MSB, Group3, 8, nil)
	if err := e.WriteRow([]byte{0x00}); !errors.Is(err, context.Canceled) {
		t.Fatalf("WriteRow: got %v, want %v", err, context.Canceled)
	}
	if err := e.Close(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close: got %v, want %v", err, context.Canceled)
	}
}

// errWriter is an io.Writer that always fails.
type errWriter struct{ err error }

func (w errWriter) Write(p []byte) (int, error) { return 0, w.err }

// shortWriter is an io.Writer that always writes one byte fewer than asked.
type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

// failAfterWriter is an io.Writer that fails after writing n bytes in total.
type failAfterWriter struct {
	n   int
	err error
}

func (w *failAfterWriter) Write(p []byte) (int, error) {
	if w.n <= 0 {
		return 0, w.err
	}
	if len(p) > w.n {
		n := w.n
		w.n = 0
		return n, w.err
	}
	w.n -= len(p)
	return len(p), nil
}

func TestEncodeFailingWriter(t *testing.T) {
	errFoo := errors.New("foo")

	// An immediate failure surfaces at the first WriteRow (for Group3, whose
	// start-of-image EOL fills the bit buffer) or at Close.
	e := NewEncoder(context.Background(), errWriter{errFoo}, MSB, Group3, 8, nil)
	if err := e.WriteRow([]byte{0xFF}); err != nil && err != errFoo {
		t.Fatalf("WriteRow: got %v, want %v or nil", err, errFoo)
	}
	if err := e.Close(); err != errFoo {
		t.Fatalf("Close: got %v, want %v", err, errFoo)
	}

	// A failure partway through surfaces at the WriteRow call that crosses
	// the failure point, and is sticky afterwards.
	w := &failAfterWriter{n: 100, err: errFoo}
	e = NewEncoder(context.Background(), w, MSB, Group3, 1024, nil)
	row := bytes.Repeat([]byte{0xAA}, 128)
	sawErr := false
	for y := 0; y < 64; y++ {
		err := e.WriteRow(row)
		if !sawErr && err == nil {
			continue
		}
		sawErr = true
		if err != errFoo {
			t.Fatalf("WriteRow(%d) after failure: got %v, want %v", y, err, errFoo)
		}
	}
	if !sawErr {
		t.Fatal("no WriteRow returned the writer's error")
	}
	if err := e.Close(); err != errFoo {
		t.Fatalf("Close: got %v, want %v", err, errFoo)
	}
}

func TestEncodeShortWriter(t *testing.T) {
	for _, sf := range []SubFormat{Group3, Group4} {
		e := NewEncoder(context.Background(), shortWriter{}, MSB, sf, 8, nil)
		e.WriteRow([]byte{0xFF})
		if err := e.Close(); err != io.ErrShortWrite {
			t.Fatalf("sf=%d: Close: got %v, want %v", sf, err, io.ErrShortWrite)
		}
	}
}

// TestEncodeDoesNotModifyRow checks that WriteRow neither modifies nor
// retains the caller's row buffer.
func TestEncodeDoesNotModifyRow(t *testing.T) {
	const width = 13
	row := []byte{0xAB, 0xCD}
	original := append([]byte(nil), row...)

	for _, sf := range []SubFormat{Group3, Group4} {
		row[0], row[1] = original[0], original[1]
		var buf bytes.Buffer
		e := NewEncoder(context.Background(), &buf, MSB, sf, width, nil)
		for y := 0; y < 4; y++ {
			if err := e.WriteRow(row); err != nil {
				t.Fatalf("sf=%d: WriteRow: %v", sf, err)
			}
			if !bytes.Equal(row, original) {
				t.Fatalf("sf=%d: WriteRow modified the row: got %X, want %X", sf, row, original)
			}
		}
		if err := e.Close(); err != nil {
			t.Fatalf("sf=%d: Close: %v", sf, err)
		}

		// Mutating the row after the WriteRow calls must not affect rows
		// that were already encoded: the decoded image is 4 identical rows.
		row[0], row[1] = 0x00, 0x00
		got, err := io.ReadAll(NewReader(bytes.NewReader(buf.Bytes()), MSB, sf, width, 4, nil))
		if err != nil {
			t.Fatalf("sf=%d: ReadAll: %v", sf, err)
		}
		maskPaddingBits(original, width)
		want := bytes.Repeat(original, 4)
		if !bytes.Equal(got, want) {
			t.Fatalf("sf=%d:\ngot:  %X\nwant: %X", sf, got, want)
		}
	}
}
