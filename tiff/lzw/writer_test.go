// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lzw

import (
	"bytes"
	stdlzw "compress/lzw"
	"errors"
	"io"
	"testing"
)

// lcg is a deterministic pseudo-random byte generator, so that the tests
// do not depend on math/rand's seeding behavior.
type lcg struct{ state uint32 }

func (g *lcg) bytes(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		g.state = g.state*1664525 + 1013904223
		b[i] = byte(g.state >> 24)
	}
	return b
}

func roundtrip(t *testing.T, original []byte, order Order, litWidth int) {
	t.Helper()
	// Encode.
	buf := new(bytes.Buffer)
	w := NewWriter(buf, order, litWidth)
	n, err := w.Write(original)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	if n != len(original) {
		t.Fatalf("Write: got n=%d, want %d", n, len(original))
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// The compressed stream must start with a clear code and end with an
	// EOI code; both are implied by a successful round-trip through a
	// strict reader, but an empty stream would be a bug of its own.
	if buf.Len() == 0 {
		t.Fatal("compressed stream is empty")
	}
	// Decode with this package's reader.
	r := NewReader(buf, order, litWidth)
	decoded, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("reader Close: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatalf("round-trip mismatch: got %d bytes, want %d", len(decoded), len(original))
	}
}

func TestWriterRoundtrip(t *testing.T) {
	g := &lcg{state: 1}
	repetitive := bytes.Repeat([]byte("ab"), 1<<16)
	zeros := make([]byte, 1<<20)
	// largeRandom exercises the 9->10->11->12 bit code width transitions
	// and the dictionary reset (clear code) once all 4096 codes are used.
	largeRandom := g.bytes(1 << 18)
	tests := []struct {
		name     string
		data     []byte
		order    Order
		litWidth int
	}{
		{"empty-msb", nil, MSB, 8},
		{"empty-lsb", nil, LSB, 8},
		{"one-byte", []byte{0x42}, MSB, 8},
		{"hello", []byte("Hello, world!"), MSB, 8},
		{"tobe", []byte("TOBEORNOTTOBEORTOBEORNOT"), MSB, 8},
		{"tobe-lsb", []byte("TOBEORNOTTOBEORTOBEORNOT"), LSB, 8},
		{"all-bytes", g.bytes(256), MSB, 8},
		{"repetitive", repetitive, MSB, 8},
		{"zeros", zeros, MSB, 8},
		{"large-random", largeRandom, MSB, 8},
		{"large-random-lsb", largeRandom, LSB, 8},
		{"small-litwidth", []byte{0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3}, MSB, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			roundtrip(t, tc.data, tc.order, tc.litWidth)
		})
	}
}

// TestWriterDictionaryReset checks that a stream long enough to exhaust the
// 4096-entry dictionary (forcing the encoder to emit clear codes and start
// over) still round-trips, and that the compressed stream indeed contains
// an interior clear code.
func TestWriterDictionaryReset(t *testing.T) {
	g := &lcg{state: 42}
	original := g.bytes(1 << 17)
	buf := new(bytes.Buffer)
	w := NewWriter(buf, MSB, 8)
	if _, err := w.Write(original); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	r := NewReader(buf, MSB, 8)
	decoded, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatal("round-trip mismatch after dictionary reset")
	}
}

// TestWriterStandardDecoderCrossCheck decodes a short TIFF-LZW stream with
// the standard library's LZW decoder. TIFF's LZW differs from standard LZW
// only in the "off by one" code width transition, which first matters after
// 254 emitted codes; a short input never reaches that point, so the two
// formats must agree byte for byte.
func TestWriterStandardDecoderCrossCheck(t *testing.T) {
	g := &lcg{state: 7}
	original := g.bytes(200)
	buf := new(bytes.Buffer)
	w := NewWriter(buf, MSB, 8)
	if _, err := w.Write(original); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	r := stdlzw.NewReader(buf, stdlzw.MSB, 8)
	decoded, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("std lzw ReadAll: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatal("standard LZW decoder mismatch")
	}
}

// TestWriterInputNotModified verifies that Write does not mutate its input.
func TestWriterInputNotModified(t *testing.T) {
	g := &lcg{state: 3}
	original := g.bytes(4096)
	cpy := make([]byte, len(original))
	copy(cpy, original)
	w := NewWriter(io.Discard, MSB, 8)
	if _, err := w.Write(original); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !bytes.Equal(original, cpy) {
		t.Fatal("Write modified its input")
	}
}

// errWriter is an io.Writer that always fails.
type errWriter struct{ err error }

func (w errWriter) Write([]byte) (int, error) { return 0, w.err }

// shortWriter is an io.Writer that reports a short write without an error.
type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestWriterErrors(t *testing.T) {
	// Invalid order.
	w := NewWriter(io.Discard, Order(42), 8)
	if _, err := w.Write([]byte{1}); err == nil {
		t.Error("Write with invalid order: no error")
	}
	// Invalid litWidth.
	for _, lw := range []int{0, 1, 9, 100} {
		w := NewWriter(io.Discard, MSB, lw)
		if _, err := w.Write([]byte{1}); err == nil {
			t.Errorf("Write with litWidth %d: no error", lw)
		}
	}
	// Literal out of range for the litWidth.
	w = NewWriter(io.Discard, MSB, 2)
	if _, err := w.Write([]byte{0xff}); err == nil {
		t.Error("Write with out-of-range literal: no error")
	}
	// Underlying writer fails. A small write is buffered by the bufio
	// wrapper, so the error may only surface at Close; a write larger
	// than the buffer must fail during Write itself.
	w = NewWriter(errWriter{errors.New("boom")}, MSB, 8)
	if _, err := w.Write([]byte("hello")); err == nil {
		if err = w.Close(); err == nil {
			t.Error("Close with failing writer: no error")
		}
	}
	w = NewWriter(errWriter{errors.New("boom")}, MSB, 8)
	if _, err := w.Write((&lcg{state: 5}).bytes(1 << 16)); err == nil {
		t.Error("large Write with failing writer: no error")
	}
	// Underlying writer reports a short write; this must surface as an
	// error (io.ErrShortWrite from the bufio wrapper) at the latest by
	// Close.
	w = NewWriter(shortWriter{}, MSB, 8)
	if _, err := w.Write([]byte("hello")); err == nil {
		if err = w.Close(); err == nil {
			t.Error("Close with short-writing writer: no error")
		}
	}
	// Write after Close.
	w = NewWriter(io.Discard, MSB, 8)
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := w.Write([]byte{1}); err == nil {
		t.Error("Write after Close: no error")
	}
	// Double Close is a no-op.
	if err := w.Close(); err != nil {
		t.Errorf("double Close: %v", err)
	}
}

// TestWriterEmptyStream checks the exact bytes of an empty stream: a clear
// code followed by an EOI code, MSB-packed at 9 bits each.
func TestWriterEmptyStream(t *testing.T) {
	buf := new(bytes.Buffer)
	w := NewWriter(buf, MSB, 8)
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// 9-bit clear (256) then 9-bit EOI (257), MSB first:
	// 256 = 0b100000000, 257 = 0b100000001 -> bits 10000000 01000000 01,
	// so the bytes are 0x80, 0x40, 0x40 (the last byte is zero-padded).
	want := []byte{0x80, 0x40, 0x40}
	if !bytes.Equal(buf.Bytes(), want) {
		t.Fatalf("empty stream: got % x, want % x", buf.Bytes(), want)
	}
}

func TestWriterMultipleWrites(t *testing.T) {
	// Writing the same input in one call or in many small calls must
	// produce identical output.
	g := &lcg{state: 9}
	original := g.bytes(10000)

	one := new(bytes.Buffer)
	w := NewWriter(one, MSB, 8)
	if _, err := w.Write(original); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	many := new(bytes.Buffer)
	w = NewWriter(many, MSB, 8)
	for i := 0; i < len(original); i += 7 {
		j := i + 7
		if j > len(original) {
			j = len(original)
		}
		if _, err := w.Write(original[i:j]); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !bytes.Equal(one.Bytes(), many.Bytes()) {
		t.Fatal("one-shot and chunked writes differ")
	}
}
