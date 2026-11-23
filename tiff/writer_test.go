// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"errors"
	"image"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

// shortWriter is an io.Writer that reports a short write without an error.
type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

var roundtripTests = []struct {
	filename string
	opts     *Options
}{
	{"video-001.tiff", nil},
	{"video-001-16bit.tiff", nil},
	{"video-001-gray.tiff", nil},
	{"video-001-gray-16bit.tiff", nil},
	{"video-001-paletted.tiff", nil},
	{"bw-packbits.tiff", nil},
	{"video-001.tiff", &Options{Predictor: true}},
	{"video-001.tiff", &Options{Compression: Deflate}},
	{"video-001.tiff", &Options{Predictor: true, Compression: Deflate}},
	{"video-001.tiff", &Options{Compression: LZW}},
	{"video-001.tiff", &Options{Predictor: true, Compression: LZW}},
	{"video-001-16bit.tiff", &Options{Compression: LZW}},
	{"video-001-16bit.tiff", &Options{Predictor: true, Compression: LZW}},
	{"video-001-gray.tiff", &Options{Compression: LZW}},
	{"video-001-gray.tiff", &Options{Predictor: true, Compression: LZW}},
	{"video-001-gray-16bit.tiff", &Options{Compression: LZW}},
	{"video-001-gray-16bit.tiff", &Options{Predictor: true, Compression: LZW}},
	{"video-001-paletted.tiff", &Options{Compression: LZW}},
	{"bw-packbits.tiff", &Options{Compression: LZW}},
}

func openImage(filename string) (image.Image, error) {
	f, err := os.Open(testdataDir + filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return Decode(f)
}

func TestRoundtrip(t *testing.T) {
	for _, rt := range roundtripTests {
		img, err := openImage(rt.filename)
		if err != nil {
			t.Fatal(err)
		}

		out := new(bytes.Buffer)
		err = Encode(out, img, rt.opts)
		if err != nil {
			t.Fatal(err)
		}

		img2, err := Decode(&buffer{buf: out.Bytes()})
		if err != nil {
			t.Fatal(err)
		}
		compare(t, img, img2)
	}
}

// TestRoundtrip2 tests that encoding and decoding an image whose
// origin is not (0, 0) gives the same thing.
func TestRoundtrip2(t *testing.T) {
	m0 := image.NewRGBA(image.Rect(3, 4, 9, 8))
	for i := range m0.Pix {
		m0.Pix[i] = byte(i)
	}
	out := new(bytes.Buffer)
	if err := Encode(out, m0, nil); err != nil {
		t.Fatal(err)
	}
	m1, err := Decode(&buffer{buf: out.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, m0, m1)
}

// TestRoundtrip3 is TestRoundtrip2 with LZW compression and the horizontal
// differencing predictor enabled.
func TestRoundtrip3(t *testing.T) {
	m0 := image.NewRGBA(image.Rect(3, 4, 9, 8))
	for i := range m0.Pix {
		m0.Pix[i] = byte(i)
	}
	out := new(bytes.Buffer)
	if err := Encode(out, m0, &Options{Compression: LZW, Predictor: true}); err != nil {
		t.Fatal(err)
	}
	m1, err := Decode(&buffer{buf: out.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, m0, m1)
}

func TestUnsupported(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 1, 1))
	out := new(bytes.Buffer)
	err := Encode(out, img, &Options{Compression: CCITTGroup3})
	if err == nil {
		t.Error("tiff.Encode(CCITTGroup3): no error returned, expected an error")
	}
}

// TestLZWTags checks that an LZW-compressed file carries the right IFD
// tags: compression 5, predictor 2 (when enabled), a single strip at
// offset 8 whose byte count matches the compressed data actually written.
func TestLZWTags(t *testing.T) {
	m := image.NewGray(image.Rect(0, 0, 17, 5))
	for i := range m.Pix {
		m.Pix[i] = byte(i * 7)
	}
	out := new(bytes.Buffer)
	if err := Encode(out, m, &Options{Compression: LZW, Predictor: true}); err != nil {
		t.Fatal(err)
	}
	// The IFD offset is stored in the 4-byte field after the 4-byte
	// header prefix; the pixel data occupies bytes [8, ifdOffset).
	ifdOffset := int(enc.Uint32(out.Bytes()[4:8]))
	d, err := newDecoder(&buffer{buf: out.Bytes()})
	if err != nil {
		t.Fatal(err)
	}
	if got := d.firstVal(tCompression); got != cLZW {
		t.Errorf("compression tag: got %d, want %d", got, cLZW)
	}
	if got := d.firstVal(tPredictor); got != prHorizontal {
		t.Errorf("predictor tag: got %d, want %d", got, prHorizontal)
	}
	if got := d.firstVal(tStripOffsets); got != 8 {
		t.Errorf("strip offset: got %d, want 8", got)
	}
	if got := d.firstVal(tStripByteCounts); got != uint(ifdOffset-8) {
		t.Errorf("strip byte count: got %d, want %d", got, ifdOffset-8)
	}
	if got := d.firstVal(tRowsPerStrip); got != 5 {
		t.Errorf("rows per strip: got %d, want 5", got)
	}
}

// TestLZWInputNotModified checks that LZW encoding (with and without the
// predictor) does not modify the input image.
func TestLZWInputNotModified(t *testing.T) {
	for _, opts := range []*Options{
		{Compression: LZW},
		{Compression: LZW, Predictor: true},
	} {
		m0 := image.NewRGBA64(image.Rect(0, 0, 13, 11))
		for i := range m0.Pix {
			m0.Pix[i] = byte(i*31 + 7)
		}
		pix := make([]byte, len(m0.Pix))
		copy(pix, m0.Pix)
		if err := Encode(io.Discard, m0, opts); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(m0.Pix, pix) {
			t.Errorf("Encode(%+v) modified the input image", opts)
		}
	}
}

// TestLZWShortWrite checks that a short write to the destination writer is
// reported as an error instead of silently truncating the file.
func TestLZWShortWrite(t *testing.T) {
	m := image.NewGray(image.Rect(0, 0, 64, 64))
	for i := range m.Pix {
		m.Pix[i] = byte(i)
	}
	if err := Encode(shortWriter{}, m, &Options{Compression: LZW}); err == nil {
		t.Error("Encode with short-writing writer: no error returned, expected an error")
	}
}

// TestLZWLargeImage round-trips images big enough to force the LZW
// encoder through its 9->12 bit code width transitions and through
// dictionary resets, as well as a highly repetitive image.
func TestLZWLargeImage(t *testing.T) {
	state := uint32(12345)
	next := func() byte {
		state = state*1664525 + 1013904223
		return byte(state >> 24)
	}
	// 512x512 RGBA of noise: ~1 MiB of hard-to-compress data, which
	// exhausts the 4096-entry LZW dictionary many times over.
	noisy := image.NewRGBA(image.Rect(0, 0, 512, 512))
	for i := range noisy.Pix {
		noisy.Pix[i] = next()
	}
	// A highly repetitive image.
	uniform := image.NewGray(image.Rect(0, 0, 1000, 1000))
	for i := range uniform.Pix {
		uniform.Pix[i] = byte(i / 1000)
	}
	for name, m := range map[string]image.Image{"noisy": noisy, "uniform": uniform} {
		for _, predictor := range []bool{false, true} {
			out := new(bytes.Buffer)
			if err := Encode(out, m, &Options{Compression: LZW, Predictor: predictor}); err != nil {
				t.Fatalf("%s predictor=%v: %v", name, predictor, err)
			}
			m1, err := Decode(&buffer{buf: out.Bytes()})
			if err != nil {
				t.Fatalf("%s predictor=%v: %v", name, predictor, err)
			}
			compare(t, m, m1)
		}
	}
}

// TestEncodeZeroSizeLZW checks that a zero-size image is rejected before
// any LZW machinery runs.
func TestEncodeZeroSizeLZW(t *testing.T) {
	out := new(bytes.Buffer)
	if err := Encode(out, image.NewRGBA(image.Rectangle{}), &Options{Compression: LZW}); err == nil {
		t.Error("Encode of zero-size image: no error returned, expected an error")
	}
}

// TestStripError checks that LZW compression errors carry strip context
// while other compression errors are passed through unchanged.
func TestStripError(t *testing.T) {
	base := errors.New("boom")
	err := stripError(cLZW, image.Pt(4, 7), base)
	if err == nil || !strings.Contains(err.Error(), "strip 0") || !strings.Contains(err.Error(), "rows 0 to 7") {
		t.Errorf("stripError(cLZW): got %v, want strip context", err)
	}
	if !errors.Is(err, base) {
		t.Errorf("stripError(cLZW): got %v, want it to wrap the original error", err)
	}
	if got := stripError(cDeflate, image.Pt(4, 7), base); got != base {
		t.Errorf("stripError(cDeflate): got %v, want %v", got, base)
	}
	if got := stripError(cNone, image.Pt(4, 7), base); got != base {
		t.Errorf("stripError(cNone): got %v, want %v", got, base)
	}
	if got := stripError(cLZW, image.Pt(4, 7), nil); got != nil {
		t.Errorf("stripError(nil): got %v, want nil", got)
	}
}

func benchmarkEncode(b *testing.B, name string, pixelSize int) {
	b.Helper()
	img, err := openImage(name)
	if err != nil {
		b.Fatal(err)
	}
	s := img.Bounds().Size()
	b.SetBytes(int64(s.X * s.Y * pixelSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Encode(ioutil.Discard, img, nil)
	}
}

func BenchmarkEncode(b *testing.B)         { benchmarkEncode(b, "video-001.tiff", 4) }
func BenchmarkEncodePaletted(b *testing.B) { benchmarkEncode(b, "video-001-paletted.tiff", 1) }
func BenchmarkEncodeGray(b *testing.B)     { benchmarkEncode(b, "video-001-gray.tiff", 1) }
func BenchmarkEncodeGray16(b *testing.B)   { benchmarkEncode(b, "video-001-gray-16bit.tiff", 2) }
func BenchmarkEncodeRGBA(b *testing.B)     { benchmarkEncode(b, "video-001.tiff", 4) }
func BenchmarkEncodeRGBA64(b *testing.B)   { benchmarkEncode(b, "video-001-16bit.tiff", 8) }
