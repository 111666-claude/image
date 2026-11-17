// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"errors"
	"image"
	"image/color"
	"io"
	"testing"
)

// TestEncodePackBitsUnit checks the exact bytes produced by the row
// encoder. PackBits is described in section 9 (p. 42) of the TIFF spec.
func TestEncodePackBitsUnit(t *testing.T) {
	tests := []struct {
		name string
		src  []byte
		want []byte
	}{{
		// All repeated bytes: header -(n-1) followed by the byte itself.
		name: "short repeat",
		src:  []byte("\xaa\xaa\xaa"),
		want: []byte{0xfe, 0xaa}, // -3 -> repeat 3 times.
	}, {
		// All distinct bytes: literal header n-1 then the bytes.
		name: "all literal",
		src:  []byte{0x00, 0x11, 0x22, 0x33},
		want: []byte{0x03, 0x00, 0x11, 0x22, 0x33},
	}, {
		// Literal prefix, repeated run, literal suffix. The first 0xff is
		// part of the literal run because the encoder only starts a repeat
		// when it sees the second equal byte.
		name: "mixed",
		src:  []byte{0x01, 0x02, 0x03, 0xff, 0xff, 0xff, 0x04},
		want: []byte{
			0x02, 0x01, 0x02, 0x03, // 3 literal bytes.
			0xfe, 0xff, // repeat 0xff 3 times.
			0x00, 0x04, // 1 literal byte.
		},
	}, {
		// A run of exactly 128 equal bytes fits in one header (-127).
		name: "run of 128",
		src:  bytes.Repeat([]byte{0x7a}, 128),
		want: []byte{0x81, 0x7a}, // -127 -> repeat 128 times.
	}, {
		// A run longer than 128 bytes must be split into multiple headers.
		name: "run of 129",
		src:  bytes.Repeat([]byte{0x7a}, 129),
		want: []byte{
			0x81, 0x7a, // repeat 128 times.
			0x00, 0x7a, // 1 literal byte.
		},
	}, {
		// 256 repeated bytes: two full repeat headers, no cross-row sharing.
		name: "run of 256",
		src:  bytes.Repeat([]byte{0x00}, 256),
		want: []byte{
			0x81, 0x00, // repeat 128.
			0x81, 0x00, // repeat 128.
		},
	}, {
		// 128 distinct bytes use one literal header at the maximum length.
		name: "literal 128",
		src: func() []byte {
			b := make([]byte, 128)
			for i := range b {
				b[i] = byte(i)
			}
			return b
		}(),
		want: append([]byte{0x7f}, func() []byte {
			b := make([]byte, 128)
			for i := range b {
				b[i] = byte(i)
			}
			return b
		}()...),
	}, {
		// Alternating pixels never form a repeat run: everything is literal.
		name: "alternating",
		src:  []byte{0x00, 0xff, 0x00, 0xff, 0x00, 0xff},
		want: []byte{0x05, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff},
	}, {
		// A pair of equal bytes is already worth a repeat header.
		name: "pairs",
		src:  []byte{0xaa, 0xaa, 0xbb, 0xbb},
		want: []byte{
			0xff, 0xaa, // repeat 2.
			0xff, 0xbb, // repeat 2.
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := encodePackBitsRow(nil, tt.src)
			if !bytes.Equal(got, tt.want) {
				t.Fatalf("encodePackBitsRow = % x, want % x", got, tt.want)
			}
			// Every encoded row must decode back to the original.
			dec, err := unpackBits(bytes.NewReader(got), int64(len(tt.src))+1)
			if err != nil {
				t.Fatalf("unpackBits: %v", err)
			}
			if !bytes.Equal(dec, tt.src) {
				t.Fatalf("round trip = % x, want % x", dec, tt.src)
			}
		})
	}
}

// TestPackBitsWriterRows checks that the writer buffers arbitrary Write
// sizes and that each scanline is encoded independently: a run of equal
// bytes never crosses a row boundary.
func TestPackBitsWriterRows(t *testing.T) {
	const rowLen = 4
	// Two identical rows of repeated bytes. The run at the end of the first
	// row must not be merged with the start of the second row.
	pix := []byte{
		0xaa, 0xaa, 0xaa, 0xaa,
		0xaa, 0xaa, 0xaa, 0xaa,
	}
	var out bytes.Buffer
	pw := newPackBitsWriter(&out, rowLen)
	// Deliberately use writes that do not line up with row boundaries.
	if _, err := pw.Write(pix[:3]); err != nil {
		t.Fatal(err)
	}
	if _, err := pw.Write(pix[3:]); err != nil {
		t.Fatal(err)
	}
	if err := pw.Close(); err != nil {
		t.Fatal(err)
	}
	want := []byte{
		0xfd, 0xaa, // row 0: repeat 4.
		0xfd, 0xaa, // row 1: repeat 4, independent header.
	}
	if !bytes.Equal(out.Bytes(), want) {
		t.Fatalf("packBitsWriter = % x, want % x", out.Bytes(), want)
	}
	dec, err := unpackBits(bytes.NewReader(out.Bytes()), int64(len(pix)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dec, pix) {
		t.Fatalf("round trip = % x, want % x", dec, pix)
	}
}

// TestPackBitsWriterBadInput checks that data which is not a whole number
// of scanlines is rejected instead of producing a corrupt stream.
func TestPackBitsWriterBadInput(t *testing.T) {
	var out bytes.Buffer
	pw := newPackBitsWriter(&out, 4)
	if _, err := pw.Write([]byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := pw.Close(); err == nil {
		t.Fatal("Close: got nil error, want non-nil for partial scanline")
	}
	if out.Len() != 0 {
		t.Fatalf("got %d output bytes after error, want 0", out.Len())
	}

	// A non-positive row length can never describe a valid scanline.
	pw = newPackBitsWriter(&out, 0)
	if err := pw.Close(); err == nil {
		t.Fatal("Close: got nil error, want non-nil for zero row length")
	}
}

// failingWriter returns errFail after accepting at most failAt bytes in
// total, to simulate short writes.
type failingWriter struct {
	failAt int
	n      int
}

var errFail = errors.New("tiff: simulated short write")

func (f *failingWriter) Write(p []byte) (int, error) {
	avail := f.failAt - f.n
	if avail <= 0 {
		return 0, errFail
	}
	if len(p) > avail {
		f.n += avail
		return avail, errFail
	}
	f.n += len(p)
	return len(p), nil
}

// TestEncodePackBitsShortWrite checks that a short or failing write is
// reported as an error rather than silently producing a truncated file.
func TestEncodePackBitsShortWrite(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for i := range img.Pix {
		img.Pix[i] = byte(i)
	}
	for _, failAt := range []int{0, 1, 7, 64} {
		fw := &failingWriter{failAt: failAt}
		err := Encode(fw, img, &Options{Compression: PackBits})
		if err == nil {
			t.Fatalf("failAt=%d: got nil error, want non-nil", failAt)
		}
	}
}

// TestEncodePackBitsRoundtrip encodes every currently encodable image type
// with PackBits and decodes it back, checking pixel equality. It also
// verifies the IFD tags against the actual written data.
func TestEncodePackBitsRoundtrip(t *testing.T) {
	const w, h = 13, 7 // Non-neat width on purpose.
	pal := color.Palette{
		color.Black, color.Opaque,
		color.RGBA{0x11, 0x22, 0x33, 0xff},
		color.RGBA{0x44, 0x55, 0x66, 0xff},
	}
	images := []struct {
		name string
		img  image.Image
	}{
		{"gray", makeGray(w, h)},
		{"paletted", makePaletted(w, h, pal)},
		{"rgba", makeRGBA(w, h)},
		{"nrgba", makeNRGBA(w, h)},
		{"gray16", makeGray16(w, h)},
		{"rgba64", makeRGBA64(w, h)},
	}
	for _, im := range images {
		t.Run(im.name, func(t *testing.T) {
			var out bytes.Buffer
			if err := Encode(&out, im.img, &Options{Compression: PackBits}); err != nil {
				t.Fatalf("Encode: %v", err)
			}
			raw := out.Bytes()

			// The Compression, StripOffsets and StripByteCounts tags must
			// describe the data actually written.
			checkPackBitsIFD(t, raw, im.img)

			img2, err := Decode(bytes.NewReader(raw))
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			compare(t, im.img, img2)
		})
	}
}

// TestEncodePackBitsRoundtripFixtures runs the existing decode fixtures
// through PackBits encoding and back.
func TestEncodePackBitsRoundtripFixtures(t *testing.T) {
	for _, name := range []string{
		"video-001.tiff",
		"video-001-gray.tiff",
		"video-001-paletted.tiff",
		"video-001-16bit.tiff",
		"video-001-gray-16bit.tiff",
	} {
		img, err := openImage(name)
		if err != nil {
			t.Fatal(err)
		}
		var out bytes.Buffer
		if err := Encode(&out, img, &Options{Compression: PackBits}); err != nil {
			t.Fatalf("Encode(%s): %v", name, err)
		}
		img2, err := Decode(bytes.NewReader(out.Bytes()))
		if err != nil {
			t.Fatalf("Decode(%s): %v", name, err)
		}
		compare(t, img, img2)
	}
}

// TestEncodePackBitsInterop checks that our PackBits output decodes to the
// same pixels as the existing PackBits-compressed fixture, and that the
// fixture itself round-trips through our encoder.
func TestEncodePackBitsInterop(t *testing.T) {
	ref, err := load("bw-packbits.tiff")
	if err != nil {
		t.Fatal(err)
	}
	// Re-encoding the decoded image with PackBits must stay readable and
	// pixel-identical.
	var out bytes.Buffer
	if err := Encode(&out, ref, &Options{Compression: PackBits}); err != nil {
		t.Fatal(err)
	}
	img, err := Decode(bytes.NewReader(out.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	compare(t, ref, img)
}

// TestEncodePackBitsErrors covers empty images and pathological dimensions.
func TestEncodePackBitsErrors(t *testing.T) {
	t.Run("zero width", func(t *testing.T) {
		err := Encode(io.Discard, image.NewGray(image.Rect(0, 0, 0, 4)), &Options{Compression: PackBits})
		if err == nil {
			t.Fatal("got nil error, want zero-size error")
		}
	})
	t.Run("zero height", func(t *testing.T) {
		err := Encode(io.Discard, image.NewGray(image.Rect(0, 0, 4, 0)), &Options{Compression: PackBits})
		if err == nil {
			t.Fatal("got nil error, want zero-size error")
		}
	})
	t.Run("negative bounds", func(t *testing.T) {
		// No concrete image type can be constructed with negative bounds,
		// but a generic image.Image can report them; Encode must reject it
		// instead of computing bogus offsets. image.Rect canonicalizes, so
		// build the malformed rectangle directly.
		err := Encode(io.Discard, fakeImage{r: image.Rectangle{
			Min: image.Pt(0, 0),
			Max: image.Pt(-4, 4),
		}}, &Options{Compression: PackBits})
		if err == nil {
			t.Fatal("got nil error, want negative-dimensions error")
		}
	})
	t.Run("huge dimensions", func(t *testing.T) {
		// The pixel data cannot fit in an int; no offsets may be written.
		// Use a fake image so no backing array is ever allocated.
		err := Encode(io.Discard, fakeImage{r: image.Rect(0, 0, 1<<30, 1<<30)}, &Options{Compression: PackBits})
		if err == nil {
			t.Fatal("got nil error, want image-too-large error")
		}
	})
}

// fakeImage is an image.Image with arbitrary bounds and no pixel storage,
// for testing dimension validation without allocating large arrays.
type fakeImage struct {
	r image.Rectangle
}

func (f fakeImage) ColorModel() color.Model { return color.RGBAModel }
func (f fakeImage) Bounds() image.Rectangle { return f.r }
func (f fakeImage) At(x, y int) color.Color { return color.Transparent }

// TestEncodePackBitsNoRegression checks that the pre-existing compression
// options keep their spec compression value.
func TestEncodePackBitsNoRegression(t *testing.T) {
	compressions := []struct {
		c    CompressionType
		spec uint32
	}{
		{Uncompressed, cNone},
		{Deflate, cDeflate},
		{PackBits, cPackBits},
	}
	for _, c := range compressions {
		if got := c.c.specValue(); got != c.spec {
			t.Errorf("specValue(%d) = %d, want %d", c.c, got, c.spec)
		}
	}
	// LZW encoding is still unsupported by the writer.
	if err := Encode(io.Discard, image.NewGray(image.Rect(0, 0, 1, 1)), &Options{Compression: LZW}); err == nil {
		t.Error("Encode(LZW): got nil error, want non-nil")
	}
}

// checkPackBitsIFD parses the IFD of raw and verifies the compression,
// strip offset and byte-count tags are consistent with the file contents.
func checkPackBitsIFD(t *testing.T, raw []byte, img image.Image) {
	t.Helper()
	if len(raw) < 8 || string(raw[:4]) != leHeader {
		t.Fatal("not a little-endian TIFF file")
	}
	ifdOff := int(binary.LittleEndian.Uint32(raw[4:8]))
	if ifdOff <= 8 || ifdOff+2 > len(raw) {
		t.Fatalf("bad IFD offset %d", ifdOff)
	}
	n := int(binary.LittleEndian.Uint16(raw[ifdOff:]))
	entries := map[uint16]uint32{}
	for i := 0; i < n; i++ {
		e := raw[ifdOff+2+i*ifdLen:]
		tag := binary.LittleEndian.Uint16(e[0:2])
		datatype := binary.LittleEndian.Uint16(e[2:4])
		count := binary.LittleEndian.Uint32(e[4:8])
		var val uint32
		if count*lengths[datatype] <= 4 {
			val = binary.LittleEndian.Uint32(e[8:12])
		} else {
			// These tags are all single SHORT/LONG values in this writer.
			val = binary.LittleEndian.Uint32(raw[binary.LittleEndian.Uint32(e[8:12]):])
		}
		if tag == tCompression || tag == tStripOffsets || tag == tStripByteCounts {
			entries[tag] = val
		}
	}
	if entries[tCompression] != cPackBits {
		t.Errorf("Compression = %d, want %d", entries[tCompression], cPackBits)
	}
	strips, ok := entries[tStripOffsets]
	if !ok {
		t.Fatal("missing StripOffsets tag")
	}
	if strips != 8 {
		t.Errorf("StripOffsets = %d, want 8 (right after header)", strips)
	}
	counts, ok := entries[tStripByteCounts]
	if !ok {
		t.Fatal("missing StripByteCounts tag")
	}
	// The single strip runs from offset 8 up to the IFD, so its byte count
	// plus 8 equals the IFD offset.
	if uint32(ifdOff) != counts+8 {
		t.Errorf("StripByteCounts = %d, IFD offset = %d, want IFD offset = count+8", counts, ifdOff)
	}
	if counts == 0 {
		t.Error("StripByteCounts is zero")
	}
	// The compressed strip must decode to the full uncompressed pixel data.
	dec, err := unpackBits(bytes.NewReader(raw[8:8+counts]), int64(ifdOff))
	if err != nil {
		t.Fatalf("unpackBits strip: %v", err)
	}
	rowLen := img.Bounds().Dx() * pixelSize(img)
	wantLen := rowLen * img.Bounds().Dy()
	if len(dec) != wantLen {
		t.Fatalf("decoded strip = %d bytes, want %d", len(dec), wantLen)
	}
}

// Image constructors with deterministic, PackBits-unfriendly content
// (repeats, alternation and non-repeating gradients).

func makeGray(w, h int) *image.Gray {
	m := image.NewGray(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetGray(x, y, color.Gray{Y: byte((x*7 + y*13) % 256)})
		}
	}
	return m
}

func makePaletted(w, h int, pal color.Palette) *image.Paletted {
	m := image.NewPaletted(image.Rect(0, 0, w, h), pal)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.Pix[y*m.Stride+x] = uint8((x + y) % len(pal))
		}
	}
	return m
}

func makeRGBA(w, h int) *image.RGBA {
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetRGBA(x, y, color.RGBA{
				R: byte(x * 17),
				G: byte(y * 29),
				B: byte((x + y) * 31),
				A: 0xff,
			})
		}
	}
	return m
}

func makeNRGBA(w, h int) *image.NRGBA {
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetNRGBA(x, y, color.NRGBA{
				R: byte(x * 19),
				G: byte(y * 23),
				B: byte((x ^ y) * 37),
				A: byte(128 + x),
			})
		}
	}
	return m
}

func makeGray16(w, h int) *image.Gray16 {
	m := image.NewGray16(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetGray16(x, y, color.Gray16{Y: uint16(x*257 + y*4099)})
		}
	}
	return m
}

func makeRGBA64(w, h int) *image.RGBA64 {
	m := image.NewRGBA64(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			m.SetRGBA64(x, y, color.RGBA64{
				R: uint16(x * 4099),
				G: uint16(y * 257),
				B: uint16((x + y) * 1031),
				A: 0xffff,
			})
		}
	}
	return m
}
