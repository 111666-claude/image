// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bmp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"testing"

	_ "image/png"
)

const testdataDir = "../testdata/"

func compare(img0, img1 image.Image) error {
	b := img1.Bounds()
	if !b.Eq(img0.Bounds()) {
		return fmt.Errorf("wrong image size: want %s, got %s", img0.Bounds(), b)
	}
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c0 := img0.At(x, y)
			c1 := img1.At(x, y)
			r0, g0, b0, a0 := c0.RGBA()
			r1, g1, b1, a1 := c1.RGBA()
			if r0 != r1 || g0 != g1 || b0 != b1 || a0 != a1 {
				return fmt.Errorf("pixel at (%d, %d) has wrong color: want %v, got %v", x, y, c0, c1)
			}
		}
	}
	return nil
}

// TestDecode tests that decoding a PNG image and a BMP image result in the
// same pixel data.
func TestDecode(t *testing.T) {
	testCases := []string{
		"colormap",
		"colormap-0",
		"colormap-251",
		"video-001",
		"yellow_rose-small",
		"yellow_rose-small-v5",
		"bmp_1bpp",
		"bmp_4bpp",
		"bmp_8bpp",
	}

	for _, tc := range testCases {
		f0, err := os.Open(testdataDir + tc + ".png")
		if err != nil {
			t.Errorf("%s: Open PNG: %v", tc, err)
			continue
		}
		defer f0.Close()
		img0, _, err := image.Decode(f0)
		if err != nil {
			t.Errorf("%s: Decode PNG: %v", tc, err)
			continue
		}

		f1, err := os.Open(testdataDir + tc + ".bmp")
		if err != nil {
			t.Errorf("%s: Open BMP: %v", tc, err)
			continue
		}
		defer f1.Close()
		img1, _, err := image.Decode(f1)
		if err != nil {
			t.Errorf("%s: Decode BMP: %v", tc, err)
			continue
		}

		if err := compare(img0, img1); err != nil {
			t.Errorf("%s: %v", tc, err)
			continue
		}
	}
}

func TestDecodeConstructed(t *testing.T) {
	for _, tc := range []struct {
		name    string
		b       []byte
		wantErr error
	}{{
		name: "1x1 paletted",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 1,
			colorsUsed:   2,
			colorTable: []colorTableEntry{
				{0, 0, 0},
				{0xff, 0xff, 0xff},
			},
			data: []byte{0, 0, 0, 0},
		}.Bytes(),
		wantErr: nil, // successful base case
	}, {
		name: "1x1 rgb",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 24,
			data: []byte{
				0, 0, 0, 0,
			},
		}.Bytes(),
		wantErr: nil, // successful base case
	}, {
		name: "invalid palette index",
		b: bmpBuilder{
			width:        1,
			height:       1,
			planes:       1,
			bitsPerPixel: 8,
			colorsUsed:   2,
			colorTable: []colorTableEntry{
				{0, 0, 0},
				{0xff, 0xff, 0xff},
			},
			data: []byte{
				2, 0, 0, 0, // index 2
			},
		}.Bytes(),
		wantErr: errInvalidPaletteIndex,
	}, {
		name: "0x1",
		b: bmpBuilder{
			width:        1,
			height:       0,
			planes:       1,
			bitsPerPixel: 24,
			data: []byte{
				0, 0, 0, 0,
			},
		}.Bytes(),
		wantErr: ErrUnsupported,
	}, {
		name: "excessive width and height",
		b: bmpBuilder{
			width:        2147483647,
			height:       2147483647,
			planes:       1,
			bitsPerPixel: 24,
		}.Bytes(),
		wantErr: ErrUnsupported,
	}} {
		img, _, err := image.Decode(bytes.NewReader(tc.b))
		if err != tc.wantErr {
			t.Errorf("%v: Decode error %v; want %v", tc.name, err, tc.wantErr)
		}
		if err != nil {
			continue
		}
		_ = img.At(0, 0) // try accessing a pixel
	}
}

// TestEOF tests that decoding a BMP image returns io.ErrUnexpectedEOF
// when there are no headers or data is empty
func TestEOF(t *testing.T) {
	_, err := Decode(bytes.NewReader(nil))
	if err != io.ErrUnexpectedEOF {
		t.Errorf("Error should be io.ErrUnexpectedEOF on nil but got %v", err)
	}
}

// v5Pixels is the pixel array of a 2x2 24bpp bottom-up BMP:
// bottom row red, green; top row blue, white.
var v5Pixels = []byte{
	0x00, 0x00, 0xff, 0x00, 0xff, 0x00, 0x00, 0x00, // bottom row: red, green (BGR)
	0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, // top row: blue, white (BGR)
}

func checkV5Pixels(t *testing.T, img image.Image) {
	t.Helper()
	want := map[image.Point]color.RGBA{
		{0, 0}: {0x00, 0x00, 0xff, 0xff}, // top-left: blue
		{1, 0}: {0xff, 0xff, 0xff, 0xff}, // top-right: white
		{0, 1}: {0xff, 0x00, 0x00, 0xff}, // bottom-left: red
		{1, 1}: {0x00, 0xff, 0x00, 0xff}, // bottom-right: green
	}
	for p, w := range want {
		r, g, b, a := img.At(p.X, p.Y).RGBA()
		got := color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), uint8(a >> 8)}
		if got != w {
			t.Errorf("pixel at %v: got %v, want %v", p, got, w)
		}
	}
}

func TestDecodeWithProfile(t *testing.T) {
	icc := []byte("hand-crafted fake ICC profile")
	link := []byte("myprofile.icc\x00padding")
	const (
		lcsSRGB    = 0x73524742 // 'sRGB'
		lcsWindows = 0x57696e20 // 'Win '
	)
	for _, tc := range []struct {
		name      string
		b         []byte
		want      *ColorProfile
		wantErr   error
		wantPixel bool // check the decoded pixels
	}{{
		name: "embedded profile",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:  lcsProfileEmbedded,
			profile: icc,
			data:    v5Pixels,
		}.Bytes(),
		want:      &ColorProfile{Embedded: icc},
		wantPixel: true,
	}, {
		name: "embedded profile after a gap",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileEmbedded,
			profileData: v5InfoHeaderLen + uint32(len(v5Pixels)) + 5,
			profile:     icc,
			data:        v5Pixels,
		}.Bytes(),
		want:      &ColorProfile{Embedded: icc},
		wantPixel: true,
	}, {
		name: "linked profile",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:  lcsProfileLinked,
			profile: link,
			data:    v5Pixels,
		}.Bytes(),
		want:      &ColorProfile{Linked: "myprofile.icc"},
		wantPixel: true,
	}, {
		name: "no profile",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType: lcsSRGB,
			data:   v5Pixels,
		}.Bytes(),
		want:      nil,
		wantPixel: true,
	}, {
		name: "unsupported color space type ignores profile fields",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsWindows,
			profileData: v5InfoHeaderLen + uint32(len(v5Pixels)),
			profileSize: 10,
			data:        v5Pixels,
		}.Bytes(),
		want:      nil,
		wantPixel: true,
	}, {
		name: "profile overlaps pixel array",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileEmbedded,
			profileData: v5InfoHeaderLen, // start of the pixel array
			profileSize: 8,
			data:        v5Pixels,
		}.Bytes(),
		wantErr: errInvalidProfile,
	}, {
		name: "profile overlaps header",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileEmbedded,
			profileData: 10,
			profileSize: 8,
			data:        v5Pixels,
		}.Bytes(),
		wantErr: errInvalidProfile,
	}, {
		name: "profile out of bounds",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileEmbedded,
			profileData: v5InfoHeaderLen + uint32(len(v5Pixels)),
			profileSize: 100, // the file ends right after the pixel array
			data:        v5Pixels,
		}.Bytes(),
		wantErr: errInvalidProfile,
	}, {
		name: "excessive profile size",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileEmbedded,
			profileData: v5InfoHeaderLen + uint32(len(v5Pixels)),
			profileSize: 0xffffffff,
			data:        v5Pixels,
		}.Bytes(),
		wantErr: errInvalidProfile,
	}, {
		name: "linked profile out of bounds",
		b: bmpV5Builder{
			width: 2, height: 2, bitsPerPixel: 24,
			csType:      lcsProfileLinked,
			profileData: v5InfoHeaderLen + uint32(len(v5Pixels)) + 1000,
			profileSize: 20,
			data:        v5Pixels,
		}.Bytes(),
		wantErr: errInvalidProfile,
	}} {
		img, p, err := DecodeWithProfile(bytes.NewReader(tc.b))
		if err != tc.wantErr {
			t.Errorf("%s: DecodeWithProfile error %v; want %v", tc.name, err, tc.wantErr)
			continue
		}
		if tc.wantErr != nil {
			continue
		}
		if tc.want == nil {
			if p != nil {
				t.Errorf("%s: got profile %+v; want nil", tc.name, p)
			}
		} else if p == nil {
			t.Errorf("%s: got nil profile; want %+v", tc.name, tc.want)
		} else if !bytes.Equal(p.Embedded, tc.want.Embedded) || p.Linked != tc.want.Linked {
			t.Errorf("%s: got profile %+v; want %+v", tc.name, p, tc.want)
		}
		if tc.wantPixel {
			checkV5Pixels(t, img)
			// Decode of the same input must produce identical pixels.
			img2, err := Decode(bytes.NewReader(tc.b))
			if err != nil {
				t.Errorf("%s: Decode: %v", tc.name, err)
			} else if err := compare(img, img2); err != nil {
				t.Errorf("%s: DecodeWithProfile vs Decode: %v", tc.name, err)
			}
		}
	}
}

// TestDecodeWithProfileV3 tests that DecodeWithProfile on a pre-V5 BMP
// returns a nil profile and the same pixels as Decode.
func TestDecodeWithProfileV3(t *testing.T) {
	b := bmpBuilder{
		width:        1,
		height:       1,
		planes:       1,
		bitsPerPixel: 24,
		data:         []byte{0x00, 0x00, 0xff, 0x00}, // red (BGR)
	}.Bytes()
	img, p, err := DecodeWithProfile(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("DecodeWithProfile: %v", err)
	}
	if p != nil {
		t.Errorf("got profile %+v; want nil", p)
	}
	img2, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(img, img2); err != nil {
		t.Error(err)
	}
}

// TestDecodeWithProfileTruncatedHeader tests that a truncated BITMAPV5HEADER
// is reported as an unexpected EOF, as it is for Decode.
func TestDecodeWithProfileTruncatedHeader(t *testing.T) {
	b := bmpV5Builder{
		width: 2, height: 2, bitsPerPixel: 24,
		csType:  lcsProfileEmbedded,
		profile: []byte("profile"),
		data:    v5Pixels,
	}.Bytes()
	_, _, err := DecodeWithProfile(bytes.NewReader(b[:fileHeaderLen+60]))
	if err != io.ErrUnexpectedEOF {
		t.Errorf("got %v; want io.ErrUnexpectedEOF", err)
	}
}

// TestDecodeWithProfileV5Testdata tests the V5 BMP in testdata, which uses
// the LCS_sRGB color space type and hence carries no profile.
func TestDecodeWithProfileV5Testdata(t *testing.T) {
	f, err := os.Open(testdataDir + "yellow_rose-small-v5.bmp")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, p, err := DecodeWithProfile(f)
	if err != nil {
		t.Fatalf("DecodeWithProfile: %v", err)
	}
	if p != nil {
		t.Errorf("got profile %+v; want nil", p)
	}
	f2, err := os.Open(testdataDir + "yellow_rose-small-v5.bmp")
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()
	img2, err := Decode(f2)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if err := compare(img, img2); err != nil {
		t.Error(err)
	}
}

type bmpV5Builder struct {
	width        int32
	height       int32
	bitsPerPixel uint16
	csType       uint32
	profileData  uint32 // bV5ProfileData; 0 means "immediately after the pixel array"
	profileSize  uint32 // bV5ProfileSize; 0 means len(profile)
	profile      []byte // appended after the pixel array (and any gap)
	data         []byte // the pixel array
}

func (b bmpV5Builder) Bytes() []byte {
	buf := []byte{
		0x42, 0x4d, // 'BM'
		0x00, 0x00, 0x00, 0x00, // file size
		0x00, 0x00, 0x00, 0x00, // reserved
		0x00, 0x00, 0x00, 0x00, // data offset
	}
	dib := make([]byte, v5InfoHeaderLen)
	binary.LittleEndian.PutUint32(dib[0:], v5InfoHeaderLen)
	binary.LittleEndian.PutUint32(dib[4:], uint32(b.width))
	binary.LittleEndian.PutUint32(dib[8:], uint32(b.height))
	binary.LittleEndian.PutUint16(dib[12:], 1) // planes
	binary.LittleEndian.PutUint16(dib[14:], b.bitsPerPixel)
	binary.LittleEndian.PutUint32(dib[56:], b.csType)
	profileData, profileSize := b.profileData, b.profileSize
	if b.profile != nil {
		if profileData == 0 {
			profileData = v5InfoHeaderLen + uint32(len(b.data))
		}
		if profileSize == 0 {
			profileSize = uint32(len(b.profile))
		}
	}
	binary.LittleEndian.PutUint32(dib[112:], profileData)
	binary.LittleEndian.PutUint32(dib[116:], profileSize)
	buf = append(buf, dib...)
	binary.LittleEndian.PutUint32(buf[10:], uint32(len(buf))) // data offset
	buf = append(buf, b.data...)
	if b.profile != nil {
		abs := fileHeaderLen + int(profileData)
		for len(buf) < abs {
			buf = append(buf, 0)
		}
		buf = append(buf, b.profile...)
	}
	binary.LittleEndian.PutUint32(buf[2:], uint32(len(buf))) // file size
	return buf
}

type bmpBuilder struct {
	width           int32
	height          int32
	planes          uint16
	bitsPerPixel    uint16
	compression     uint32
	imageSize       uint32
	xppm            uint32
	yppm            uint32
	colorsUsed      uint32
	colorsImportant uint32
	colorTable      []colorTableEntry
	data            []byte
}

type colorTableEntry struct {
	r, g, b byte
}

func (b bmpBuilder) Bytes() []byte {
	buf := []byte{
		0x42, 0x4d, // 'BM'
		0x00, 0x00, 0x00, 0x00, // file size
		0x00, 0x00, 0x00, 0x00, // reserved
		0x00, 0x00, 0x00, 0x00, // data offset
		0x28, 0x00, 0x00, 0x00, // header size (40)
	}
	buf = binary.LittleEndian.AppendUint32(buf, uint32(b.width))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(b.height))
	buf = binary.LittleEndian.AppendUint16(buf, b.planes)
	buf = binary.LittleEndian.AppendUint16(buf, b.bitsPerPixel)
	buf = binary.LittleEndian.AppendUint32(buf, b.compression)
	buf = binary.LittleEndian.AppendUint32(buf, b.imageSize)
	buf = binary.LittleEndian.AppendUint32(buf, b.xppm)
	buf = binary.LittleEndian.AppendUint32(buf, b.yppm)
	buf = binary.LittleEndian.AppendUint32(buf, b.colorsUsed)
	buf = binary.LittleEndian.AppendUint32(buf, b.colorsImportant)
	for _, e := range b.colorTable {
		buf = append(buf, e.r, e.g, e.b, 0)
	}
	binary.LittleEndian.PutUint32(buf[10:], uint32(len(buf))) // data offset
	buf = append(buf, b.data...)
	binary.LittleEndian.PutUint32(buf[2:], uint32(len(buf))) // file size
	return buf
}
