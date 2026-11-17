// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"testing"
)

// buildTIFF constructs an uncompressed TIFF from raw strip data.
// The on-disk layout is: header, strip pixel data, IFD (entries followed by the
// next-IFD terminator), then an external data area for entries whose values do
// not fit in the inline 4-byte value field. entries supplies the IFD tags
// (ImageWidth, ImageLength, StripOffsets and StripByteCounts are added
// automatically); supported value types are uint16, uint32, []uint16 and
// []uint32. If orient is non-zero, an Orientation tag with that value is
// added; orient == 0 omits the tag.
func buildTIFF(t *testing.T, enc byteOrder, w, h int, orient uint16, strips [][]byte, entries map[uint16]any) []byte {
	t.Helper()

	// Start with the header and all strip pixel data.
	var data []byte
	switch enc.Uint16([]byte{1, 0}) {
	case 0x1:
		data = append(data, 'I', 'I', 42, 0)
	default:
		data = append(data, 'M', 'M', 0, 42)
	}
	data = append(data, 0, 0, 0, 0) // IFD offset, patched later.

	offsets := make([]uint32, len(strips))
	counts := make([]uint32, len(strips))
	for i, strip := range strips {
		offsets[i] = uint32(len(data))
		counts[i] = uint32(len(strip))
		data = append(data, strip...)
	}

	all := map[uint16]any{
		tImageWidth:      uint32(w),
		tImageLength:     uint32(h),
		tCompression:     uint16(cNone),
		tStripOffsets:    offsets,
		tStripByteCounts: counts,
	}
	if orient != 0 {
		all[tOrientation] = orient
	}
	for tag, val := range entries {
		all[tag] = val
	}

	var tags []uint16
	for tag := range all {
		tags = append(tags, tag)
	}
	sortUint16s(tags)

	ifdOff := uint32(len(data))
	// External data begins after the entry count, the entries and the
	// four-byte next-IFD terminator.
	externalOff := ifdOff + 2 + uint32(len(tags))*ifdLen + 4
	var external []byte

	data = enc.AppendUint16(data, uint16(len(tags)))
	for _, tag := range tags {
		var datatype uint16
		var values []uint32
		switch v := all[tag].(type) {
		case uint16:
			datatype, values = dtShort, []uint32{uint32(v)}
		case uint32:
			datatype, values = dtLong, []uint32{v}
		case []uint16:
			datatype = dtShort
			values = make([]uint32, len(v))
			for i, e := range v {
				values[i] = uint32(e)
			}
		case []uint32:
			datatype, values = dtLong, v
		default:
			t.Fatalf("tag %d: unsupported entry type %T", tag, all[tag])
		}

		data = enc.AppendUint16(data, tag)
		data = enc.AppendUint16(data, datatype)
		data = enc.AppendUint32(data, uint32(len(values)))

		datalen := uint32(len(values)) * lengths[datatype]
		if datalen <= 4 {
			start := len(data)
			for _, v := range values {
				switch datatype {
				case dtShort:
					data = enc.AppendUint16(data, uint16(v))
				case dtLong:
					data = enc.AppendUint32(data, v)
				}
			}
			for len(data)-start < 4 {
				data = append(data, 0)
			}
		} else {
			ptr := externalOff + uint32(len(external))
			data = enc.AppendUint32(data, ptr)
			for _, v := range values {
				switch datatype {
				case dtShort:
					external = enc.AppendUint16(external, uint16(v))
				case dtLong:
					external = enc.AppendUint32(external, v)
				}
			}
		}
	}
	data = enc.AppendUint32(data, 0) // no next IFD
	data = append(data, external...)

	enc.PutUint32(data[4:8], ifdOff)
	return data
}

// sortUint16s sorts a slice of uint16 in ascending order.
func sortUint16s(s []uint16) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

// pixOf returns the backing pixel slice, stride and pixel size for the
// concrete image types the decoder produces.
func pixOf(img image.Image) (pix []byte, stride, pixelSize int, ok bool) {
	switch img := img.(type) {
	case *image.Gray:
		return img.Pix, img.Stride, 1, true
	case *image.Gray16:
		return img.Pix, img.Stride, 2, true
	case *image.Paletted:
		return img.Pix, img.Stride, 1, true
	case *image.NRGBA:
		return img.Pix, img.Stride, 4, true
	case *image.NRGBA64:
		return img.Pix, img.Stride, 8, true
	case *image.RGBA:
		return img.Pix, img.Stride, 4, true
	case *image.RGBA64:
		return img.Pix, img.Stride, 8, true
	}
	return nil, 0, 0, false
}

// allocLike allocates a new image of the same concrete type as src with the
// given dimensions, preserving a Paletted image's palette.
func allocLike(src image.Image, w, h int) image.Image {
	rect := image.Rect(0, 0, w, h)
	switch src := src.(type) {
	case *image.Gray:
		return image.NewGray(rect)
	case *image.Gray16:
		return image.NewGray16(rect)
	case *image.Paletted:
		return image.NewPaletted(rect, src.Palette)
	case *image.NRGBA:
		return image.NewNRGBA(rect)
	case *image.NRGBA64:
		return image.NewNRGBA64(rect)
	case *image.RGBA:
		return image.NewRGBA(rect)
	case *image.RGBA64:
		return image.NewRGBA64(rect)
	}
	return nil
}

// refOrientation is an independent reference implementation written directly
// from the TIFF 6.0 Orientation semantics. It uses the forward mapping (where
// each stored pixel ends up), whereas the production code uses the inverse
// mapping (which stored pixel fills each output position), so agreement
// between the two validates both derivations. Pixel bytes are copied whole to
// preserve the source precision.
func refOrientation(t *testing.T, src image.Image, o Orientation) image.Image {
	t.Helper()
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dw, dh := w, h
	if o >= LeftTop {
		dw, dh = h, w
	}
	dst := allocLike(src, dw, dh)
	if dst == nil {
		t.Fatalf("unsupported image type %T", src)
	}
	spix, sstride, ps, _ := pixOf(src)
	dpix, dstride, _, _ := pixOf(dst)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var dx, dy int
			switch o {
			case TopLeft:
				dx, dy = x, y
			case TopRight:
				dx, dy = w-1-x, y
			case BottomRight:
				dx, dy = w-1-x, h-1-y
			case BottomLeft:
				dx, dy = x, h-1-y
			case LeftTop:
				dx, dy = y, x
			case RightTop:
				dx, dy = h-1-y, x
			case RightBottom:
				dx, dy = h-1-y, w-1-x
			case LeftBottom:
				dx, dy = y, w-1-x
			default:
				t.Fatalf("unknown orientation %d", o)
			}
			sOff := y*sstride + x*ps
			dOff := dy*dstride + dx*ps
			copy(dpix[dOff:dOff+ps], spix[sOff:sOff+ps])
		}
	}
	return dst
}

// assertSamePix verifies that two images have the same concrete type, bounds
// and identical backing pixel bytes.
func assertSamePix(t *testing.T, got, want image.Image) {
	t.Helper()
	if fmt.Sprintf("%T", got) != fmt.Sprintf("%T", want) {
		t.Fatalf("image type: got %T, want %T", got, want)
	}
	if got.Bounds() != want.Bounds() {
		t.Fatalf("bounds: got %v, want %v", got.Bounds(), want.Bounds())
	}
	gpix, _, _, ok := pixOf(got)
	if !ok {
		t.Fatalf("unsupported image type %T", got)
	}
	wpix, _, _, _ := pixOf(want)
	if !bytes.Equal(gpix, wpix) {
		t.Fatalf("pixel data differs:\n got  %x\n want %x", gpix, wpix)
	}
}

// inverseOrientation maps each Orientation to the orientation that undoes it.
// The reflections (2, 4, 5, 7) are self-inverse; the two 90-degree rotations
// (6 and 8) undo each other.
var inverseOrientation = map[Orientation]Orientation{
	TopLeft:     TopLeft,
	TopRight:    TopRight,
	BottomRight: BottomRight,
	BottomLeft:  BottomLeft,
	LeftTop:     LeftTop,
	RightTop:    LeftBottom,
	RightBottom: RightBottom,
	LeftBottom:  RightTop,
}

// rgbaToPackedRGB returns the packed 3-bytes-per-pixel strip data of an
// 8-bit RGB image.
func rgbaToPackedRGB(img *image.RGBA) []byte {
	b := img.Bounds()
	out := make([]byte, 0, b.Dx()*b.Dy()*3)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.RGBAAt(x, y)
			out = append(out, c.R, c.G, c.B)
		}
	}
	return out
}

// TestOrientationAllTransforms uses a fixed 5x3 RGBA TIFF whose four corners
// have distinct colors and whose every pixel has a unique color, and checks
// all eight Orientation values in two complementary ways:
//
// Design A feeds the same stored buffer through every tag and verifies the
// normalized output against an independent reference implementation written
// from the TIFF 6.0 forward mapping (whereas the production code derives the
// inverse mapping), in both byte orders.
//
// Design B models a real scan batch: for each orientation, a file is written
// whose stored pixels are the canonical scene pre-transformed by the inverse
// orientation and whose dimensions are swapped when the orientation requires
// it. Normalizing every such file must reproduce the identical canonical
// image, including all four corner colors and the canonical dimensions.
func TestOrientationAllTransforms(t *testing.T) {
	const w, h = 5, 3
	// Canonical (orientation 1) raw RGB pixel data. Every pixel gets a unique
	// color so that any wrong source/destination mapping is detectable.
	raw := make([]byte, 0, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			raw = append(raw, byte(x*40+1), byte(y*80+2), byte(x*16+y*32+4))
		}
	}
	rgbEntries := map[uint16]any{
		tBitsPerSample:             []uint16{8, 8, 8},
		tPhotometricInterpretation: uint16(pRGB),
	}

	for _, enc := range []struct {
		name string
		bo   byteOrder
	}{
		{"little-endian", binary.LittleEndian},
		{"big-endian", binary.BigEndian},
	} {
		t.Run(enc.name, func(t *testing.T) {
			// Decode an orientation-1 file in this byte order to obtain the
			// canonical image exactly as the decoder sees it, rather than
			// assuming a particular in-memory byte order.
			baseData := buildTIFF(t, enc.bo, w, h, 1, [][]byte{raw}, rgbEntries)
			decoded, err := Decode(bytes.NewReader(baseData))
			if err != nil {
				t.Fatal(err)
			}
			canonical := decoded.(*image.RGBA)
			corners := map[string]color.RGBA{
				"top-left":     canonical.RGBAAt(0, 0),
				"top-right":    canonical.RGBAAt(w-1, 0),
				"bottom-left":  canonical.RGBAAt(0, h-1),
				"bottom-right": canonical.RGBAAt(w-1, h-1),
			}

			for orient := TopLeft; orient <= LeftBottom; orient++ {
				t.Run(orient.String(), func(t *testing.T) {
					// Design A: identical stored bytes, only the tag changes.
					dataA := buildTIFF(t, enc.bo, w, h, uint16(orient), [][]byte{raw}, rgbEntries)
					gotA, gotOrient, err := DecodeWithOrientation(bytes.NewReader(dataA))
					if err != nil {
						t.Fatalf("Design A DecodeWithOrientation: %v", err)
					}
					if gotOrient != orient {
						t.Fatalf("reported orientation: got %d, want %d", gotOrient, orient)
					}
					assertSamePix(t, gotA, refOrientation(t, canonical, orient))

					// Dimensions must swap exactly for orientations 5-8.
					gw, gh := gotA.Bounds().Dx(), gotA.Bounds().Dy()
					if orient >= LeftTop {
						if gw != h || gh != w {
							t.Fatalf("swapped bounds: got %dx%d, want %dx%d", gw, gh, h, w)
						}
					} else if gw != w || gh != h {
						t.Fatalf("bounds: got %dx%d, want %dx%d", gw, gh, w, h)
					}

					// Design B: the stored pixels already depict the scene in
					// this orientation, as real scanning software would write.
					stored := refOrientation(t, canonical, inverseOrientation[orient])
					sb := stored.Bounds()
					dataB := buildTIFF(t, enc.bo, sb.Dx(), sb.Dy(), uint16(orient),
						[][]byte{rgbaToPackedRGB(stored.(*image.RGBA))}, rgbEntries)
					gotB, gotOrient, err := DecodeWithOrientation(bytes.NewReader(dataB))
					if err != nil {
						t.Fatalf("Design B DecodeWithOrientation: %v", err)
					}
					if gotOrient != orient {
						t.Fatalf("Design B reported orientation: got %d, want %d", gotOrient, orient)
					}
					// Every normalized image must be byte-identical to the
					// canonical scene, regardless of the stored orientation.
					assertSamePix(t, gotB, canonical)
					if b := gotB.Bounds(); b.Dx() != w || b.Dy() != h {
						t.Fatalf("Design B normalized bounds: got %v, want %dx%d", b, w, h)
					}
					norm := gotB.(*image.RGBA)
					for name, want := range corners {
						var x, y int
						switch name {
						case "top-left":
							x, y = 0, 0
						case "top-right":
							x, y = w-1, 0
						case "bottom-left":
							x, y = 0, h-1
						case "bottom-right":
							x, y = w-1, h-1
						}
						if got := norm.RGBAAt(x, y); got != want {
							t.Fatalf("Design B corner %s at (%d,%d): got %v, want %v", name, x, y, got, want)
						}
					}
				})
			}
		})
	}
}

// TestDecodeIgnoresOrientation checks that the plain Decode path keeps its
// historical behavior: the Orientation tag is neither applied nor rejected,
// and the decoded pixels and dimensions are exactly the stored ones.
func TestDecodeIgnoresOrientation(t *testing.T) {
	const w, h = 4, 3
	raw := make([]byte, 0, w*h*3)
	for i := 0; i < w*h; i++ {
		raw = append(raw, uint8(i*7), uint8(i*11), uint8(i*13+1))
	}
	base := map[uint16]any{
		tBitsPerSample:             []uint16{8, 8, 8},
		tPhotometricInterpretation: uint16(pRGB),
	}

	untagged := buildTIFF(t, binary.LittleEndian, w, h, 0, [][]byte{raw}, base)
	want, err := Decode(bytes.NewReader(untagged))
	if err != nil {
		t.Fatal(err)
	}

	// An out-of-range orientation is tolerated by Decode but rejected by
	// DecodeWithOrientation.
	for _, orient := range []uint16{1, 2, 3, 4, 5, 6, 7, 8, 9} {
		data := buildTIFF(t, binary.LittleEndian, w, h, orient, [][]byte{raw}, base)
		got, err := Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("orientation %d: Decode: %v", orient, err)
		}
		assertSamePix(t, got, want)
		if got.Bounds().Dx() != w || got.Bounds().Dy() != h {
			t.Fatalf("orientation %d: Decode changed bounds to %v", orient, got.Bounds())
		}
	}

	if _, _, err := DecodeWithOrientation(bytes.NewReader(
		buildTIFF(t, binary.LittleEndian, w, h, 9, [][]byte{raw}, base),
	)); err == nil {
		t.Fatal("DecodeWithOrientation with orientation 9: got nil error, want non-nil")
	}
}

// TestOrientationMissingDefault checks that a file without an Orientation tag
// is reported as TopLeft and returned byte-identical to Decode.
func TestOrientationMissingDefault(t *testing.T) {
	const w, h = 3, 2
	raw := []byte{
		10, 20, 30, 40, 50, 60, 70, 80, 90,
		11, 21, 31, 41, 51, 61, 71, 81, 91,
	}
	data := buildTIFF(t, binary.LittleEndian, w, h, 0, [][]byte{raw}, map[uint16]any{
		tBitsPerSample:             []uint16{8, 8, 8},
		tPhotometricInterpretation: uint16(pRGB),
	})
	decoded, err := Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	img, orient, err := DecodeWithOrientation(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if orient != TopLeft {
		t.Fatalf("default orientation: got %d, want 1", orient)
	}
	assertSamePix(t, img, decoded)
}

// TestOrientationPrecision exercises every supported concrete image type,
// including 8- and 16-bit gray and RGB(A), with multi-strip storage, and
// verifies that normalization neither loses precision nor alters samples.
func TestOrientationPrecision(t *testing.T) {
	const w, h = 4, 3
	type testCase struct {
		name      string
		pixelSize int
		entries   map[uint16]any
		pix       []byte
		wantType  string
	}
	cases := []testCase{{
		name:      "gray8",
		pixelSize: 1,
		entries: map[uint16]any{
			tBitsPerSample:             uint16(8),
			tPhotometricInterpretation: uint16(pBlackIsZero),
		},
		wantType: "*image.Gray",
	}, {
		name:      "gray16",
		pixelSize: 2,
		entries: map[uint16]any{
			tBitsPerSample:             uint16(16),
			tPhotometricInterpretation: uint16(pBlackIsZero),
		},
		wantType: "*image.Gray16",
	}, {
		name:      "rgb8",
		pixelSize: 3,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{8, 8, 8},
			tPhotometricInterpretation: uint16(pRGB),
		},
		wantType: "*image.RGBA",
	}, {
		name:      "rgb16",
		pixelSize: 6,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{16, 16, 16},
			tPhotometricInterpretation: uint16(pRGB),
		},
		wantType: "*image.RGBA64",
	}, {
		name:      "rgba8",
		pixelSize: 4,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{8, 8, 8, 8},
			tPhotometricInterpretation: uint16(pRGB),
			tExtraSamples:              uint16(1),
		},
		wantType: "*image.RGBA",
	}, {
		name:      "rgba16",
		pixelSize: 8,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{16, 16, 16, 16},
			tPhotometricInterpretation: uint16(pRGB),
			tExtraSamples:              uint16(1),
		},
		wantType: "*image.RGBA64",
	}, {
		name:      "nrgba8",
		pixelSize: 4,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{8, 8, 8, 8},
			tPhotometricInterpretation: uint16(pRGB),
			tExtraSamples:              uint16(2),
		},
		wantType: "*image.NRGBA",
	}, {
		name:      "nrgba16",
		pixelSize: 8,
		entries: map[uint16]any{
			tBitsPerSample:             []uint16{16, 16, 16, 16},
			tPhotometricInterpretation: uint16(pRGB),
			tExtraSamples:              uint16(2),
		},
		wantType: "*image.NRGBA64",
	}, {
		name:      "paletted8",
		pixelSize: 1,
		entries: map[uint16]any{
			tBitsPerSample:             uint16(8),
			tPhotometricInterpretation: uint16(pPaletted),
			tColorMap: []uint16{
				0, 0x4000, 0x8000, 0xffff, // red channel, 4 colors
				0, 0x1111, 0x2222, 0x3333, // green channel
				0, 0x5555, 0xaaaa, 0xffff, // blue channel
			},
		},
		wantType: "*image.Paletted",
	}}

	enc := binary.LittleEndian
	for _, tc := range cases {
		// Distinct sample values, including values that would collide if the
		// high byte of a 16-bit sample were dropped. Paletted images can only
		// address their four-entry palette, so use valid indices there.
		tc.pix = make([]byte, w*h*tc.pixelSize)
		for i := range tc.pix {
			if tc.name == "paletted8" {
				tc.pix[i] = byte(i % 4)
			} else {
				tc.pix[i] = byte(i*37 + 9)
			}
		}
		t.Run(tc.name, func(t *testing.T) {
			// Decode the orientation-1 file once to obtain the canonical image.
			base := buildTIFF(t, enc, w, h, 1, [][]byte{tc.pix}, tc.entries)
			canonical, err := Decode(bytes.NewReader(base))
			if err != nil {
				t.Fatal(err)
			}
			if got := fmt.Sprintf("%T", canonical); got != tc.wantType {
				t.Fatalf("canonical type: got %s, want %s", got, tc.wantType)
			}

			for _, orient := range []Orientation{TopRight, BottomRight, BottomLeft, LeftTop, RightTop, RightBottom, LeftBottom} {
				t.Run(orient.String(), func(t *testing.T) {
					// Store the pixels one row per strip so that strip
					// assembly is exercised independently of normalization.
					strips := make([][]byte, h)
					rowBytes := w * tc.pixelSize
					for y := range strips {
						strips[y] = tc.pix[y*rowBytes : (y+1)*rowBytes]
					}
					entries := map[uint16]any{tRowsPerStrip: uint32(1)}
					for k, v := range tc.entries {
						entries[k] = v
					}
					data := buildTIFF(t, enc, w, h, uint16(orient), strips, entries)

					got, gotOrient, err := DecodeWithOrientation(bytes.NewReader(data))
					if err != nil {
						t.Fatalf("DecodeWithOrientation: %v", err)
					}
					if gotOrient != orient {
						t.Fatalf("reported orientation: got %d, want %d", gotOrient, orient)
					}
					assertSamePix(t, got, refOrientation(t, canonical, orient))
				})
			}
		})
	}
}

// TestOrientationErrors covers the deterministic outcomes for malformed
// Orientation metadata: unknown values, conflicting duplicate tags and
// truncated external tag data.
func TestOrientationErrors(t *testing.T) {
	const w, h = 2, 2
	raw := []byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	base := map[uint16]any{
		tBitsPerSample:             []uint16{8, 8, 8},
		tPhotometricInterpretation: uint16(pRGB),
	}

	t.Run("unknown-values", func(t *testing.T) {
		for _, orient := range []uint16{9, 256, 65535} {
			data := buildTIFF(t, binary.LittleEndian, w, h, orient, [][]byte{raw}, base)
			img, _, err := DecodeWithOrientation(bytes.NewReader(data))
			if err == nil {
				t.Fatalf("orientation %d: got nil error, want non-nil", orient)
			}
			if img != nil {
				t.Fatalf("orientation %d: got non-nil image on error", orient)
			}
		}

		// Value 0 must be present in the file to be rejected; buildTIFF would
		// otherwise treat 0 as "omit the tag", which defaults to TopLeft.
		data := buildTIFF(t, binary.LittleEndian, w, h, uint16(TopRight), [][]byte{raw}, base)
		marker := binary.LittleEndian.AppendUint16(nil, tOrientation)
		i := bytes.Index(data, marker)
		if i < 0 {
			t.Fatal("Orientation entry not found")
		}
		binary.LittleEndian.PutUint16(data[i+8:i+10], 0)
		if _, _, err := DecodeWithOrientation(bytes.NewReader(data)); err == nil {
			t.Fatal("orientation 0: got nil error, want non-nil")
		}
	})

	t.Run("duplicate-conflicting-tags", func(t *testing.T) {
		// Hand-build an IFD containing two Orientation entries (values 6 and
		// 3) plus the mandatory dimension tags. IFD tags must be unique and
		// sorted, so the decoder rejects this before reading pixels.
		var buf bytes.Buffer
		buf.Write([]byte{'I', 'I', 42, 0})
		buf.Write(binary.LittleEndian.AppendUint32(nil, 8)) // IFD offset
		entry := func(tag uint16, typ uint16, count uint32, val uint32) []byte {
			b := binary.LittleEndian.AppendUint16(nil, tag)
			b = binary.LittleEndian.AppendUint16(b, typ)
			b = binary.LittleEndian.AppendUint32(b, count)
			b = binary.LittleEndian.AppendUint32(b, val)
			return b
		}
		var ifd []byte
		ifd = binary.LittleEndian.AppendUint16(ifd, 4) // four entries
		ifd = append(ifd, entry(tImageWidth, dtShort, 1, w)...)
		ifd = append(ifd, entry(tImageLength, dtShort, 1, h)...)
		ifd = append(ifd, entry(tOrientation, dtShort, 1, uint32(RightTop))...)
		ifd = append(ifd, entry(tOrientation, dtShort, 1, uint32(BottomRight))...)
		ifd = binary.LittleEndian.AppendUint32(ifd, 0) // no next IFD
		buf.Write(ifd)

		if _, err := Decode(bytes.NewReader(buf.Bytes())); err == nil {
			t.Fatal("Decode with duplicate Orientation: got nil error, want non-nil")
		}
		if _, _, err := DecodeWithOrientation(bytes.NewReader(buf.Bytes())); err == nil {
			t.Fatal("DecodeWithOrientation with duplicate Orientation: got nil error, want non-nil")
		}
	})

	t.Run("truncated-external-tag", func(t *testing.T) {
		data := buildTIFF(t, binary.LittleEndian, w, h, uint16(RightTop), [][]byte{raw}, base)
		// Locate the Orientation entry (tag 0x0112) inside the IFD region and
		// turn it into an entry whose three values would be stored beyond the
		// end of file.
		ifdOff := int(binary.LittleEndian.Uint32(data[4:8]))
		numEntries := int(binary.LittleEndian.Uint16(data[ifdOff : ifdOff+2]))
		ifdEnd := ifdOff + 2 + numEntries*ifdLen
		marker := binary.LittleEndian.AppendUint16(nil, tOrientation)
		i := bytes.Index(data[ifdOff:ifdEnd], marker)
		if i < 0 {
			t.Fatal("Orientation entry not found in IFD")
		}
		i += ifdOff
		binary.LittleEndian.PutUint32(data[i+4:i+8], 3) // count = 3, forces external data
		binary.LittleEndian.PutUint32(data[i+8:i+12], uint32(len(data)+1000))

		if _, err := Decode(bytes.NewReader(data)); err == nil {
			t.Fatal("Decode with truncated Orientation: got nil error, want non-nil")
		}
		if _, _, err := DecodeWithOrientation(bytes.NewReader(data)); err == nil {
			t.Fatal("DecodeWithOrientation with truncated Orientation: got nil error, want non-nil")
		}
	})

	t.Run("multi-value-inline-tag", func(t *testing.T) {
		// Two SHORT values fit inline, so this exercises the count check
		// rather than the external-data read path. The Orientation entry is
		// mutated in place, which keeps every IFD offset and pointer valid.
		data := buildTIFF(t, binary.LittleEndian, w, h, uint16(TopRight), [][]byte{raw}, base)
		ifdOff := int(binary.LittleEndian.Uint32(data[4:8]))
		numEntries := int(binary.LittleEndian.Uint16(data[ifdOff : ifdOff+2]))
		ifdEnd := ifdOff + 2 + numEntries*ifdLen
		marker := binary.LittleEndian.AppendUint16(nil, tOrientation)
		i := bytes.Index(data[ifdOff:ifdEnd], marker)
		if i < 0 {
			t.Fatal("Orientation entry not found in IFD")
		}
		i += ifdOff
		binary.LittleEndian.PutUint32(data[i+4:i+8], 2) // count = 2, still inline
		binary.LittleEndian.PutUint16(data[i+8:i+10], uint16(RightTop))
		binary.LittleEndian.PutUint16(data[i+10:i+12], uint16(BottomRight))

		if _, _, err := DecodeWithOrientation(bytes.NewReader(data)); err == nil {
			t.Fatal("DecodeWithOrientation with multi-value Orientation: got nil error, want non-nil")
		}
	})
}

// TestOrientationAbnormalDimensions checks edge-case dimensions: single-row
// and single-column images, a 1x1 image, and non-square swaps.
func TestOrientationAbnormalDimensions(t *testing.T) {
	for _, dims := range []struct{ w, h int }{
		{1, 1},
		{1, 4},
		{4, 1},
		{2, 7},
		{7, 2},
	} {
		w, h := dims.w, dims.h
		raw := make([]byte, w*h*3)
		for i := range raw {
			raw[i] = byte(i*29 + 3)
		}
		entries := map[uint16]any{
			tBitsPerSample:             []uint16{8, 8, 8},
			tPhotometricInterpretation: uint16(pRGB),
			tRowsPerStrip:              uint32(1),
		}
		for orient := TopLeft; orient <= LeftBottom; orient++ {
			data := buildTIFF(t, binary.LittleEndian, w, h, uint16(orient),
				singleRowStrips(raw, w, h, 3), entries)

			base := buildTIFF(t, binary.LittleEndian, w, h, 1, [][]byte{raw}, map[uint16]any{
				tBitsPerSample:             []uint16{8, 8, 8},
				tPhotometricInterpretation: uint16(pRGB),
			})
			canonical, err := Decode(bytes.NewReader(base))
			if err != nil {
				t.Fatal(err)
			}
			got, gotOrient, err := DecodeWithOrientation(bytes.NewReader(data))
			if err != nil {
				t.Fatalf("%dx%d orientation %d: %v", w, h, orient, err)
			}
			if gotOrient != orient {
				t.Fatalf("%dx%d: reported orientation %d, want %d", w, h, gotOrient, orient)
			}
			assertSamePix(t, got, refOrientation(t, canonical, orient))
		}
	}
}

// singleRowStrips splits packed RGB pixel data into one strip per row.
func singleRowStrips(pix []byte, w, h, channels int) [][]byte {
	rowBytes := w * channels
	strips := make([][]byte, h)
	for y := range strips {
		strips[y] = pix[y*rowBytes : (y+1)*rowBytes]
	}
	return strips
}

// TestOrientationString exercises the Orientation stringer for valid and
// invalid values.
func TestOrientationString(t *testing.T) {
	want := map[Orientation]string{
		TopLeft:        "TopLeft",
		TopRight:       "TopRight",
		BottomRight:    "BottomRight",
		BottomLeft:     "BottomLeft",
		LeftTop:        "LeftTop",
		RightTop:       "RightTop",
		RightBottom:    "RightBottom",
		LeftBottom:     "LeftBottom",
		Orientation(9): "invalid Orientation(9)",
	}
	for o, s := range want {
		if got := o.String(); got != s {
			t.Errorf("Orientation(%d).String() = %q, want %q", uint16(o), got, s)
		}
	}
}
