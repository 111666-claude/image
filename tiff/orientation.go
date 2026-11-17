// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff // import "golang.org/x/image/tiff"

import (
	"fmt"
	"image"
	"io"
)

// Orientation is the value of the TIFF Orientation tag (tag 274), as defined
// on page 33 of the TIFF 6.0 specification. It describes the orientation of
// the image's rows and columns relative to the stored pixel data.
type Orientation uint16

const (
	// TopLeft is Orientation 1: row 0 is the visual top, column 0 is the
	// visual left. No transformation is needed.
	TopLeft Orientation = 1
	// TopRight is Orientation 2: mirror horizontally.
	TopRight Orientation = 2
	// BottomRight is Orientation 3: rotate 180 degrees.
	BottomRight Orientation = 3
	// BottomLeft is Orientation 4: mirror vertically.
	BottomLeft Orientation = 4
	// LeftTop is Orientation 5: mirror horizontally, then rotate 270
	// degrees clockwise (equivalently, transpose across the main diagonal).
	LeftTop Orientation = 5
	// RightTop is Orientation 6: rotate 90 degrees clockwise.
	RightTop Orientation = 6
	// RightBottom is Orientation 7: mirror horizontally, then rotate 90
	// degrees clockwise (equivalently, transpose across the anti-diagonal).
	RightBottom Orientation = 7
	// LeftBottom is Orientation 8: rotate 270 degrees clockwise.
	LeftBottom Orientation = 8
)

// String returns the name of the orientation from the TIFF 6.0 specification,
// such as "TopLeft" or "RightTop". Values outside the valid range return a
// string describing the invalid value.
func (o Orientation) String() string {
	switch o {
	case TopLeft:
		return "TopLeft"
	case TopRight:
		return "TopRight"
	case BottomRight:
		return "BottomRight"
	case BottomLeft:
		return "BottomLeft"
	case LeftTop:
		return "LeftTop"
	case RightTop:
		return "RightTop"
	case RightBottom:
		return "RightBottom"
	case LeftBottom:
		return "LeftBottom"
	}
	return fmt.Sprintf("invalid Orientation(%d)", uint16(o))
}

// orientation returns the value of the Orientation tag.
// A missing tag defaults to TopLeft (1), as specified by TIFF 6.0.
// A malformed or unknown value produces a FormatError so that callers get a
// deterministic result instead of silently misinterpreting the pixels.
func (d *decoder) orientation() (Orientation, error) {
	vals := d.features[tOrientation]
	if len(vals) == 0 {
		// The Orientation tag is optional and defaults to 1.
		return TopLeft, nil
	}
	if len(vals) != 1 {
		return 0, FormatError("bad Orientation tag count")
	}
	v := vals[0]
	if v == 0 || v > 8 {
		return 0, FormatError(fmt.Sprintf("unknown Orientation value %d", v))
	}
	return Orientation(v), nil
}

// DecodeWithOrientation reads a TIFF image from r and returns a normalized
// image together with the Orientation value stored in the file.
//
// The returned image is transformed so that its rows and columns have the
// TopLeft orientation: row 0 is the visual top and column 0 is the visual
// left. All eight Orientation values (1 through 8), including the mirrored
// ones, are supported. Orientations 5 through 8 swap the image's width and
// height.
//
// The transformation operates on the decoded pixel bytes, so it preserves
// the image's concrete type and precision, including 16-bit-per-sample gray
// and RGB(A) images. If the Orientation tag is absent, the returned
// orientation is TopLeft and the image is returned unchanged.
//
// An unknown or malformed Orientation value results in an error and no
// image. Unlike Decode, this function validates the Orientation tag even
// when no transformation would otherwise be required.
func DecodeWithOrientation(r io.Reader) (image.Image, Orientation, error) {
	d, err := newDecoder(r)
	if err != nil {
		return nil, 0, err
	}
	orient, err := d.orientation()
	if err != nil {
		return nil, 0, err
	}
	img, err := d.decodeImage()
	if err != nil {
		return nil, orient, err
	}
	img, err = applyOrientation(img, orient)
	if err != nil {
		return nil, orient, err
	}
	return img, orient, nil
}

// pixLayout returns the backing pixel slice, byte stride and the number of
// bytes in one pixel for the concrete image types the decoder produces.
func pixLayout(img image.Image) (pix []byte, stride, pixelSize int, ok bool) {
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

// newImageLike allocates a new image of the same concrete type as img with
// the given bounds. Paletted images keep the source palette.
func newImageLike(img image.Image, rect image.Rectangle) (image.Image, error) {
	switch img := img.(type) {
	case *image.Gray:
		return image.NewGray(rect), nil
	case *image.Gray16:
		return image.NewGray16(rect), nil
	case *image.Paletted:
		return image.NewPaletted(rect, img.Palette), nil
	case *image.NRGBA:
		return image.NewNRGBA(rect), nil
	case *image.NRGBA64:
		return image.NewNRGBA64(rect), nil
	case *image.RGBA:
		return image.NewRGBA(rect), nil
	case *image.RGBA64:
		return image.NewRGBA64(rect), nil
	}
	return nil, UnsupportedError(fmt.Sprintf("cannot normalize image type %T", img))
}

// applyOrientation returns a copy of img transformed to the TopLeft
// orientation according to the TIFF Orientation value o. It preserves the
// concrete image type and its precision by moving whole, undecoded pixel
// values rather than re-encoding colors through the color model.
//
// Decoded images always have bounds with Min at (0, 0), so all coordinates
// below are relative to the source's top-left corner.
func applyOrientation(img image.Image, o Orientation) (image.Image, error) {
	if o == TopLeft {
		return img, nil
	}
	pix, stride, pixelSize, ok := pixLayout(img)
	if !ok {
		return nil, UnsupportedError(fmt.Sprintf("cannot normalize image type %T", img))
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()

	// Orientations 5-8 rotate by 90 or 270 degrees and therefore swap the
	// width and height. The swapped dimensions were already validated as
	// representable when the source image was decoded (their product is
	// unchanged), so no additional overflow check is needed here.
	dw, dh := w, h
	if o >= LeftTop {
		dw, dh = h, w
	}
	dst, err := newImageLike(img, image.Rect(0, 0, dw, dh))
	if err != nil {
		return nil, err
	}
	dpix, dstride, _, _ := pixLayout(dst)

	switch o {
	// Same dimensions: copy one source row at a time, optionally reversing
	// the order of the rows and of the pixels within a row.
	case TopRight, BottomRight, BottomLeft:
		flipH := o == TopRight || o == BottomRight
		flipV := o == BottomRight || o == BottomLeft
		for dy := 0; dy < h; dy++ {
			sy := dy
			if flipV {
				sy = h - 1 - dy
			}
			sRow := sy * stride
			dRow := dy * dstride
			if flipH {
				for dx := 0; dx < w; dx++ {
					sx := w - 1 - dx
					copy(dpix[dRow+dx*pixelSize:dRow+(dx+1)*pixelSize],
						pix[sRow+sx*pixelSize:sRow+(sx+1)*pixelSize])
				}
			} else {
				copy(dpix[dRow:dRow+w*pixelSize], pix[sRow:sRow+w*pixelSize])
			}
		}

	// Swapped dimensions: each destination pixel is placed individually.
	// (sx, sy) is the coordinate, relative to the source bounds, of the
	// pixel that ends up at (dx, dy). The two flips are independent:
	// orientation 5 is a plain transpose, 6 and 8 are 90/270-degree
	// rotations, and 7 is the anti-diagonal transpose.
	case LeftTop, RightTop, RightBottom, LeftBottom:
		var flipSx, flipSy bool
		switch o {
		case LeftTop: // transpose: (dx, dy) <- (dy, dx)
		case RightTop: // rotate 90 CW: (dx, dy) <- (dy, h-1-dx)
			flipSy = true
		case RightBottom: // anti-diagonal transpose: (dx, dy) <- (w-1-dy, h-1-dx)
			flipSx, flipSy = true, true
		case LeftBottom: // rotate 270 CW: (dx, dy) <- (w-1-dy, dx)
			flipSx = true
		}
		for dy := 0; dy < dh; dy++ {
			for dx := 0; dx < dw; dx++ {
				sx, sy := dy, dx
				if flipSx {
					sx = w - 1 - sx
				}
				if flipSy {
					sy = h - 1 - sy
				}
				sOff := sy*stride + sx*pixelSize
				dOff := dy*dstride + dx*pixelSize
				copy(dpix[dOff:dOff+pixelSize], pix[sOff:sOff+pixelSize])
			}
		}

	default:
		// Unreachable: orientation() already rejected values outside 1-8.
		return nil, FormatError(fmt.Sprintf("unknown Orientation value %d", uint16(o)))
	}

	return dst, nil
}
