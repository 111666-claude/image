// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package draw provides image composition functions.
//
// See "The Go image/draw package" for an introduction to this package:
// http://golang.org/doc/articles/image_draw.html
//
// This package is a superset of and a drop-in replacement for the image/draw
// package in the standard library.
package draw

// This file just contains the API exported by the image/draw package in the
// standard library. Other files in this package provide additional features.

import (
	"image"
	"image/draw"
)

// Draw calls DrawMask with a nil mask.
func Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point, op Op) {
	if porterDuffOp(op) {
		drawMaskPorterDuff(dst, r, src, sp, nil, image.Point{}, op)
		return
	}
	draw.Draw(dst, r, src, sp, draw.Op(op))
}

// DrawMask aligns r.Min in dst with sp in src and mp in mask and then
// replaces the rectangle r in dst with the result of a Porter-Duff
// composition. A nil mask is treated as opaque.
func DrawMask(dst Image, r image.Rectangle, src image.Image, sp image.Point, mask image.Image, mp image.Point, op Op) {
	if porterDuffOp(op) {
		drawMaskPorterDuff(dst, r, src, sp, mask, mp, op)
		return
	}
	draw.DrawMask(dst, r, src, sp, mask, mp, draw.Op(op))
}

// Drawer contains the Draw method.
type Drawer = draw.Drawer

// FloydSteinberg is a Drawer that is the Src Op with Floyd-Steinberg error
// diffusion.
var FloydSteinberg Drawer = floydSteinberg{}

type floydSteinberg struct{}

func (floydSteinberg) Draw(dst Image, r image.Rectangle, src image.Image, sp image.Point) {
	draw.FloydSteinberg.Draw(dst, r, src, sp)
}

// Image is an image.Image with a Set method to change a single pixel.
type Image = draw.Image

// RGBA64Image extends both the Image and image.RGBA64Image interfaces with a
// SetRGBA64 method to change a single pixel. SetRGBA64 is equivalent to
// calling Set, but it can avoid allocations from converting concrete color
// types to the color.Color interface type.
type RGBA64Image = draw.RGBA64Image

// Op is a Porter-Duff compositing operator.
type Op = draw.Op

const (
	// Over specifies ``(src in mask) over dst''.
	Over Op = draw.Over
	// Src specifies ``src in mask''.
	Src Op = draw.Src

	// SrcIn specifies ``(src in mask) in dst'': the part of the source
	// lying inside the destination's alpha. The result's color and alpha
	// are the source's, scaled by the destination's alpha.
	//
	// SrcIn is not provided by the standard library's image/draw package.
	// It is supported by this package's Draw, DrawMask, Copy, Scale and
	// Transform.
	SrcIn Op = 2
	// SrcOut specifies ``(src in mask) out dst'': the part of the source
	// lying outside the destination's alpha. The result's color and alpha
	// are the source's, scaled by one minus the destination's alpha.
	//
	// SrcOut is not provided by the standard library's image/draw package.
	// It is supported by this package's Draw, DrawMask, Copy, Scale and
	// Transform.
	SrcOut Op = 3
	// Atop specifies ``(src in mask) atop dst'': the part of the source
	// lying inside the destination's alpha, plus the part of the
	// destination lying outside the source's alpha. The result's alpha
	// equals the destination's alpha.
	//
	// Atop is not provided by the standard library's image/draw package.
	// It is supported by this package's Draw, DrawMask, Copy, Scale and
	// Transform.
	Atop Op = 4
	// Xor specifies ``(src in mask) xor dst'': the part of the source
	// lying outside the destination's alpha, plus the part of the
	// destination lying outside the source's alpha.
	//
	// Xor is not provided by the standard library's image/draw package.
	// It is supported by this package's Draw, DrawMask, Copy, Scale and
	// Transform.
	Xor Op = 5
)

// Quantizer produces a palette for an image.
type Quantizer = draw.Quantizer
