// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector_test

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"os"

	"golang.org/x/image/vector"
)

func Example_draw() {
	const (
		width  = 30
		height = 20
	)

	// Define a closed shape with three edges: two linear and one quadratic.
	// One of its vertices is at the top-left corner of the (1, 2) pixel, which
	// is also the bottom-right corner of the (0, 1) pixel.
	//
	// Co-ordinates can be floating point numbers, not just integers. They can
	// also be outside the vector.Rasterizer's dimensions. The shapes will be
	// clipped during rasterization.
	r := vector.NewRasterizer(width, height)
	r.DrawOp = draw.Src
	r.MoveTo(1, 2)
	r.LineTo(20, 2)
	r.QuadTo(40.5, 15, 10, 20)
	r.ClosePath()

	// Finish the rasterization: the conversion from vector graphics (shapes)
	// to raster graphics (pixels). Co-ordinates are now integers.
	dst := image.NewAlpha(image.Rect(0, 0, width, height))
	r.Draw(dst, dst.Bounds(), image.Opaque, image.Point{})

	// Visualize the pixels.
	const asciiArt = ".++8"
	buf := make([]byte, 0, height*(width+1))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			a := dst.AlphaAt(x, y).A
			buf = append(buf, asciiArt[a>>6])
		}
		buf = append(buf, '\n')
	}
	os.Stdout.Write(buf)

	// Output:
	// ..............................
	// ..............................
	// .8888888888888888888+.........
	// .+88888888888888888888+.......
	// ..888888888888888888888+......
	// ..+888888888888888888888+.....
	// ...8888888888888888888888+....
	// ...+8888888888888888888888+...
	// ....88888888888888888888888+..
	// ....+88888888888888888888888..
	// .....88888888888888888888888..
	// .....+8888888888888888888888..
	// ......8888888888888888888888..
	// ......+88888888888888888888+..
	// .......8888888888888888888+...
	// .......+88888888888888888.....
	// ........888888888888888+......
	// ........+88888888888+.........
	// .........8888888++............
	// .........+8+++................
}

// Example_coverage shows how to export a Rasterizer's per-pixel coverage into a
// reusable CoverageBuffer instead of immediately compositing colors. The
// buffer's *image.Alpha can then be reused as a mask by image/draw, for
// example to render the same shape in several colors without re-rasterizing.
func Example_coverage() {
	const width, height = 12, 8

	r := vector.NewRasterizer(width, height)
	r.MoveTo(1, 1)
	r.LineTo(10, 1)
	r.LineTo(10, 6)
	r.LineTo(1, 6)
	r.ClosePath()

	// Export the geometric coverage once. This does not depend on DrawOp or on
	// any source image, and it does not consume the paths.
	buf, err := vector.NewCoverageBuffer(r.Bounds(), 0, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	if _, err := r.Coverage(context.Background(), buf); err != nil {
		fmt.Println("error:", err)
		return
	}

	// Reuse the same coverage mask twice, with different colors, via
	// draw.DrawMask. No second Rasterizer or path feed is needed.
	for _, c := range []color.Color{
		color.RGBA{R: 0xff, A: 0xff}, // red
		color.RGBA{B: 0xff, A: 0xff}, // blue
	} {
		dst := image.NewRGBA(r.Bounds())
		draw.DrawMask(dst, dst.Bounds(), image.NewUniform(c), image.Point{},
			buf.Alpha(), image.Point{}, draw.Over)
		// Report the color of one fully-covered interior pixel.
		fmt.Println(dst.RGBAAt(5, 3))
	}

	// Clear and re-export; the geometry is still available.
	buf.Clear()
	if _, err := r.Coverage(nil, buf); err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println("re-exported interior alpha:", buf.AlphaAt(5, 3).A)
	fmt.Println("exterior alpha:", buf.AlphaAt(0, 0).A)

	// Output:
	// {255 0 0 255}
	// {0 0 255 255}
	// re-exported interior alpha: 255
	// exterior alpha: 0
}
