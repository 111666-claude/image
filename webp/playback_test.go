// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"testing"
)

func TestAnimationPlayback(t *testing.T) {
	left := image.Rect(0, 0, 2, 2)
	right := image.Rect(2, 0, 4, 2)
	black := color.NRGBA{A: 255}
	data := animationTestRIFF(
		animationTestVP8X(4, 2, 0x12),
		animationTestANIM(black, 1),
		animationTestANMF(left, 10, 3, animationTestLossless(t, "green")),
		animationTestANMF(right, 20, 0, animationTestLossless(t, "red")),
		animationTestANMF(right, 30, 2, animationTestLossless(t, "blue")),
		animationTestANMF(right, 0, 2, animationTestLossless(t, "transparent")),
	)
	a, err := DecodeAll(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	want := [][2]color.NRGBA{
		// Replace the left region with opaque green.
		{{G: 255, A: 255}, black},
		// Dispose the left region, then blend red over the black right region.
		{black, {R: 128, A: 255}},
		// Replace the right region, preserving the new frame's transparency.
		{black, {B: 255, A: 128}},
		// A fully transparent replacement clears the previous frame.
		{black, {}},
	}
	if len(a.Frames) != len(want) {
		t.Fatalf("len(Frames) = %d, want %d", len(a.Frames), len(want))
	}

	// Reconstruct the display frames using only the exported animation data.
	canvas := image.NewNRGBA(image.Rect(0, 0, a.Config.Width, a.Config.Height))
	background := image.NewUniform(a.BackgroundColor)
	draw.Draw(canvas, canvas.Bounds(), background, image.Point{}, draw.Src)
	for i, frame := range a.Frames {
		op := draw.Src
		if frame.Blend {
			op = draw.Over
		}
		r := frame.Image.Bounds()
		draw.Draw(canvas, r, frame.Image, r.Min, op)
		for y := 0; y < 2; y++ {
			for x := 0; x < 4; x++ {
				if got := canvas.NRGBAAt(x, y); got != want[i][x/2] {
					t.Errorf("frame %d pixel (%d, %d) = %v, want %v", i, x, y, got, want[i][x/2])
				}
			}
		}
		if frame.DisposeToBackground {
			draw.Draw(canvas, r, background, image.Point{}, draw.Src)
		}
	}
}

// TestAnimationPlaybackTwoFrameOverlay specifically exercises two
// overlapping frames with alpha: the second semi-transparent frame is blended
// over the first on the shared rectangle, while the rest of the canvas keeps
// the first frame's pixels. It mirrors how an independent RIFF player walks
// ANMF chunks, applying Blend then Dispose itself.
func TestAnimationPlaybackTwoFrameOverlay(t *testing.T) {
	// Canvas 2x2. Frame 0 fills the canvas with opaque green. Frame 1 covers
	// the same rectangle with red at alpha 128 and blending enabled.
	data := animationTestRIFF(
		animationTestVP8X(2, 2, 0x12),
		animationTestANIM(color.NRGBA{}, 0),
		animationTestANMF(image.Rect(0, 0, 2, 2), 10, 2, animationTestLossless(t, "green")),
		animationTestANMF(image.Rect(0, 0, 2, 2), 20, 0, animationTestLossless(t, "red")),
	)
	a, err := DecodeAll(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Frames) != 2 || !a.Frames[1].Blend || a.Frames[1].DisposeToBackground {
		t.Fatalf("frame playback settings = %+v", a.Frames)
	}

	canvas := image.NewNRGBA(image.Rect(0, 0, 2, 2))
	draw.Draw(canvas, canvas.Bounds(), image.NewUniform(a.BackgroundColor), image.Point{}, draw.Src)

	// Frame 0: replace.
	f0 := a.Frames[0]
	draw.Draw(canvas, f0.Image.Bounds(), f0.Image, f0.Image.Bounds().Min, draw.Src)
	if got := canvas.NRGBAAt(0, 0); got != (color.NRGBA{G: 255, A: 255}) {
		t.Fatalf("after frame 0: pixel = %v, want opaque green", got)
	}

	// Frame 1: alpha blend red (255,0,0,128) over opaque green. image/draw
	// operates in premultiplied space and uses the integer "over" formula
	// out = src + dst*(255-srcA)/255, with asymmetric rounding. Compute the
	// expected channels with that same arithmetic instead of hard-coding.
	f1 := a.Frames[1]
	op := draw.Src
	if f1.Blend {
		op = draw.Over
	}
	draw.Draw(canvas, f1.Image.Bounds(), f1.Image, f1.Image.Bounds().Min, op)
	// src premultiplied is (128, 0, 0, 128); dst is opaque (0, 255, 0, 255).
	over := func(sp, dp uint8) uint8 {
		return uint8(int(sp) + int(dp)*(255-128)/255)
	}
	want := color.NRGBA{R: over(128, 0), G: over(0, 255), B: over(0, 0), A: 255}
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if got := canvas.NRGBAAt(x, y); got != want {
				t.Errorf("after frame 1: pixel (%d,%d) = %v, want %v", x, y, got, want)
			}
		}
	}
}
