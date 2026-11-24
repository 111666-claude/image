// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webp

// This file implements decoding a sub-rectangle of a WEBP image.
//
// A thumbnail service only needs one rectangle of a large picture, so
// DecodeRegion avoids materializing the whole canvas. For lossy images it
// decodes a horizontal band of macroblock rows; for lossless images it streams
// the entropy-coded rows through a fixed-size ring buffer; and for a VP8X
// image with an alpha channel it decodes only the alpha rows it returns.
//
// The decoded pixels are identical to the corresponding pixels of Decode, and
// the returned image's Bounds equal the requested rectangle clipped to the
// image.

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"

	"golang.org/x/image/riff"
	"golang.org/x/image/vp8"
	"golang.org/x/image/vp8l"
)

// ErrRegionMemory is returned when a region decode would exceed the configured
// memory limit.
var ErrRegionMemory = errors.New("webp: region decode exceeds memory limit")

// defaultRegionMemory is the memory cap applied when the caller does not
// supply one.
const defaultRegionMemory = 256 << 20

// maxCanvasPixels bounds the canvas area accepted from a VP8X header. It
// matches the limit that Decode enforces, and keeps all row and rectangle
// arithmetic within a signed 32-bit pixel count.
const maxCanvasPixels = 1<<31 - 1

// RegionOptions are the optional parameters to DecodeRegion. The zero value
// uses a 256 MiB memory cap.
type RegionOptions struct {
	// MaxMemory bounds the total number of bytes that a region decode may
	// allocate for its working buffers and its result. If zero, a 256 MiB
	// limit is used. Negative values are treated as zero.
	MaxMemory int64
}

// Decoder decodes WEBP region requests. Unlike DecodeRegion, a Decoder keeps
// its lossy decoder state between calls, so reusing one avoids re-allocating
// the VP8 working buffers. A Decoder is not safe for concurrent use.
type Decoder struct {
	vp8 *vp8.Decoder
}

// NewDecoder returns a new region Decoder.
func NewDecoder() *Decoder {
	return &Decoder{vp8: vp8.NewDecoder()}
}

// DecodeRegion decodes the part of the WEBP image in r that intersects rect.
//
// rect is given in the image's own coordinate space and is clipped to it; a
// negative origin and a rectangle that extends past the image are both
// allowed. An empty clipped rectangle yields an empty image (of the same
// concrete type that a non-empty decode would return) without decoding pixel
// data.
//
// The returned concrete type matches Decode: *image.YCbCr for lossy images,
// *image.NYCbCrA for lossy images with an alpha channel, and *image.NRGBA for
// lossless images. Decoding is checked for cancellation between rows; a
// canceled context makes DecodeRegion return ctx.Err().
func DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, opts *RegionOptions) (image.Image, error) {
	return NewDecoder().DecodeRegion(ctx, r, rect, opts)
}

// DecodeRegion decodes the region of the WEBP image in r that intersects rect.
// See the package-level DecodeRegion for details. A Decoder can be reused for
// any number of calls; a failed or canceled call leaves it ready for the next.
func (d *Decoder) DecodeRegion(ctx context.Context, r io.Reader, rect image.Rectangle, opts *RegionOptions) (image.Image, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	limit := int64(defaultRegionMemory)
	if opts != nil && opts.MaxMemory > 0 {
		limit = opts.MaxMemory
	}
	budget := &regionBudget{remaining: limit}

	formType, riffReader, err := riff.NewReader(r)
	if err != nil {
		return nil, err
	}
	if formType != fccWEBP {
		return nil, errInvalidFormat
	}

	var (
		alpha          []byte
		alphaStride    int
		alphaPresent   bool
		wantAlpha      bool
		seenVP8X       bool
		haveCanvas     bool
		canvas         image.Rectangle
		widthMinusOne  uint32
		heightMinusOne uint32
		buf            [10]byte
	)
	for {
		chunkID, chunkLen, chunkData, err := riffReader.Next()
		if err == io.EOF {
			err = errInvalidFormat
		}
		if err != nil {
			return nil, err
		}

		switch chunkID {
		case fccALPH:
			if !wantAlpha || !haveCanvas {
				return nil, errInvalidFormat
			}
			wantAlpha = false
			alphaPresent = true
			if _, err := io.ReadFull(chunkData, buf[:1]); err != nil {
				if err == io.EOF {
					err = errInvalidFormat
				}
				return nil, err
			}
			// An empty request needs no alpha at all; skip the payload.
			if clip(rect, canvas).Empty() {
				continue
			}
			alpha, alphaStride, err = readAlphaRegion(ctx, chunkData, canvas, rect, buf[0], budget)
			if err != nil {
				return nil, wrapRegionError(err)
			}

		case fccVP8:
			if wantAlpha || int32(chunkLen) < 0 {
				return nil, errInvalidFormat
			}
			d.vp8.Init(chunkData, int(chunkLen))
			fh, err := d.vp8.DecodeFrameHeader()
			if err != nil {
				return nil, wrapRegionError(err)
			}
			frameRect := image.Rect(0, 0, fh.Width, fh.Height)
			if haveCanvas && frameRect != canvas {
				return nil, errInvalidFormat
			}
			out := clip(rect, frameRect)
			if out.Empty() {
				if alphaPresent {
					return &image.NYCbCrA{YCbCr: image.YCbCr{SubsampleRatio: image.YCbCrSubsampleRatio420}}, nil
				}
				return &image.YCbCr{SubsampleRatio: image.YCbCrSubsampleRatio420}, nil
			}
			// Both codecs treat a zero MaxMemory as "use the default", so an
			// exhausted budget must be rejected here rather than passed on.
			if budget.remaining <= 0 {
				return nil, wrapRegionError(ErrRegionMemory)
			}
			m, err := d.vp8.DecodeFrameRegion(ctx, out, &vp8.RegionOptions{MaxMemory: budget.remaining})
			if err != nil {
				return nil, wrapRegionError(err)
			}
			if alphaPresent {
				return &image.NYCbCrA{
					YCbCr:   *m,
					A:       alpha,
					AStride: alphaStride,
				}, nil
			}
			return m, nil

		case fccVP8L:
			if wantAlpha || alphaPresent {
				return nil, errInvalidFormat
			}
			// Peek the 5-byte VP8L header so that the canvas check and the
			// empty-rectangle fast path work without decoding the stream.
			br := bufio.NewReader(chunkData)
			vp8lHeader, err := br.Peek(5)
			if err != nil {
				if err == io.EOF {
					err = errInvalidFormat
				}
				return nil, err
			}
			c, err := vp8l.DecodeConfig(bytes.NewReader(vp8lHeader))
			if err != nil {
				return nil, wrapRegionError(err)
			}
			lRect := image.Rect(0, 0, c.Width, c.Height)
			if haveCanvas && lRect != canvas {
				return nil, errInvalidFormat
			}
			out := clip(rect, lRect)
			if out.Empty() {
				return &image.NRGBA{}, nil
			}
			m, err := vp8l.DecodeRegion(ctx, br, out, &vp8l.Options{MaxMemory: budget.remaining})
			if err != nil {
				return nil, wrapRegionError(err)
			}
			return m, nil

		case fccVP8X:
			if seenVP8X {
				return nil, errInvalidFormat
			}
			seenVP8X = true
			if chunkLen != 10 {
				return nil, errInvalidFormat
			}
			if _, err := io.ReadFull(chunkData, buf[:10]); err != nil {
				return nil, err
			}
			const (
				animationBit    = 1 << 1
				xmpMetadataBit  = 1 << 2
				exifMetadataBit = 1 << 3
				alphaBit        = 1 << 4
				iccProfileBit   = 1 << 5
			)
			wantAlpha = (buf[0] & alphaBit) != 0
			widthMinusOne = uint32(buf[4]) | uint32(buf[5])<<8 | uint32(buf[6])<<16
			heightMinusOne = uint32(buf[7]) | uint32(buf[8])<<8 | uint32(buf[9])<<16
			w := uint64(widthMinusOne) + 1
			h := uint64(heightMinusOne) + 1
			if w*h > maxCanvasPixels {
				return nil, errInvalidFormat
			}
			haveCanvas = true
			canvas = image.Rect(0, 0, int(w), int(h))
		}
	}
}

// wrapRegionError adds decode context to an error while preserving errors.Is
// and errors.As. Memory-limit errors from either codec are additionally
// reported as ErrRegionMemory, so callers have one sentinel to check.
func wrapRegionError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrRegionMemory) ||
		errors.Is(err, vp8l.ErrMemoryLimitExceeded) ||
		errors.Is(err, vp8.ErrRegionTooLarge) {
		return &regionError{err: err}
	}
	return fmt.Errorf("webp: region decode: %w", err)
}

// regionError wraps a memory-limit error so that both the underlying sentinel
// and ErrRegionMemory match under errors.Is.
type regionError struct{ err error }

func (e *regionError) Error() string {
	return "webp: region decode: " + e.err.Error()
}

func (e *regionError) Unwrap() error { return e.err }

func (e *regionError) Is(target error) bool { return target == ErrRegionMemory }

// regionBudget tracks the bytes charged against a region decode's limit.
type regionBudget struct {
	remaining int64
}

func (b *regionBudget) charge(n int64) error {
	if n <= 0 {
		return nil
	}
	if b.remaining < n {
		return ErrRegionMemory
	}
	b.remaining -= n
	return nil
}

// clip returns r clipped to bounds without computing Dx or Dy first, so that
// rectangles with extreme coordinates cannot overflow.
func clip(r, bounds image.Rectangle) image.Rectangle {
	if r.Min.X < bounds.Min.X {
		r.Min.X = bounds.Min.X
	}
	if r.Min.Y < bounds.Min.Y {
		r.Min.Y = bounds.Min.Y
	}
	if r.Max.X > bounds.Max.X {
		r.Max.X = bounds.Max.X
	}
	if r.Max.Y > bounds.Max.Y {
		r.Max.Y = bounds.Max.Y
	}
	if r.Min.X >= r.Max.X || r.Min.Y >= r.Max.Y {
		return image.Rectangle{}
	}
	return r
}

// readAlphaRegion decodes the alpha channel rows of a VP8X image that the
// requested rectangle needs. canvas is the full image rectangle and rect is
// the caller's request; rect must already intersect canvas.
//
// The returned slice is rect-sized with stride equal to its width, so it can
// be used directly as an image.NYCbCrA's A plane.
func readAlphaRegion(ctx context.Context, chunkData io.Reader, canvas, rect image.Rectangle, header byte, budget *regionBudget) (alpha []byte, stride int, err error) {
	out := clip(rect, canvas)
	if out.Empty() {
		return nil, 0, nil
	}
	w, h := canvas.Dx(), canvas.Dy()
	dx, dy := out.Dx(), out.Dy()
	if err := budget.charge(int64(dx) * int64(dy)); err != nil {
		return nil, 0, err
	}
	alpha = make([]byte, dx*dy)

	switch compression := header & 0x03; compression {
	case 0:
		filter := (header >> 2) & 0x03
		if err := readUncompressedAlphaRows(ctx, chunkData, w, h, out, filter, alpha, budget); err != nil {
			return nil, 0, err
		}
		return alpha, dx, nil

	case 1:
		// The compressed alpha channel is itself a VP8L image whose green
		// channel holds the alpha values. Synthesize its 5-byte header and
		// decode only the requested rectangle.
		if canvas.Dx() > 0x4000 || canvas.Dy() > 0x4000 {
			return nil, 0, errInvalidFormat
		}
		wm1 := uint32(canvas.Dx() - 1)
		hm1 := uint32(canvas.Dy() - 1)
		headerBytes := []byte{
			0x2f, // VP8L magic number.
			uint8(wm1),
			uint8(wm1>>8) | uint8(hm1<<6),
			uint8(hm1 >> 2),
			uint8(hm1 >> 10),
		}
		img, err := vp8l.DecodeRegion(ctx, io.MultiReader(bytes.NewReader(headerBytes), chunkData), out,
			&vp8l.Options{MaxMemory: budget.remaining})
		if err != nil {
			return nil, 0, err
		}
		for i := range alpha {
			alpha[i] = img.Pix[4*i+1]
		}
		return alpha, dx, nil
	}
	return nil, 0, errInvalidFormat
}

// readUncompressedAlphaRows reads and reverses the ALPH filter for every row
// of the canvas, but stores only the rows and columns inside out.
func readUncompressedAlphaRows(ctx context.Context, chunkData io.Reader, w, h int, out image.Rectangle, filter byte, dst []byte, budget *regionBudget) error {
	if err := budget.charge(int64(2 * w)); err != nil {
		return err
	}
	prev := make([]byte, w) // Reconstructed previous row.
	cur := make([]byte, w)
	for y := 0; y < h; y++ {
		if y&0xff == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if _, err := io.ReadFull(chunkData, cur); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return err
		}
		switch {
		case y == 0:
			// The first row always uses the horizontal predictor, whichever
			// filter mode the header names.
			if filter != 0 {
				for i := 1; i < w; i++ {
					cur[i] += cur[i-1]
				}
			}
		case filter == 1: // Horizontal.
			// The first column of every row below the first uses the vertical
			// predictor instead; see unfilterAlpha.
			cur[0] += prev[0]
			for i := 1; i < w; i++ {
				cur[i] += cur[i-1]
			}
		case filter == 2: // Vertical.
			for i := 0; i < w; i++ {
				cur[i] += prev[i]
			}
		case filter == 3: // Gradient.
			cur[0] += prev[0]
			for i := 1; i < w; i++ {
				x := int(cur[i-1]) + int(prev[i]) - int(prev[i-1])
				if x < 0 {
					x = 0
				} else if x > 255 {
					x = 255
				}
				cur[i] += uint8(x)
			}
		}
		if y >= out.Min.Y && y < out.Max.Y {
			row := dst[(y-out.Min.Y)*out.Dx():]
			copy(row, cur[out.Min.X:out.Max.X])
		}
		prev, cur = cur, prev
	}
	return nil
}
