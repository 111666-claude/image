// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8

// This file implements incremental, cancellable frame decoding. Whereas
// DecodeFrame produces the complete frame in one call,
// DecodeFrameIncremental reports progress at frame header, partition and
// macroblock-row boundaries, so that a caller such as a live video thumbnailer
// can display partial rows as soon as they are available and cancel a frame
// that is no longer needed.

import (
	"bytes"
	"context"
	"errors"
	"image"
	"io"
	"sync"
)

// ProgressStage identifies a boundary at which an incremental decode reports
// progress. The stages are reported in increasing order while a frame is
// decoded.
type ProgressStage int

const (
	// StageFrameHeader means that the frame header (section 9.1) has been
	// parsed. No image data is available yet.
	StageFrameHeader ProgressStage = 1 + iota

	// StagePartitions means that the first partition and all coefficient
	// partitions (sections 9.2-9.8) have been parsed. No image data is
	// available yet, but the quantizer, probabilities and loop filter
	// parameters are finalized.
	StagePartitions

	// StageMacroblockRow means that one more macroblock row has been
	// reconstructed. Progress.RowsAvailable counts the rows that are both
	// reconstructed and already separated, by the loop filter, from every row
	// still to be processed. The pixels of those rows will never be modified
	// again, neither by later rows nor by the final loop filter.
	//
	// Loop filtering a macroblock reads pixels in the row below it, so a
	// macroblock cannot be filtered until that row exists, and a row cannot be
	// declared immutable until the row below has applied its horizontal
	// edges. With a loop filter active, the first two macroblock-row
	// boundaries therefore emit no event; subsequent events report
	// RowsAvailable 1, 2, ..., mbh-3, and the final boundary flushes the two
	// deferred rows and reports all mbh rows at once. (Frames with only one or
	// two macroblock rows emit a single all-rows event.) Frames whose loop
	// filter level is 0 deliver each row immediately after reconstruction.
	StageMacroblockRow

	// StageComplete means that the entire frame has been reconstructed and
	// loop-filtered. RowsAvailable equals the number of macroblock rows and
	// the image is identical to the one returned by DecodeFrame.
	StageComplete
)

// String returns the stage's name.
func (s ProgressStage) String() string { return stageName(s) }

// Progress describes how far an incremental frame decode has advanced.
//
// Image, when non-nil, is the decoder's working image. Its YCbCr strides and
// backing arrays are the same ones that DecodeFrame would return. The caller
// must not modify the image; the pixels above the row range reported by
// RowsAvailable are valid and immutable, the remaining pixels are still being
// decoded.
type Progress struct {
	// Stage is the boundary that has just been reached.
	Stage ProgressStage

	// FrameHeader is the frame header, valid from StageFrameHeader onwards.
	FrameHeader FrameHeader

	// RowsAvailable is the number of fully decoded macroblock rows whose
	// pixels will never change again. It is zero for StageFrameHeader and
	// StagePartitions. For StageMacroblockRow it is zero on the very first
	// boundary of a loop-filtered frame (no event is emitted in that case,
	// see StageMacroblockRow), strictly increases afterwards, and equals the
	// number of macroblock rows for StageComplete.
	RowsAvailable int

	// MacroblockRows is the total number of macroblock rows in the frame.
	MacroblockRows int

	// Image is the image being decoded. It is nil at StageFrameHeader and
	// non-nil from StagePartitions onwards. The image's Rect is the visible
	// (width, height) rectangle; the underlying buffers are padded to whole
	// macroblocks, exactly as for DecodeFrame.
	Image *image.YCbCr
}

// ProgressFunc is the callback invoked by DecodeFrameIncremental at every
// progress boundary. Returning a non-nil error cancels the decode: the error
// is returned (wrapped in a *DecodeError) and the decoder is left in a clean
// state, ready to be re-initialized and to decode another frame. In
// particular, returning context.Canceled (for example by checking ctx.Err())
// cancels the decode without it being treated as a malformed bitstream.
type ProgressFunc func(progress Progress) error

// DecodeError wraps an error encountered while incrementally decoding a frame
// with the frame/partition context in which it occurred.
type DecodeError struct {
	// Stage is the progress stage at which the error occurred.
	Stage ProgressStage

	// Frame is the zero-based index of the frame within the incremental
	// session. It is 0 for Decoder.DecodeFrameIncremental.
	Frame int

	// Partition is the zero-based coefficient partition implicated in the
	// error, or -1 when the error is not specific to a coefficient partition.
	// The first (header) partition is reported as partition -1, because the
	// coefficient partitions are numbered 0..nOP-1.
	Partition int

	// Err is the underlying error.
	Err error
}

func (e *DecodeError) Error() string {
	if e.Partition >= 0 {
		return "vp8: incremental decode error at stage " + stageName(e.Stage) +
			", frame " + itoa(e.Frame) + ", partition " + itoa(e.Partition) +
			": " + e.Err.Error()
	}
	return "vp8: incremental decode error at stage " + stageName(e.Stage) +
		", frame " + itoa(e.Frame) + ": " + e.Err.Error()
}

func (e *DecodeError) Unwrap() error { return e.Err }

// Is reports whether the target is the underlying error. It makes errors.Is
// work for the standard sentinels such as io.ErrUnexpectedEOF and
// context.Canceled.
func (e *DecodeError) Is(target error) bool { return e.Err == target }

func stageName(s ProgressStage) string {
	switch s {
	case StageFrameHeader:
		return "frame-header"
	case StagePartitions:
		return "partitions"
	case StageMacroblockRow:
		return "macroblock-row"
	case StageComplete:
		return "complete"
	}
	return "unknown"
}

// itoa is a tiny strconv.Itoa replacement that avoids importing strconv solely
// for error messages.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	n := len(buf)
	for i > 0 {
		n--
		buf[n] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		n--
		buf[n] = '-'
	}
	return string(buf[n:])
}

// maxFrameDimension is the largest frame width or height accepted. The VP8
// frame header can in principle encode 14-bit dimensions (up to 16383), but
// decoding such a frame would allocate hundreds of megabytes, so the
// incremental API rejects anything above this practical bound before any
// allocation happens. The one-shot DecodeFrame keeps its historical behavior
// and does not apply this guard.
const maxFrameDimension = 4096

// ErrFrameTooLarge is returned when a frame header declares a width or height
// above the decoder's practical limit.
var ErrFrameTooLarge = errors.New("vp8: frame dimensions are too large")

// ErrZeroSizeFrame is returned when a frame header declares a zero width or
// height.
var ErrZeroSizeFrame = errors.New("vp8: zero-size frame")

// errIncrementalInProgress is returned when an incremental decode is started
// while another one is still in progress on the same Decoder.
var errIncrementalInProgress = errors.New("vp8: incremental decode already in progress")

// DecodeFrameIncremental decodes one frame, invoking cb at every progress
// boundary. The decoder must have been initialized with Init and the frame
// header must not have been consumed yet: the typical call sequence is
// NewDecoder, Init and then DecodeFrameIncremental (which parses the frame
// header itself).
//
// Cancellation, including a non-nil error returned from cb, leaves the
// decoder reset to a clean state: calling Init followed by
// DecodeFrameIncremental or DecodeFrame on the next frame is safe and no
// goroutines are started.
//
// On success, the returned image is identical, pixel for pixel, to the image
// DecodeFrame would produce for the same bitstream.
func (d *Decoder) DecodeFrameIncremental(cb ProgressFunc) (img *image.YCbCr, err error) {
	if d.busy {
		return nil, &DecodeError{Stage: StageFrameHeader, Frame: 0, Partition: -1, Err: errIncrementalInProgress}
	}
	d.busy = true
	defer func() {
		d.busy = false
		if err != nil {
			// Drop all per-frame state so that a canceled or failed
			// incremental decode cannot leak partial rows into the next frame.
			d.resetFrameState()
		}
	}()

	frameHeader, err := d.DecodeFrameHeader()
	if err != nil {
		return nil, d.wrapError(StageFrameHeader, -1, err)
	}
	// The dimension guards only apply to key frames: an inter frame's
	// three-byte header carries no dimensions - it inherits the width and
	// height of the most recent key frame - so its zero header values are not
	// evidence of a zero-size frame. Inter frames instead reach the decoder's
	// existing Golden/AltRef handling during parseOtherHeaders.
	if frameHeader.KeyFrame {
		if frameHeader.Width <= 0 || frameHeader.Height <= 0 {
			return nil, d.wrapError(StageFrameHeader, -1, ErrZeroSizeFrame)
		}
		if frameHeader.Width > maxFrameDimension || frameHeader.Height > maxFrameDimension {
			return nil, d.wrapError(StageFrameHeader, -1, ErrFrameTooLarge)
		}
	}
	if cb != nil {
		if err = cb(Progress{
			Stage:          StageFrameHeader,
			FrameHeader:    frameHeader,
			MacroblockRows: d.mbh,
		}); err != nil {
			return nil, d.wrapError(StageFrameHeader, -1, err)
		}
	}

	d.ensureImg()
	if err = d.parseOtherHeaders(); err != nil {
		return nil, d.wrapError(StagePartitions, d.failingPartition(), err)
	}
	if cb != nil {
		if err = cb(Progress{
			Stage:          StagePartitions,
			FrameHeader:    frameHeader,
			MacroblockRows: d.mbh,
			Image:          d.img,
		}); err != nil {
			return nil, d.wrapError(StagePartitions, -1, err)
		}
	}

	// Reconstruct the macroblock rows with the loop filter interleaved. The
	// whole-frame filter walks macroblocks in raster order, applying each
	// macroblock's vertical edges before its horizontal edges. A macroblock's
	// horizontal edges read pixels in the row below it, so it cannot be fully
	// filtered until that row exists. The incremental schedule therefore
	// filters, right after reconstructing row mby, every macroblock in row
	// mby-1 left-to-right (the row-0 horizontal boundary above the frame is a
	// no-op), and flushes the final row after the last reconstruction:
	//
	//	reconstruct 0
	//	reconstruct 1, filter row 0
	//	reconstruct 2, filter row 1
	//	...
	//	reconstruct n-1, filter rows n-2 and n-1
	//
	// Each macroblock keeps its vertical-before-horizontal order and rows are
	// filtered left-to-right, so the edge-operation sequence is exactly the
	// whole-frame one and the result is bit-identical to DecodeFrame's. A row
	// is written only by its own filtering and by the horizontal edges of the
	// row below; once both have happened the row is immutable.
	for mbx := 0; mbx < d.mbw; mbx++ {
		d.upMB[mbx] = mb{}
	}
	for mby := 0; mby < d.mbh; mby++ {
		d.leftMB = mb{}
		for mbx := 0; mbx < d.mbw; mbx++ {
			skip := d.reconstruct(mbx, mby)
			fs := d.filterParams[d.segment][btou(!d.usePredY16)]
			fs.inner = fs.inner || !skip
			d.perMBFilterParams[d.mbw*mby+mbx] = fs
		}
		if d.fp.unexpectedEOF {
			return nil, d.wrapError(StageMacroblockRow, -1, io.ErrUnexpectedEOF)
		}
		if p := d.failingPartition(); p >= 0 {
			return nil, d.wrapError(StageMacroblockRow, p, io.ErrUnexpectedEOF)
		}

		if d.filterHeader.level != 0 {
			// Filter the row whose below-neighbor now exists. Row -1 is a
			// no-op because the macroblock filters guard mbx > 0 and mby > 0.
			if d.filterHeader.simple {
				for mbx := 0; mbx < d.mbw; mbx++ {
					d.simpleFilterMacroblock(mbx, mby-1)
				}
			} else {
				for mbx := 0; mbx < d.mbw; mbx++ {
					d.normalFilterMacroblock(mbx, mby-1)
				}
			}
		}

		// Determine which rows are immutable after this boundary. Row k is
		// final once it has been filtered itself and the row below (k+1) has
		// applied its horizontal edges, which touch row k's bottom boundary.
		rowsAvailable := 0
		switch {
		case d.filterHeader.level == 0:
			// Without a loop filter the just-reconstructed row is final.
			rowsAvailable = mby + 1
		case mby == d.mbh-1:
			// Flush the final row, whose below-neighbor is the frame edge.
			if d.filterHeader.simple {
				for mbx := 0; mbx < d.mbw; mbx++ {
					d.simpleFilterMacroblock(mbx, mby)
				}
			} else {
				for mbx := 0; mbx < d.mbw; mbx++ {
					d.normalFilterMacroblock(mbx, mby)
				}
			}
			rowsAvailable = d.mbh
		case mby >= 2:
			// Row mby-1 was just filtered, finalizing row mby-2's bottom
			// boundary, so rows 0..mby-2 are immutable.
			rowsAvailable = mby - 1
		default:
			// mby 0 or 1 with filtering active: no row has completed both its
			// own and the below-row's filter phase yet, so no event is
			// emitted.
		}
		if rowsAvailable > 0 && cb != nil {
			if err = cb(Progress{
				Stage:          StageMacroblockRow,
				FrameHeader:    frameHeader,
				RowsAvailable:  rowsAvailable,
				MacroblockRows: d.mbh,
				Image:          d.img,
			}); err != nil {
				return nil, d.wrapError(StageMacroblockRow, -1, err)
			}
		}
	}

	if cb != nil {
		if err = cb(Progress{
			Stage:          StageComplete,
			FrameHeader:    frameHeader,
			RowsAvailable:  d.mbh,
			MacroblockRows: d.mbh,
			Image:          d.img,
		}); err != nil {
			return nil, d.wrapError(StageComplete, -1, err)
		}
	}
	return d.img, nil
}

// wrapError annotates err with the current frame/partition context. Errors
// already wrapped (for example those produced by a callback) are passed
// through unchanged.
func (d *Decoder) wrapError(stage ProgressStage, partition int, err error) error {
	if err == nil {
		return nil
	}
	var de *DecodeError
	if errors.As(err, &de) {
		return err
	}
	return &DecodeError{Stage: stage, Frame: 0, Partition: partition, Err: err}
}

// failingPartition returns the index of the first coefficient partition whose
// arithmetic decoder hit an unexpected EOF, or -1 if none did.
func (d *Decoder) failingPartition() int {
	for i := 0; i < d.nOP; i++ {
		if d.op[i].unexpectedEOF {
			return i
		}
	}
	return -1
}

// resetFrameState discards the per-frame working state so that the decoder
// can be re-used for the next frame after a cancellation or a malformed-frame
// error. It returns every field that a frame can mutate to the zero value a
// fresh Decoder starts with; the next frame's DecodeFrameHeader then
// re-establishes the key-frame defaults (segment probabilities, token
// probabilities, dimensions) exactly as it would on a new Decoder. The
// limitReader is left untouched because the caller re-inits the decoder with
// a fresh payload before the next frame anyway.
func (d *Decoder) resetFrameState() {
	d.frameHeader = FrameHeader{}
	d.mbw, d.mbh = 0, 0
	d.segmentHeader = segmentHeader{}
	d.filterHeader = filterHeader{}
	d.img = nil
	d.fp = partition{}
	for i := range d.op {
		d.op[i] = partition{}
	}
	d.nOP = 0
	for i := range d.quant {
		d.quant[i] = quant{}
	}
	d.tokenProb = [nPlane][nBand][nContext][nProb]uint8{}
	d.useSkipProb = false
	d.skipProb = 0
	d.filterParams = [nSegment][2]filterParam{}
	d.perMBFilterParams = nil
	d.upMB = nil
	d.leftMB = mb{}
	d.segment = 0
	d.nzDCMask, d.nzACMask = 0, 0
	d.usePredY16 = false
	d.predY16, d.predC8 = 0, 0
	d.predY4 = [4][4]uint8{}
	for i := range d.coeff {
		d.coeff[i] = 0
	}
	d.ybr = [1 + 16 + 1 + 8][32]uint8{}
}

// FrameReader supplies the encoded bytes of successive VP8 frames. A bare VP8
// bitstream is not self-delimiting: the surrounding container (for example an
// IVF or WebM/WebP chunk) knows each frame's length, so an implementation
// returns one frame's payload per ReadFrame call. The returned slice must
// remain valid until the next ReadFrame call.
type FrameReader interface {
	// ReadFrame returns the bytes of the next frame. Returning io.EOF signals
	// that there are no more frames.
	ReadFrame() ([]byte, error)
}

// FrameReaderFunc is an adapter to allow the use of an ordinary function as a
// FrameReader.
type FrameReaderFunc func() ([]byte, error)

// ReadFrame calls f.
func (f FrameReaderFunc) ReadFrame() ([]byte, error) { return f() }

// IncrementalSession decodes a sequence of frames incrementally on a separate
// goroutine. It is the cancellable, streaming counterpart to calling
// DecodeFrameHeader and DecodeFrame for each frame in turn. A session must be
// closed with Close (or fully drained with Wait) to release its goroutine.
type IncrementalSession struct {
	decoder *Decoder

	progress chan Progress
	errOnce  sync.Once
	err      error

	cancel chan struct{}
	done   chan struct{}

	frameMu sync.Mutex
	frame   int
	closed  bool
}

// NewIncrementalSession starts an incremental decoding session whose frames
// are supplied by frames.
//
// Progress events are delivered on the channel returned by Progress, strictly
// in order, one frame at a time. The channel is closed when the session
// finishes, either because frames returned io.EOF, because a frame failed to
// decode, or because Close was called.
//
// The session runs on exactly one goroutine, which owns the underlying
// Decoder; no other goroutine touches it.
func NewIncrementalSession(frames FrameReader) *IncrementalSession {
	s := &IncrementalSession{
		decoder: NewDecoder(),
		// A small buffer decouples a slow thumbnail consumer from the decode
		// goroutine without accumulating an unbounded number of partial rows.
		progress: make(chan Progress, 8),
		cancel:   make(chan struct{}),
		done:     make(chan struct{}),
	}
	go s.run(frames)
	return s
}

// Progress returns the channel on which progress events are delivered. The
// channel is closed when the session ends.
func (s *IncrementalSession) Progress() <-chan Progress { return s.progress }

// Close asks the session to stop decoding and waits for its goroutine to exit,
// so it is safe to call repeatedly and from a different goroutine than the one
// draining Progress. It never starts additional goroutines and guarantees that
// the session's goroutine does not leak: after Close returns, the goroutine
// has terminated.
func (s *IncrementalSession) Close() error {
	s.frameMu.Lock()
	if !s.closed {
		s.closed = true
		close(s.cancel)
	}
	s.frameMu.Unlock()
	return s.Wait()
}

// Wait blocks until the session's goroutine has exited and returns the error
// that terminated it. It returns nil if the input ended cleanly after a
// complete frame and context.Canceled if Close canceled the session.
func (s *IncrementalSession) Wait() error {
	<-s.done
	return s.err
}

// FrameIndex returns the zero-based index of the frame currently being
// decoded (or the frame after the last completed one).
func (s *IncrementalSession) FrameIndex() int {
	s.frameMu.Lock()
	defer s.frameMu.Unlock()
	return s.frame
}

// run is the session's goroutine. It owns the decoder; no other goroutine
// touches it.
func (s *IncrementalSession) run(frames FrameReader) {
	defer close(s.done)
	defer close(s.progress)

	for {
		select {
		case <-s.cancel:
			s.fail(context.Canceled)
			return
		default:
		}

		payload, err := frames.ReadFrame()
		if err == io.EOF {
			s.fail(nil)
			return
		}
		if err != nil {
			s.fail(&DecodeError{
				Stage:     StageFrameHeader,
				Frame:     s.frame,
				Partition: -1,
				Err:       err,
			})
			return
		}
		if len(payload) == 0 {
			s.fail(&DecodeError{
				Stage:     StageFrameHeader,
				Frame:     s.frame,
				Partition: -1,
				Err:       io.ErrUnexpectedEOF,
			})
			return
		}

		frame := s.frame
		s.decoder.Init(bytes.NewReader(payload), len(payload))
		_, err = s.decoder.DecodeFrameIncremental(func(p Progress) error {
			return s.deliver(p)
		})
		if err != nil {
			s.fail(&DecodeError{
				Stage:     errorStage(err),
				Frame:     frame,
				Partition: errorPartition(err),
				Err:       underlyingError(err),
			})
			return
		}

		s.frameMu.Lock()
		s.frame++
		s.frameMu.Unlock()
	}
}

// deliver sends one progress event, honoring a concurrent Close. It is the
// only place the goroutine blocks on the consumer, which is why cancellation
// can always make progress and the goroutine can never be stuck forever.
func (s *IncrementalSession) deliver(p Progress) error {
	select {
	case s.progress <- p:
		return nil
	case <-s.cancel:
		return context.Canceled
	}
}

// fail records the terminal session error. A nil error means a clean end of
// stream.
func (s *IncrementalSession) fail(err error) {
	s.errOnce.Do(func() {
		s.err = err
	})
}

func errorStage(err error) ProgressStage {
	var de *DecodeError
	if errors.As(err, &de) {
		return de.Stage
	}
	return StageFrameHeader
}

func errorPartition(err error) int {
	var de *DecodeError
	if errors.As(err, &de) {
		return de.Partition
	}
	return -1
}

func underlyingError(err error) error {
	var de *DecodeError
	if errors.As(err, &de) {
		return de.Err
	}
	return err
}
