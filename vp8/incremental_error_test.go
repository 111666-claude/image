// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/image/vp8"
)

// keyFrame returns a fixed key-frame fixture used across the error/cancel
// tests, so failures are reproducible against one known bitstream.
func keyFrame(t *testing.T) []byte {
	t.Helper()
	return extractVP8Frame(t,
		filepath.Join("..", "testdata", "blue-purple-pink-large.normal-filter.lossy.webp"))
}

// numGoroutines is a small helper used by the leak tests.
func numGoroutines() int {
	// Let parked goroutines settle before counting.
	runtime.Gosched()
	return runtime.NumGoroutine()
}

// TestCancelMidFrameAndReuse cancels a decode at a fixed macroblock row, then
// reuses the same Decoder for a complete frame and checks the pixels against
// a fresh one-shot decode. Requirement 1: cancellation must reset safely.
func TestCancelMidFrameAndReuse(t *testing.T) {
	frame := keyFrame(t)
	want := decodeFull(t, frame)

	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))
	const cancelAtRows = 2
	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
		if p.Stage == vp8.StageMacroblockRow && p.RowsAvailable >= cancelAtRows {
			return context.Canceled
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected cancellation error, got nil")
	}
	var de *vp8.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("error is not *DecodeError: %T %v", err, err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(err, context.Canceled) = false; err=%v", err)
	}
	if de.Stage != vp8.StageMacroblockRow {
		t.Errorf("stage = %v, want macroblock-row", de.Stage)
	}
	if !strings.Contains(de.Error(), "macroblock-row") {
		t.Errorf("error message lacks stage context: %q", de.Error())
	}

	// Re-initialize the very same decoder and decode the next frame fully.
	// As per the package contract, the one-shot path requires an explicit
	// DecodeFrameHeader before DecodeFrame; that header parse also
	// re-establishes the key-frame state cleared by the cancellation reset.
	d.Init(bytes.NewReader(frame), len(frame))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatalf("DecodeFrameHeader after cancellation: %v", err)
	}
	got, err := d.DecodeFrame()
	if err != nil {
		t.Fatalf("DecodeFrame after cancellation: %v", err)
	}
	if err := ycbcrEqual(got, want); err != nil {
		t.Errorf("frame after cancellation differs: %v", err)
	}

	// The incremental API must work on the reused decoder as well.
	d.Init(bytes.NewReader(frame), len(frame))
	var img *image.YCbCr
	_, err = d.DecodeFrameIncremental(func(p vp8.Progress) error {
		if p.Stage == vp8.StageComplete {
			img = p.Image
		}
		return nil
	})
	if err != nil {
		t.Fatalf("incremental decode after reuse: %v", err)
	}
	if err := ycbcrEqual(img, want); err != nil {
		t.Errorf("incremental frame after reuse differs: %v", err)
	}
}

// TestCancelAtEveryBoundary cancels at each distinct stage in turn and
// verifies the decoder always recovers.
func TestCancelAtEveryBoundary(t *testing.T) {
	frame := keyFrame(t)
	cancelStages := []vp8.ProgressStage{
		vp8.StageFrameHeader,
		vp8.StagePartitions,
		vp8.StageMacroblockRow,
		vp8.StageComplete,
	}
	for _, stage := range cancelStages {
		t.Run(stage.String(), func(t *testing.T) {
			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(frame), len(frame))
			callbackErr := errors.New("stop here")
			_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
				if p.Stage == stage {
					return callbackErr
				}
				return nil
			})
			if !errors.Is(err, callbackErr) {
				t.Fatalf("stage %v: err = %v, want callback error", stage, err)
			}
			// Recovery: full decode on the same decoder, re-parsing the frame
			// header first as the one-shot API requires.
			d.Init(bytes.NewReader(frame), len(frame))
			if _, err := d.DecodeFrameHeader(); err != nil {
				t.Fatalf("recovery header after stage %v: %v", stage, err)
			}
			if _, err := d.DecodeFrame(); err != nil {
				t.Fatalf("recovery after stage %v: %v", stage, err)
			}
		})
	}
}

// TestTruncatedPartitions builds partition-failure fixtures by truncating the
// frame at several points past the (complete) frame header. Each must fail
// with frame/partition context rather than panic, and the decoder's reset must
// leave it able to decode a complete frame afterwards.
func TestTruncatedPartitions(t *testing.T) {
	frame := keyFrame(t)
	// A key-frame header is 10 bytes. Every cut below keeps it intact so the
	// failure is genuinely in the partition data rather than in frame
	// framing, and spans the first partition and the coefficient partitions.
	cuts := []int{10, 20, len(frame) / 4, len(frame) / 2, 3 * len(frame) / 4}
	for _, cut := range cuts {
		if cut >= len(frame) {
			continue
		}
		t.Run(fmt.Sprintf("cut=%d", cut), func(t *testing.T) {
			truncated := frame[:cut]
			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(truncated), len(truncated))
			img, err := d.DecodeFrameIncremental(func(p vp8.Progress) error { return nil })
			if err == nil {
				t.Fatalf("cut %d: expected error, got image %v", cut, img != nil)
			}
			var de *vp8.DecodeError
			if !errors.As(err, &de) {
				t.Fatalf("cut %d: error is not *DecodeError: %v", cut, err)
			}
			if !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
				t.Errorf("cut %d: unexpected underlying error %v", cut, err)
			}
			// Reuse the very same decoder after the reset. Re-initializing and
			// re-parsing the frame header re-establishes key-frame state; the
			// subsequent decode must succeed, proving no partial state leaked.
			d.Init(bytes.NewReader(frame), len(frame))
			if _, err := d.DecodeFrameHeader(); err != nil {
				t.Fatalf("cut %d: header after reset: %v", cut, err)
			}
			if _, err := d.DecodeFrame(); err != nil {
				t.Fatalf("cut %d: decoder not reusable one-shot: %v", cut, err)
			}
			// And the incremental entry point must recover too.
			d.Init(bytes.NewReader(frame), len(frame))
			if _, err := d.DecodeFrameIncremental(nil); err != nil {
				t.Fatalf("cut %d: decoder not reusable incrementally: %v", cut, err)
			}
		})
	}
}

// TestTruncatedFrameHeader covers a frame whose 3-byte tag or 10-byte key
// header is itself cut short. The decoder must report a wrapped error without
// panicking and still be reusable for a complete frame.
func TestTruncatedFrameHeader(t *testing.T) {
	frame := keyFrame(t)
	for _, cut := range []int{0, 1, 3, 7, 9} {
		t.Run(fmt.Sprintf("cut=%d", cut), func(t *testing.T) {
			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(frame[:cut]), cut)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("cut %d panicked: %v", cut, r)
				}
			}()
			if _, err := d.DecodeFrameIncremental(nil); err == nil {
				t.Fatalf("cut %d: expected error", cut)
			}
			// Re-initialize and re-parse the header on the same decoder; the
			// reset must leave it fully functional.
			d.Init(bytes.NewReader(frame), len(frame))
			if _, err := d.DecodeFrameHeader(); err != nil {
				t.Fatalf("cut %d: header after reset: %v", cut, err)
			}
			if _, err := d.DecodeFrame(); err != nil {
				t.Fatalf("cut %d: decoder not reusable: %v", cut, err)
			}
		})
	}
}

// TestTruncatedFirstPartition corrupts only the first (arithmetic-coded
// header) partition and checks that its unexpected EOF is surfaced with
// context, without touching any coefficient-partition index.
func TestTruncatedFirstPartition(t *testing.T) {
	frame := keyFrame(t)
	// FirstPartitionLen is encoded in the first 3 bytes.
	fpl := int(uint32(frame[0])>>5 | uint32(frame[1])<<3 | uint32(frame[2])<<11)
	if fpl <= 1 || fpl >= len(frame) {
		t.Fatalf("unexpected first partition length %d", fpl)
	}
	// Keep the header but cut inside the first partition payload.
	cut := 3 + fpl/2
	corrupt := append([]byte(nil), frame[:cut]...)
	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(corrupt), len(corrupt))
	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error { return nil })
	if err == nil {
		t.Fatal("expected error from truncated first partition")
	}
	var de *vp8.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("not a *DecodeError: %v", err)
	}
	if de.Partition != -1 {
		t.Errorf("first-partition failure reported partition %d, want -1", de.Partition)
	}
}

// TestZeroSizeAndOversizedFrames covers the dimension guards.
func TestZeroSizeAndOversizedFrames(t *testing.T) {
	frame := keyFrame(t)

	// A key-frame header starts at frame byte 3 (after the 3-byte frame tag).
	// Per section 9.1 its layout is:
	//	3+0..2: magic sync code
	//	3+3..4: width-1  (low byte, then 6 bits in the high byte)
	//	3+5..6: height-1
	// so width is at frame bytes 6-7 and height at 8-9. The high bytes' top
	// two bits carry the X/Y scale flags and must be preserved.
	const (
		widthLo  = 6
		widthHi  = 7
		heightLo = 8
		heightHi = 9
	)

	// Zero width: clear only the 14 width bits, preserving the scale bits.
	zero := append([]byte(nil), frame...)
	zero[widthLo] = 0
	zero[widthHi] &^= 0x3f
	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(zero), len(zero))
	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error { return nil })
	if !errors.Is(err, vp8.ErrZeroSizeFrame) {
		t.Errorf("zero width: err = %v, want ErrZeroSizeFrame", err)
	}
	var de *vp8.DecodeError
	if !errors.As(err, &de) || de.Stage != vp8.StageFrameHeader {
		t.Errorf("zero-size error context = %v", err)
	}

	// Zero height as well.
	zeroH := append([]byte(nil), frame...)
	zeroH[heightLo] = 0
	zeroH[heightHi] &^= 0x3f
	d = vp8.NewDecoder()
	d.Init(bytes.NewReader(zeroH), len(zeroH))
	if _, err := d.DecodeFrameIncremental(func(p vp8.Progress) error { return nil }); !errors.Is(err, vp8.ErrZeroSizeFrame) {
		t.Errorf("zero height: err = %v, want ErrZeroSizeFrame", err)
	}

	// Dimensions above maxFrameDimension (4096): the guard must reject the
	// frame at the frame-header stage, before allocating image buffers. Set
	// the 14-bit width/height to their maximum representable value (16383).
	big := append([]byte(nil), frame...)
	big[widthLo] = 0xff
	big[widthHi] = big[widthHi]&0xc0 | 0x3f
	big[heightLo] = 0xff
	big[heightHi] = big[heightHi]&0xc0 | 0x3f
	d = vp8.NewDecoder()
	d.Init(bytes.NewReader(big), len(big))
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("decoder panicked on oversized dimensions: %v", r)
		}
	}()
	_, err = d.DecodeFrameIncremental(func(p vp8.Progress) error { return nil })
	if !errors.Is(err, vp8.ErrFrameTooLarge) {
		t.Errorf("oversized frame: err = %v, want ErrFrameTooLarge", err)
	}
	if !errors.As(err, &de) {
		t.Fatalf("oversized error is not a *DecodeError: %v", err)
	}
	if de.Stage != vp8.StageFrameHeader {
		t.Errorf("oversized error stage = %v, want frame-header", de.Stage)
	}
}

// TestDuplicateIncrementalCalls verifies that re-entering an incremental
// decode from inside its own callback (or while one is active) fails cleanly
// instead of corrupting state.
func TestDuplicateIncrementalCalls(t *testing.T) {
	frame := keyFrame(t)
	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))

	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
		if p.Stage != vp8.StagePartitions {
			return nil
		}
		// Nested incremental attempt on the same decoder.
		_, nestedErr := d.DecodeFrameIncremental(func(q vp8.Progress) error { return nil })
		if nestedErr == nil {
			t.Error("nested DecodeFrameIncremental succeeded unexpectedly")
		}
		// And a nested one-shot attempt.
		if _, nestedErr := d.DecodeFrame(); nestedErr == nil {
			t.Error("nested DecodeFrame succeeded unexpectedly")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("outer decode failed: %v", err)
	}
}

// TestSessionMultiFrame decodes a sequence of fixed key frames through an
// IncrementalSession and checks every final image and the per-frame event
// ordering and frame indexing.
func TestSessionMultiFrame(t *testing.T) {
	frame := keyFrame(t)
	want := decodeFull(t, frame)
	const nFrames = 3
	idx := 0
	frames := vp8.FrameReaderFunc(func() ([]byte, error) {
		if idx >= nFrames {
			return nil, io.EOF
		}
		idx++
		return frame, nil
	})

	before := numGoroutines()
	s := vp8.NewIncrementalSession(frames)

	var completed int
	var stagesPerFrame [][]vp8.ProgressStage
	var current []vp8.ProgressStage
	for p := range s.Progress() {
		current = append(current, p.Stage)
		if p.Stage == vp8.StageComplete {
			if err := ycbcrEqual(p.Image, want); err != nil {
				t.Errorf("frame %d pixels: %v", completed, err)
			}
			completed++
			stagesPerFrame = append(stagesPerFrame, current)
			current = nil
		}
	}
	if err := s.Wait(); err != nil {
		t.Fatalf("session ended with error: %v", err)
	}
	if completed != nFrames {
		t.Errorf("completed %d frames, want %d", completed, nFrames)
	}
	if len(stagesPerFrame) != nFrames {
		t.Fatalf("got %d frame event groups", len(stagesPerFrame))
	}
	for i, st := range stagesPerFrame {
		if st[0] != vp8.StageFrameHeader || st[len(st)-1] != vp8.StageComplete {
			t.Errorf("frame %d bad stage sequence: %v", i, st)
		}
	}
	if s.FrameIndex() != nFrames {
		t.Errorf("FrameIndex = %d, want %d", s.FrameIndex(), nFrames)
	}
	// Close after natural completion must be a harmless no-op.
	if err := s.Close(); err != nil {
		t.Errorf("Close after completion: %v", err)
	}
	if got := numGoroutines(); got > before+1 {
		t.Errorf("goroutine leak: before=%d after=%d", before, got)
	}
}

// TestSessionCancelGoroutineLeak cancels a session mid-frame through Close and
// proves the goroutine exits promptly and no goroutine leaks.
func TestSessionCancelGoroutineLeak(t *testing.T) {
	frame := keyFrame(t)
	release := make(chan struct{})
	frames := vp8.FrameReaderFunc(func() ([]byte, error) {
		return frame, nil
	})

	before := numGoroutines()
	s := vp8.NewIncrementalSession(frames)

	// Wait until at least one event proves the goroutine started, then Close
	// without draining further events. The buffered channel may fill; Close
	// must still unblock the goroutine via the cancel channel.
	<-s.Progress()
	done := make(chan error, 1)
	go func() { done <- s.Close() }()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Close error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close did not return: goroutine likely leaked")
	}
	close(release)
	if got := numGoroutines(); got > before+1 {
		t.Errorf("goroutine leak after cancel: before=%d after=%d", before, got)
	}
	// Repeated Close is safe and returns the same result.
	if err := s.Close(); !errors.Is(err, context.Canceled) {
		t.Errorf("second Close = %v", err)
	}
}

// TestSessionTruncatedFrame feeds a good frame then a truncated one and
// checks the session terminates with a framed *DecodeError.
func TestSessionTruncatedFrame(t *testing.T) {
	frame := keyFrame(t)
	idx := 0
	frames := vp8.FrameReaderFunc(func() ([]byte, error) {
		idx++
		switch idx {
		case 1:
			return frame, nil
		case 2:
			return frame[:len(frame)/3], nil
		default:
			return nil, io.EOF
		}
	})
	s := vp8.NewIncrementalSession(frames)
	var completed int
	for range s.Progress() {
	}
	err := s.Wait()
	if err == nil {
		t.Fatal("expected session error on truncated frame")
	}
	var de *vp8.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("not a *DecodeError: %v", err)
	}
	if de.Frame != 1 {
		t.Errorf("error frame index = %d, want 1", de.Frame)
	}
	_ = completed
}

// TestSessionCallbacklessCoreNoGoroutine confirms the synchronous core starts
// no goroutines of its own: the count must be unchanged across a decode.
func TestSessionCallbacklessCoreNoGoroutine(t *testing.T) {
	frame := keyFrame(t)
	before := numGoroutines()
	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))
	if _, err := d.DecodeFrameIncremental(nil); err != nil {
		t.Fatalf("decode with nil callback: %v", err)
	}
	if got := numGoroutines(); got != before {
		t.Errorf("synchronous decode changed goroutine count: before=%d after=%d", before, got)
	}
}

// TestConcurrentSessionsAreIndependent runs several sessions in parallel on
// separate decoders (the API does not promise a single Decoder being
// thread-safe) to shake out data races under the race detector.
func TestConcurrentSessionsAreIndependent(t *testing.T) {
	frame := keyFrame(t)
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			frames := vp8.FrameReaderFunc(func() ([]byte, error) {
				return frame, io.EOF // exactly one frame
			})
			s := vp8.NewIncrementalSession(frames)
			for range s.Progress() {
			}
			if err := s.Wait(); err != nil {
				t.Errorf("session: %v", err)
			}
		}()
	}
	wg.Wait()
}

// TestIncrementalGoldenAltRefSemantics verifies that non-key (inter) frames
// follow the same Golden/AltRef semantics through the incremental API as
// through the one-shot API: both reject them, and the incremental error carries
// partition-stage context rather than a panic or state corruption.
func TestIncrementalGoldenAltRefSemantics(t *testing.T) {
	frame := keyFrame(t)
	// Flip the frame tag's least-significant bit: 0 means key frame, 1 means
	// inter frame. Everything else stays byte-identical.
	inter := append([]byte(nil), frame...)
	inter[0] |= 0x01

	// One-shot behavior, for reference: parse the header then attempt a frame.
	d0 := vp8.NewDecoder()
	d0.Init(bytes.NewReader(inter), len(inter))
	fh, err := d0.DecodeFrameHeader()
	if err != nil {
		t.Fatalf("inter-frame header: %v", err)
	}
	if fh.KeyFrame {
		t.Fatal("frame tag flip did not produce an inter frame")
	}
	if _, err := d0.DecodeFrame(); err == nil {
		t.Fatal("one-shot decode unexpectedly accepted an inter frame")
	}

	// Incremental behavior: the same limitation, surfaced with context.
	d1 := vp8.NewDecoder()
	d1.Init(bytes.NewReader(inter), len(inter))
	_, err = d1.DecodeFrameIncremental(nil)
	if err == nil {
		t.Fatal("incremental decode unexpectedly accepted an inter frame")
	}
	var de *vp8.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("inter-frame error is not a *DecodeError: %v", err)
	}
	if de.Stage != vp8.StagePartitions {
		t.Errorf("inter-frame error stage = %v, want partitions", de.Stage)
	}

	// The decoder must still recover on a real key frame afterwards.
	d1.Init(bytes.NewReader(frame), len(frame))
	if _, err := d1.DecodeFrameIncremental(nil); err != nil {
		t.Fatalf("recovery after inter-frame rejection: %v", err)
	}
}
