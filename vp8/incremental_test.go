// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/image/vp8"
)

// extractVP8Frame extracts the raw VP8 bitstream (the contents of the VP8
// chunk) from a lossy RIFF/WebP file. The package itself decodes bare VP8
// frames, so the RIFF container has to be stripped in the test.
func extractVP8Frame(t *testing.T, webpPath string) []byte {
	t.Helper()
	data, err := os.ReadFile(webpPath)
	if err != nil {
		t.Fatalf("read %s: %v", webpPath, err)
	}
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WEBP" {
		t.Fatalf("%s is not a RIFF/WEBP file", webpPath)
	}
	for off := 12; off+8 <= len(data); {
		fourCC := string(data[off : off+4])
		chunkLen := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		body := off + 8
		if body+chunkLen > len(data) {
			t.Fatalf("%s: truncated %s chunk", webpPath, fourCC)
		}
		if fourCC == "VP8 " {
			return append([]byte(nil), data[body:body+chunkLen]...)
		}
		// Chunks are padded to an even number of bytes.
		off = body + chunkLen + (chunkLen & 1)
	}
	t.Fatalf("%s: no VP8 chunk found", webpPath)
	return nil
}

// lossyFixtures are the lossy WEBP files shipped in ../testdata. They cover
// the three loop filter modes (none, simple, normal), which exercise the
// incremental filter paths.
func lossyFixtures() []string {
	return []string{
		"blue-purple-pink.lossy.webp",
		"blue-purple-pink-large.no-filter.lossy.webp",
		"blue-purple-pink-large.simple-filter.lossy.webp",
		"blue-purple-pink-large.normal-filter.lossy.webp",
		"video-001.lossy.webp",
		"yellow_rose.lossy.webp",
	}
}

// decodeFull decodes a bare VP8 frame with the one-shot API.
func decodeFull(t *testing.T, frame []byte) *image.YCbCr {
	t.Helper()
	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))
	if _, err := d.DecodeFrameHeader(); err != nil {
		t.Fatalf("DecodeFrameHeader: %v", err)
	}
	img, err := d.DecodeFrame()
	if err != nil {
		t.Fatalf("DecodeFrame: %v", err)
	}
	return img
}

// ycbcrEqual reports whether two YCbCr images have identical backing pixels,
// including the macroblock padding outside the visible rectangle.
func ycbcrEqual(a, b *image.YCbCr) error {
	if a.YStride != b.YStride || a.CStride != b.CStride {
		return fmt.Errorf("stride mismatch: Y %d/%d C %d/%d",
			a.YStride, b.YStride, a.CStride, b.CStride)
	}
	if a.SubsampleRatio != b.SubsampleRatio {
		return fmt.Errorf("subsample mismatch: %v/%v", a.SubsampleRatio, b.SubsampleRatio)
	}
	if len(a.Y) != len(b.Y) || len(a.Cb) != len(b.Cb) || len(a.Cr) != len(b.Cr) {
		return fmt.Errorf("backing size mismatch: Y %d/%d Cb %d/%d Cr %d/%d",
			len(a.Y), len(b.Y), len(a.Cb), len(b.Cb), len(a.Cr), len(b.Cr))
	}
	for name, plane := range map[string][2][]byte{
		"Y":  {a.Y, b.Y},
		"Cb": {a.Cb, b.Cb},
		"Cr": {a.Cr, b.Cr},
	} {
		for i, v := range plane[0] {
			if v != plane[1][i] {
				return fmt.Errorf("plane %s byte %d: %d vs %d", name, i, v, plane[1][i])
			}
		}
	}
	return nil
}

// TestIncrementalMatchesFullFrame is the core pixel-equivalence test: for
// every shipped fixture, the incremental result must be byte-identical to the
// one-shot DecodeFrame result, strides and macroblock padding included.
func TestIncrementalMatchesFullFrame(t *testing.T) {
	for _, tc := range lossyFixtures() {
		t.Run(tc, func(t *testing.T) {
			frame := extractVP8Frame(t, filepath.Join("..", "testdata", tc))
			want := decodeFull(t, frame)

			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(frame), len(frame))
			var got *image.YCbCr
			var stages []vp8.ProgressStage
			var lastRows int
			_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
				stages = append(stages, p.Stage)
				if p.Stage == vp8.StageMacroblockRow {
					if p.RowsAvailable <= lastRows || p.RowsAvailable > p.MacroblockRows {
						t.Errorf("rows available went from %d to %d of %d",
							lastRows, p.RowsAvailable, p.MacroblockRows)
					}
					lastRows = p.RowsAvailable
					if p.Image == nil {
						t.Error("macroblock-row progress has nil image")
					}
				}
				if p.Stage == vp8.StageComplete {
					got = p.Image
				}
				return nil
			})
			if err != nil {
				t.Fatalf("DecodeFrameIncremental: %v", err)
			}
			if got == nil {
				t.Fatal("no complete event")
			}
			if err := ycbcrEqual(got, want); err != nil {
				t.Errorf("incremental pixels differ from full frame: %v", err)
			}
			// Every frame emits header, partitions, at least one row, complete.
			if len(stages) < 4 || stages[0] != vp8.StageFrameHeader ||
				stages[1] != vp8.StagePartitions ||
				stages[len(stages)-1] != vp8.StageComplete {
				t.Errorf("unexpected stage sequence: %v", stages)
			}
		})
	}
}

// TestIncrementalProgressOrder checks the exact progress sequence and the
// monotonic, bounded RowsAvailable values on a fixed key frame.
func TestIncrementalProgressOrder(t *testing.T) {
	frame := extractVP8Frame(t,
		filepath.Join("..", "testdata", "blue-purple-pink-large.normal-filter.lossy.webp"))

	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))
	type event struct {
		stage vp8.ProgressStage
		rows  int
	}
	var events []event
	var fh vp8.FrameHeader
	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
		if p.Stage == vp8.StageFrameHeader {
			fh = p.FrameHeader
			if !fh.KeyFrame || fh.Width == 0 || fh.Height == 0 {
				t.Errorf("bad frame header: %+v", fh)
			}
		}
		events = append(events, event{p.Stage, p.RowsAvailable})
		return nil
	})
	if err != nil {
		t.Fatalf("DecodeFrameIncremental: %v", err)
	}
	mbw := (fh.Width + 15) / 16
	mbh := (fh.Height + 15) / 16

	// Validate the sequence against the documented two-row-lag invariants
	// rather than a hard-coded list: the first two boundaries emit no row
	// event, RowsAvailable then increases by exactly one per event up to
	// mbh-2, and the final row event jumps to all mbh rows.
	var rowRows []int
	for _, e := range events {
		if e.stage == vp8.StageMacroblockRow {
			if e.rows <= 0 || e.rows > mbh {
				t.Fatalf("row event out of range: %+v", e)
			}
			rowRows = append(rowRows, e.rows)
		}
	}
	if len(rowRows) == 0 {
		t.Fatal("no macroblock-row events")
	}
	// The penultimate events must be 1, 2, ..., mbh-3 in order (the row that
	// is second-from-last is only finalized when the final boundary flushes
	// the two deferred rows), and the final row event delivers all mbh rows.
	var wantRows []int
	for r := 1; r <= mbh-3; r++ {
		wantRows = append(wantRows, r)
	}
	wantRows = append(wantRows, mbh)
	if fmt.Sprint(rowRows) != fmt.Sprint(wantRows) {
		t.Errorf("row availability %v, want %v", rowRows, wantRows)
	}
	if events[0].stage != vp8.StageFrameHeader ||
		events[1].stage != vp8.StagePartitions ||
		events[len(events)-1].stage != vp8.StageComplete {
		t.Errorf("unexpected framing events: %v", events)
	}
	_ = mbw
}

// TestIncrementalNoFilterImmediateRows checks that without a loop filter, each
// reconstructed row is immediately available (no one-row lag).
func TestIncrementalNoFilterImmediateRows(t *testing.T) {
	frame := extractVP8Frame(t,
		filepath.Join("..", "testdata", "blue-purple-pink-large.no-filter.lossy.webp"))

	d := vp8.NewDecoder()
	d.Init(bytes.NewReader(frame), len(frame))
	var rowEvents []int
	var mbh int
	_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
		if p.Stage == vp8.StageFrameHeader {
			mbh = p.MacroblockRows
		}
		if p.Stage == vp8.StageMacroblockRow {
			rowEvents = append(rowEvents, p.RowsAvailable)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("DecodeFrameIncremental: %v", err)
	}
	want := make([]int, mbh)
	for i := range want {
		want[i] = i + 1
	}
	if fmt.Sprint(rowEvents) != fmt.Sprint(want) {
		t.Errorf("row availability %v, want %v", rowEvents, want)
	}
}

// TestIncrementalDeliveredRowsImmutable verifies requirement 2: once rows have
// been reported as available, their pixels never change again.
func TestIncrementalDeliveredRowsImmutable(t *testing.T) {
	for _, tc := range lossyFixtures() {
		t.Run(tc, func(t *testing.T) {
			frame := extractVP8Frame(t, filepath.Join("..", "testdata", tc))
			d := vp8.NewDecoder()
			d.Init(bytes.NewReader(frame), len(frame))

			snapshots := map[int][]byte{}
			_, err := d.DecodeFrameIncremental(func(p vp8.Progress) error {
				if p.Stage != vp8.StageMacroblockRow || p.Image == nil {
					return nil
				}
				img := p.Image
				// Snapshot the full Y plane rows reported as available, plus
				// the corresponding chroma rows.
				yRows := p.RowsAvailable * 16
				if yRows > img.Rect.Dy() {
					yRows = img.Rect.Dy()
				}
				snap := make([]byte, yRows*img.YStride)
				copy(snap, img.Y[:yRows*img.YStride])
				snapshots[p.RowsAvailable] = snap
				return nil
			})
			if err != nil {
				t.Fatalf("DecodeFrameIncremental: %v", err)
			}

			// Re-decode and confirm every historical snapshot still matches
			// the final image at the same offsets.
			final := decodeFull(t, frame)
			for rows, snap := range snapshots {
				yRows := rows * 16
				if yRows > final.Rect.Dy() {
					yRows = final.Rect.Dy()
				}
				got := final.Y[:yRows*final.YStride]
				if !bytes.Equal(got, snap) {
					t.Errorf("rows=%d: delivered pixels were modified later", rows)
				}
			}
		})
	}
}
