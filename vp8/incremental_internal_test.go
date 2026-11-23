// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vp8

import (
	"errors"
	"strings"
	"testing"
)

// testSentinel is a package-internal sentinel used by TestDecodeErrorContext.
var testSentinel = errors.New("vp8-test: sentinel")

// TestDecodeErrorContext checks the framing of a DecodeError, including the
// partition-less form, the wrapped-sentinel form and the stage name.
func TestDecodeErrorContext(t *testing.T) {
	e := &DecodeError{Stage: StageMacroblockRow, Frame: 7, Partition: 3, Err: testSentinel}
	if e.Error() == "" {
		t.Fatal("empty error message")
	}
	if e.Unwrap() != testSentinel {
		t.Error("Unwrap did not return the underlying error")
	}
	if !e.Is(testSentinel) {
		t.Error("Is failed for the wrapped sentinel")
	}
	if stageName(ProgressStage(99)) != "unknown" {
		t.Error("unknown stage did not render as unknown")
	}
	if StageFrameHeader.String() != "frame-header" {
		t.Errorf("stage String = %q", StageFrameHeader.String())
	}

	// A partition-less error must not carry the partition clause.
	noPart := &DecodeError{Stage: StagePartitions, Frame: 0, Partition: -1, Err: testSentinel}
	if got := noPart.Error(); got == "" || strings.Contains(got, ", partition ") {
		t.Errorf("partition-less error message wrong: %q", got)
	}
}

// TestRowsAvailableSchedule exercises the row-immutability accounting used by
// DecodeFrameIncremental without needing a fully encoded frame. It mirrors the
// switch in the decode loop and pins the boundary behavior for the smallest
// macroblock grids, where the two-row filter lag matters most.
func TestRowsAvailableSchedule(t *testing.T) {
	cases := []struct {
		name   string
		mbh    int
		filter bool
		want   []int // RowsAvailable after each macroblock-row boundary
	}{
		{"no-filter 1 row", 1, false, []int{1}},
		{"no-filter 2 rows", 2, false, []int{1, 2}},
		{"no-filter 4 rows", 4, false, []int{1, 2, 3, 4}},
		// With filtering, the first two boundaries (mby 0 and 1) emit no
		// event; mby>=2 emits rowsAvailable=mby-1; the final boundary flushes
		// all mbh rows. For a single row the final flush is the only event.
		{"filter 1 row", 1, true, []int{1}},
		// Two rows: mby 0 is silent, mby 1 is the final flush to 2.
		{"filter 2 rows", 2, true, []int{2}},
		// Three rows: mby 0,1 silent, mby 2 is final (flush to 3). The mby>=2
		// intermediate case never fires on the last row.
		{"filter 3 rows", 3, true, []int{3}},
		// Four rows: mby 2 emits rows=1, mby 3 flushes to all 4.
		{"filter 4 rows", 4, true, []int{1, 4}},
		// Five rows: mby 2 -> 1, mby 3 -> 2, mby 4 flushes to 5.
		{"filter 5 rows", 5, true, []int{1, 2, 5}},
		// Six rows: intermediates mby 2,3,4 give 1,2,3 (=1..mbh-3), then the
		// final boundary flushes to all rows.
		{"filter 6 rows", 6, true, []int{1, 2, 3, 6}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got []int
			for mby := 0; mby < tc.mbh; mby++ {
				rowsAvailable := 0
				switch {
				case !tc.filter:
					rowsAvailable = mby + 1
				case mby == tc.mbh-1:
					rowsAvailable = tc.mbh
				case mby >= 2:
					rowsAvailable = mby - 1
				}
				if rowsAvailable > 0 {
					got = append(got, rowsAvailable)
				}
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// TestResetFrameStateIsIdempotent confirms that resetting a zero-value
// decoder stays a zero-value decoder, i.e. reset does not assume prior
// allocation and can run repeatedly.
func TestResetFrameStateIsIdempotent(t *testing.T) {
	d := NewDecoder()
	d.resetFrameState()
	d.resetFrameState()
	if d.img != nil || d.nOP != 0 || d.mbw != 0 || d.mbh != 0 || d.busy {
		t.Fatalf("reset left non-zero state: img=%v nOP=%d mbw=%d mbh=%d busy=%v",
			d.img, d.nOP, d.mbw, d.mbh, d.busy)
	}
	if d.upMB != nil || d.perMBFilterParams != nil {
		t.Fatal("reset did not drop per-macroblock slices")
	}
}
