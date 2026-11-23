// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io/ioutil"
	"path/filepath"
	"testing"

	"golang.org/x/image/font"
	"golang.org/x/image/math/fixed"
)

// readFvarTestFile reads the named file from ../testdata and also returns a
// mutable copy, for tests that corrupt the font data.
func readFvarTestFile(t *testing.T, filename string) []byte {
	t.Helper()
	data, err := ioutil.ReadFile(filepath.FromSlash("../testdata/" + filename))
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	return data
}

// fvarDirEntry returns the byte offset of the fvar table's directory entry
// and of the table data itself, in an SFNT font that starts at data[0].
func fvarDirEntry(t *testing.T, data []byte) (entryOffset, tableOffset int) {
	t.Helper()
	numTables := int(binary.BigEndian.Uint16(data[4:]))
	for i := 0; i < numTables; i++ {
		e := 12 + 16*i
		if string(data[e:e+4]) == "fvar" {
			return e, int(binary.BigEndian.Uint32(data[e+8:]))
		}
	}
	t.Fatalf("no fvar table found")
	return 0, 0
}

func TestFvarAxes(t *testing.T) {
	data := readFvarTestFile(t, "fvarTest.ttf")
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	axes, err := f.VariationAxes()
	if err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	want := []VariationAxis{
		{Tag: "wght", Minimum: 100, Default: 400, Maximum: 900, Flags: 0, NameID: 256},
		{Tag: "wdth", Minimum: 50, Default: 100, Maximum: 200, Flags: VariationAxisFlagHidden, NameID: 257},
	}
	if len(axes) != len(want) {
		t.Fatalf("got %d axes, want %d", len(axes), len(want))
	}
	for i, w := range want {
		if axes[i] != w {
			t.Errorf("axis %d: got %+v, want %+v", i, axes[i], w)
		}
	}

	// The axes' name IDs should resolve via the existing Name method.
	for i, name := range []string{"Weight", "Width"} {
		got, err := f.Name(nil, axes[i].NameID)
		if err != nil {
			t.Errorf("Name(%d): %v", axes[i].NameID, err)
		} else if got != name {
			t.Errorf("Name(%d): got %q, want %q", axes[i].NameID, got, name)
		}
	}

	// The returned slice is a copy: mutating it must not affect the Font.
	axes[0].Tag = "XXXX"
	again, err := f.VariationAxes()
	if err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	if again[0].Tag != "wght" {
		t.Errorf("VariationAxes result aliases the Font's cached data")
	}
}

func TestFvarInstances(t *testing.T) {
	data := readFvarTestFile(t, "fvarTest.ttf")
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	instances, err := f.VariationInstances()
	if err != nil {
		t.Fatalf("VariationInstances: %v", err)
	}
	if len(instances) != 3 {
		t.Fatalf("got %d instances, want 3", len(instances))
	}

	// The first instance sits at the default coordinates.
	if got, want := instances[0].Coordinates, []float64{400, 100}; !float64sEqual(got, want) {
		t.Errorf("instance 0 coordinates: got %v, want %v", got, want)
	}
	if instances[0].HasPostScriptName {
		t.Errorf("instance 0: unexpected PostScript name")
	}

	// The second instance has fractional-free coordinates and a PostScript
	// name ID.
	if got, want := instances[1].Coordinates, []float64{700, 100}; !float64sEqual(got, want) {
		t.Errorf("instance 1 coordinates: got %v, want %v", got, want)
	}
	if !instances[1].HasPostScriptName {
		t.Errorf("instance 1: missing PostScript name")
	} else if got, err := f.Name(nil, instances[1].PostScriptNameID); err != nil {
		t.Errorf("Name(%d): %v", instances[1].PostScriptNameID, err)
	} else if want := "FvarTest-Bold"; got != want {
		t.Errorf("PostScript name: got %q, want %q", got, want)
	}

	// The third instance exercises the 16.16 fixed-point conversion with a
	// fractional coordinate: 75.5 is exactly representable.
	if got, want := instances[2].Coordinates, []float64{700, 75.5}; !float64sEqual(got, want) {
		t.Errorf("instance 2 coordinates: got %v, want %v", got, want)
	}

	// The instances' name IDs should resolve via the existing Name method.
	for i, name := range []string{"Regular", "Bold", "Condensed Bold"} {
		got, err := f.Name(nil, instances[i].NameID)
		if err != nil {
			t.Errorf("Name(%d): %v", instances[i].NameID, err)
		} else if got != name {
			t.Errorf("Name(%d): got %q, want %q", instances[i].NameID, got, name)
		}
	}

	// The returned slice is a deep copy.
	instances[0].Coordinates[0] = -1
	again, err := f.VariationInstances()
	if err != nil {
		t.Fatalf("VariationInstances: %v", err)
	}
	if again[0].Coordinates[0] != 400 {
		t.Errorf("VariationInstances result aliases the Font's cached data")
	}
}

func float64sEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFvarCompatibleWithOtherAPIs(t *testing.T) {
	data := readFvarTestFile(t, "fvarTest.ttf")
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	if got, want := f.UnitsPerEm(), Units(1000); got != want {
		t.Errorf("UnitsPerEm: got %v, want %v", got, want)
	}
	if got, want := f.NumGlyphs(), 3; got != want {
		t.Errorf("NumGlyphs: got %v, want %v", got, want)
	}
	x, err := f.GlyphIndex(nil, 'A')
	if err != nil {
		t.Errorf("GlyphIndex: %v", err)
	} else if x == 0 {
		t.Errorf("GlyphIndex('A'): got 0")
	}
	if _, err := f.Bounds(nil, fixed.Int26_6(f.UnitsPerEm()), font.HintingNone); err != nil {
		t.Errorf("Bounds: %v", err)
	}
	if _, err := f.LoadGlyph(nil, x, fixed.Int26_6(f.UnitsPerEm()), nil); err != nil {
		t.Errorf("LoadGlyph: %v", err)
	}
}

func TestFvarMissingTable(t *testing.T) {
	// glyfTest.ttf is not a variable font.
	data := readFvarTestFile(t, "glyfTest.ttf")
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	axes, err := f.VariationAxes()
	if err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	if axes != nil {
		t.Errorf("VariationAxes: got %v, want nil", axes)
	}
	instances, err := f.VariationInstances()
	if err != nil {
		t.Fatalf("VariationInstances: %v", err)
	}
	if instances != nil {
		t.Errorf("VariationInstances: got %v, want nil", instances)
	}
}

func TestFvarBadTables(t *testing.T) {
	orig := readFvarTestFile(t, "fvarTest.ttf")
	entryOffset, tableOffset := fvarDirEntry(t, orig)
	axesOffset := tableOffset + int(binary.BigEndian.Uint16(orig[tableOffset+4:]))

	testCases := []struct {
		name   string
		mutate func(data []byte)
		want   error
	}{
		{
			name: "truncated table",
			mutate: func(data []byte) {
				// Shrink the fvar table's length below the header size.
				binary.BigEndian.PutUint32(data[entryOffset+12:], 10)
			},
			want: errInvalidFvarTable,
		},
		{
			name: "unsupported version",
			mutate: func(data []byte) {
				binary.BigEndian.PutUint16(data[tableOffset:], 2)
			},
			want: errUnsupportedFvarTable,
		},
		{
			name: "duplicate axis tag",
			mutate: func(data []byte) {
				// Copy the first axis's tag ("wght") over the second's.
				copy(data[axesOffset+20:], data[axesOffset:axesOffset+4])
			},
			want: errInvalidFvarTable,
		},
		{
			name: "inverted range",
			mutate: func(data []byte) {
				// Set the first axis's default (1000) above its max (900).
				binary.BigEndian.PutUint32(data[axesOffset+8:], 1000<<16)
			},
			want: errInvalidFvarTable,
		},
		{
			name: "too many axes",
			mutate: func(data []byte) {
				binary.BigEndian.PutUint16(data[tableOffset+8:], maxNumVariationAxes+1)
			},
			want: errUnsupportedNumberOfVariationAxes,
		},
		{
			name: "too many instances",
			mutate: func(data []byte) {
				binary.BigEndian.PutUint16(data[tableOffset+12:], maxNumVariationInstances+1)
			},
			want: errUnsupportedNumberOfVariationInstances,
		},
		{
			name: "axes beyond table length",
			mutate: func(data []byte) {
				// Claim 200 axes (below the limit) without growing the table.
				binary.BigEndian.PutUint16(data[tableOffset+8:], 200)
			},
			want: errInvalidFvarTable,
		},
	}
	for _, tc := range testCases {
		data := append([]byte(nil), orig...)
		tc.mutate(data)
		f, err := Parse(data)
		if err != nil {
			t.Errorf("%s: Parse: %v", tc.name, err)
			continue
		}
		// A bad fvar table must not break the fixed-font APIs.
		if got, want := f.UnitsPerEm(), Units(1000); got != want {
			t.Errorf("%s: UnitsPerEm: got %v, want %v", tc.name, got, want)
		}
		if _, err := f.VariationAxes(); !errors.Is(err, tc.want) {
			t.Errorf("%s: VariationAxes: got %v, want %v", tc.name, err, tc.want)
		}
		if _, err := f.VariationInstances(); !errors.Is(err, tc.want) {
			t.Errorf("%s: VariationInstances: got %v, want %v", tc.name, err, tc.want)
		}
	}
}

func TestFvarBadNameID(t *testing.T) {
	orig := readFvarTestFile(t, "fvarTest.ttf")
	_, tableOffset := fvarDirEntry(t, orig)
	axesOffset := tableOffset + int(binary.BigEndian.Uint16(orig[tableOffset+4:]))

	// Corrupt the first axis's name ID. The fvar metadata itself remains
	// valid; resolving the dangling name ID reports ErrNotFound.
	data := append([]byte(nil), orig...)
	binary.BigEndian.PutUint16(data[axesOffset+18:], 0xffff)
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	axes, err := f.VariationAxes()
	if err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	if got, want := axes[0].NameID, NameID(0xffff); got != want {
		t.Fatalf("NameID: got %v, want %v", got, want)
	}
	if _, err := f.Name(nil, axes[0].NameID); !errors.Is(err, ErrNotFound) {
		t.Errorf("Name(0xffff): got %v, want %v", err, ErrNotFound)
	}
}

func TestFvarCollection(t *testing.T) {
	data := readFvarTestFile(t, "fvarTest.ttc")
	c, err := ParseCollection(data)
	if err != nil {
		t.Fatalf("ParseCollection: %v", err)
	}
	if got, want := c.NumFonts(), 2; got != want {
		t.Fatalf("NumFonts: got %v, want %v", got, want)
	}

	// The first face is the variable font.
	f0, err := c.Font(0)
	if err != nil {
		t.Fatalf("Font(0): %v", err)
	}
	axes, err := f0.VariationAxes()
	if err != nil {
		t.Fatalf("Font(0) VariationAxes: %v", err)
	}
	if len(axes) != 2 || axes[0].Tag != "wght" || axes[1].Tag != "wdth" {
		t.Errorf("Font(0) axes: got %+v", axes)
	}
	instances, err := f0.VariationInstances()
	if err != nil {
		t.Fatalf("Font(0) VariationInstances: %v", err)
	}
	if len(instances) != 3 {
		t.Errorf("Font(0) instances: got %d, want 3", len(instances))
	}

	// The second face is a static font.
	f1, err := c.Font(1)
	if err != nil {
		t.Fatalf("Font(1): %v", err)
	}
	axes, err = f1.VariationAxes()
	if err != nil {
		t.Fatalf("Font(1) VariationAxes: %v", err)
	}
	if axes != nil {
		t.Errorf("Font(1) axes: got %+v, want nil", axes)
	}
}

func TestFvarSourceUnmodified(t *testing.T) {
	// Parsing and querying must not modify the source data.
	orig := readFvarTestFile(t, "fvarTest.ttf")
	data := append([]byte(nil), orig...)
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if _, err := f.VariationAxes(); err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	if _, err := f.VariationInstances(); err != nil {
		t.Fatalf("VariationInstances: %v", err)
	}
	if !bytes.Equal(data, orig) {
		t.Errorf("source data was modified")
	}

	// The same holds when the source is an io.ReaderAt.
	f, err = ParseReaderAt(bytes.NewReader(orig))
	if err != nil {
		t.Fatalf("ParseReaderAt: %v", err)
	}
	axes, err := f.VariationAxes()
	if err != nil {
		t.Fatalf("VariationAxes: %v", err)
	}
	if len(axes) != 2 {
		t.Errorf("got %d axes, want 2", len(axes))
	}
	instances, err := f.VariationInstances()
	if err != nil {
		t.Fatalf("VariationInstances: %v", err)
	}
	if len(instances) != 3 {
		t.Errorf("got %d instances, want 3", len(instances))
	}
}
