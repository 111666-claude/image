// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfnt

import (
	"fmt"
)

// VariationAxisFlagHidden indicates that a variation axis should not be
// exposed directly to users in application UIs.
//
// See the "VariationAxisRecord" section of
// https://www.microsoft.com/typography/otspec/fvar.htm
const VariationAxisFlagHidden uint16 = 0x0001

// VariationAxis describes a single design axis of a variable font, as
// declared by the font's fvar table.
type VariationAxis struct {
	// Tag is the 4-byte axis tag, such as "wght" or "wdth".
	Tag string
	// Minimum, Default and Maximum are the axis's value range and default
	// value, converted from the fvar table's 16.16 fixed-point
	// representation. Minimum <= Default <= Maximum holds for every axis
	// returned without an error.
	Minimum float64
	Default float64
	Maximum float64
	// Flags are the axis's flags, such as VariationAxisFlagHidden.
	Flags uint16
	// NameID is the name table ID of the axis's localized name. It can be
	// resolved to a string with the Font.Name method.
	NameID NameID
}

// VariationInstance describes a named instance of a variable font: a
// pre-defined combination of axis coordinates, as declared by the font's
// fvar table.
type VariationInstance struct {
	// NameID is the name table ID of the instance's localized subfamily
	// (style) name. It can be resolved to a string with the Font.Name
	// method.
	NameID NameID
	// Flags are the instance's flags. No flags are currently defined by the
	// specification.
	Flags uint16
	// Coordinates holds the instance's axis coordinates, one per variation
	// axis, in the same order as the Font.VariationAxes result.
	Coordinates []float64
	// PostScriptNameID is the name table ID of the instance's PostScript
	// name. It is only valid if HasPostScriptName is true.
	PostScriptNameID NameID
	// HasPostScriptName reports whether the instance record contains a
	// PostScript name ID.
	HasPostScriptName bool
}

// VariationAxes returns the variation axes declared by f's fvar table, in
// table order.
//
// If f has no fvar table (i.e. it is not a variable font), it returns nil,
// nil. If f has a malformed fvar table, it returns a non-nil error; f can
// still be used as a fixed font in that case.
//
// The returned slice is a copy; the caller may modify it.
func (f *Font) VariationAxes() ([]VariationAxis, error) {
	if f.cached.variationErr != nil {
		return nil, f.cached.variationErr
	}
	return append([]VariationAxis(nil), f.cached.variationAxes...), nil
}

// VariationInstances returns the named instances declared by f's fvar table,
// in table order.
//
// If f has no fvar table (i.e. it is not a variable font), it returns nil,
// nil. If f has a malformed fvar table, it returns a non-nil error; f can
// still be used as a fixed font in that case.
//
// The returned slice is a deep copy; the caller may modify it.
func (f *Font) VariationInstances() ([]VariationInstance, error) {
	if f.cached.variationErr != nil {
		return nil, f.cached.variationErr
	}
	if len(f.cached.variationInstances) == 0 {
		return nil, nil
	}
	instances := make([]VariationInstance, len(f.cached.variationInstances))
	for i, in := range f.cached.variationInstances {
		instances[i] = in
		instances[i].Coordinates = append([]float64(nil), in.Coordinates...)
	}
	return instances, nil
}

// fixed1616ToFloat64 converts a 16.16 fixed-point number, as used by the
// fvar table, to a float64. The conversion is exact: every int32 maps to a
// finite float64.
func fixed1616ToFloat64(x uint32) float64 {
	return float64(int32(x)) / 65536
}

func (f *Font) parseFvar(buf []byte) (buf1 []byte, axes []VariationAxis, instances []VariationInstance, err error) {
	// https://www.microsoft.com/typography/otspec/fvar.htm

	if f.fvar.length == 0 {
		// Not a variable font.
		return buf, nil, nil, nil
	}
	const headerSize = 16
	if f.fvar.length < headerSize {
		return nil, nil, nil, errInvalidFvarTable
	}
	buf, err = f.src.view(buf, int(f.fvar.offset), headerSize)
	if err != nil {
		return nil, nil, nil, err
	}
	if major, minor := u16(buf[0:]), u16(buf[2:]); major != 1 {
		return nil, nil, nil, fmt.Errorf("%w: version %d.%d", errUnsupportedFvarTable, major, minor)
	}
	axesOffset := uint32(u16(buf[4:]))
	axisCount := int(u16(buf[8:]))
	axisSize := uint32(u16(buf[10:]))
	instanceCount := int(u16(buf[12:]))
	instanceSize := uint32(u16(buf[14:]))

	if axisCount > maxNumVariationAxes {
		return nil, nil, nil, errUnsupportedNumberOfVariationAxes
	}
	if instanceCount > maxNumVariationInstances {
		return nil, nil, nil, errUnsupportedNumberOfVariationInstances
	}

	const axisRecordSize = 20
	if axisCount > 0 {
		if axisSize < axisRecordSize {
			return nil, nil, nil, errInvalidFvarTable
		}
		if axesOffset < headerSize || axesOffset+uint32(axisCount)*axisSize > f.fvar.length {
			return nil, nil, nil, errInvalidFvarTable
		}
	}
	instancesOffset := axesOffset + uint32(axisCount)*axisSize
	if instanceCount > 0 {
		if instanceSize < uint32(4+4*axisCount) {
			return nil, nil, nil, errInvalidFvarTable
		}
		if instancesOffset+uint32(instanceCount)*instanceSize > f.fvar.length {
			return nil, nil, nil, errInvalidFvarTable
		}
	}

	if axisCount > 0 {
		axes = make([]VariationAxis, axisCount)
		buf, err = f.src.view(buf, int(f.fvar.offset+axesOffset), int(uint32(axisCount)*axisSize))
		if err != nil {
			return nil, nil, nil, err
		}
		seen := make(map[string]struct{}, axisCount)
		for i := range axes {
			b := buf[int(uint32(i)*axisSize):]
			tag := string(b[:4])
			if _, ok := seen[tag]; ok {
				return nil, nil, nil, fmt.Errorf("%w: duplicate axis tag %q", errInvalidFvarTable, tag)
			}
			seen[tag] = struct{}{}
			minimum := fixed1616ToFloat64(u32(b[4:]))
			def := fixed1616ToFloat64(u32(b[8:]))
			maximum := fixed1616ToFloat64(u32(b[12:]))
			if minimum > def || def > maximum {
				return nil, nil, nil, fmt.Errorf("%w: axis %q has min %v, default %v, max %v",
					errInvalidFvarTable, tag, minimum, def, maximum)
			}
			axes[i] = VariationAxis{
				Tag:     tag,
				Minimum: minimum,
				Default: def,
				Maximum: maximum,
				Flags:   u16(b[16:]),
				NameID:  NameID(u16(b[18:])),
			}
		}
	}

	if instanceCount > 0 {
		instances = make([]VariationInstance, instanceCount)
		buf, err = f.src.view(buf, int(f.fvar.offset+instancesOffset), int(uint32(instanceCount)*instanceSize))
		if err != nil {
			return nil, nil, nil, err
		}
		hasPostScriptName := instanceSize >= uint32(4+4*axisCount+2)
		for i := range instances {
			b := buf[int(uint32(i)*instanceSize):]
			in := VariationInstance{
				NameID:      NameID(u16(b[0:])),
				Flags:       u16(b[2:]),
				Coordinates: make([]float64, axisCount),
			}
			for j := range in.Coordinates {
				in.Coordinates[j] = fixed1616ToFloat64(u32(b[4+4*j:]))
			}
			if hasPostScriptName {
				// The specification says that 0xFFFF means "no PostScript
				// name", even when the instance record includes the field.
				if id := NameID(u16(b[4+4*axisCount:])); id != 0xffff {
					in.PostScriptNameID = id
					in.HasPostScriptName = true
				}
			}
			instances[i] = in
		}
	}

	return buf, axes, instances, nil
}
