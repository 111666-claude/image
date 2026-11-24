#!/usr/bin/env python3
"""Generates the vertical metrics fixtures for the
golang.org/x/image/font/sfnt package's vhea/vmtx tests.

verticalTest.ttf is a minimal TrueType font with a 'vhea' table (version 1.1)
and a 'vmtx' table. It has eight glyphs, of which the first four (including
'.notdef' and 'A') have full (advance height, top side bearing) records and the
remaining four are described only by the trailing top side bearing array. Two
of the top side bearings are negative, and glyph 7 is unmapped by the cmap so
that tests can address it by index.

verticalTestNoVhea.ttf is the same font without either table.

verticalTest.ttc is a collection whose first face is verticalTest.ttf and whose
second face is verticalTestNoVhea.ttf.

verticalTestCFF.otf is a CFF-flavored font with vertical metrics, exercising
the PostScript outline path.

Tests that need malformed tables (a wrong table length, an out-of-range
numOfLongVerMetrics, and so on) patch the bytes of verticalTest.ttf directly
rather than carrying more fixture files.

Run this script with the output directory as its only argument.
"""

import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection

UPM = 1000

GLYPH_ORDER = [
    ".notdef", "space", "A", "uni4E00", "uni4E2D", "uni3042", "uni3001", "glyph7",
]
CMAP = {
    0x20: "space",
    0x41: "A",
    0x4E00: "uni4E00",
    0x4E2D: "uni4E2D",
    0x3042: "uni3042",
    0x3001: "uni3001",
}
# (xMin, yMin, xMax, yMax) or None for an empty glyph.
BOXES = {
    ".notdef": (50, 0, 550, 700),
    "space": None,
    "A": (0, 0, 600, 700),
    "uni4E00": (100, 350, 900, 450),
    "uni4E2D": (100, -100, 900, 800),
    "uni3042": (90, -80, 910, 820),
    "uni3001": (400, -120, 600, 100),
    "glyph7": (100, 0, 900, 600),
}
H_METRICS = {
    ".notdef": (1000, 100), "space": (500, 0), "A": (600, 0),
    "uni4E00": (1000, 100), "uni4E2D": (1000, 100), "uni3042": (1000, 90),
    "uni3001": (1000, 400), "glyph7": (800, 100),
}
# The first four are full metrics; the last four are in the trailing array.
V_METRICS = {
    ".notdef": (1100, 880),
    "space": (1000, 880),
    "A": (900, -100),
    "uni4E00": (1000, 550),
    "uni4E2D": (1000, 200),
    "uni3042": (1000, 180),
    "uni3001": (1000, -50),
    "glyph7": (1000, 400),
}
NUM_V_METRICS = 4


def tt_glyph(box):
    pen = TTGlyphPen(None)
    if box is not None:
        x0, y0, x1, y1 = box
        pen.moveTo((x0, y0))
        pen.lineTo((x1, y0))
        pen.lineTo((x1, y1))
        pen.lineTo((x0, y1))
        pen.closePath()
    return pen.glyph()


def t2_charstring(box):
    pen = T2CharStringPen(500, None)
    if box is not None:
        x0, y0, x1, y1 = box
        pen.moveTo((x0, y0))
        pen.lineTo((x1, y0))
        pen.lineTo((x1, y1))
        pen.lineTo((x0, y1))
        pen.closePath()
    return pen.getCharString()


def common_tables(fb):
    fb.setupHorizontalMetrics(H_METRICS)
    fb.setupHorizontalHeader(
        ascent=880, descent=-120, lineGap=0, caretSlopeRise=1, caretSlopeRun=0
    )
    fb.setupOS2(
        sTypoAscender=800, sTypoDescender=-200,
        usWinAscent=880, usWinDescent=120, sxHeight=500, sCapHeight=700,
    )
    fb.setupPost()
    fb.setupMaxp()


def setup_names(fb, family):
    fb.setupNameTable({
        "familyName": family,
        "styleName": "Regular",
        "uniqueFontIdentifier": family + "-test",
        "fullName": family + " Regular",
        "psName": family + "-Regular",
    })


def setup_vertical(fb):
    fb.setupVerticalHeader(
        ascent=500, descent=-500, lineGap=50,
        caretSlopeRise=0, caretSlopeRun=1,
    )
    # fontTools' compiler emits long (advance height, top side bearing)
    # records only for the run of glyphs whose advances differ from the last
    # glyph's; the trailing advances are stored once and the remaining top
    # side bearings form the second array. V_METRICS is chosen so that this
    # split lands at NUM_V_METRICS.
    fb.setupVerticalMetrics(V_METRICS)


def build_ttf(family, vertical):
    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(GLYPH_ORDER)
    fb.setupCharacterMap(CMAP)
    fb.setupGlyf({g: tt_glyph(BOXES[g]) for g in GLYPH_ORDER})
    common_tables(fb)
    setup_names(fb, family)
    if vertical:
        setup_vertical(fb)
    return fb.font


def build_otf(family):
    fb = FontBuilder(UPM, isTTF=False)
    fb.setupGlyphOrder(GLYPH_ORDER)
    fb.setupCharacterMap(CMAP)
    charstrings = {g: t2_charstring(BOXES[g]) for g in GLYPH_ORDER}
    fb.setupCFF(
        "VertTestCFF-Regular",
        {"FullName": "VertTestCFF Regular", "FamilyName": "VertTestCFF"},
        charstrings,
        {},
    )
    common_tables(fb)
    setup_names(fb, family)
    setup_vertical(fb)
    return fb.font


def main():
    out_dir = sys.argv[1]

    vertical = build_ttf("VertTest", vertical=True)
    vertical.save(out_dir + "/verticalTest.ttf")
    print("wrote", out_dir + "/verticalTest.ttf")

    no_vhea = build_ttf("VertTestNoVhea", vertical=False)
    no_vhea.save(out_dir + "/verticalTestNoVhea.ttf")
    print("wrote", out_dir + "/verticalTestNoVhea.ttf")

    ttc = TTCollection()
    ttc.fonts = [vertical, no_vhea]
    ttc.save(out_dir + "/verticalTest.ttc")
    print("wrote", out_dir + "/verticalTest.ttc")

    otf = build_otf("VertTestCFF")
    otf.save(out_dir + "/verticalTestCFF.otf")
    print("wrote", out_dir + "/verticalTestCFF.otf")


if __name__ == "__main__":
    main()
