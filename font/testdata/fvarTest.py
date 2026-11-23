#!/usr/bin/env python3
"""Generates fvarTest.ttf and fvarTest.ttc fixtures for the
golang.org/x/image/font/sfnt package's fvar (variable font) tests.

fvarTest.ttf is a minimal TrueType variable font with two axes (wght and
wdth, the latter hidden) and three named instances, one of which ("Regular")
sits at the default coordinates and one of which ("Bold") carries a
postScriptNameID.

fvarTest.ttc is a TrueType collection whose first face is that variable font
and whose second face is a static font without an fvar table.
"""

import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection

UPM = 1000


def build_font(family, variable):
    fb = FontBuilder(UPM, isTTF=True)
    glyph_order = [".notdef", "A", "space"]
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({0x20: "space", 0x41: "A"})

    pen = TTGlyphPen(None)
    notdef = pen.glyph()

    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((300, 700))
    pen.lineTo((550, 0))
    pen.closePath()
    glyph_a = pen.glyph()

    pen = TTGlyphPen(None)
    glyph_space = pen.glyph()

    glyphs = {".notdef": notdef, "A": glyph_a, "space": glyph_space}
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({g: (600, 0) for g in glyph_order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        usWinAscent=800,
        usWinDescent=200,
        sxHeight=500,
        sCapHeight=700,
    )
    fb.setupNameTable(
        {
            "familyName": family,
            "styleName": "Regular",
            "uniqueFontIdentifier": family + "-test",
            "fullName": family + " Regular",
            "psName": family + "-Regular",
        }
    )
    fb.setupPost()
    fb.setupMaxp()

    if variable:
        fb.setupFvar(
            [
                ("wght", 100.0, 400.0, 900.0, "Weight"),
                ("wdth", 50.0, 100.0, 200.0, "Width"),
            ],
            [
                {
                    "location": {"wght": 400.0, "wdth": 100.0},
                    "stylename": "Regular",
                },
                {
                    "location": {"wght": 700.0, "wdth": 100.0},
                    "stylename": "Bold",
                    "postscriptfontname": family + "-Bold",
                },
                {
                    "location": {"wght": 700.0, "wdth": 75.5},
                    "stylename": "Condensed Bold",
                },
            ],
        )
        # Mark the wdth axis as hidden (AXIS_FLAG_HIDDEN = 0x0001).
        fb.font["fvar"].axes[1].flags = 0x0001

    return fb.font


def main():
    out_dir = sys.argv[1]

    variable = build_font("FvarTest", variable=True)
    variable.save(out_dir + "/fvarTest.ttf")

    static = build_font("FvarStaticTest", variable=False)

    ttc = TTCollection()
    # Re-load the saved variable font so the TTC contains exactly the bytes
    # that fvarTest.ttf round-trips.
    ttc.fonts = [variable, static]
    ttc.save(out_dir + "/fvarTest.ttc")

    print("wrote", out_dir + "/fvarTest.ttf")
    print("wrote", out_dir + "/fvarTest.ttc")


if __name__ == "__main__":
    main()
