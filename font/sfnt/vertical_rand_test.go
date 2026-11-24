package sfnt

import (
	"math/rand"
	"testing"
)

// TestVerticalRandomCorruption hammers the vertical metrics code with random
// corruption of an otherwise valid font, checking that it never panics.
func TestVerticalRandomCorruption(t *testing.T) {
	t.Parallel()
	base := readVerticalTestFile(t, "verticalTest.ttf")
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 30000; iter++ {
		data := append([]byte(nil), base...)
		for n := rng.Intn(5) + 1; n > 0; n-- {
			data[rng.Intn(len(data))] = byte(rng.Intn(256))
		}
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("iter %d: panic: %v", iter, r)
				}
			}()
			f, err := Parse(data)
			if err != nil {
				return
			}
			_ = f.HasVerticalMetrics()
			f.VerticalMetrics(nil, 1000, 0)
			f.VerticalMetrics(nil, 37, 1)
			for x := GlyphIndex(0); int(x) < f.NumGlyphs() && int(x) < 64; x++ {
				f.GlyphVerticalAdvance(nil, x, 1000, 0)
				f.GlyphVerticalOrigin(nil, x, 1000, 0)
				f.GlyphVerticalBounds(nil, x, 1000, 0)
				f.GlyphVerticalAdvance(nil, x, 7, 1)
				f.GlyphVerticalBounds(nil, x, 7, 1)
			}
		}()
	}
}
