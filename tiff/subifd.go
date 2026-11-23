// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tiff

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"io"
	"sync"
)

// Errors reported when enumerating associated (Sub) IFDs. They can be tested
// for with errors.Is; the wrapping *IFDError carries the offending file offset
// and the chain of page indices that led to it.
var (
	// ErrIFDLoop reports an IFD graph that revisits an offset that is still
	// being walked, i.e. a true cycle.
	ErrIFDLoop = errors.New("tiff: IFD graph contains a loop")

	// ErrIFDDuplicate reports an IFD offset reached by more than one path
	// when it is not part of a cycle. Two pages cannot unambiguously share
	// one IFD, so enumeration rejects this rather than picking a parent.
	// (Pages may still share strip or tile data.)
	ErrIFDDuplicate = errors.New("tiff: IFD offset is reachable more than once")

	// ErrIFDDepth reports nesting deeper than DefaultMaxIFDDepth.
	ErrIFDDepth = errors.New("tiff: IFD nesting is too deep")

	// ErrIFDType reports a SubIFDs tag whose type cannot hold an offset.
	ErrIFDType = errors.New("tiff: invalid SubIFDs tag type")
)

// DefaultMaxIFDDepth is the limit on SubIFD nesting used by EnumerateSubIFDs
// and NewReader. A main image with a thumbnail and a depth map is two levels
// deep; the limit leaves room for unusual files while bounding the work a
// malicious input can cause.
const DefaultMaxIFDDepth = 32

// SubIFD describes one image page reachable from a TIFF file's main image
// through the SubIFDs tag.
//
// Its fields are populated by EnumerateSubIFDs without decoding any pixel
// data. Reader.Decode obtains the pixels lazily, one page at a time.
type SubIFD struct {
	// Parent is the index of the parent page in the slice returned by
	// EnumerateSubIFDs or Reader.Pages. The main image, whose index is 0,
	// has Parent -1.
	Parent int

	// Index is the page's position within its parent's SubIFDs array, in file
	// order. It is -1 for the main image.
	Index int

	// Depth is the nesting level: 0 for the main image, 1 for the main
	// image's direct SubIFDs, and so on.
	Depth int

	// Offset is the file offset of the page's IFD.
	Offset int64

	// Width and Height are the ImageWidth and ImageLength tag values.
	Width, Height int

	// Compression is the Compression tag value, or 0 if the tag is absent
	// (which this package treats as uncompressed).
	Compression uint

	// BitsPerSample, SamplesPerPixel and PhotometricInterpretation are the
	// corresponding tag values, or 0 if absent.
	BitsPerSample             uint
	SamplesPerPixel           uint
	PhotometricInterpretation uint

	// NewSubfileType is the NewSubfileType tag, whose bits distinguish a
	// reduced-resolution thumbnail (0x1) and a transparency mask (0x4).
	NewSubfileType uint

	// Description is the ImageDescription tag, if any.
	Description string
}

// ReducedResolution reports whether the page is marked as reduced-resolution
// (NewSubfileType bit 0), as camera thumbnails usually are.
func (p SubIFD) ReducedResolution() bool { return p.NewSubfileType&0x1 != 0 }

// IsMask reports whether the page is marked as a transparency mask
// (NewSubfileType bit 2), as some depth or alpha planes are.
func (p SubIFD) IsMask() bool { return p.NewSubfileType&0x4 != 0 }

// IFDError reports an error found while walking the IFD tree. It wraps the
// underlying cause (for example ErrIFDLoop) and records where in the tree the
// problem occurred.
type IFDError struct {
	// Offset is the file offset of the IFD at which the problem occurred.
	Offset int64
	// Path is the chain of SubIFD array indices from the main image to the
	// offending page. A nil Path refers to the main image.
	Path []int
	// Err is the underlying cause.
	Err error
}

func (e *IFDError) Error() string {
	if len(e.Path) == 0 {
		return fmt.Sprintf("%v (at IFD offset %d)", e.Err, e.Offset)
	}
	return fmt.Sprintf("%v (at IFD offset %d, SubIFD path %v)", e.Err, e.Offset, e.Path)
}

// Unwrap returns the underlying cause.
func (e *IFDError) Unwrap() error { return e.Err }

// EnumerateSubIFDs parses a TIFF file's IFD tree without decoding pixel data.
// The returned slice's element 0 is always the main image (the IFD pointed to
// by the file header), followed by every associated IFD reachable through
// SubIFDs, in a stable depth-first order that follows file order within each
// SubIFDs array. SubIFD.Parent, SubIFD.Index and SubIFD.Depth describe the
// position in that tree.
//
// To decode the pixels of a page as well, use a Reader obtained from NewReader:
// its page indices match the returned slice. See ExampleReader_Decode.
//
// Enumeration reads only IFD metadata (and the tag arrays those offsets point
// to). A malformed or unreadable IFD is reported as an error and no pages are
// returned, so a partial result is never mistaken for a complete tree. Errors
// are of type *IFDError and wrap ErrIFDLoop, ErrIFDDuplicate, ErrIFDDepth or
// ErrIFDType where those apply.
//
// The main image's IFD is parsed exactly as Decode would; the SubIFDs tag is
// validated as well, which Decode itself does not need to do.
func EnumerateSubIFDs(r io.Reader) ([]SubIFD, error) {
	return enumerateSubIFDs(r, DefaultMaxIFDDepth)
}

func enumerateSubIFDs(r io.Reader, maxDepth int) ([]SubIFD, error) {
	ra, byteOrder, ifdOffset, err := readHeader(r)
	if err != nil {
		return nil, err
	}
	w, err := walkIFDs(ra, byteOrder, ifdOffset, maxDepth)
	if err != nil {
		return nil, err
	}
	return w.subs, nil
}

// subIFDReader holds the state of an IFD tree walk.
type subIFDReader struct {
	r         io.ReaderAt
	byteOrder binary.ByteOrder
	maxDepth  int

	pages []*ifdInfo
	subs  []SubIFD

	// visited records offsets that are being walked (inProgress) or that have
	// been fully walked (done). Re-entering an in-progress offset is a cycle;
	// reaching a done offset from a second path is a duplicate. The two are
	// distinguished so that each can be reported precisely.
	visited map[int64]ifdVisitState
}

type ifdVisitState int

const (
	inProgress ifdVisitState = iota + 1
	done
)

// walkIFDs parses the IFD tree rooted at ifdOffset.
func walkIFDs(ra io.ReaderAt, byteOrder binary.ByteOrder, ifdOffset int64, maxDepth int) (*subIFDReader, error) {
	if maxDepth < 0 {
		maxDepth = 0
	}
	w := &subIFDReader{
		r:         ra,
		byteOrder: byteOrder,
		maxDepth:  maxDepth,
		visited:   make(map[int64]ifdVisitState),
	}

	main, err := w.read(ifdOffset, 0, nil)
	if err != nil {
		return nil, err
	}
	w.add(main, -1, -1, 0)
	if err := w.walk(main, 0, 0, nil); err != nil {
		return nil, err
	}
	return w, nil
}

// add records page info as the next page of the tree.
func (w *subIFDReader) add(info *ifdInfo, parent, index, depth int) int {
	n := len(w.subs)
	w.pages = append(w.pages, info)
	w.subs = append(w.subs, SubIFD{
		Parent:                    parent,
		Index:                     index,
		Depth:                     depth,
		Offset:                    info.offset,
		Width:                     int(info.firstVal(tImageWidth)),
		Height:                    int(info.firstVal(tImageLength)),
		Compression:               info.firstVal(tCompression),
		BitsPerSample:             info.firstVal(tBitsPerSample),
		SamplesPerPixel:           info.firstVal(tSamplesPerPixel),
		PhotometricInterpretation: info.firstVal(tPhotometricInterpretation),
		NewSubfileType:            info.firstVal(tNewSubfileType),
		Description:               info.desc,
	})
	return n
}

func (i *ifdInfo) firstVal(tag int) uint {
	if f := i.features[tag]; len(f) > 0 {
		return f[0]
	}
	return 0
}

// read parses the IFD at offset and guards against loops and excessive depth.
// path is the chain of SubIFD array indices leading to this IFD. The caller
// must call finish once the IFD's own SubIFDs have been walked.
func (w *subIFDReader) read(offset int64, depth int, path []int) (*ifdInfo, error) {
	if depth > w.maxDepth {
		return nil, &IFDError{Offset: offset, Path: path, Err: ErrIFDDepth}
	}
	switch w.visited[offset] {
	case inProgress:
		return nil, &IFDError{Offset: offset, Path: path, Err: ErrIFDLoop}
	case done:
		return nil, &IFDError{Offset: offset, Path: path, Err: ErrIFDDuplicate}
	}
	w.visited[offset] = inProgress

	info, err := readIFDAt(w.r, w.byteOrder, offset, true)
	if err != nil {
		return nil, &IFDError{Offset: offset, Path: path, Err: err}
	}
	return info, nil
}

// finish marks an IFD previously returned by read as fully walked.
func (w *subIFDReader) finish(info *ifdInfo) {
	w.visited[info.offset] = done
}

// walk recursively enumerates the SubIFDs of info. parent and depth are the
// page's own position; its children are one level deeper.
func (w *subIFDReader) walk(info *ifdInfo, parent, depth int, path []int) error {
	defer w.finish(info)
	for index, off := range info.subIFDs {
		if off == 0 {
			// A zero offset is a placeholder for an absent page, not an IFD
			// at the file header.
			continue
		}
		childPath := make([]int, len(path)+1)
		copy(childPath, path)
		childPath[len(path)] = index

		child, err := w.read(off, depth+1, childPath)
		if err != nil {
			return err
		}
		childIndex := w.add(child, parent, index, depth+1)
		if err := w.walk(child, childIndex, depth+1, childPath); err != nil {
			return err
		}
	}
	return nil
}

// Reader reads the pages of a TIFF file, including the associated IFDs that
// camera files store in the SubIFDs tag. After construction, its methods may
// be called concurrently.
type Reader struct {
	r         io.ReaderAt
	byteOrder binary.ByteOrder
	pages     []*ifdInfo
	subs      []SubIFD
}

// NewReader returns a Reader for the TIFF file in r. Construction reads the
// file header and enumerates the IFD tree (see EnumerateSubIFDs) but decodes
// no pixel data. If r does not implement io.ReaderAt its bytes are buffered on
// demand, under a lock, as pages are decoded.
func NewReader(r io.Reader) (*Reader, error) {
	ra := newReaderAt(r)
	if _, ok := ra.(*buffer); ok {
		// Our own buffer only supports one reader at a time.
		ra = &lockedReaderAt{r: ra}
	}
	byteOrder, ifdOffset, err := readHeaderFrom(ra)
	if err != nil {
		return nil, err
	}
	return newPageReader(ra, byteOrder, ifdOffset)
}

// NewReaderAt is like NewReader for an existing io.ReaderAt. byteOrder must be
// the file's byte order and ifdOffset the offset of the first IFD. Prefer
// NewReader, which reads both from the file header, unless the caller already
// has them. The ReaderAt must support concurrent ReadAt calls, as the
// io.ReaderAt contract requires.
func NewReaderAt(ra io.ReaderAt, byteOrder binary.ByteOrder, ifdOffset int64) (*Reader, error) {
	if ra == nil {
		return nil, errors.New("tiff: nil ReaderAt")
	}
	if byteOrder == nil {
		return nil, errors.New("tiff: nil ByteOrder")
	}
	return newPageReader(ra, byteOrder, ifdOffset)
}

func newPageReader(ra io.ReaderAt, byteOrder binary.ByteOrder, ifdOffset int64) (*Reader, error) {
	if ifdOffset < 0 {
		return nil, &IFDError{Offset: ifdOffset, Err: FormatError("negative IFD offset")}
	}
	w, err := walkIFDs(ra, byteOrder, ifdOffset, DefaultMaxIFDDepth)
	if err != nil {
		return nil, err
	}
	return &Reader{r: ra, byteOrder: byteOrder, pages: w.pages, subs: w.subs}, nil
}

// lockedReaderAt serializes ReadAt on a non-thread-safe io.ReaderAt.
type lockedReaderAt struct {
	mu sync.Mutex
	r  io.ReaderAt
}

func (l *lockedReaderAt) ReadAt(p []byte, off int64) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.r.ReadAt(p, off)
}

// NumPages returns the number of pages: one main image plus its associated
// IFDs.
func (r *Reader) NumPages() int { return len(r.subs) }

// Page returns a description of page i. The value is a copy; modifying it does
// not affect the Reader.
func (r *Reader) Page(i int) (SubIFD, bool) {
	if i < 0 || i >= len(r.subs) {
		return SubIFD{}, false
	}
	return r.subs[i], true
}

// Pages returns a copy of all page descriptions, in enumeration order.
func (r *Reader) Pages() []SubIFD {
	return append([]SubIFD(nil), r.subs...)
}

// Config returns the image configuration of page i without decoding it.
func (r *Reader) Config(i int) (image.Config, error) {
	if i < 0 || i >= len(r.pages) {
		return image.Config{}, r.indexError(i)
	}
	d, err := newDecoderFromIFD(r.r, r.byteOrder, r.pages[i])
	if err != nil {
		return image.Config{}, r.pageError(i, err)
	}
	return d.config, nil
}

// Decode returns the image for page i. Pixel data is read and decoded at call
// time; a failure leaves the Reader untouched, so other pages remain readable.
// Page 0 is the main image and behaves like package-level Decode.
func (r *Reader) Decode(i int) (image.Image, error) {
	if i < 0 || i >= len(r.pages) {
		return nil, r.indexError(i)
	}
	img, err := decodeIFD(r.r, r.byteOrder, r.pages[i])
	if err != nil {
		return nil, r.pageError(i, err)
	}
	return img, nil
}

func (r *Reader) indexError(i int) error {
	return fmt.Errorf("tiff: page index %d out of range [0, %d)", i, len(r.pages))
}

// pageError wraps err with the page's IFD offset and its position in the tree,
// so that a failure decoding an associated IFD can be located in the file.
func (r *Reader) pageError(i int, err error) error {
	var ifdErr *IFDError
	if errors.As(err, &ifdErr) {
		return err
	}
	return &IFDError{Offset: r.pages[i].offset, Path: r.pagePath(i), Err: err}
}

// pagePath returns the SubIFD array indices leading from the main image to
// page i, or nil for the main image itself. Unlike the page index, this is
// stable with respect to the tree shape.
func (r *Reader) pagePath(i int) []int {
	if i == 0 {
		return nil
	}
	var rev []int
	for i > 0 && i < len(r.subs) {
		rev = append(rev, r.subs[i].Index)
		i = r.subs[i].Parent
	}
	// rev is leaf-to-root; reverse it.
	path := make([]int, len(rev))
	for j, v := range rev {
		path[len(rev)-1-j] = v
	}
	return path
}
