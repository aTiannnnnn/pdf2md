# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install marker-pdf          # default backend
pip install pymupdf4llm         # optional fallback
pip install nougat-ocr pix2tex  # optional OCR backends
```

## Running the CLI

```bash
pdf2md "paper.pdf"                          # default marker backend
pdf2md "paper.pdf" --backend latexocr       # hybrid equation OCR
pdf2md "paper.pdf" --backend pdftotext      # offline fallback
pdf2md "paper.pdf" -o output.md --force

pdf2md-review "paper.pdf" "paper.md"        # review equation quality
```

## Architecture

Two CLI entry points defined in `pyproject.toml`:
- `pdf2md` → `pdf2md_app.cli:main`
- `pdf2md-review` → `pdf2md_app.review_cli:main`

**Conversion flow** (`converter.py`): `convert_pdf()` routes to one of five backends, then applies a shared post-processing pipeline to `pdftotext`/`latexocr` output:
1. Unicode normalization
2. Header/footer removal (frequency-based heuristic)
3. Hyphenation fixing
4. Inline heading splitting
5. `_lines_to_markdown()` — structure inference with TOC generation

**Backends:**
- `marker` / `nougat` / `pymupdf4llm` — delegate to external tools/libs, output markdown directly (post-processing skipped)
- `pdftotext` — offline text extraction, full post-processing pipeline applied
- `latexocr` — hybrid: `pdftohtml` extracts layout, heuristics in `_looks_like_display_equation_line()` identify equations, `sips` (macOS) crops images, `pix2tex` runs LaTeX-OCR; full post-processing applied after

**Review flow** (`reviewer.py`): extracts numbered display equations from both PDF (via `pdftotext`) and markdown (via regex), cross-references them with similarity scoring, and outputs a comparison report.

## Key implementation notes

- `latexocr` depends on macOS `sips` for image cropping — not portable
- External binaries required: `marker_single`, `nougat`, `pdftohtml`, `pdftotext`
- `_looks_like_display_equation_line()` in `converter.py` is the core heuristic for equation detection — symbol/letter ratio, operator presence, equation reference patterns
- `_is_reasonable_latex()` validates pix2tex output before including it
- `ConversionError` is the user-facing exception type
