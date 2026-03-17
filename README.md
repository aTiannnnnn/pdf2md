# PDF to Markdown (Research-Paper Focus)

A CLI tool to convert PDF files into Markdown, optimized for research papers.

## Fidelity expectations

No open-source tool guarantees perfect 1:1 conversion for all PDFs. For papers with equations, tables, and figures, the best practical option is usually `marker` (`marker-pdf` package). This app defaults to that backend.
For OCR-heavy papers (especially scanned PDFs), `nougat` is often better than plain text extraction and can output LaTeX-style equations in Markdown.
You can also use a hybrid `latexocr` backend that detects likely equation lines and runs `pix2tex` (LaTeX-OCR) on those regions.

## Install

```bash
git clone https://github.com/aTiannnnnn/pdf2md.git
cd pdf2md
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install marker-pdf
```

Optional fallback backend:

```bash
pip install pymupdf4llm
```

Optional OCR backend for equations:

```bash
pip install nougat-ocr
pip install pix2tex
```

## Usage

Basic:

```bash
pdf2md "paper.pdf"
```

Specify output path:

```bash
pdf2md "paper.pdf" -o "paper.md"
```

Choose backend:

```bash
pdf2md "paper.pdf" --backend marker
pdf2md "paper.pdf" --backend nougat
pdf2md "paper.pdf" --backend latexocr
pdf2md "paper.pdf" --backend pymupdf4llm
pdf2md "paper.pdf" --backend pdftotext
```

Review display equations (equation-number based) against source PDF:

```bash
pdf2md-review "paper.pdf" "paper.md"
pdf2md-review "paper.pdf" "paper.md" -o "equation_review.md"
```

Overwrite existing output:

```bash
pdf2md "paper.pdf" --force
```

## Notes for better quality

- Born-digital PDFs convert better than scanned PDFs.
- If a PDF is scanned, add an OCR step before conversion.
- Review and patch equation blocks after conversion for publication-grade output.
- `pdftotext` backend is an offline fallback and lower fidelity than `marker`, but this app post-processes it into one-column Markdown with section heading and TOC heuristics.
- For equation quality, prefer `--backend marker` first, then try `--backend nougat` for OCR-based reconstruction.
- `latexocr` prioritizes display equations with equation numbers (e.g., `(1)`, `(2)`) and inserts `\tag{n}` in generated LaTeX blocks.
- `latexocr` depends on `pdftohtml` + `pix2tex`; inline equations are intentionally not the primary target in this mode.

## Acknowledgements

This project relies on several excellent open-source tools and libraries:

- **[Marker](https://github.com/VikParuchuri/marker)** — High-fidelity PDF-to-Markdown converter (default backend)
- **[Nougat](https://github.com/facebookresearch/nougat)** — Neural OCR for academic documents (Meta Research)
- **[PyMuPDF4LLM](https://github.com/pymupdf/PyMuPDF)** — PyMuPDF-based Markdown extraction
- **[pix2tex (LaTeX-OCR)](https://github.com/lukas-blecher/LaTeX-OCR)** — Image-to-LaTeX equation recognition
- **[pdftotext / pdftohtml](https://poppler.freedesktop.org/)** — Poppler PDF utilities for text and layout extraction
