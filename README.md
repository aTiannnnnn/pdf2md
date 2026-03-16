# PDF to Markdown (Research-Paper Focus)

This project provides a CLI app to convert PDF files into Markdown, optimized for research papers.

## Fidelity expectations

No open-source tool guarantees perfect 1:1 conversion for all PDFs. For papers with equations, tables, and figures, the best practical option is usually `marker` (`marker-pdf` package). This app defaults to that backend.
For OCR-heavy papers (especially scanned PDFs), `nougat` is often better than plain text extraction and can output LaTeX-style equations in Markdown.
You can also use a hybrid `latexocr` backend that detects likely equation lines and runs `pix2tex` (LaTeX-OCR) on those regions.

## Install

```bash
cd /Users/tian/Work/Funnnn/pdf_2_md
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
pdf2md "/path/to/paper.pdf"
```

Specify output path:

```bash
pdf2md "/path/to/paper.pdf" -o "/path/to/paper.md"
```

Choose backend:

```bash
pdf2md "/path/to/paper.pdf" --backend marker
pdf2md "/path/to/paper.pdf" --backend nougat
pdf2md "/path/to/paper.pdf" --backend latexocr
pdf2md "/path/to/paper.pdf" --backend pymupdf4llm
pdf2md "/path/to/paper.pdf" --backend pdftotext
```

Review display equations (equation-number based) against source PDF:

```bash
pdf2md-review "/path/to/paper.pdf" "/path/to/paper.md"
pdf2md-review "/path/to/paper.pdf" "/path/to/paper.md" -o "/path/to/equation_review.md"
```

Overwrite existing output:

```bash
pdf2md "/path/to/paper.pdf" --force
```

## Example (from this repo)

```bash
pdf2md "example/2019_Batson_Royer_Noise2Self Blind Denoising by Self-Supervision_2019_Batson_Royer_Noise2Self Blind Denoising by Self-Supervision_2019_Batson_Royer_Noise2Self Blind Denoising by Self-Supervision.pdf" --force
```

## Notes for better quality

- Born-digital PDFs convert better than scanned PDFs.
- If a PDF is scanned, add an OCR step before conversion.
- Review and patch equation blocks after conversion for publication-grade output.
- `pdftotext` backend is an offline fallback and lower fidelity than `marker`, but this app post-processes it into one-column Markdown with section heading and TOC heuristics.
- For equation quality, prefer `--backend marker` first, then try `--backend nougat` for OCR-based reconstruction.
- `latexocr` now prioritizes display equations with equation numbers (e.g., `(1)`, `(2)`) and inserts `\tag{n}` in generated LaTeX blocks.
- `latexocr` depends on `pdftohtml` + `pix2tex`; inline equations are intentionally not the primary target in this mode.
