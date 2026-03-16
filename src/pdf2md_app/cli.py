from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .converter import ConversionError, ConvertOptions, convert_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert research-paper PDFs to Markdown.",
    )
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument(
        "-o",
        "--output",
        help="Output Markdown path (default: same name next to input)",
    )
    parser.add_argument(
        "--backend",
        default="marker",
        choices=["marker", "nougat", "latexocr", "pymupdf4llm", "pdftotext"],
        help="Conversion backend (default: marker)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_pdf = Path(args.input_pdf)
    output = Path(args.output) if args.output else input_pdf.with_suffix(".md")

    options = ConvertOptions(backend=args.backend, force=args.force)
    try:
        out_path = convert_pdf(input_pdf, output, options)
    except ConversionError as exc:
        print(f"[pdf2md] error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # defensive catch for user-facing CLI
        print(f"[pdf2md] unexpected error: {exc}", file=sys.stderr)
        return 3

    print(f"[pdf2md] markdown written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
