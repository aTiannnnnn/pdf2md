from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .converter import ConversionError, ConvertOptions, convert_pdf
from .html_bundle import bundle_html


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert research-paper PDFs to Markdown (or HTML).",
    )
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: same name with .md or .html suffix)",
    )
    parser.add_argument(
        "--backend",
        default="marker",
        choices=["marker", "nougat", "latexocr", "pymupdf4llm", "pdftotext"],
        help="Conversion backend (default: marker)",
    )
    parser.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "html"],
        help="Output format (default: markdown)",
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
    if args.output:
        output = Path(args.output)
    else:
        suffix = ".html" if args.format == "html" else ".md"
        output = input_pdf.with_suffix(suffix)

    # Always convert to markdown first (intermediate step for HTML)
    md_output = output.with_suffix(".md") if args.format == "html" else output

    options = ConvertOptions(backend=args.backend, force=args.force)
    try:
        md_path = convert_pdf(input_pdf, md_output, options)
    except ConversionError as exc:
        print(f"[pdf2md] error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # defensive catch for user-facing CLI
        print(f"[pdf2md] unexpected error: {exc}", file=sys.stderr)
        return 3

    if args.format == "html":
        html_path = bundle_html(md_path, output)
        print(f"[pdf2md] HTML written: {html_path}")
    else:
        print(f"[pdf2md] Markdown written: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
