from __future__ import annotations

import argparse
import sys

from .reviewer import ReviewError, review_display_equations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md-review",
        description="Review display equations generated in Markdown against a source PDF.",
    )
    parser.add_argument("input_pdf", help="Source PDF path")
    parser.add_argument("markdown_file", help="Generated Markdown path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output report path (default: <markdown>_display_eq_review.md)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = review_display_equations(args.input_pdf, args.markdown_file, args.output)
    except ReviewError as exc:
        print(f"[pdf2md-review] error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[pdf2md-review] unexpected error: {exc}", file=sys.stderr)
        return 3

    print(f"[pdf2md-review] report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
