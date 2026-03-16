from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class ReviewError(RuntimeError):
    pass


@dataclass
class EquationRow:
    ref: str
    status: str
    similarity: float | None
    pdf_text: str
    md_text: str


def _find_command(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    user_bin = (
        Path.home()
        / "Library"
        / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "bin"
        / name
    )
    if user_bin.exists():
        return str(user_bin)
    return None


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _extract_equation_ref(text: str) -> str | None:
    tag = re.search(r"\\+tag\{(\d{1,4})\}", text)
    if tag:
        return tag.group(1)
    m = re.search(r"\((\d{1,4})\)\s*$", text)
    if m:
        return m.group(1)
    return None


def _strip_equation_ref(text: str) -> str:
    text = re.sub(r"\\+tag\{\d{1,4}\}", "", text)
    text = re.sub(r"\s*,?\s*\(\d{1,4}\)\s*$", "", text)
    return _normalize_line(text)


def _looks_like_display_equation_text(line: str) -> bool:
    if not line:
        return False
    if _extract_equation_ref(line) is None:
        return False
    if len(line) < 8 or len(line) > 240:
        return False
    lower = line.lower()
    if lower.startswith(("figure ", "table ", "algorithm ")):
        return False

    core = _strip_equation_ref(line)
    token_count = len(core.split())
    if token_count > 45:
        return False

    has_operator = any(op in core for op in ("=", "≈", "≤", "≥", "∈", "∑", "∫", "->", "=>"))
    symbol_count = sum((not ch.isalnum() and not ch.isspace()) for ch in core)
    symbol_ratio = symbol_count / max(1, len(core))
    return has_operator and symbol_ratio >= 0.05


def _extract_display_equations_from_pdf(pdf_path: Path) -> dict[str, str]:
    pdftotext_bin = _find_command("pdftotext")
    if pdftotext_bin is None:
        raise ReviewError("Review needs `pdftotext` installed.")

    proc = subprocess.run(
        [pdftotext_bin, "-raw", "-nopgbrk", str(pdf_path), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ReviewError(
            "Failed to extract PDF text for review.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )

    text = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    equations: dict[str, str] = {}
    for raw in text.split("\n"):
        line = _normalize_line(raw)
        if not _looks_like_display_equation_text(line):
            continue
        ref = _extract_equation_ref(line)
        if ref is None:
            continue
        body = _strip_equation_ref(line)
        if not body:
            continue
        # Keep the longest candidate for a given ref.
        if ref not in equations or len(body) > len(equations[ref]):
            equations[ref] = body
    return equations


def _extract_display_equations_from_markdown(md_path: Path) -> dict[str, str]:
    content = md_path.read_text(encoding="utf-8", errors="replace")
    equations: dict[str, str] = {}
    for m in re.finditer(r"\$\$(.*?)\$\$", content, flags=re.DOTALL):
        body_raw = _normalize_line(m.group(1))
        if not body_raw:
            continue
        ref = _extract_equation_ref(body_raw)
        if ref is None:
            continue
        body = _strip_equation_ref(body_raw)
        if not body:
            continue
        if ref not in equations or len(body) > len(equations[ref]):
            equations[ref] = body
    return equations


def _norm_compare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-z]+", " ", text)
    text = re.sub(r"[^a-z0-9=+\-*/^_]", "", text)
    return text


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_compare(a), _norm_compare(b)).ratio()


def _sort_ref_key(ref: str) -> tuple[int, int | str]:
    if ref.isdigit():
        return (0, int(ref))
    return (1, ref)


def _truncate(text: str, max_len: int = 80) -> str:
    text = text.replace("|", "\\|")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _build_rows(pdf_map: dict[str, str], md_map: dict[str, str]) -> list[EquationRow]:
    refs = sorted(set(pdf_map) | set(md_map), key=_sort_ref_key)
    rows: list[EquationRow] = []
    for ref in refs:
        pdf_text = pdf_map.get(ref, "")
        md_text = md_map.get(ref, "")
        if pdf_text and not md_text:
            rows.append(EquationRow(ref, "missing_in_markdown", None, pdf_text, ""))
            continue
        if md_text and not pdf_text:
            rows.append(EquationRow(ref, "extra_in_markdown", None, "", md_text))
            continue
        score = _similarity(pdf_text, md_text)
        status = "match" if score >= 0.72 else "different"
        rows.append(EquationRow(ref, status, score, pdf_text, md_text))
    return rows


def _render_report(pdf_path: Path, md_path: Path, rows: list[EquationRow]) -> str:
    match_count = sum(1 for row in rows if row.status == "match")
    diff_count = sum(1 for row in rows if row.status == "different")
    missing_count = sum(1 for row in rows if row.status == "missing_in_markdown")
    extra_count = sum(1 for row in rows if row.status == "extra_in_markdown")

    lines = [
        "# Display Equation Review",
        "",
        f"- PDF: `{pdf_path}`",
        f"- Markdown: `{md_path}`",
        "",
        "## Summary",
        f"- total_equation_refs: {len(rows)}",
        f"- match: {match_count}",
        f"- different: {diff_count}",
        f"- missing_in_markdown: {missing_count}",
        f"- extra_in_markdown: {extra_count}",
        "",
        "## Results",
        "",
        "| Ref | Status | Similarity | PDF Equation | Markdown Equation |",
        "|---|---|---:|---|---|",
    ]
    for row in rows:
        similarity = "" if row.similarity is None else f"{row.similarity:.2f}"
        lines.append(
            f"| {row.ref} | {row.status} | {similarity} | "
            f"`{_truncate(row.pdf_text)}` | `{_truncate(row.md_text)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def review_display_equations(pdf_path: str | Path, markdown_path: str | Path, report_path: str | Path | None = None) -> Path:
    pdf = Path(pdf_path).expanduser().resolve()
    md = Path(markdown_path).expanduser().resolve()
    if not pdf.exists():
        raise ReviewError(f"PDF not found: {pdf}")
    if not md.exists():
        raise ReviewError(f"Markdown not found: {md}")

    report = Path(report_path).expanduser().resolve() if report_path else md.with_name(f"{md.stem}_display_eq_review.md")
    report.parent.mkdir(parents=True, exist_ok=True)

    pdf_map = _extract_display_equations_from_pdf(pdf)
    md_map = _extract_display_equations_from_markdown(md)
    rows = _build_rows(pdf_map, md_map)
    report.write_text(_render_report(pdf, md, rows), encoding="utf-8")
    return report
