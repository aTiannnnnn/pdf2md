from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


class ConversionError(RuntimeError):
    pass


@dataclass
class ConvertOptions:
    backend: str = "marker"
    force: bool = False


@dataclass
class _PageLine:
    page: int
    x: int
    y: int
    text: str
    raw_html: str
    page_width: int
    page_height: int


_HEADING_WORD_WHITELIST = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "vs",
    "was",
    "were",
    "with",
}

_MATH_SYMBOLS = set("=+-*/^_<>≈≠≤≥∑∏∫√∞∂∇±×÷∈∉∝")


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


def _ensure_pdf(path: Path) -> None:
    if not path.exists():
        raise ConversionError(f"Input PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ConversionError(f"Input must be a .pdf file: {path}")


def _prepare_output(output_md: Path, force: bool) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if output_md.exists() and not force:
        raise ConversionError(
            f"Output exists: {output_md}. Use --force to overwrite."
        )


def _convert_with_marker(input_pdf: Path, output_md: Path, force: bool) -> Path:
    marker_bin = _find_command("marker_single")
    if marker_bin is None:
        raise ConversionError(
            "marker backend selected, but `marker_single` is not installed.\n"
            "Install with: pip install marker-pdf"
        )

    tmp_dir = output_md.parent / f".{output_md.stem}_marker_tmp"
    if tmp_dir.exists() and force:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        marker_bin,
        str(input_pdf),
        "--output_dir",
        str(tmp_dir),
        "--output_format",
        "markdown",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(
            "marker conversion failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    generated = list(tmp_dir.rglob("*.md"))
    if not generated:
        raise ConversionError(
            "marker conversion completed but no Markdown file was produced."
        )

    chosen = sorted(generated, key=lambda p: len(str(p)))[0]
    md_content = chosen.read_text(encoding="utf-8")

    # Copy extracted images to sit alongside the output markdown file.
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
    for img_file in chosen.parent.rglob("*"):
        if img_file.suffix.lower() not in _IMAGE_EXTS:
            continue
        dest = output_md.parent / img_file.name
        if dest.exists() and force:
            dest.unlink()
        if not dest.exists():
            shutil.copy2(img_file, dest)
        # Fix any sub-path references in the markdown so they point to just the filename.
        rel = img_file.relative_to(chosen.parent)
        if str(rel) != img_file.name:
            md_content = md_content.replace(str(rel), img_file.name)

    if output_md.exists():
        output_md.unlink()
    output_md.write_text(md_content, encoding="utf-8")
    return output_md


def _convert_with_pymupdf4llm(input_pdf: Path, output_md: Path) -> Path:
    try:
        import pymupdf4llm  # type: ignore
    except ImportError as exc:
        raise ConversionError(
            "pymupdf4llm backend selected but dependency missing.\n"
            "Install with: pip install pymupdf4llm"
        ) from exc

    md = pymupdf4llm.to_markdown(
        str(input_pdf),
        write_images=True,
        image_path=str(output_md.parent),
    )
    output_md.write_text(md, encoding="utf-8")
    return output_md


def _convert_with_nougat(input_pdf: Path, output_md: Path, force: bool) -> Path:
    nougat_bin = _find_command("nougat")
    if nougat_bin is None:
        raise ConversionError(
            "nougat backend selected, but `nougat` is not installed.\n"
            "Install with: pip install nougat-ocr"
        )

    tmp_dir = output_md.parent / f".{output_md.stem}_nougat_tmp"
    if tmp_dir.exists() and force:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [nougat_bin, str(input_pdf), "-o", str(tmp_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(
            "nougat conversion failed.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    candidates = (
        list(tmp_dir.rglob("*.mmd"))
        + list(tmp_dir.rglob("*.md"))
        + list(tmp_dir.rglob("*.markdown"))
        + list(tmp_dir.rglob("*.txt"))
    )
    if not candidates:
        raise ConversionError(
            "nougat conversion completed but no text/markdown file was produced."
        )

    chosen = sorted(candidates, key=lambda p: len(str(p)))[0]
    text = chosen.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").strip() + "\n"
    output_md.write_text(text, encoding="utf-8")
    return output_md


def _parse_html_int(value: str) -> int:
    try:
        return int(round(float(value)))
    except ValueError:
        return 0


def _strip_html_tags(fragment: str) -> str:
    no_tags = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(no_tags)


def _parse_pdftohtml_page(page_html: Path, page_number: int) -> list[_PageLine]:
    content = page_html.read_text(encoding="utf-8", errors="replace")
    img_match = re.search(r'<img[^>]*width="(\d+)"[^>]*height="(\d+)"', content)
    page_width = int(img_match.group(1)) if img_match else 612
    page_height = int(img_match.group(2)) if img_match else 792

    lines: list[_PageLine] = []
    div_pattern = re.compile(
        r'<div class="txt" style="position:absolute; left:(\d+(?:\.\d+)?)px; top:(\d+(?:\.\d+)?)px;">(.*?)</div>',
        flags=re.DOTALL,
    )
    for m in div_pattern.finditer(content):
        x = _parse_html_int(m.group(1))
        y = _parse_html_int(m.group(2))
        raw_html = m.group(3)
        text = _normalize_line(_strip_html_tags(raw_html))
        if not text:
            continue
        lines.append(
            _PageLine(
                page=page_number,
                x=x,
                y=y,
                text=text,
                raw_html=raw_html,
                page_width=page_width,
                page_height=page_height,
            )
        )
    return lines


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
    return text.strip()


def _text_equation_to_latex(text: str) -> str:
    eq = _strip_equation_ref(text)
    replacements = {
        "−": "-",
        "≤": r" \leq ",
        "≥": r" \geq ",
        "≠": r" \neq ",
        "≈": r" \approx ",
        "∈": r" \in ",
        "∉": r" \notin ",
        "∑": r" \sum ",
        "∫": r" \int ",
        "∞": r" \infty ",
        "∂": r" \partial ",
        "×": r" \times ",
        "÷": r" \div ",
    }
    for src, dst in replacements.items():
        eq = eq.replace(src, dst)
    eq = re.sub(r"\s+", " ", eq).strip()
    return eq


def _looks_like_display_equation_line(text: str, raw_html: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 220:
        return False

    eq_ref = _extract_equation_ref(stripped)
    if eq_ref is None:
        return False

    lower = stripped.lower()
    if lower.startswith(("figure ", "table ", "algorithm ", "references", "appendix")):
        return False

    core = _strip_equation_ref(stripped)
    if len(core) < 6:
        return False

    letters = sum(ch.isalpha() for ch in stripped)
    symbols = sum((ch in _MATH_SYMBOLS) for ch in stripped)
    punct = sum((not ch.isalnum() and not ch.isspace()) for ch in stripped)
    token_count = len(core.split())
    has_super_sub = ("vertical-align:super" in raw_html) or ("vertical-align:sub" in raw_html)
    has_operator = any(op in core for op in ("=", "≈", "≤", "≥", "∈", "∑", "∫", "->", "=>"))
    symbol_ratio = (symbols + punct) / max(1, len(stripped))
    letter_ratio = letters / max(1, len(stripped))

    if token_count > 42:
        return False
    if has_super_sub and has_operator and token_count <= 42 and symbol_ratio >= 0.06:
        return True
    if has_operator and token_count <= 42 and symbol_ratio >= 0.06:
        return True
    if has_operator and token_count <= 28 and letter_ratio <= 0.78:
        return True
    if re.search(r"\b(E|Var|arg\s*max|arg\s*min|log|exp|min|max)\b", core):
        return has_operator and symbol_ratio >= 0.06
    return False


def _estimate_crop_size(line: _PageLine) -> tuple[int, int]:
    font_match = re.search(r"font-size:(\d+(?:\.\d+)?)px", line.raw_html)
    font_size = float(font_match.group(1)) if font_match else 10.0
    width = int(font_size * 0.62 * len(line.text) + 52)
    width = max(120, min(width, line.page_width - 4))
    height = int(max(24, min(120, font_size * 3.0)))
    return width, height


def _crop_with_sips(src_image: Path, dest_image: Path, x: int, y: int, w: int, h: int) -> None:
    cmd = [
        "sips",
        "-c",
        str(h),
        str(w),
        "--cropOffset",
        str(y),
        str(x),
        str(src_image),
        "--out",
        str(dest_image),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(
            "sips crop failed while preparing equation OCR.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def _load_pix2tex_model():
    try:
        from pix2tex.cli import LatexOCR  # type: ignore
    except ImportError as exc:
        raise ConversionError(
            "latexocr backend selected, but `pix2tex` is not installed.\n"
            "Install with: pip install pix2tex"
        ) from exc
    return LatexOCR()


def _run_pix2tex_model(model, image_path: Path) -> str | None:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise ConversionError(
            "latexocr backend needs Pillow. Install with: pip install Pillow"
        ) from exc

    try:
        with Image.open(image_path) as img:
            latex = model(img)
    except Exception:
        return None

    if not isinstance(latex, str):
        return None
    latex = latex.strip().strip("$")
    if not _is_reasonable_latex(latex):
        return None
    return latex or None


def _is_reasonable_latex(latex: str) -> bool:
    text = latex.strip()
    if not text:
        return False
    if len(text) < 3 or len(text) > 220:
        return False
    if re.search(r"(.)\1{7,}", text):
        return False
    if re.search(r"(?:[A-Za-z]\s+){10,}", text):
        return False
    if text.count(" ") > int(len(text) * 0.33):
        return False
    if re.search(r"(?:\\[A-Za-z]+\s*){8,}", text) and "=" not in text:
        return False
    if re.search(r"\b(?:bigstar|square|triangle)\b", text, flags=re.IGNORECASE):
        return False
    if not re.search(r"(\\[A-Za-z]+|=|\^|_|\\frac|\\sum|\\int|\\left|\\right)", text):
        return False
    return True


def _column_order(lines: list[_PageLine], page_width: int) -> list[_PageLine]:
    if not lines:
        return []

    split_x = page_width * 0.5
    left = [line for line in lines if line.x < split_x]
    right = [line for line in lines if line.x >= split_x]
    has_two_columns = bool(left and right and (min(line.x for line in right) - max(line.x for line in left) > 40))

    if not has_two_columns:
        return sorted(lines, key=lambda line: (line.y, line.x))

    ordered: list[_PageLine] = []
    ordered.extend(sorted(left, key=lambda line: (line.y, line.x)))
    # Marker for blank line between columns.
    ordered.append(
        _PageLine(
            page=lines[0].page,
            x=0,
            y=10**9,
            text="",
            raw_html="",
            page_width=page_width,
            page_height=lines[0].page_height,
        )
    )
    ordered.extend(sorted(right, key=lambda line: (line.y, line.x)))
    return ordered


def _convert_with_latexocr(input_pdf: Path, output_md: Path, force: bool) -> Path:
    pdftohtml_bin = _find_command("pdftohtml")
    sips_bin = _find_command("sips")

    if pdftohtml_bin is None:
        raise ConversionError("latexocr backend needs `pdftohtml` installed.")
    if sips_bin is None:
        raise ConversionError("latexocr backend needs `sips` (macOS tool) installed.")

    warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*", category=RuntimeWarning)
    latex_model = _load_pix2tex_model()

    tmp_dir = output_md.parent / f".{output_md.stem}_latexocr_tmp"
    if tmp_dir.exists() and force:
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [pdftohtml_bin, "-overwrite", str(input_pdf), str(tmp_dir)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise ConversionError(
            "pdftohtml failed for latexocr backend.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    page_files = sorted(
        tmp_dir.glob("page*.html"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 10**9,
    )
    if not page_files:
        raise ConversionError("latexocr backend could not find page HTML files from pdftohtml.")

    extracted_lines: list[str] = []
    equation_count = 0
    equation_cap = 120

    for page_file in page_files:
        page_match = re.search(r"(\d+)", page_file.stem)
        page_num = int(page_match.group(1)) if page_match else 1
        page_lines = _parse_pdftohtml_page(page_file, page_num)
        if not page_lines:
            continue

        page_width = page_lines[0].page_width
        ordered = _column_order(page_lines, page_width)
        prev_y: int | None = None

        for idx, line in enumerate(ordered):
            if not line.text:
                extracted_lines.append("")
                prev_y = None
                continue

            if prev_y is not None and abs(line.y - prev_y) > 18:
                extracted_lines.append("")
            prev_y = line.y

            if equation_count < equation_cap and _looks_like_display_equation_line(line.text, line.raw_html):
                eq_ref = _extract_equation_ref(line.text)
                fallback = _text_equation_to_latex(line.text)
                width, height = _estimate_crop_size(line)
                x0 = max(0, min(line.page_width - width, line.x - 8))
                y0 = max(0, min(line.page_height - height, line.y - 4))

                page_png = tmp_dir / f"page{page_num}.png"
                crop_path = tmp_dir / f"eq_{page_num}_{idx}.png"
                if page_png.exists():
                    try:
                        _crop_with_sips(page_png, crop_path, x0, y0, width, height)
                        latex = _run_pix2tex_model(latex_model, crop_path)
                    except ConversionError:
                        latex = None
                    chosen = latex or fallback
                    if chosen:
                        if eq_ref and "\\tag{" not in chosen:
                            chosen = f"{chosen} \\tag{{{eq_ref}}}"
                        if extracted_lines and extracted_lines[-1] != "":
                            extracted_lines.append("")
                        extracted_lines.append(f"$$ {chosen} $$")
                        extracted_lines.append("")
                        equation_count += 1
                        continue

            extracted_lines.append(line.text)
        extracted_lines.append("")

    lines = [_normalize_line(line) for line in extracted_lines]
    lines = _remove_repeated_headers_footers(lines)
    lines = _fix_hyphenation(lines)
    lines = _split_inline_headings(lines)

    compact: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                compact.append("")
            prev_blank = True
            continue
        prev_blank = False
        compact.append(line)

    md = _lines_to_markdown(compact)
    output_md.write_text(md, encoding="utf-8")
    return output_md


def _normalize_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2212", "-")
    text = re.sub(r"(\d)(https?://)", r"\1 \2", text)
    return text


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _remove_repeated_headers_footers(lines: list[str]) -> list[str]:
    line_counts: Counter[str] = Counter()
    for line in lines:
        if not line:
            continue
        if 8 <= len(line) <= 120:
            line_counts[line] += 1

    repeated = {
        line
        for line, count in line_counts.items()
        if count >= 3 and len(line.split()) <= 10
    }

    cleaned: list[str] = []
    repeated_seen: Counter[str] = Counter()
    for line in lines:
        lower = line.lower()
        if re.fullmatch(r"\d+", line):
            continue
        if lower.startswith("arxiv:"):
            continue
        if lower.startswith("*equal contribution"):
            continue
        if lower.startswith("proceedings of the "):
            continue
        if "correspondence to:" in lower and "@" in lower:
            continue
        if "dence to:" in lower and "@" in lower:
            continue
        if "<" in line and "@" in line:
            continue
        if "pmlr" in lower and "copyright" in lower:
            continue
        if "learning, long beach, california" in lower and "pmlr" in lower:
            continue
        if "by the author(s)." in lower:
            continue
        if line in repeated:
            repeated_seen[line] += 1
            # Keep the first occurrence, drop later repeated headers/footers.
            if repeated_seen[line] > 1:
                continue
        cleaned.append(line)
    return cleaned


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbol_count / len(text)


def _is_reasonable_section_number(number: str) -> bool:
    parts = number.split(".")
    if len(parts) > 4:
        return False
    if any(not part.isdigit() for part in parts):
        return False
    ints = [int(part) for part in parts]
    if any(val <= 0 for val in ints):
        return False
    if ints[0] > 30:
        return False
    return True


def _is_likely_heading_title(title: str) -> bool:
    if len(title) > 100:
        return False
    if len(title.split()) > 16:
        return False
    if title.endswith("."):
        return False
    if not re.search(r"[A-Za-z]", title):
        return False
    if not re.match(r"^[A-Za-z]", title):
        return False
    if _symbol_ratio(title) > 0.18:
        return False
    if not _looks_like_title_case(title.split()):
        return False
    return True


def _detect_heading(line: str) -> tuple[int, str] | None:
    fixed = line.lower()
    if fixed in {
        "abstract",
        "introduction",
        "related work",
        "discussion",
        "conclusion",
        "conclusions",
        "acknowledgements",
        "acknowledgments",
        "references",
        "appendix",
    }:
        return 2, line

    numbered = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$", line)
    if not numbered:
        appendix = re.match(r"^(appendix)\s*[:.]?\s*(.+)?$", line, flags=re.IGNORECASE)
        if appendix:
            suffix = appendix.group(2) or ""
            label = "Appendix" + (f": {suffix}" if suffix else "")
            return 2, label.strip()
        return None

    number = numbered.group(1)
    title = numbered.group(2).strip()
    if not _is_reasonable_section_number(number):
        return None
    if not _is_likely_heading_title(title):
        return None

    level = 2 + max(0, len(number.split(".")) - 1)
    return level, f"{number}. {title}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"\s+", "-", slug).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug


def _looks_like_caption(line: str) -> bool:
    return bool(re.match(r"^(figure|table|algorithm)\s+\d+[.:]?\s+", line, flags=re.IGNORECASE))


def _split_inline_headings(lines: list[str]) -> list[str]:
    split_lines: list[str] = []
    for line in lines:
        if not line:
            split_lines.append(line)
            continue

        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+(.+?\?)\s+(.+)$", line)
        if not match:
            split_lines.append(line)
            continue

        heading_line = f"{match.group(1)}. {match.group(2).strip()}"
        if _detect_heading(heading_line) is None:
            mid = re.match(r"^(.*?)(\d+(?:\.\d+)*)\.?\s+(.+?\?)\s+(.+)$", line)
            if not mid:
                split_lines.append(line)
                continue
            prefix = mid.group(1).strip()
            if prefix and prefix[-1] not in {".", ":", ";"}:
                split_lines.append(line)
                continue

            heading_line = f"{mid.group(2)}. {mid.group(3).strip()}"
            if _detect_heading(heading_line) is None:
                split_lines.append(line)
                continue

            if prefix:
                split_lines.append(prefix)
            split_lines.append(heading_line)
            split_lines.append(mid.group(4).strip())
            continue

        split_lines.append(heading_line)
        split_lines.append(match.group(3).strip())
    return split_lines


def _lines_to_markdown(lines: list[str]) -> str:
    if not any(lines):
        return "# Converted Document\n"

    first_idx = next(i for i, line in enumerate(lines) if line)
    title = lines[first_idx]
    cursor = first_idx + 1
    authors = ""

    while cursor < len(lines) and not lines[cursor]:
        cursor += 1
    if cursor < len(lines):
        candidate = lines[cursor]
        heading = _detect_heading(candidate)
        if heading is None and len(candidate) <= 140 and not _looks_like_caption(candidate):
            authors = candidate
            cursor += 1

    body_lines = lines[cursor:]
    headings: list[tuple[int, str]] = []
    body_blocks: list[tuple[str, str, int]] = []
    para_parts: list[str] = []

    def flush_paragraph() -> None:
        if para_parts:
            paragraph = " ".join(part.strip() for part in para_parts if part.strip())
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                body_blocks.append(("paragraph", paragraph, 0))
            para_parts.clear()

    for line in body_lines:
        if not line:
            flush_paragraph()
            continue

        if re.match(r"^\$\$.*\$\$$", line):
            flush_paragraph()
            body_blocks.append(("equation", line, 0))
            continue

        detected = _detect_heading(line)
        if detected:
            flush_paragraph()
            level, text = detected
            headings.append((level, text))
            body_blocks.append(("heading", text, level))
            continue

        if _looks_like_caption(line):
            flush_paragraph()
            body_blocks.append(("caption", line, 0))
            continue

        para_parts.append(line)

    flush_paragraph()

    md_lines = [f"# {title}", ""]
    if authors:
        md_lines.extend([f"*{authors}*", ""])

    if headings:
        md_lines.append("## Contents")
        seen_anchors: set[str] = set()
        for _, heading_text in headings:
            anchor = _slugify(heading_text)
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            md_lines.append(f"- [{heading_text}](#{anchor})")
        md_lines.append("")

    for block_type, text, level in body_blocks:
        if block_type == "heading":
            hashes = "#" * min(level, 6)
            md_lines.extend([f"{hashes} {text}", ""])
            continue
        if block_type == "caption":
            md_lines.extend([f"**{text}**", ""])
            continue
        if block_type == "equation":
            md_lines.extend([text, ""])
            continue
        md_lines.extend([text, ""])

    return "\n".join(md_lines).strip() + "\n"


def _fix_hyphenation(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            merged.append(line)
            i += 1
            continue

        while line.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1].lstrip()
            if nxt and nxt[0].islower():
                line = line[:-1] + nxt
                i += 1
            else:
                break
        merged.append(line)
        i += 1
    return merged


def _looks_like_title_case(words: list[str]) -> bool:
    if not words:
        return False
    good = 0
    total = 0
    for word in words:
        token = word.strip("()[]{}.,:;!?`'\"")
        if not token:
            continue
        total += 1
        low = token.lower()
        if low in _HEADING_WORD_WHITELIST:
            good += 1
            continue
        if token[:1].isupper():
            good += 1
            continue
        if token.isupper() and len(token) <= 5:
            good += 1
    if total == 0:
        return False
    return (good / total) >= 0.45


def _convert_with_pdftotext(input_pdf: Path, output_md: Path) -> Path:
    pdftotext_bin = _find_command("pdftotext")
    if pdftotext_bin is None:
        raise ConversionError(
            "pdftotext backend selected but `pdftotext` is not installed."
        )

    cmd = [pdftotext_bin, "-raw", "-nopgbrk", str(input_pdf), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise ConversionError(
            "pdftotext conversion failed.\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )

    text = _normalize_text(proc.stdout.decode("utf-8", errors="replace"))
    lines = [_normalize_line(line) for line in text.split("\n")]
    lines = _remove_repeated_headers_footers(lines)
    lines = _fix_hyphenation(lines)
    lines = _split_inline_headings(lines)

    compact: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                compact.append("")
            prev_blank = True
            continue
        prev_blank = False
        compact.append(line)

    md = _lines_to_markdown(compact)
    output_md.write_text(md, encoding="utf-8")
    return output_md


def convert_pdf(input_pdf: str | Path, output_md: str | Path, options: ConvertOptions) -> Path:
    in_path = Path(input_pdf).expanduser().resolve()
    out_path = Path(output_md).expanduser().resolve()

    _ensure_pdf(in_path)
    _prepare_output(out_path, options.force)

    backend = options.backend.strip().lower()
    if backend == "marker":
        return _convert_with_marker(in_path, out_path, options.force)
    if backend == "nougat":
        return _convert_with_nougat(in_path, out_path, options.force)
    if backend == "latexocr":
        return _convert_with_latexocr(in_path, out_path, options.force)
    if backend == "pymupdf4llm":
        return _convert_with_pymupdf4llm(in_path, out_path)
    if backend == "pdftotext":
        return _convert_with_pdftotext(in_path, out_path)

    raise ConversionError(
        f"Unknown backend: {options.backend}. Use `marker`, `nougat`, `latexocr`, `pymupdf4llm`, or `pdftotext`."
    )
