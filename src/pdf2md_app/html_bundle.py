"""Convert a markdown file + sibling images into a single self-contained HTML."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown as md


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}

# Match markdown image syntax: ![alt](path)
_MD_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def bundle_html(md_path: Path, output_path: Path | None = None) -> Path:
    """Read *md_path*, inline all sibling images as base64, render to HTML.

    Returns the path to the written HTML file.
    """
    md_text = md_path.read_text(encoding="utf-8")
    parent = md_path.parent

    def _replace_image(match: re.Match) -> str:
        alt = match.group(1)
        src = match.group(2)
        img_path = parent / src
        if not img_path.exists() or img_path.suffix.lower() not in _IMAGE_EXTS:
            return match.group(0)  # leave as-is
        mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f'![{alt}](data:{mime};base64,{b64})'

    md_text = _MD_IMG_RE.sub(_replace_image, md_text)

    extensions = ["tables", "fenced_code", "codehilite", "toc", "md_in_html"]
    html_body = md.markdown(md_text, extensions=extensions)

    title = md_path.stem.replace("_", " ").replace("-", " ")

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{
    max-width: 48rem;
    margin: 2rem auto;
    padding: 0 1.5rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    color: #1a1a1a;
  }}
  img {{ max-width: 100%; height: auto; }}
  pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; border-radius: 6px; }}
  code {{ font-size: 0.9em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  blockquote {{ border-left: 3px solid #ddd; margin: 1rem 0; padding-left: 1rem; color: #555; }}
  h1, h2, h3, h4 {{ margin-top: 1.5em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    if output_path is None:
        output_path = md_path.with_suffix(".html")
    output_path.write_text(html, encoding="utf-8")
    return output_path
