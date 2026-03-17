"""Flask web interface for pdf2md."""

from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
)

from pdf2md_app.converter import ConversionError, ConvertOptions, convert_pdf
from pdf2md_app.html_bundle import bundle_html

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB limit

# Store conversion results keyed by job id so the user can download after convert
_results: dict[str, Path] = {}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify(error="No file uploaded."), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify(error="Please upload a PDF file."), 400

    backend = request.form.get("backend", "marker")
    output_format = request.form.get("format", "markdown")

    tmpdir = tempfile.mkdtemp(prefix="pdf2md_web_")
    input_path = Path(tmpdir) / file.filename
    file.save(input_path)

    output_path = input_path.with_suffix(".md")
    options = ConvertOptions(backend=backend, force=True)

    try:
        result = convert_pdf(input_path, output_path, options)
    except ConversionError as exc:
        return jsonify(error=str(exc)), 422
    except Exception as exc:
        return jsonify(error=f"Conversion failed: {exc}"), 500

    job_id = uuid.uuid4().hex

    if output_format == "html":
        # Single self-contained HTML with base64-embedded images
        html_path = bundle_html(result)
        _results[job_id] = html_path
        filename = html_path.name
    else:
        # Markdown — zip with images if any exist
        images = [
            f for f in result.parent.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and f.is_file()
        ]
        if images:
            zip_path = result.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(result, result.name)
                for img in images:
                    zf.write(img, img.name)
            _results[job_id] = zip_path
            filename = zip_path.name
        else:
            _results[job_id] = result
            filename = result.name

    return jsonify(
        job_id=job_id,
        filename=filename,
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    path = _results.get(job_id)
    if path is None or not path.exists():
        return jsonify(error="File not found or expired."), 404

    return send_file(path, as_attachment=True, download_name=path.name)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="pdf2md web interface")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Starting pdf2md web UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
