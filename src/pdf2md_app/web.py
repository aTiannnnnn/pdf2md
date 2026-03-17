"""Flask web interface for pdf2md."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
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

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
_MAX_AGE_SECONDS = 21 * 24 * 60 * 60  # 3 weeks

# Job tracking: job_id -> {status, stage, progress, filename, path, error}
_jobs: dict[str, dict] = {}


def _cleanup_old_tmp():
    """Remove pdf2md_web_ temp directories older than 3 weeks."""
    tmp_root = Path(tempfile.gettempdir())
    now = time.time()
    for d in tmp_root.glob("pdf2md_web_*"):
        if not d.is_dir():
            continue
        age = now - d.stat().st_mtime
        if age > _MAX_AGE_SECONDS:
            shutil.rmtree(d, ignore_errors=True)


def _make_progress_callback(job_id: str):
    """Return a progress callback that updates the job dict."""
    def on_progress(stage: str, frac: float):
        _jobs[job_id]["stage"] = stage
        _jobs[job_id]["progress"] = round(frac, 3)
    return on_progress


def _run_conversion(job_id: str, input_path: Path, output_path: Path,
                    options: ConvertOptions, output_format: str):
    """Run conversion in a background thread and update job status."""
    try:
        result = convert_pdf(input_path, output_path, options)

        if output_format == "html":
            _jobs[job_id].update(stage="Bundling HTML", progress=0.95)
            html_path = bundle_html(result)
            _jobs[job_id].update(
                status="done", path=html_path, filename=html_path.name,
                stage="Done", progress=1.0)
        else:
            images = [
                f for f in result.parent.iterdir()
                if f.suffix.lower() in _IMAGE_EXTS and f.is_file()
            ]
            if images:
                _jobs[job_id].update(stage="Packaging files", progress=0.95)
                zip_path = result.with_suffix(".zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(result, result.name)
                    for img in images:
                        zf.write(img, img.name)
                _jobs[job_id].update(
                    status="done", path=zip_path, filename=zip_path.name,
                    stage="Done", progress=1.0)
            else:
                _jobs[job_id].update(
                    status="done", path=result, filename=result.name,
                    stage="Done", progress=1.0)

    except ConversionError as exc:
        _jobs[job_id].update(status="error", error=str(exc))
    except Exception as exc:
        _jobs[job_id].update(status="error", error=f"Conversion failed: {exc}")


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

    _cleanup_old_tmp()

    backend = request.form.get("backend", "marker")
    output_format = request.form.get("format", "markdown")

    tmpdir = tempfile.mkdtemp(prefix="pdf2md_web_")
    input_path = Path(tmpdir) / file.filename
    file.save(input_path)

    output_path = input_path.with_suffix(".md")
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "converting", "stage": "Starting", "progress": 0.0}

    options = ConvertOptions(
        backend=backend, force=True,
        on_progress=_make_progress_callback(job_id),
    )

    thread = threading.Thread(
        target=_run_conversion,
        args=(job_id, input_path, output_path, options, output_format),
        daemon=True,
    )
    thread.start()

    return jsonify(job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify(error="Unknown job."), 404

    if job["status"] == "converting":
        return jsonify(
            status="converting",
            stage=job.get("stage", ""),
            progress=job.get("progress", 0.0),
        )
    elif job["status"] == "error":
        return jsonify(status="error", error=job["error"])
    else:
        return jsonify(status="done", filename=job["filename"])


@app.route("/system-stats")
def system_stats():
    # CPU usage (per-core average over 0.1s sampling)
    try:
        with open("/proc/stat") as f:
            line1 = f.readline()
        time.sleep(0.1)
        with open("/proc/stat") as f:
            line2 = f.readline()
        v1 = list(map(int, line1.split()[1:]))
        v2 = list(map(int, line2.split()[1:]))
        delta = [b - a for a, b in zip(v1, v2)]
        idle = delta[3]
        total = sum(delta)
        cpu_pct = round(100 * (1 - idle / max(1, total)), 1)
    except Exception:
        cpu_pct = None

    # Memory
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        mem_total = meminfo.get("MemTotal", 0)
        mem_avail = meminfo.get("MemAvailable", 0)
        mem_used = mem_total - mem_avail
        mem_pct = round(100 * mem_used / max(1, mem_total), 1)
        mem_used_gb = round(mem_used / 1048576, 1)
        mem_total_gb = round(mem_total / 1048576, 1)
    except Exception:
        mem_pct = mem_used_gb = mem_total_gb = None

    # GPU via nvidia-smi
    gpu = None
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 5:
                gpu = {
                    "name": parts[0],
                    "util_pct": float(parts[1]),
                    "mem_used_mb": float(parts[2]),
                    "mem_total_mb": float(parts[3]),
                    "mem_pct": round(100 * float(parts[2]) / max(1, float(parts[3])), 1),
                    "temp_c": float(parts[4]),
                }
    except Exception:
        pass

    return jsonify(
        cpu_pct=cpu_pct,
        mem_pct=mem_pct,
        mem_used_gb=mem_used_gb,
        mem_total_gb=mem_total_gb,
        gpu=gpu,
    )


@app.route("/download/<job_id>")
def download(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job.get("status") != "done":
        return jsonify(error="File not found or not ready."), 404

    path = job["path"]
    if not path.exists():
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
