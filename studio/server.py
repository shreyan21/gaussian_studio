import asyncio
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from studio.config import DATA, DEPTH_DIR, MAX_PIXELS, MAX_UPLOAD, ROOT, SHARP_SOURCE, SHARP_WEIGHTS
from studio.gaussians import export_scene, make_demo

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def focal_length_35mm(image):
    """Read SHARP's preferred full-frame focal length, with its official fallback."""
    exif = image.getexif()
    value = exif.get(41989)  # FocalLengthIn35mmFilm
    try:
        value = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        value = 0.0
    if value >= 1:
        return value
    value = exif.get(37386)  # FocalLength
    try:
        value = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 30.0
    if value < 1:
        return 30.0
    return value * 8.4 if value < 10 else value


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


class Jobs:
    def __init__(self, root):
        self.root = root
        self.lock = threading.RLock()
        self.active = None
        self.process = None
        self.cancelled = set()
        self.root.mkdir(parents=True, exist_ok=True)
        for file in self.root.glob("*/job.json"):
            try:
                job = json.loads(file.read_text())
                if job["status"] in ("running", "queued"):
                    job.update(status="failed", message="The app stopped during reconstruction. Please start a new conversion.")
                    atomic_json(file, job)
            except (OSError, ValueError, KeyError):
                continue

    def path(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Scene not found")
        path = self.root / job_id
        if not (path / "job.json").is_file():
            raise HTTPException(404, "Scene not found")
        return path

    def read(self, job_id):
        with self.lock:
            path = self.path(job_id)
            job = json.loads((path / "job.json").read_text())
            if job["status"] == "running" and (path / "progress.json").is_file():
                try:
                    job.update(json.loads((path / "progress.json").read_text()))
                except (OSError, ValueError):
                    pass
            if job["status"] == "completed":
                job["scene"] = json.loads((path / "scene.json").read_text())
            return job

    def create(self, image, filename, options):
        with self.lock:
            if self.active:
                raise HTTPException(409, "A reconstruction is already running. Wait for it or cancel it first.")
            job_id = uuid.uuid4().hex
            path = self.root / job_id
            path.mkdir()
            image.save(path / "input.png")
            thumb = image.copy()
            thumb.thumbnail((480, 360))
            thumb.save(path / "thumbnail.jpg", quality=85)
            atomic_json(path / "request.json", options)
            job = {"id": job_id, "name": Path(filename.replace("\\", "/")).name[:120], "created": datetime.now(timezone.utc).isoformat(), "status": "running", "progress": 0, "message": "Starting reconstruction", "engine": options["engine"]}
            atomic_json(path / "job.json", job)
            self.active = job_id
            threading.Thread(target=self.run, args=(job_id,), daemon=True).start()
            return job

    def run(self, job_id):
        path = self.root / job_id
        try:
            with (path / "worker.log").open("w", encoding="utf-8") as log:
                with self.lock:
                    if job_id in self.cancelled:
                        return
                    self.process = subprocess.Popen([sys.executable, "-m", "studio.worker", str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    process = self.process
                try:
                    returncode = process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("Reconstruction exceeded 30 minutes. Use the lightweight model or a faster device.")
            with self.lock:
                if job_id in self.cancelled:
                    return
                job = json.loads((path / "job.json").read_text())
                if returncode == 0 and (path / "scene.json").is_file() and (path / "scene.gsb").is_file():
                    job.update(status="completed", progress=100, message="Scene ready")
                else:
                    error_path = path / "error.json"
                    error = json.loads(error_path.read_text())["error"] if error_path.is_file() else f"Inference process exited ({returncode}). Open the job log for details; check available RAM and installed dependencies."
                    job.update(status="failed", message=error)
                atomic_json(path / "job.json", job)
        except Exception as exc:
            with self.lock:
                if job_id not in self.cancelled:
                    job = json.loads((path / "job.json").read_text())
                    job.update(status="failed", message=str(exc))
                    atomic_json(path / "job.json", job)
        finally:
            with self.lock:
                if self.active == job_id:
                    self.active, self.process = None, None
                self.cancelled.discard(job_id)

    def cancel(self, job_id):
        with self.lock:
            path = self.path(job_id)
            if self.active != job_id:
                raise HTTPException(409, "This reconstruction is no longer running")
            self.cancelled.add(job_id)
            if self.process and self.process.poll() is None:
                self.process.terminate()
            job = json.loads((path / "job.json").read_text())
            job.update(status="cancelled", message="Reconstruction cancelled")
            atomic_json(path / "job.json", job)
            return job

    def stop(self):
        with self.lock:
            if self.active:
                self.cancel(self.active)
            process = self.process
        if process:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def create_app(data_dir=None):
    storage = Path(data_dir) if data_dir else DATA

    @asynccontextmanager
    async def lifespan(app):
        app.state.jobs = Jobs(storage / "jobs")
        demo = storage / "demo"
        if not (demo / "scene.json").exists():
            demo.mkdir(parents=True, exist_ok=True)
            export_scene(demo, make_demo(), {"engine": "Procedural calibration scene", "method": "demo", "fov_y": 50, "image_size": [1024,768], "limitation": "Synthetic renderer demo. This is not reconstructed from an image.", "seconds": 0, "device": "none"})
        app.state.hardware = {"status": "checking", "cuda": False}
        def probe():
            try:
                p = subprocess.run([sys.executable, "scripts/doctor.py", "--json"], cwd=ROOT, capture_output=True, text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                app.state.hardware = json.loads(p.stdout)
            except Exception as exc:
                app.state.hardware = {"status": "unknown", "cuda": False, "error": str(exc)}
        if os.environ.get("GSS_SKIP_PROBE") == "1":
            app.state.hardware = {"status": "test", "cuda": False}
        else:
            threading.Thread(target=probe, daemon=True).start()
        yield
        app.state.jobs.stop()

    app = FastAPI(title="Gaussian Scene Studio", lifespan=lifespan)

    @app.middleware("http")
    async def local_guard(request: Request, call_next):
        # Local-only service: reject cross-origin writes and oversized upload bodies,
        # including chunked requests, before multipart parsing writes a temp file.
        hostname = request.url.hostname
        if hostname not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"detail": "Use the local app address"}, status_code=403)
        if request.method in ("POST", "DELETE", "PUT"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
            if request.url.path == "/api/jobs":
                raw = bytearray()
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > MAX_UPLOAD + 1024*1024:
                        return JSONResponse({"detail": "Upload must be under 20 MB"}, status_code=413)
                request._body = bytes(raw)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health():
        sharp_source = SHARP_SOURCE / "sharp" / "models" / "__init__.py"
        return {"app": "Gaussian Scene Studio", "version": "1.0.0", "instance_id": os.environ.get("GSS_INSTANCE_ID"), "hardware": app.state.hardware, "models": {"depth": (DEPTH_DIR / "model.safetensors").is_file(), "sharp": SHARP_WEIGHTS.is_file() and sharp_source.is_file()}, "active_job": app.state.jobs.active}

    @app.post("/api/jobs", status_code=202)
    async def upload(image: UploadFile = File(...), engine: str = Form("depth"), device: str = Form("auto"), resolution: int = Form(512), depth_strength: float = Form(1.0), research_use: bool = Form(False)):
        if engine not in ("depth", "sharp") or device not in ("auto", "cpu", "cuda"):
            raise HTTPException(422, "Invalid model or device")
        if resolution not in (384, 512, 768) or not 0.25 <= depth_strength <= 1.5:
            raise HTTPException(422, "Invalid quality or depth range")
        if engine == "sharp" and not research_use:
            raise HTTPException(422, "SHARP weights are limited to non-commercial scientific research. Confirm your permitted use or choose Depth Anything V2 Small.")
        payload = await image.read(MAX_UPLOAD+1)
        await image.close()
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Upload must be under 20 MB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(payload)) as source:
                    if source.format not in ("JPEG", "PNG", "WEBP"):
                        raise HTTPException(415, "Please use a JPEG, PNG or WebP image")
                    w, h = source.size
                    if w*h > MAX_PIXELS or min(w,h) < 32 or max(w,h)/min(w,h) > 8:
                        raise HTTPException(422, "Use an image of at least 32 pixels per side, at most 24 megapixels, and aspect ratio under 8:1")
                    focal_35mm = focal_length_35mm(source)
                    source.load()
                    oriented = ImageOps.exif_transpose(source)
                    rgba = oriented.convert("RGBA")
                    clean = Image.new("RGBA", rgba.size, "white")
                    clean.alpha_composite(rgba)
                    clean = clean.convert("RGB")
                    clean.thumbnail((2048,2048), Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(415, "The image is invalid, damaged or too large")
        return app.state.jobs.create(clean, image.filename or "image.png", {"engine": engine, "device": device, "resolution": resolution, "depth_strength": depth_strength, "focal_35mm": focal_35mm})

    @app.get("/api/jobs")
    def history():
        files = sorted((storage / "jobs").glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
        return [app.state.jobs.read(p.parent.name) for p in files]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return app.state.jobs.read(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return app.state.jobs.cancel(job_id)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def asset(job_id: str, filename: str):
        allowed = {"scene.ply", "scene.gsb", "scene.json", "depth.png", "input.png", "thumbnail.jpg", "worker.log"}
        if filename not in allowed:
            raise HTTPException(404, "File not found")
        folder = app.state.jobs.path(job_id)
        job = app.state.jobs.read(job_id)
        if filename.startswith("scene.") and job["status"] != "completed":
            raise HTTPException(409, "Scene is not ready")
        file = folder / filename
        if not file.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(file, filename=filename if filename.endswith((".ply", ".log")) else None)

    @app.get("/api/demo/{filename}")
    def demo(filename: str):
        if filename not in ("scene.gsb", "scene.json", "scene.ply"):
            raise HTTPException(404, "File not found")
        return FileResponse(storage / "demo" / filename)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static/index.html")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


app = create_app()
