import asyncio
import io
import json
import os
import re
import secrets
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from studio.anysplat_runtime import model_ready as anysplat_model_ready
from studio.config import DATA, DEPTH_DIR, MAX_IMAGES, MAX_PIXELS, MAX_UPLOAD, ROOT
from studio.gaussians import export_scene, make_demo

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


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

    def create(self, images, filenames, options):
        with self.lock:
            if self.active:
                raise HTTPException(409, "A reconstruction is already running. Wait for it or cancel it first.")
            job_id = uuid.uuid4().hex
            path = self.root / job_id
            path.mkdir()
            for index, image in enumerate(images):
                image.save(path / ("input.png" if index == 0 else f"input_{index}.png"))
            thumb = images[0].copy()
            thumb.thumbnail((480, 360))
            thumb.save(path / "thumbnail.jpg", quality=85)
            atomic_json(path / "request.json", options)
            first_name = Path(filenames[0].replace("\\", "/")).name[:100]
            name = first_name if len(images) == 1 else f"{first_name} + {len(images)-1} views"
            job = {"id": job_id, "name": name, "created": datetime.now(timezone.utc).isoformat(), "status": "running", "progress": 0, "message": "Starting reconstruction", "engine": options["engine"], "image_count": len(images)}
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
    access_token = os.environ.get("GSS_ACCESS_TOKEN", "").strip()

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
        # Default is loopback-only. Public-tunnel mode requires a generated token
        # before any page, API, upload, or result can be read.
        hostname = request.url.hostname
        query_token = request.query_params.get("token", "")
        cookie_token = request.cookies.get("gss_access", "")
        authorization = request.headers.get("authorization", "")
        bearer_token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
        token_ok = bool(access_token) and any(
            secrets.compare_digest(access_token, candidate)
            for candidate in (query_token, cookie_token, bearer_token)
            if candidate
        )
        if access_token and not token_ok:
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Valid access token required"}, status_code=401)
            return HTMLResponse(
                "<h1>Gaussian Scene Studio</h1><p>Access denied. Use complete protected link printed by Start Public Link.cmd.</p>",
                status_code=401,
            )
        if not access_token and hostname not in ("127.0.0.1", "localhost", "::1", "testserver"):
            return JSONResponse({"detail": "Use the local app address"}, status_code=403)
        if access_token and query_token and request.url.path == "/":
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                "gss_access",
                access_token,
                httponly=True,
                secure=request.headers.get("x-forwarded-proto", "").lower() == "https",
                samesite="strict",
                max_age=12 * 60 * 60,
            )
            response.headers["Referrer-Policy"] = "no-referrer"
            return response
        if request.method in ("POST", "DELETE", "PUT"):
            origin = request.headers.get("origin")
            if origin:
                origin_host = urlparse(origin).netloc.lower()
                allowed_hosts = {request.headers.get("host", "").lower()}
                # A protected public deployment may sit behind a reverse proxy
                # that connects to loopback while preserving the browser-facing
                # host in X-Forwarded-Host. Never trust that header in the
                # unauthenticated local-only mode.
                if access_token:
                    forwarded_host = request.headers.get("x-forwarded-host", "")
                    allowed_hosts.update(
                        item.strip().lower()
                        for item in forwarded_host.split(",")
                        if item.strip()
                    )
                if not origin_host or origin_host not in allowed_hosts:
                    return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
            if request.url.path == "/api/jobs":
                raw = bytearray()
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > MAX_UPLOAD * MAX_IMAGES + 1024*1024:
                        return JSONResponse({"detail": f"Each image must be under 20 MB and at most {MAX_IMAGES} images may be uploaded"}, status_code=413)
                request._body = bytes(raw)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health():
        return {"app": "Gaussian Scene Studio", "version": "2.1.0", "instance_id": os.environ.get("GSS_INSTANCE_ID"), "hardware": app.state.hardware, "models": {"depth": (DEPTH_DIR / "model.safetensors").is_file(), "anysplat": anysplat_model_ready()}, "active_job": app.state.jobs.active, "max_images": MAX_IMAGES, "remote_access": bool(access_token)}

    @app.post("/api/jobs", status_code=202)
    async def upload(
        images: list[UploadFile] | None = File(None),
        image: UploadFile | None = File(None),
        engine: str = Form("depth"),
        device: str = Form("auto"),
        resolution: int = Form(512),
        depth_strength: float = Form(1.0),
        view_limit: str = Form("auto"),
    ):
        if engine not in ("depth", "anysplat") or device not in ("auto", "cpu", "cuda"):
            raise HTTPException(422, "Invalid model or device")
        if resolution not in (384, 512, 768) or not 0.25 <= depth_strength <= 1.5:
            raise HTTPException(422, "Invalid quality or depth range")
        if view_limit not in ("auto", "all", "2", "4", "6", "8", "10", "12", "16"):
            raise HTTPException(422, "Invalid AnySplat view budget")
        if engine == "anysplat" and device == "cpu":
            raise HTTPException(422, "AnySplat requires NVIDIA CUDA; choose Auto or NVIDIA CUDA")
        uploads = list(images or [])
        if image is not None:
            uploads.insert(0, image)
        if not uploads:
            raise HTTPException(422, "Upload at least one image")
        if len(uploads) > MAX_IMAGES:
            raise HTTPException(422, f"Upload at most {MAX_IMAGES} images")
        if engine == "depth" and len(uploads) != 1:
            raise HTTPException(422, "Depth Anything accepts one image; choose AnySplat for multiple images")
        if engine == "anysplat" and len(uploads) < 2:
            raise HTTPException(422, "AnySplat needs at least two overlapping images")

        cleaned, names = [], []
        for upload_file in uploads:
            payload = await upload_file.read(MAX_UPLOAD+1)
            await upload_file.close()
            if len(payload) > MAX_UPLOAD:
                raise HTTPException(413, "Each image must be under 20 MB")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(payload)) as source:
                        if source.format not in ("JPEG", "PNG", "WEBP"):
                            raise HTTPException(415, "Please use JPEG, PNG or WebP images")
                        w, h = source.size
                        if w*h > MAX_PIXELS or min(w,h) < 32 or max(w,h)/min(w,h) > 8:
                            raise HTTPException(422, "Use images of at least 32 pixels per side, at most 24 megapixels, and aspect ratio under 8:1")
                        source.load()
                        oriented = ImageOps.exif_transpose(source)
                        rgba = oriented.convert("RGBA")
                        clean = Image.new("RGBA", rgba.size, "white")
                        clean.alpha_composite(rgba)
                        clean = clean.convert("RGB")
                        clean.thumbnail((2048,2048), Image.Resampling.LANCZOS)
                        cleaned.append(clean)
                        names.append(upload_file.filename or "image.png")
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                raise HTTPException(415, "An uploaded image is invalid, damaged or too large")
        inputs = [
            {"file": "input.png" if i == 0 else f"input_{i}.png", "original_name": Path(names[i].replace("\\", "/")).name[:100]}
            for i in range(len(cleaned))
        ]
        options = {"engine": engine, "device": device, "resolution": resolution, "depth_strength": depth_strength, "inputs": inputs, "view_limit": view_limit}
        return app.state.jobs.create(cleaned, names, options)

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
        fixed = {"scene.ply", "scene.gsb", "scene.json", "depth.png", "thumbnail.jpg", "worker.log"}
        is_input = filename == "input.png" or bool(re.fullmatch(r"input_(?:[1-9]|1[0-5])\.png", filename))
        if filename not in fixed and not is_input:
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
