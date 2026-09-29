import asyncio
import io
import json
import os
import re
import secrets
import shutil
import signal
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

from studio.config import DATA, MAX_IMAGES, MAX_PIXELS, MAX_TOTAL_UPLOAD, MAX_UPLOAD, MAX_VIDEO_UPLOAD, ROOT
from studio.custom_sfm import engine_ready
from studio.gsplat_runtime import gsplat_ready
from studio.gaussians import export_scene, make_demo, read_ply
from studio.triposplat_runtime import triposplat_ready, triposplat_status

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
FIXED_RESOLUTION = 768
FIXED_FOCUS_SUBJECT = False
VIDEO_UPLOAD_CHUNK = 8 * 1024 * 1024
CHUNKED_VIDEO_UPLOAD_LIMIT = max(MAX_VIDEO_UPLOAD, 500 * 1024 * 1024)


def terminate_process_tree(process):
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


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

    def create(self, images, filenames, options, video_stream=None, video_limit=MAX_VIDEO_UPLOAD):
        with self.lock:
            if self.active:
                raise HTTPException(409, "A reconstruction is already running. Wait for it or cancel it first.")
            job_id = uuid.uuid4().hex
            path = self.root / job_id
            path.mkdir()
            if video_stream is not None:
                destination = path / options["video"]["file"]
                total = 0
                with destination.open("wb") as output:
                    while chunk := video_stream.read(1024 * 1024):
                        total += len(chunk)
                        if total > video_limit:
                            raise HTTPException(413, f"Video must be under {video_limit // 1024**2} MB")
                        output.write(chunk)
            else:
                for index, image in enumerate(images):
                    image.save(path / ("input.png" if index == 0 else f"input_{index}.png"))
                thumb = images[0].copy()
                thumb.thumbnail((480, 360))
                if thumb.mode != "RGB":
                    rgba = thumb.convert("RGBA")
                    flattened = Image.new("RGBA", rgba.size, "white")
                    flattened.alpha_composite(rgba)
                    thumb = flattened.convert("RGB")
                thumb.save(path / "thumbnail.jpg", quality=85)
            atomic_json(path / "request.json", options)
            first_name = Path(filenames[0].replace("\\", "/")).name[:100]
            name = first_name if video_stream is not None or len(images) == 1 else f"{first_name} + {len(images)-1} views"
            generated = options["engine"] == "triposplat"
            job = {"id": job_id, "name": name, "created": datetime.now(timezone.utc).isoformat(), "status": "running", "progress": 0, "message": "Starting single-image generation" if generated else "Starting reconstruction", "engine": options["engine"], "image_count": 1 if video_stream is not None else len(images), "source_type": "image" if generated else ("video" if video_stream is not None else "photos")}
            atomic_json(path / "job.json", job)
            self.active = job_id
            threading.Thread(target=self.run, args=(job_id,), daemon=True).start()
            return job

    def retry(self, job_id, resolution=None, focus_subject=None):
        """Re-run a saved capture with current code and optional current UI settings."""
        with self.lock:
            if self.active:
                raise HTTPException(409, "A reconstruction is already running. Wait for it or cancel it first.")
            source = self.path(job_id)
            previous = json.loads((source / "job.json").read_text(encoding="utf-8"))
            options = json.loads((source / "request.json").read_text(encoding="utf-8"))
            if resolution is not None:
                if resolution not in (384, 512, 768):
                    raise HTTPException(422, "Unsupported reconstruction resolution")
                options["resolution"] = resolution
            if focus_subject is not None:
                options["focus_subject"] = focus_subject
            expected = []
            if options.get("video"):
                expected.append(options["video"]["file"])
            else:
                expected.extend(item["file"] for item in options.get("inputs", []))
            missing = [name for name in expected if not (source / name).is_file()]
            if missing:
                raise HTTPException(409, "The original upload is no longer available; upload the capture again.")

            new_id = uuid.uuid4().hex
            destination = self.root / new_id
            destination.mkdir()
            atomic_json(destination / "request.json", options)
            for name in expected:
                shutil.copy2(source / name, destination / name)
            job = {
                "id": new_id,
                "name": previous.get("name", "Saved capture") + " - rerun",
                "created": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "progress": 0,
                "message": "Re-running saved capture with current code",
                "engine": options["engine"],
                "image_count": previous.get("image_count", len(expected)),
                "source_type": previous.get("source_type", "video" if options.get("video") else "photos"),
            }
            atomic_json(destination / "job.json", job)
            self.active = new_id
            threading.Thread(target=self.run, args=(new_id,), daemon=True).start()
            return job

    def clean(self, job_id):
        """Create a cleaned copy of a completed focused scene without retraining."""
        with self.lock:
            source = self.path(job_id)
            previous = json.loads((source / "job.json").read_text(encoding="utf-8"))
            if previous.get("status") != "completed":
                raise HTTPException(409, "The scene must finish before floating fragments can be removed.")
            metadata = json.loads((source / "scene.json").read_text(encoding="utf-8"))
            if not metadata.get("subject_focus_applied"):
                raise HTTPException(409, "This scene was not reconstructed in central-subject mode.")
            if metadata.get("component_filter_evaluated"):
                raise HTTPException(409, "Connected-subject cleanup was already applied to this scene.")

            new_id = uuid.uuid4().hex
            destination = self.root / new_id
            destination.mkdir()
            for name in ("request.json", "thumbnail.jpg", "dense.ply"):
                if (source / name).is_file():
                    shutil.copy2(source / name, destination / name)
            if (source / "request.json").is_file():
                options = json.loads((source / "request.json").read_text(encoding="utf-8"))
                input_names = [item["file"] for item in options.get("inputs", [])]
                if options.get("video"):
                    input_names.append(options["video"]["file"])
                for name in input_names:
                    if (source / name).is_file():
                        shutil.copy2(source / name, destination / name)

            metadata = dict(metadata)
            metadata["cleaned_from_job"] = job_id
            export_scene(destination, read_ply(source / "scene.ply"), metadata)
            cleaned = json.loads((destination / "scene.json").read_text(encoding="utf-8"))
            removed = int(cleaned.get("component_removed_gaussians", 0))
            job = {
                "id": new_id,
                "name": previous.get("name", "Saved scene") + " - cleaned",
                "created": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "progress": 100,
                "message": f"Scene ready; removed {removed:,} detached Gaussians",
                "engine": previous.get("engine", "custom"),
                "image_count": previous.get("image_count", metadata.get("input_count", 0)),
                "source_type": previous.get("source_type", metadata.get("source_type", "photos")),
            }
            atomic_json(destination / "job.json", job)
            return self.read(new_id)

    def run(self, job_id):
        path = self.root / job_id
        try:
            with (path / "worker.log").open("w", encoding="utf-8") as log:
                with self.lock:
                    if job_id in self.cancelled:
                        return
                    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    worker_environment = os.environ.copy()
                    worker_environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                    self.process = subprocess.Popen(
                        [sys.executable, "-m", "studio.worker", str(path)],
                        cwd=ROOT,
                        env=worker_environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=creationflags,
                        start_new_session=os.name != "nt",
                    )
                    process = self.process
                try:
                    timeout_minutes = max(10, int(os.environ.get("GSS_JOB_TIMEOUT_MINUTES", "120")))
                    returncode = process.wait(timeout=timeout_minutes * 60)
                except subprocess.TimeoutExpired:
                    terminate_process_tree(process)
                    process.wait()
                    raise RuntimeError(f"Reconstruction exceeded {timeout_minutes} minutes and was stopped.")
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
            terminate_process_tree(self.process)
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
                terminate_process_tree(process)


def create_app(data_dir=None):
    storage = Path(data_dir) if data_dir else DATA
    upload_root = storage / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    upload_lock = threading.Lock()
    access_token = os.environ.get("GSS_ACCESS_TOKEN", "").strip()

    def video_upload_path(upload_id):
        if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise HTTPException(404, "Video upload not found")
        path = upload_root / upload_id
        if not (path / "upload.json").is_file():
            raise HTTPException(404, "Video upload not found")
        return path

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
                content_length = request.headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > MAX_TOTAL_UPLOAD:
                    return JSONResponse({"detail": f"Upload is too large. Videos must be under {MAX_VIDEO_UPLOAD // 1024**2} MB; each photo must be under 20 MB."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health():
        single_ready, single_message = triposplat_status()
        return {"app": "Gaussian Scene Studio", "version": "5.0.0", "instance_id": os.environ.get("GSS_INSTANCE_ID"), "hardware": app.state.hardware, "engines": {"custom": engine_ready(), "gsplat": gsplat_ready(), "triposplat": single_ready}, "engine_messages": {"triposplat": single_message}, "active_job": app.state.jobs.active, "max_images": MAX_IMAGES, "max_video_mb": CHUNKED_VIDEO_UPLOAD_LIMIT // 1024**2, "video_chunk_mb": VIDEO_UPLOAD_CHUNK // 1024**2, "remote_access": bool(access_token)}

    @app.post("/api/video-uploads", status_code=201)
    def start_video_upload(
        filename: str = Form(...),
        size: int = Form(...),
    ):
        if app.state.jobs.active:
            raise HTTPException(409, "A reconstruction is already running. Wait for it or cancel it first.")
        clean_name = Path(filename.replace("\\", "/")).name[:100]
        suffix = Path(clean_name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".webm"}:
            raise HTTPException(415, "Use MP4, MOV, M4V, or WebM video")
        if size < 1024:
            raise HTTPException(415, "The uploaded video is empty or invalid")
        if size > CHUNKED_VIDEO_UPLOAD_LIMIT:
            raise HTTPException(413, f"Video must be under {CHUNKED_VIDEO_UPLOAD_LIMIT // 1024**2} MB")
        upload_id = uuid.uuid4().hex
        path = upload_root / upload_id
        path.mkdir()
        atomic_json(path / "upload.json", {
            "id": upload_id,
            "filename": clean_name,
            "suffix": suffix,
            "size": size,
            "received": 0,
            "created": time.time(),
        })
        (path / "video.part").touch()
        return {"id": upload_id, "chunk_bytes": VIDEO_UPLOAD_CHUNK, "max_video_mb": CHUNKED_VIDEO_UPLOAD_LIMIT // 1024**2}

    @app.put("/api/video-uploads/{upload_id}")
    async def append_video_upload(upload_id: str, request: Request, offset: int):
        path = video_upload_path(upload_id)
        parts = []
        total = 0
        async for part in request.stream():
            total += len(part)
            if total > VIDEO_UPLOAD_CHUNK:
                raise HTTPException(413, f"Each video chunk must be at most {VIDEO_UPLOAD_CHUNK // 1024**2} MB")
            parts.append(part)
        payload = b"".join(parts)
        if not payload:
            raise HTTPException(413, f"Each video chunk must be between 1 byte and {VIDEO_UPLOAD_CHUNK // 1024**2} MB")
        with upload_lock:
            metadata = json.loads((path / "upload.json").read_text(encoding="utf-8"))
            if offset != metadata["received"]:
                raise HTTPException(409, f"Expected upload offset {metadata['received']}")
            if metadata["received"] + len(payload) > metadata["size"]:
                raise HTTPException(413, "Video upload exceeds its declared size")
            with (path / "video.part").open("ab") as output:
                output.write(payload)
            metadata["received"] += len(payload)
            atomic_json(path / "upload.json", metadata)
        return {"received": metadata["received"], "size": metadata["size"]}

    @app.post("/api/video-uploads/{upload_id}/finish", status_code=202)
    def finish_video_upload(
        upload_id: str,
        device: str = Form("auto"),
    ):
        if device not in ("auto", "cuda"):
            raise HTTPException(422, "Dense custom reconstruction requires NVIDIA CUDA; choose Auto or NVIDIA CUDA")
        path = video_upload_path(upload_id)
        with upload_lock:
            metadata = json.loads((path / "upload.json").read_text(encoding="utf-8"))
            if metadata["received"] != metadata["size"]:
                raise HTTPException(409, f"Video upload is incomplete: {metadata['received']} of {metadata['size']} bytes")
        options = {"engine": "custom", "device": device, "resolution": FIXED_RESOLUTION, "depth_strength": 1.0, "inputs": [], "video": {"file": "input-video" + metadata["suffix"], "original_name": metadata["filename"]}, "view_limit": "auto", "focus_subject": FIXED_FOCUS_SUBJECT}
        with (path / "video.part").open("rb") as video_stream:
            job = app.state.jobs.create(
                [],
                [metadata["filename"]],
                options,
                video_stream=video_stream,
                video_limit=CHUNKED_VIDEO_UPLOAD_LIMIT,
            )
        shutil.rmtree(path)
        return job

    @app.delete("/api/video-uploads/{upload_id}", status_code=204)
    def discard_video_upload(upload_id: str):
        path = video_upload_path(upload_id)
        with upload_lock:
            shutil.rmtree(path)

    @app.post("/api/jobs", status_code=202)
    async def upload(
        images: list[UploadFile] | None = File(None),
        image: UploadFile | None = File(None),
        engine: str = Form("auto"),
        device: str = Form("auto"),
        depth_strength: float = Form(1.0),
        view_limit: str = Form("auto"),
    ):
        if engine not in ("auto", "custom", "triposplat") or device not in ("auto", "cpu", "cuda"):
            raise HTTPException(422, "Invalid model or device")
        if not 0.25 <= depth_strength <= 1.5:
            raise HTTPException(422, "Invalid depth range")
        if view_limit not in ("auto", "all", "2", "4", "6", "8", "10", "12", "16"):
            raise HTTPException(422, "Invalid view option")
        if device == "cpu":
            raise HTTPException(422, "Dense custom reconstruction requires NVIDIA CUDA; choose Auto or NVIDIA CUDA")
        uploads = list(images or [])
        if image is not None:
            uploads.insert(0, image)
        if not uploads:
            raise HTTPException(422, "Upload one photograph, one video, or at least 12 overlapping photos")
        video_extensions = {".mp4", ".mov", ".m4v", ".webm"}
        video_uploads = [item for item in uploads if (item.content_type or "").startswith("video/") or Path(item.filename or "").suffix.lower() in video_extensions]
        if video_uploads:
            if len(video_uploads) != 1 or len(uploads) != 1:
                raise HTTPException(422, "Upload either one video or a set of photographs, not both")
            upload_file = video_uploads[0]
            suffix = Path(upload_file.filename or "video.mp4").suffix.lower()
            if suffix not in video_extensions:
                raise HTTPException(415, "Use MP4, MOV, M4V, or WebM video")
            if engine == "triposplat":
                raise HTTPException(422, "TripoSplat accepts one photograph, not video")
            upload_file.file.seek(0, os.SEEK_END)
            size = upload_file.file.tell()
            upload_file.file.seek(0)
            if size > MAX_VIDEO_UPLOAD:
                raise HTTPException(413, f"Video must be under {MAX_VIDEO_UPLOAD // 1024**2} MB")
            if size < 1024:
                raise HTTPException(415, "The uploaded video is empty or invalid")
            options = {"engine": "custom", "device": device, "resolution": FIXED_RESOLUTION, "depth_strength": depth_strength, "inputs": [], "video": {"file": "input-video" + suffix, "original_name": Path(upload_file.filename or "video").name[:100]}, "view_limit": view_limit, "focus_subject": FIXED_FOCUS_SUBJECT}
            try:
                return app.state.jobs.create([], [upload_file.filename or "video"], options, video_stream=upload_file.file)
            finally:
                await upload_file.close()
        if len(uploads) > MAX_IMAGES:
            raise HTTPException(422, f"Upload at most {MAX_IMAGES} images")
        if engine == "auto":
            engine = "triposplat" if len(uploads) == 1 else "custom"
        if engine == "triposplat" and len(uploads) != 1:
            raise HTTPException(422, "TripoSplat needs exactly one photograph")
        if engine == "custom" and len(uploads) < 12:
            raise HTTPException(422, "Use exactly one photograph for AI generation, or at least 12 overlapping photographs for measured reconstruction")
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
                        if engine == "triposplat" and rgba.getchannel("A").getextrema()[0] < 255:
                            clean = rgba
                        else:
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
        if engine == "triposplat" and not triposplat_ready():
            raise HTTPException(503, triposplat_status()[1] + " Run the TripoSplat setup and restart Studio.")
        options = {"engine": engine, "device": device, "resolution": FIXED_RESOLUTION, "depth_strength": depth_strength, "inputs": inputs, "video": None, "view_limit": view_limit, "focus_subject": FIXED_FOCUS_SUBJECT}
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

    @app.post("/api/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str):
        return app.state.jobs.retry(job_id, FIXED_RESOLUTION, FIXED_FOCUS_SUBJECT)

    @app.post("/api/jobs/{job_id}/clean", status_code=201)
    def clean(job_id: str):
        return app.state.jobs.clean(job_id)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def asset(job_id: str, filename: str):
        fixed = {"scene.ply", "scene.gsb", "scene.json", "dense.ply", "depth.png", "thumbnail.jpg", "worker.log"}
        is_mask = bool(re.fullmatch(r"object-mask-(?:[0-9]|1[0-5])\.png", filename))
        is_input = filename == "input.png" or bool(re.fullmatch(r"input_(?:[1-9]|[1-7][0-9])\.png", filename))
        if filename not in fixed and not is_input and not is_mask:
            raise HTTPException(404, "File not found")
        folder = app.state.jobs.path(job_id)
        job = app.state.jobs.read(job_id)
        if (filename.startswith("scene.") or filename == "dense.ply") and job["status"] != "completed":
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
