"""Pretrained-free multi-view reconstruction and adaptive Gaussian conversion.

COLMAP estimates cameras and dense geometry only from the uploaded photographs.
This module converts the fused oriented point cloud into portable 3D Gaussians;
it does not download or execute a learned checkpoint.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

from studio.config import COLMAP_ROOT
from studio.gaussians import validate

Progress = Callable[[int, str], None]
MIN_IMAGES = 12
MAX_VIDEO_FRAMES = 80


def find_colmap() -> Path | None:
    configured = os.environ.get("GSS_COLMAP", "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file():
            return candidate
    if COLMAP_ROOT.is_dir():
        for name in ("COLMAP.bat", "colmap.exe", "colmap"):
            matches = list(COLMAP_ROOT.rglob(name))
            if matches:
                return matches[0]
    found = shutil.which("colmap") or shutil.which("COLMAP.bat")
    return Path(found).resolve() if found else None


def engine_ready() -> bool:
    if find_colmap() is not None:
        return True
    try:
        import pycolmap
        return bool(getattr(pycolmap, "has_cuda", False))
    except Exception:
        return False


def _command(executable: Path, arguments: Iterable[str]) -> list[str]:
    args = [str(item) for item in arguments]
    if os.name == "nt" and executable.suffix.lower() in (".bat", ".cmd"):
        return ["cmd.exe", "/d", "/c", str(executable), *args]
    return [str(executable), *args]


def _run(executable: Path, *arguments: str) -> None:
    result = subprocess.run(
        _command(executable, arguments),
        check=False,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(
            f"COLMAP step '{arguments[0]}' failed with exit code {result.returncode}. "
            "Download the job log and check photo overlap, blur, reflections, and CUDA memory."
        )


def _sharpness(path: Path) -> float:
    with Image.open(path) as source:
        gray = np.asarray(source.convert("L").resize((320, 240), Image.Resampling.BILINEAR), np.float32)
    laplacian = -4 * gray
    laplacian[1:] += gray[:-1]
    laplacian[:-1] += gray[1:]
    laplacian[:, 1:] += gray[:, :-1]
    laplacian[:, :-1] += gray[:, 1:]
    return float(laplacian[2:-2, 2:-2].var())


def extract_video_frames(video_path: Path, output: Path, max_frames: int = MAX_VIDEO_FRAMES) -> list[Path]:
    """Decode a video and keep the sharpest frame in each time interval."""
    import imageio_ffmpeg

    raw = output.parent / "video-frames-raw"
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(video_path),
        "-t", "90", "-vf", "fps=2,scale=1600:-2:force_original_aspect_ratio=decrease",
        "-frames:v", "180", "-q:v", "2", str(raw / "candidate_%04d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError("Video decoding failed. Upload a standard MP4, MOV, M4V, or WebM file.")
    candidates = sorted(raw.glob("candidate_*.jpg"))
    if len(candidates) < MIN_IMAGES:
        raise RuntimeError(
            f"Video yielded only {len(candidates)} frames. Record at least 8 seconds while moving slowly through the scene."
        )
    count = min(max_frames, len(candidates))
    edges = np.linspace(0, len(candidates), count + 1, dtype=int)
    selected = []
    for start, stop in zip(edges[:-1], edges[1:]):
        group = candidates[start:max(stop, start + 1)]
        source = max(group, key=_sharpness)
        target = output / f"input_{len(selected):03d}.jpg"
        shutil.copy2(source, target)
        selected.append(target)
    return selected


def _prepare_images(paths: list[Path], output: Path, max_side: int) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (max_side, max_side), (127, 127, 127))
            canvas.paste(image, ((max_side - image.width) // 2, (max_side - image.height) // 2))
            target = output / f"{index:04d}.jpg"
            canvas.save(target, quality=95, subsampling=0)
            prepared.append(target)
    return prepared


def _largest_model(sparse: Path) -> Path:
    models = [path for path in sparse.iterdir() if path.is_dir() and (path / "images.bin").is_file()]
    if not models:
        raise RuntimeError(
            "Camera alignment failed. Use at least 12 sharp ordered photos, or a slow video, with 70-85% overlap; "
            "keep the subject, background, zoom, lighting, and focus unchanged."
        )
    return max(models, key=lambda path: (path / "images.bin").stat().st_size)


def _cli_model_stats(executable: Path, model: Path, work: Path) -> tuple[int, int]:
    text_model = work / "model-statistics"
    text_model.mkdir(exist_ok=True)
    _run(executable, "model_converter", "--input_path", str(model), "--output_path", str(text_model), "--output_type", "TXT")
    images_text = (text_model / "images.txt").read_text(encoding="utf-8")
    points_text = (text_model / "points3D.txt").read_text(encoding="utf-8")
    image_match = re.search(r"# Number of images:\s*(\d+)", images_text)
    point_match = re.search(r"# Number of points:\s*(\d+)", points_text)
    if not image_match or not point_match:
        raise RuntimeError("COLMAP model statistics could not be read.")
    return int(image_match.group(1)), int(point_match.group(1))


def _validate_registration(registered: int, total: int, sparse_points: int) -> dict:
    ratio = registered / max(total, 1)
    if registered < 8 or ratio < 0.55 or sparse_points < 500:
        raise RuntimeError(
            f"Capture rejected: COLMAP registered {registered}/{total} frames with {sparse_points} sparse points. "
            "Move slowly, keep 70-85% neighbouring overlap, and avoid moving objects, blur, zoom, or exposure changes."
        )
    return {"registered_images": registered, "registration_ratio": round(ratio, 4), "sparse_points": sparse_points}


def _reconstruct_cli(
    executable: Path,
    images: Path,
    work: Path,
    max_side: int,
    use_gpu: bool,
    progress: Progress,
) -> tuple[Path, str, dict]:
    database, sparse, dense = work / "database.db", work / "sparse", work / "dense"
    sparse.mkdir(parents=True, exist_ok=True)
    gpu = "1" if use_gpu else "0"

    progress(12, "Extracting photo features")
    _run(
        executable,
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(images),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.type", "SIFT",
        "--FeatureExtraction.use_gpu", gpu,
        "--FeatureExtraction.gpu_index", "0",
        "--FeatureExtraction.max_image_size", str(max_side),
    )
    progress(27, "Matching overlapping views")
    _run(
        executable,
        "exhaustive_matcher",
        "--database_path", str(database),
        "--FeatureMatching.use_gpu", gpu,
        "--FeatureMatching.gpu_index", "0",
    )
    progress(42, "Solving cameras and sparse geometry")
    _run(
        executable,
        "mapper",
        "--database_path", str(database),
        "--image_path", str(images),
        "--output_path", str(sparse),
    )
    model = _largest_model(sparse)
    registered, sparse_points = _cli_model_stats(executable, model, work)
    quality = _validate_registration(registered, len(list(images.glob("*.jpg"))), sparse_points)
    progress(55, "Undistorting registered photographs")
    _run(
        executable,
        "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(model),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_side),
    )
    if not use_gpu:
        raise RuntimeError("Dense custom reconstruction requires NVIDIA CUDA. Select NVIDIA CUDA and retry.")
    progress(64, "Estimating dense multi-view depth")
    _run(
        executable,
        "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", "0",
        "--PatchMatchStereo.max_image_size", str(max_side),
        "--PatchMatchStereo.geom_consistency", "1",
    )
    fused = dense / "fused.ply"
    progress(80, "Fusing consistent surfaces")
    _run(
        executable,
        "stereo_fusion",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", str(fused),
        "--StereoFusion.max_image_size", str(max_side),
    )
    if not fused.is_file():
        raise RuntimeError("Dense fusion completed without a point cloud. Capture more overlapping, textured views.")
    return fused, "COLMAP 4.2 CUDA command-line", quality


def _reconstruct_pycolmap(
    images: Path,
    work: Path,
    max_side: int,
    use_gpu: bool,
    progress: Progress,
) -> tuple[Path, str, dict]:
    import pycolmap

    if not use_gpu or not getattr(pycolmap, "has_cuda", False):
        raise RuntimeError(
            "No CUDA COLMAP backend found. Windows: rerun Setup NVIDIA Workstation.cmd. "
            "Kaggle/Linux: install pycolmap-cuda12==4.2.0."
        )
    database, sparse, dense = work / "database.db", work / "sparse", work / "dense"
    sparse.mkdir(parents=True, exist_ok=True)
    device = pycolmap.Device.cuda
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = max_side
    extraction_options.type = pycolmap.FeatureExtractorType.SIFT
    progress(12, "Extracting photo features")
    pycolmap.extract_features(
        database,
        images,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_options,
        device=device,
    )
    progress(27, "Matching overlapping views")
    pycolmap.match_exhaustive(database, device=device)
    progress(42, "Solving cameras and sparse geometry")
    maps = pycolmap.incremental_mapping(database, images, sparse)
    if not maps:
        raise RuntimeError("Camera alignment failed. Capture a slow video or 20-40 ordered photos with 70-85% overlap.")
    reconstruction = max(maps.values(), key=lambda item: item.num_reg_images())
    quality = _validate_registration(reconstruction.num_reg_images(), len(list(images.glob("*.jpg"))), reconstruction.num_points3D())
    model = sparse / "best"
    model.mkdir(exist_ok=True)
    reconstruction.write(model)
    progress(55, "Undistorting registered photographs")
    undistort_options = pycolmap.UndistortCameraOptions()
    undistort_options.max_image_size = max_side
    pycolmap.undistort_images(
        dense,
        model,
        images,
        undistort_options=undistort_options,
    )
    progress(64, "Estimating dense multi-view depth")
    patch_options = pycolmap.PatchMatchOptions()
    patch_options.gpu_index = "0"
    patch_options.max_image_size = max_side
    patch_options.geom_consistency = True
    pycolmap.patch_match_stereo(
        dense,
        options=patch_options,
    )
    fused = dense / "fused.ply"
    progress(80, "Fusing consistent surfaces")
    fusion_options = pycolmap.StereoFusionOptions()
    fusion_options.max_image_size = max_side
    pycolmap.stereo_fusion(
        fused,
        dense,
        input_type="geometric",
        output_type="PLY",
        options=fusion_options,
    )
    if not fused.is_file():
        raise RuntimeError("Dense fusion completed without a point cloud. Capture more overlapping, textured views.")
    return fused, f"PyCOLMAP {pycolmap.__version__} CUDA", quality


def _field(vertex, *names: str, default=None):
    available = set(vertex.data.dtype.names or ())
    for name in names:
        if name in available:
            return np.asarray(vertex[name])
    if default is not None:
        return default
    raise ValueError(f"Dense point cloud is missing fields: {', '.join(names)}")


def dense_cloud_to_gaussians(path: Path, max_gaussians: int = 1_250_000) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    total = len(vertex)
    if total < 10_000:
        raise ValueError(f"Dense reconstruction produced only {total:,} points. Add more sharp overlapping views.")
    stride = max(1, math.ceil(total / max_gaussians))
    indices = np.arange(0, total, stride, dtype=np.int64)[:max_gaussians]
    xyz = np.column_stack([_field(vertex, axis)[indices] for axis in ("x", "y", "z")]).astype(np.float32)
    normals = np.column_stack([
        _field(vertex, "nx", default=np.zeros(total))[indices],
        _field(vertex, "ny", default=np.zeros(total))[indices],
        _field(vertex, "nz", default=np.ones(total))[indices],
    ]).astype(np.float32)
    colors = np.column_stack([
        _field(vertex, "red", "r", default=np.full(total, 127))[indices],
        _field(vertex, "green", "g", default=np.full(total, 127))[indices],
        _field(vertex, "blue", "b", default=np.full(total, 127))[indices],
    ]).astype(np.float32)
    if colors.max(initial=0) > 1.5:
        colors /= 255.0

    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(normals).all(axis=1) & np.isfinite(colors).all(axis=1)
    xyz, normals, colors = xyz[finite], normals[finite], colors[finite]
    # COLMAP camera coordinates use +Y down and +Z forward; this viewer uses
    # +Y up and -Z forward. Keep the first reconstructed camera near the origin.
    xyz *= np.array([1, -1, -1], dtype=np.float32)
    normals *= np.array([1, -1, -1], dtype=np.float32)
    center = np.median(xyz, axis=0)
    radius = np.linalg.norm(xyz - center, axis=1)
    xyz, normals, colors = (array[radius <= np.percentile(radius, 99.5)] for array in (xyz, normals, colors))
    if len(xyz) < 100:
        raise ValueError("Dense reconstruction collapsed after outlier filtering.")

    neighbors = min(6, len(xyz))
    distances, _ = cKDTree(xyz).query(xyz, k=neighbors, workers=-1)
    local = np.median(distances[:, 1:], axis=1)
    positive = local[np.isfinite(local) & (local > 0)]
    if not len(positive):
        raise ValueError("Dense reconstruction points have no usable spacing.")
    median_spacing = float(np.median(positive))
    keep = np.isfinite(local) & (local > 0) & (local < median_spacing * 8)
    xyz, normals, colors, local = xyz[keep], normals[keep], colors[keep], local[keep]

    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(normal_length > 1e-6, normals / np.maximum(normal_length, 1e-6), np.array([0, 0, 1]))
    quaternion = np.column_stack((1 + normals[:, 2], -normals[:, 1], normals[:, 0], np.zeros(len(normals))))
    opposite = np.linalg.norm(quaternion, axis=1) < 1e-6
    quaternion[opposite] = (0, 1, 0, 0)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)

    low, high = np.percentile(local, [1, 99])
    tangent = np.clip(local * 0.72, max(low * 0.5, 1e-6), max(high * 1.2, 1e-5))
    gaussians = np.zeros((len(xyz), 16), dtype=np.float32)
    gaussians[:, :3] = xyz
    gaussians[:, 3] = 0.86
    gaussians[:, 4:7] = np.column_stack((tangent, tangent, tangent * 0.24))
    gaussians[:, 8:12] = quaternion
    gaussians[:, 12:15] = np.clip(colors, 0, 1)
    return validate(gaussians)


def reconstruct(
    directory: Path,
    input_paths: list[Path],
    resolution: int,
    use_gpu: bool,
    progress: Progress,
) -> tuple[np.ndarray, dict]:
    if len(input_paths) < MIN_IMAGES:
        raise RuntimeError(f"Custom reconstruction needs at least {MIN_IMAGES} overlapping views; 20-60 are recommended.")
    max_side = {384: 1200, 512: 1600, 768: 2000}[resolution]
    work = directory / "custom-work"
    images = work / "images"
    progress(7, f"Preparing {len(input_paths)} ordered photographs")
    _prepare_images(input_paths, images, max_side)
    executable = find_colmap()
    if executable is not None:
        fused, backend, quality = _reconstruct_cli(executable, images, work, max_side, use_gpu, progress)
    else:
        fused, backend, quality = _reconstruct_pycolmap(images, work, max_side, use_gpu, progress)
    progress(88, "Building adaptive oriented Gaussians")
    dense_points = len(PlyData.read(str(fused))["vertex"])
    gaussians = dense_cloud_to_gaussians(fused)
    shutil.copy2(fused, directory / "dense.ply")
    return gaussians, {
        "fov_y": 50.0,
        "image_size": [max_side, max_side],
        "engine": "Custom SfM + adaptive Gaussian splatting",
        "method": "custom",
        "device": "cuda" if use_gpu else "cpu",
        "backend": backend,
        "input_count": len(input_paths),
        "dense_points": dense_points,
        "dense_file": "dense.ply",
        **quality,
        "pretrained_weights": False,
        "licence": "Application MIT; COLMAP BSD-3-Clause. See docs/MODEL-LICENSES.md.",
        "limitation": "Measured per-scene reconstruction from registered views. Unseen, reflective, transparent, moving, textureless, or poorly matched regions remain incomplete; dense.ply is the raw COLMAP cloud.",
    }
