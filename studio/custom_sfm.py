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
import sys
import time
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
DENSE_TARGET_POINTS = 10_000
DENSE_MINIMUM_POINTS = 2_000
PATCH_MATCH_PROFILES = {
    1200: {"iterations": 3, "samples": 10, "cache_gb": 4, "max_sources": 8},
    1600: {"iterations": 4, "samples": 12, "cache_gb": 8, "max_sources": 12},
    2000: {"iterations": 5, "samples": 15, "cache_gb": 8, "max_sources": 16},
}
FUSION_PROFILES = (
    {
        "name": "strict-geometric",
        "input_type": "geometric",
        "min_num_pixels": 5,
        "max_reproj_error": 2.0,
        "max_depth_error": 0.01,
        "max_normal_error": 10.0,
    },
    {
        "name": "relaxed-geometric",
        "input_type": "geometric",
        "min_num_pixels": 3,
        "max_reproj_error": 3.0,
        "max_depth_error": 0.02,
        "max_normal_error": 20.0,
    },
    {
        "name": "photometric-recovery",
        "input_type": "photometric",
        "min_num_pixels": 3,
        "max_reproj_error": 3.0,
        "max_depth_error": 0.03,
        "max_normal_error": 25.0,
    },
)


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


def _patch_match_profile(max_side: int) -> dict:
    return PATCH_MATCH_PROFILES[max_side]


def _gpu_indices() -> str:
    configured = os.environ.get("GSS_GPU_INDEX", "").strip()
    if configured:
        return configured
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        count = len([line for line in probe.stdout.splitlines() if line.strip().lower().startswith("gpu ")])
        if probe.returncode == 0 and count:
            return ",".join(str(index) for index in range(count))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "0"


def _limit_patch_match_sources(workspace: Path, maximum: int) -> int:
    """Bound automatically selected source views per reference image."""
    config = workspace / "stereo" / "patch-match.cfg"
    if not config.is_file():
        raise RuntimeError("COLMAP did not create stereo/patch-match.cfg during undistortion.")
    lines = config.read_text(encoding="utf-8").splitlines()
    replacements = 0
    for index, line in enumerate(lines):
        if re.fullmatch(r"__auto__(?:\s*,\s*\d+)?", line.strip()):
            lines[index] = f"__auto__, {maximum}"
            replacements += 1
    if not replacements:
        raise RuntimeError("COLMAP stereo source-view configuration is invalid.")
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return replacements


def _run_dense_process(command: list[str], progress: Progress) -> None:
    """Run PatchMatch out-of-process with a visible heartbeat and hard timeout."""
    timeout_minutes = max(5, int(os.environ.get("GSS_DENSE_TIMEOUT_MINUTES", "60")))
    process = subprocess.Popen(
        command,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    started = time.monotonic()
    while True:
        try:
            returncode = process.wait(timeout=15)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_minutes * 60:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError(
                    f"Dense stereo exceeded {timeout_minutes} minutes and was stopped. "
                    "Retry with Quick - 1200 px and a 20-30 second video."
                )
            percent = min(77, 64 + max(1, int(elapsed // 90)))
            progress(percent, f"Dense CUDA stereo is running ({int(elapsed // 60)} min elapsed)")
    if returncode:
        raise RuntimeError(
            f"COLMAP dense stereo failed with exit code {returncode}. "
            "Retry with Quick - 1200 px; if it fails again, download the job log."
        )


def _ply_point_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(PlyData.read(str(path))["vertex"])
    except Exception:
        return 0


def _run_fusion_recovery(dense: Path, fuse, progress: Progress) -> tuple[Path, dict]:
    """Prefer consistent geometry, then recover a sparse result without rerunning PatchMatch."""
    best_path, best_count, best_name = None, 0, None
    failures = []
    for index, profile in enumerate(FUSION_PROFILES):
        if index == 0:
            message = "Fusing consistent surfaces"
        elif index == 1:
            message = f"Dense cloud has only {best_count:,} points; relaxing fusion confidence"
        else:
            message = f"Dense cloud has only {best_count:,} points; trying photometric recovery"
        progress(80 + index * 3, message)
        target = dense / f"fused-{profile['name']}.ply"
        try:
            fuse(target, profile)
        except Exception as exc:
            failures.append(f"{profile['name']}: {exc}")
            continue
        count = _ply_point_count(target)
        if count > best_count:
            best_path, best_count, best_name = target, count, profile["name"]
        if count >= DENSE_TARGET_POINTS:
            break
    if best_path is None:
        detail = "; ".join(failures) if failures else "no point cloud file was written"
        raise RuntimeError(f"Dense fusion failed after automatic recovery: {detail}")
    if best_count < DENSE_MINIMUM_POINTS:
        raise RuntimeError(
            f"Dense reconstruction produced only {best_count:,} points after automatic recovery. "
            "Record a slower orbit with the flower and background completely still."
        )
    return best_path, {
        "fusion_profile": best_name,
        "fusion_recovered": best_name != "strict-geometric",
        "fusion_points": best_count,
    }


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


def _focus_from_camera_rays(centers, directions) -> dict | None:
    centers = np.asarray(centers, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if centers.shape != directions.shape or centers.ndim != 2 or centers.shape[0] < 3 or centers.shape[1] != 3:
        return None
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    good = np.isfinite(centers).all(axis=1) & np.isfinite(directions).all(axis=1) & (lengths[:, 0] > 1e-8)
    centers, directions = centers[good], directions[good] / lengths[good]
    if len(centers) < 3:
        return None
    projectors = np.eye(3)[None, :, :] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    if not np.isfinite(system).all() or np.linalg.cond(system) > 1e5:
        return None
    target = np.linalg.solve(system, np.einsum("nij,nj->i", projectors, centers))
    camera_distance = float(np.median(np.linalg.norm(centers - target, axis=1)))
    if not np.isfinite(target).all() or not np.isfinite(camera_distance) or camera_distance <= 0:
        return None
    return {"target": target, "camera_distance": camera_distance, "source_camera": centers[0]}


def _rotation_from_qvec(qvec) -> np.ndarray:
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def _cli_camera_focus(work: Path) -> dict | None:
    text = (work / "model-statistics" / "images.txt").read_text(encoding="utf-8")
    centers, directions = [], []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 10 or not fields[9].lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        rotation = _rotation_from_qvec([float(value) for value in fields[1:5]])
        translation = np.array([float(value) for value in fields[5:8]])
        centers.append(-rotation.T @ translation)
        directions.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
    return _focus_from_camera_rays(centers, directions)


def _pycolmap_camera_focus(reconstruction) -> dict | None:
    images = [image for image in reconstruction.images.values() if image.has_pose]
    return _focus_from_camera_rays(
        [image.projection_center() for image in images],
        [image.viewing_direction() for image in images],
    )


def _reconstruct_cli(
    executable: Path,
    images: Path,
    work: Path,
    max_side: int,
    use_gpu: bool,
    progress: Progress,
) -> tuple[Path, str, dict, dict | None]:
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
    focus = _cli_camera_focus(work)
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
    profile = _patch_match_profile(max_side)
    dense_gpu_index = _gpu_indices()
    _limit_patch_match_sources(dense, profile["max_sources"])
    progress(64, f"Starting dense CUDA stereo ({profile['iterations']} iterations)")
    _run_dense_process(
        _command(executable, (
        "patch_match_stereo",
        "--workspace_path", str(dense),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.gpu_index", dense_gpu_index,
        "--PatchMatchStereo.max_image_size", str(max_side),
        "--PatchMatchStereo.num_iterations", str(profile["iterations"]),
        "--PatchMatchStereo.num_samples", str(profile["samples"]),
        "--PatchMatchStereo.cache_size", str(profile["cache_gb"]),
        "--PatchMatchStereo.geom_consistency", "1",
        )),
        progress,
    )
    def fuse(target: Path, fusion: dict) -> None:
        _run(
            executable,
            "stereo_fusion",
            "--workspace_path", str(dense),
            "--workspace_format", "COLMAP",
            "--input_type", fusion["input_type"],
            "--output_path", str(target),
            "--StereoFusion.max_image_size", str(max_side),
            "--StereoFusion.min_num_pixels", str(fusion["min_num_pixels"]),
            "--StereoFusion.max_reproj_error", str(fusion["max_reproj_error"]),
            "--StereoFusion.max_depth_error", str(fusion["max_depth_error"]),
            "--StereoFusion.max_normal_error", str(fusion["max_normal_error"]),
        )

    fused, fusion_stats = _run_fusion_recovery(dense, fuse, progress)
    quality.update(fusion_stats)
    return fused, "COLMAP 4.2 CUDA command-line", quality, focus


def _reconstruct_pycolmap(
    images: Path,
    work: Path,
    max_side: int,
    use_gpu: bool,
    progress: Progress,
) -> tuple[Path, str, dict, dict | None]:
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
    focus = _pycolmap_camera_focus(reconstruction)
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
    profile = _patch_match_profile(max_side)
    dense_gpu_index = _gpu_indices()
    _limit_patch_match_sources(dense, profile["max_sources"])
    progress(64, f"Starting dense CUDA stereo ({profile['iterations']} iterations)")
    _run_dense_process(
        [
            sys.executable,
            "-m",
            "studio.patchmatch_worker",
            str(dense),
            str(max_side),
            str(profile["iterations"]),
            str(profile["samples"]),
            str(profile["cache_gb"]),
            dense_gpu_index,
        ],
        progress,
    )
    def fuse(target: Path, fusion: dict) -> None:
        fusion_options = pycolmap.StereoFusionOptions()
        fusion_options.max_image_size = max_side
        fusion_options.min_num_pixels = fusion["min_num_pixels"]
        fusion_options.max_reproj_error = fusion["max_reproj_error"]
        fusion_options.max_depth_error = fusion["max_depth_error"]
        fusion_options.max_normal_error = fusion["max_normal_error"]
        pycolmap.stereo_fusion(
            target,
            dense,
            input_type=fusion["input_type"],
            output_type="PLY",
            options=fusion_options,
        )

    fused, fusion_stats = _run_fusion_recovery(dense, fuse, progress)
    quality.update(fusion_stats)
    return fused, f"PyCOLMAP {pycolmap.__version__} CUDA", quality, focus


def _field(vertex, *names: str, default=None):
    available = set(vertex.data.dtype.names or ())
    for name in names:
        if name in available:
            return np.asarray(vertex[name])
    if default is not None:
        return default
    raise ValueError(f"Dense point cloud is missing fields: {', '.join(names)}")


def _remove_small_components(xyz, normals, colors, local, spacing: float):
    """Discard disconnected voxel islands while preserving substantial surfaces."""
    keys = np.floor((xyz - xyz.min(axis=0)) / max(spacing * 4, 1e-6)).astype(np.int64)
    voxels, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    if len(voxels) < 2:
        return xyz, normals, colors, local, 0
    lookup = {tuple(key): index for index, key in enumerate(voxels)}
    parent = np.arange(len(voxels))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = root(left), root(right)
        if left != right:
            parent[right] = left

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0)
    ]
    for index, key in enumerate(voxels):
        base = tuple(key)
        for offset in offsets:
            neighbor = lookup.get(tuple(base[axis] + offset[axis] for axis in range(3)))
            if neighbor is not None:
                union(index, neighbor)
    roots = np.array([root(index) for index in range(len(voxels))])
    component_sizes = np.bincount(roots, weights=counts, minlength=len(voxels))
    minimum = max(100, int(component_sizes.max(initial=0) * 0.02))
    keep = component_sizes[roots[inverse]] >= minimum
    if keep.sum() < max(800, int(len(xyz) * 0.35)):
        return xyz, normals, colors, local, 0
    removed = int(len(xyz) - keep.sum())
    return xyz[keep], normals[keep], colors[keep], local[keep], removed


def dense_cloud_to_gaussians(
    path: Path,
    max_gaussians: int = 1_250_000,
    focus: dict | None = None,
    focus_stats: dict | None = None,
) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    total = len(vertex)
    if total < DENSE_MINIMUM_POINTS:
        raise ValueError(
            f"Dense reconstruction produced only {total:,} points after automatic recovery. "
            "Add more sharp overlapping views."
        )
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
    focus_applied = False
    focus_radius = None
    if focus is not None:
        focus_target = np.asarray(focus["target"], dtype=np.float32) * np.array([1, -1, -1], dtype=np.float32)
        focus_radius = float(focus["camera_distance"] * 0.42)
        subject_distance = np.linalg.norm(xyz - focus_target, axis=1)
        subject_keep = subject_distance <= focus_radius
        minimum = max(800, int(len(xyz) * 0.05))
        if subject_keep.sum() >= minimum:
            xyz, normals, colors = xyz[subject_keep], normals[subject_keep], colors[subject_keep]
            focus_applied = True
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
    removed_components = 0
    if focus_applied and len(xyz) <= 400_000:
        xyz, normals, colors, local, removed_components = _remove_small_components(
            xyz, normals, colors, local, median_spacing
        )
    if focus_stats is not None:
        focus_stats.update({
            "subject_focus_requested": focus is not None,
            "subject_focus_applied": focus_applied,
            "subject_focus_radius": focus_radius,
            "removed_fragment_points": removed_components,
            "focused_points": int(len(xyz)),
        })

    normal_length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(normal_length > 1e-6, normals / np.maximum(normal_length, 1e-6), np.array([0, 0, 1]))
    quaternion = np.column_stack((1 + normals[:, 2], -normals[:, 1], normals[:, 0], np.zeros(len(normals))))
    opposite = np.linalg.norm(quaternion, axis=1) < 1e-6
    quaternion[opposite] = (0, 1, 0, 0)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)

    low, high = np.percentile(local, [1, 99])
    coverage_boost = float(np.clip((DENSE_TARGET_POINTS / max(len(xyz), 1)) ** 0.25, 1.0, 1.6))
    tangent = np.clip(
        local * 0.72 * coverage_boost,
        max(low * 0.5, 1e-6),
        max(high * 1.2 * coverage_boost, 1e-5),
    )
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
    focus_subject: bool = True,
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
        fused, backend, quality, focus = _reconstruct_cli(executable, images, work, max_side, use_gpu, progress)
    else:
        fused, backend, quality, focus = _reconstruct_pycolmap(images, work, max_side, use_gpu, progress)
    progress(88, "Filtering background fragments and building Gaussians" if focus_subject else "Building adaptive oriented Gaussians")
    dense_points = len(PlyData.read(str(fused))["vertex"])
    focus_stats = {}
    gaussians = dense_cloud_to_gaussians(
        fused,
        focus=focus if focus_subject else None,
        focus_stats=focus_stats,
    )
    shutil.copy2(fused, directory / "dense.ply")
    viewer_flip = np.array([1, -1, -1], dtype=np.float64)
    source_camera = (np.asarray(focus["source_camera"]) * viewer_flip).tolist() if focus else [0, 0, 0]
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
        "source_camera": source_camera,
        **focus_stats,
        **quality,
        "pretrained_weights": False,
        "licence": "Application MIT; COLMAP BSD-3-Clause. See docs/MODEL-LICENSES.md.",
        "limitation": "Measured per-scene reconstruction from registered views. Central-subject focus removes distant fragments but cannot recover unseen or moving petals; dense.ply retains the uncropped COLMAP cloud.",
    }
