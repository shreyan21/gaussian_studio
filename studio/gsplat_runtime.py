"""Optional true 3D Gaussian-splat training on registered COLMAP views.

gsplat is imported lazily so the existing COLMAP dense path remains available on
machines where the CUDA extension is not installed.  The training loop uses only
the public gsplat rasterization and densification APIs.
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from studio.gaussians import validate

Progress = Callable[[int, str], None]

TRAINING_PROFILES = {
    1200: {"image_side": 720, "steps": 7_000, "max_splats": 500_000},
    1600: {"image_side": 900, "steps": 11_000, "max_splats": 750_000},
    2000: {"image_side": 1080, "steps": 15_000, "max_splats": 1_000_000},
}
SUBJECT_CROP_RATIO = 0.68
MIN_SUBJECT_CROP_RATIO = 0.48
MAX_SUBJECT_CROP_RATIO = 0.86
MAX_TRAINING_SCALE_RATIO = 6.0
ANISOTROPY_PENALTY_START_RATIO = 4.5
MIN_EXPORT_OPACITY = 0.04
FOCUSED_MIN_EXPORT_OPACITY = 0.02
MIN_VALID_EXPORT_OPACITY = 0.0051
EXPORT_SCALE_MEDIAN_MULTIPLIER = 6.0
EXPORT_SCALE_PERCENTILE = 99.0
EXPORT_SCENE_SCALE_RATIO = 0.05
FOCUS_EXPORT_RADIUS_RATIO = 0.72
FOCUS_MASK_SUPPORT_THRESHOLD = 0.55
MIN_EXPORT_GAUSSIANS = 60_000
MIN_EXPORT_RETENTION_RATIO = 0.18


def gsplat_ready() -> bool:
    if os.environ.get("GSS_DISABLE_GSPLAT", "").strip() == "1":
        return False
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("gsplat") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _training_profile(max_side: int) -> dict:
    profile = TRAINING_PROFILES[max_side].copy()
    configured = os.environ.get("GSS_GSPLAT_STEPS", "").strip()
    if configured:
        profile["steps"] = max(500, int(configured))
    return profile


def _image_pose_matrix(image) -> np.ndarray:
    pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :4] = pose
    return matrix


def _subject_crop_bounds(
    width: int,
    height: int,
    K: np.ndarray,
    viewmat: np.ndarray,
    target: np.ndarray,
    ratio: float = SUBJECT_CROP_RATIO,
) -> tuple[int, int, int, int]:
    """Return an in-frame crop centered on the reconstructed subject projection."""
    crop_width = min(width, max(32, round(width * ratio)))
    crop_height = min(height, max(32, round(height * ratio)))
    camera_point = viewmat[:3, :3] @ np.asarray(target, np.float32) + viewmat[:3, 3]
    if np.isfinite(camera_point).all() and camera_point[2] > 1e-6:
        center_x = float(K[0, 0] * camera_point[0] / camera_point[2] + K[0, 2])
        center_y = float(K[1, 1] * camera_point[1] / camera_point[2] + K[1, 2])
    else:
        center_x, center_y = width / 2, height / 2
    left = int(np.clip(round(center_x - crop_width / 2), 0, width - crop_width))
    top = int(np.clip(round(center_y - crop_height / 2), 0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def _adaptive_subject_crop_ratio(camera_center: np.ndarray, focus: dict) -> float:
    """Keep a reconstructed subject at a similar pixel scale as camera distance changes."""
    distance = float(
        np.linalg.norm(
            np.asarray(camera_center, np.float32)
            - np.asarray(focus["target"], np.float32)
        )
    )
    reference = float(focus["camera_distance"])
    if not np.isfinite(distance) or not np.isfinite(reference) or distance <= 1e-6 or reference <= 1e-6:
        return SUBJECT_CROP_RATIO
    return float(
        np.clip(
            SUBJECT_CROP_RATIO * reference / distance,
            MIN_SUBJECT_CROP_RATIO,
            MAX_SUBJECT_CROP_RATIO,
        )
    )


def _automatic_subject_mask(pixels: np.ndarray) -> tuple[np.ndarray, str]:
    """Segment a centered object without pretrained weights.

    GrabCut receives a conservative central-object prior rather than a fixed
    rectangle.  That preserves thin leaves while declaring the image border as
    definite background.  A bounded geometric prior is retained as a safe
    fallback for unusually flat or low-contrast frames.
    """
    height, width = pixels.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    nx = (xx + 0.5 - width * 0.5) / max(width * 0.5, 1)
    ny = (yy + 0.5 - height * 0.53) / max(height * 0.5, 1)
    broad = (nx / 0.91) ** 2 + (ny / 0.98) ** 2 <= 1.0
    core = (nx / 0.28) ** 2 + (ny / 0.63) ** 2 <= 1.0
    fallback = ((nx / 0.78) ** 2 + (ny / 0.93) ** 2 <= 1.0).astype(np.uint8)
    try:
        import cv2

        labels = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        labels[broad] = cv2.GC_PR_FGD
        labels[core] = cv2.GC_FGD
        border_x, border_y = max(2, width // 40), max(2, height // 40)
        labels[:border_y] = cv2.GC_BGD
        labels[-border_y:] = cv2.GC_BGD
        labels[:, :border_x] = cv2.GC_BGD
        labels[:, -border_x:] = cv2.GC_BGD
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            np.ascontiguousarray(pixels),
            labels,
            None,
            background,
            foreground,
            5,
            cv2.GC_INIT_WITH_MASK,
        )
        mask = np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask &= broad.astype(np.uint8)
        ratio = float(mask.mean())
        if 0.04 <= ratio <= 0.82:
            return mask, "grabcut-central-prior"
    except Exception:
        pass
    return fallback, "geometric-central-prior"


def _project_mask_support(means: np.ndarray, views: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score each 3D mean by agreement with foreground masks across cameras."""
    hits = np.zeros(len(means), dtype=np.uint16)
    seen = np.zeros(len(means), dtype=np.uint16)
    for view in views:
        mask = view.get("mask")
        if mask is None:
            continue
        viewmat = np.asarray(view["viewmat"], dtype=np.float32)
        K = np.asarray(view["K"], dtype=np.float32)
        camera = means @ viewmat[:3, :3].T + viewmat[:3, 3]
        depth = camera[:, 2]
        good = np.isfinite(camera).all(axis=1) & (depth > 1e-5)
        safe_depth = np.where(good, depth, 1.0)
        x = np.rint(K[0, 0] * camera[:, 0] / safe_depth + K[0, 2]).astype(np.int64)
        y = np.rint(K[1, 1] * camera[:, 1] / safe_depth + K[1, 2]).astype(np.int64)
        height, width = mask.shape
        good &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        indices = np.flatnonzero(good)
        seen[indices] += 1
        hits[indices] += (mask[y[indices], x[indices]] >= 0.5).astype(np.uint16)
    support = np.divide(hits, np.maximum(seen, 1), dtype=np.float32)
    support[seen < 2] = 0
    return support, hits, seen


def _load_training_views(workspace: Path, image_side: int, focus: dict | None):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(workspace / "sparse")
    registered = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: image.name,
    )
    if len(registered) < 8:
        raise RuntimeError("True 3DGS training needs at least eight registered COLMAP cameras.")

    views = []
    crop_ratios = []
    foreground_ratios = []
    mask_backends = []
    for image in registered:
        camera = reconstruction.cameras[image.camera_id]
        path = workspace / "images" / image.name
        if not path.is_file():
            raise RuntimeError(f"COLMAP undistorted image is missing: {image.name}")
        with Image.open(path) as source:
            frame = source.convert("RGB")
        width, height = frame.size
        scale_x, scale_y = width / camera.width, height / camera.height
        K = np.array(
            [
                [camera.focal_length_x * scale_x, 0, camera.principal_point_x * scale_x],
                [0, camera.focal_length_y * scale_y, camera.principal_point_y * scale_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        viewmat = _image_pose_matrix(image)
        if focus is not None:
            crop_ratio = _adaptive_subject_crop_ratio(image.projection_center(), focus)
            left, top, right, bottom = _subject_crop_bounds(
                width,
                height,
                K,
                viewmat,
                np.asarray(focus["target"], np.float32),
                ratio=crop_ratio,
            )
            crop_ratios.append(crop_ratio)
            frame = frame.crop((left, top, right, bottom))
            K[0, 2] -= left
            K[1, 2] -= top
            width, height = frame.size
        resize = min(1.0, image_side / max(width, height))
        output_size = (max(16, round(width * resize)), max(16, round(height * resize)))
        if output_size != frame.size:
            frame = frame.resize(output_size, Image.Resampling.LANCZOS)
            K[0, :] *= output_size[0] / width
            K[1, :] *= output_size[1] / height
        pixels = np.asarray(frame, dtype=np.uint8).copy()
        mask = None
        if focus is not None:
            mask, mask_backend = _automatic_subject_mask(pixels)
            foreground_ratios.append(float(mask.mean()))
            mask_backends.append(mask_backend)
        views.append(
            {
                "name": image.name,
                "pixels": pixels,
                "mask": mask,
                "K": K,
                "viewmat": viewmat,
            }
        )

    points = list(reconstruction.points3D.values())
    xyz = np.asarray([point.xyz for point in points], dtype=np.float32)
    colors = np.asarray([point.color for point in points], dtype=np.float32) / 255.0
    errors = np.asarray([point.error for point in points], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(colors).all(axis=1) & np.isfinite(errors)
    if finite.sum() >= 500:
        cutoff = np.percentile(errors[finite], 97)
        finite &= errors <= cutoff
    xyz, colors = xyz[finite], np.clip(colors[finite], 1 / 255, 254 / 255)
    if len(xyz) < 500:
        raise RuntimeError(f"COLMAP produced only {len(xyz):,} usable sparse seeds for 3DGS training.")
    seed_points_before_focus = len(xyz)
    focus_seed_candidates = 0
    if focus is not None:
        target = np.asarray(focus["target"], np.float32)
        focused = np.linalg.norm(xyz - target, axis=1) <= float(focus["camera_distance"] * 0.65)
        focus_seed_candidates = int(np.count_nonzero(focused))
        # Do not discard the other registered COLMAP points before optimization.
        # Sparse foregrounds (flowers, foliage, fur) otherwise start from only a
        # few hundred Gaussians, which expand into long translucent sheets.  The
        # projected image crop focuses the loss and export performs the final
        # spatial subject filter after the geometry has converged.

    centers = np.asarray([image.projection_center() for image in registered], dtype=np.float32)
    scene_center = np.mean(centers, axis=0)
    scene_scale = max(float(np.linalg.norm(centers - scene_center, axis=1).max()), 1e-3)
    front_height, front_width = views[0]["pixels"].shape[:2]
    front_fov_y = math.degrees(2.0 * math.atan(front_height / (2.0 * float(views[0]["K"][1, 1]))))
    return views, xyz, colors, scene_scale, {
        "training_subject_crop_ratio": round(float(np.median(crop_ratios)), 4) if crop_ratios else 1.0,
        "training_subject_crop_ratio_range": (
            [round(float(min(crop_ratios)), 4), round(float(max(crop_ratios)), 4)]
            if crop_ratios else [1.0, 1.0]
        ),
        "fov_y": round(front_fov_y, 4),
        "image_size": [front_width, front_height],
        "seed_points_before_focus": seed_points_before_focus,
        "focus_seed_candidates": focus_seed_candidates,
        "seed_focus_applied": False,
        "foreground_mask_backend": (
            max(set(mask_backends), key=mask_backends.count) if mask_backends else None
        ),
        "training_foreground_ratio": (
            round(float(np.median(foreground_ratios)), 4) if foreground_ratios else 1.0
        ),
        "training_foreground_ratio_range": (
            [round(float(min(foreground_ratios)), 4), round(float(max(foreground_ratios)), 4)]
            if foreground_ratios else [1.0, 1.0]
        ),
    }


def _ssim(prediction, target):
    import torch.nn.functional as F

    prediction = prediction.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    mu_x = F.avg_pool2d(prediction, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(prediction * prediction, 11, stride=1, padding=5) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, 11, stride=1, padding=5) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def _clamp_log_scale_anisotropy_(log_scales, maximum_ratio: float = MAX_TRAINING_SCALE_RATIO) -> None:
    """Bound every Gaussian axis ratio while preserving its overall footprint."""
    import torch

    if maximum_ratio <= 1:
        raise ValueError("maximum_ratio must be greater than one")
    with torch.no_grad():
        lower = log_scales.amin(dim=1, keepdim=True)
        upper = log_scales.amax(dim=1, keepdim=True)
        midpoint = (lower + upper) * 0.5
        half_span = 0.5 * math.log(maximum_ratio)
        log_scales.clamp_(min=midpoint - half_span, max=midpoint + half_span)


def _initial_parameters(xyz: np.ndarray, colors: np.ndarray, scene_scale: float, device: str):
    import torch

    neighbors = min(4, len(xyz))
    distances, _ = cKDTree(xyz).query(xyz, k=neighbors, workers=-1)
    local = np.mean(np.square(distances[:, 1:]), axis=1)
    fallback = max(float(np.median(local[local > 0])) if np.any(local > 0) else scene_scale * 1e-4, 1e-8)
    local = np.sqrt(np.maximum(local, fallback)).astype(np.float32)
    typical = float(np.median(local[np.isfinite(local) & (local > 0)]))
    local = np.minimum(local, max(typical * 4.0, 1e-7))
    tensors = {
        "means": torch.from_numpy(xyz),
        "scales": torch.from_numpy(np.log(local)[:, None].repeat(3, axis=1)),
        # Random initial axes break the symmetry of initially isotropic splats;
        # rasterization normalizes these quaternions internally.
        "quats": torch.rand((len(xyz), 4), dtype=torch.float32),
        "opacities": torch.full((len(xyz),), torch.logit(torch.tensor(0.1)).item()),
        "colors": torch.from_numpy(np.log(colors / (1 - colors))),
    }
    splats = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value.to(device=device, dtype=torch.float32)) for name, value in tensors.items()}
    )
    rates = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": rate, "name": name}], eps=1e-15)
        for name, rate in rates.items()
    }
    return splats, optimizers


def _viewer_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Left-compose COLMAP-world rotations with a 180 degree X rotation."""
    w, x, y, z = quaternions.T
    return np.column_stack((-x, w, -z, y)).astype(np.float32)


def _select_export_indices(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    focus: dict | None,
    scene_scale: float | None,
    maximum: int,
    subject_support: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, dict]:
    """Select useful splats without collapsing a trained subject to a sparse shell."""
    input_gaussians = len(means)
    finite = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(opacities)
    )
    valid = finite & (opacities >= MIN_VALID_EXPORT_OPACITY) & (scales > 0).all(axis=1)
    scale_limit = None
    maximum_scale = np.max(scales, axis=1)
    minimum_scale = np.min(scales, axis=1)
    anisotropy = maximum_scale / np.maximum(minimum_scale, 1e-8)
    valid_scales = maximum_scale[valid]
    scale_keep = valid.copy()
    if len(valid_scales):
        robust_limit = min(
            float(np.median(valid_scales) * EXPORT_SCALE_MEDIAN_MULTIPLIER),
            float(np.percentile(valid_scales, EXPORT_SCALE_PERCENTILE)),
        )
        scale_limit = (
            robust_limit
            if scene_scale is None
            else min(robust_limit, scene_scale * EXPORT_SCENE_SCALE_RATIO)
        )
        scale_keep &= maximum_scale <= scale_limit
        scale_keep &= anisotropy <= MAX_TRAINING_SCALE_RATIO
    candidate = scale_keep.copy()
    focus_applied = False
    mask_focus_count = None
    if focus is not None:
        target = np.asarray(focus["target"], dtype=np.float32)
        distance = np.linalg.norm(means - target, axis=1)
        focused = distance <= float(focus["camera_distance"] * FOCUS_EXPORT_RADIUS_RATIO)
        if subject_support is not None:
            if len(subject_support) != input_gaussians:
                raise ValueError("subject_support must match the Gaussian count")
            mask_focused = np.asarray(subject_support) >= FOCUS_MASK_SUPPORT_THRESHOLD
            mask_focus_count = int(np.count_nonzero(candidate & mask_focused))
            focused &= mask_focused
        if np.count_nonzero(candidate & focused) >= max(1_500, int(np.count_nonzero(candidate) * 0.03)):
            candidate &= focused
            focus_applied = True
    opacity_minimum = FOCUSED_MIN_EXPORT_OPACITY if focus is not None else MIN_EXPORT_OPACITY
    keep = candidate & (opacities >= opacity_minimum)
    retention_floor = 0
    if input_gaussians >= MIN_EXPORT_GAUSSIANS:
        retention_floor = min(
            int(np.count_nonzero(candidate)),
            max(MIN_EXPORT_GAUSSIANS, int(math.ceil(input_gaussians * MIN_EXPORT_RETENTION_RATIO))),
        )
        if np.count_nonzero(keep) < retention_floor:
            eligible = np.flatnonzero(candidate)
            selected = eligible[
                np.argpartition(opacities[eligible], -retention_floor)[-retention_floor:]
            ]
            keep[selected] = True
    indices = np.flatnonzero(keep)
    if len(indices) > maximum:
        indices = indices[np.argpartition(opacities[indices], -maximum)[-maximum:]]
    return indices, focus_applied, {
        "optimized_gaussians": input_gaussians,
        "exported_gaussians": len(indices),
        "export_removed_gaussians": input_gaussians - len(indices),
        "export_valid_gaussians": int(np.count_nonzero(valid)),
        "export_scale_gaussians": int(np.count_nonzero(scale_keep)),
        "export_focus_gaussians": int(np.count_nonzero(candidate)),
        "export_confident_gaussians": int(np.count_nonzero(candidate & (opacities >= opacity_minimum))),
        "export_retention_floor": retention_floor,
        "export_scale_limit": round(float(scale_limit), 7) if scale_limit is not None else None,
        "export_opacity_minimum": opacity_minimum,
        "export_focus_radius_ratio": FOCUS_EXPORT_RADIUS_RATIO if focus is not None else None,
        "export_mask_support_threshold": FOCUS_MASK_SUPPORT_THRESHOLD if subject_support is not None else None,
        "export_mask_supported_gaussians": mask_focus_count,
        "export_anisotropy_limit": MAX_TRAINING_SCALE_RATIO,
    }


def _export_arrays(
    splats,
    focus: dict | None,
    scene_scale: float | None = None,
    maximum: int = 1_250_000,
    export_stats: dict | None = None,
    subject_support: np.ndarray | None = None,
):
    import torch

    means = splats["means"].detach().cpu().numpy().astype(np.float32)
    scales = torch.exp(splats["scales"]).detach().cpu().numpy().astype(np.float32)
    quats = splats["quats"].detach().cpu().numpy().astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacities = torch.sigmoid(splats["opacities"]).detach().cpu().numpy().astype(np.float32)
    colors = torch.sigmoid(splats["colors"]).detach().cpu().numpy().astype(np.float32)
    finite_attributes = np.isfinite(quats).all(axis=1) & np.isfinite(colors).all(axis=1)
    safe_opacities = np.where(finite_attributes, opacities, -np.inf)
    input_gaussians = len(means)
    indices, focus_applied, filter_stats = _select_export_indices(
        means, scales, safe_opacities, focus, scene_scale, maximum, subject_support
    )
    means, scales, quats, opacities, colors = (
        array[indices] for array in (means, scales, quats, opacities, colors)
    )
    means *= np.array([1, -1, -1], dtype=np.float32)
    quats = _viewer_quaternions(quats)
    gaussians = np.zeros((len(means), 16), dtype=np.float32)
    gaussians[:, :3] = means
    gaussians[:, 3] = np.clip(opacities, 0.01, 0.999)
    gaussians[:, 4:7] = np.maximum(scales, 1e-7)
    gaussians[:, 8:12] = quats
    gaussians[:, 12:15] = np.clip(colors, 0, 1)
    gaussians = validate(gaussians)
    if export_stats is not None:
        filter_stats["exported_gaussians"] = len(gaussians)
        filter_stats["export_removed_gaussians"] = input_gaussians - len(gaussians)
        export_stats.update(filter_stats)
    return gaussians, focus_applied


def write_support_cloud(path: Path, gaussians: np.ndarray) -> None:
    data = np.zeros(
        len(gaussians),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    data["x"], data["y"], data["z"] = gaussians[:, 0], gaussians[:, 1], gaussians[:, 2]
    rgb = np.uint8(np.clip(gaussians[:, 12:15] * 255, 0, 255))
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def train_gaussian_scene(
    workspace: Path,
    max_side: int,
    focus: dict | None,
    focus_subject: bool,
    progress: Progress,
) -> tuple[np.ndarray, dict]:
    if not gsplat_ready():
        raise RuntimeError("gsplat with CUDA is not installed or CUDA is unavailable.")
    import gsplat
    import torch
    import torch.nn.functional as F
    from gsplat import DefaultStrategy, rasterization

    profile = _training_profile(max_side)
    progress(58, "Loading registered views for true 3D Gaussian training")
    views, xyz, seed_colors, scene_scale, load_stats = _load_training_views(
        workspace, profile["image_side"], focus if focus_subject else None
    )
    device = "cuda:0"
    torch.manual_seed(42)
    np.random.seed(42)
    splats, optimizers = _initial_parameters(xyz, seed_colors, scene_scale, device)
    steps = profile["steps"]
    strategy = DefaultStrategy(
        grow_scale3d=0.006,
        grow_scale2d=0.03,
        prune_scale3d=0.035,
        prune_scale2d=0.10,
        refine_scale2d_stop_iter=max(1_000, int(steps * 0.75)),
        refine_start_iter=min(500, max(150, steps // 10)),
        refine_stop_iter=max(500, int(steps * 0.9)),
        reset_every=min(3_000, max(1_200, steps // 2)),
        refine_every=100,
        pause_refine_after_reset=len(views) + 100,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / steps)
    )
    report_every = max(50, steps // 25)
    last_loss = None
    splat_budget_reached = False
    for step in range(steps):
        view = views[np.random.randint(0, len(views))]
        pixels = torch.from_numpy(view["pixels"]).to(device=device, dtype=torch.float32)[None] / 255.0
        K = torch.from_numpy(view["K"]).to(device=device)[None]
        viewmat = torch.from_numpy(view["viewmat"]).to(device=device)[None]
        height, width = view["pixels"].shape[:2]
        rendered, rendered_alpha, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.sigmoid(splats["colors"]),
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=True,
            absgrad=strategy.absgrad,
            rasterize_mode="classic",
        )
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        mask_array = view.get("mask")
        if mask_array is not None:
            mask = torch.from_numpy(mask_array).to(device=device, dtype=torch.float32)[None, :, :, None]
            background = torch.rand((1, 1, 1, 3), device=device)
            target_composite = pixels * mask + background * (1.0 - mask)
            rendered_composite = rendered + background * (1.0 - rendered_alpha)
            alpha_loss = F.l1_loss(rendered_alpha, mask)
        else:
            target_composite = pixels
            rendered_composite = rendered
            alpha_loss = rendered.new_zeros(())
        l1 = F.l1_loss(rendered_composite, target_composite)
        log_scale_span = splats["scales"].amax(dim=1) - splats["scales"].amin(dim=1)
        anisotropy_penalty = torch.relu(
            log_scale_span - math.log(ANISOTROPY_PENALTY_START_RATIO)
        ).mean()
        loss = (
            0.75 * l1
            + 0.20 * (1.0 - _ssim(rendered_composite, target_composite))
            + 0.04 * alpha_loss
            + 0.01 * anisotropy_penalty
        )
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        _clamp_log_scale_anisotropy_(splats["scales"])
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=True)
        _clamp_log_scale_anisotropy_(splats["scales"])
        if not splat_budget_reached and len(splats["means"]) >= profile["max_splats"]:
            strategy.refine_stop_iter = step
            splat_budget_reached = True
            progress(
                min(91, 60 + round(31 * (step + 1) / steps)),
                f"Reached the {profile['max_splats']:,}-splat GPU budget; refining appearance",
            )
        last_loss = float(loss.detach().cpu())
        if step % report_every == 0 or step == steps - 1:
            percent = min(91, 60 + round(31 * (step + 1) / steps))
            progress(percent, f"Training true 3D Gaussians: {step + 1:,}/{steps:,} steps, {len(splats['means']):,} splats")
    export_stats = {}
    subject_support = None
    if focus_subject:
        progress(92, "Removing background with multi-view foreground agreement")
        trained_means = splats["means"].detach().cpu().numpy().astype(np.float32)
        subject_support, support_hits, support_seen = _project_mask_support(trained_means, views)
        export_stats.update({
            "mask_support_views_median": round(float(np.median(support_seen)), 2),
            "mask_support_hits_median": round(float(np.median(support_hits)), 2),
        })
    gaussians, focus_applied = _export_arrays(
        splats,
        focus if focus_subject else None,
        scene_scale=scene_scale,
        export_stats=export_stats,
        subject_support=subject_support,
    )
    version = getattr(gsplat, "__version__", "1.5.3")
    stats = {
        "representation": "trained-3dgs",
        "trainer": f"gsplat {version}",
        "training_steps": steps,
        "training_loss": round(last_loss or 0.0, 6),
        "seed_points": len(xyz),
        "trained_gaussians": export_stats.get("optimized_gaussians", len(gaussians)),
        "training_image_side": profile["image_side"],
        "training_splat_budget": profile["max_splats"],
        "recommended_splat_scale": 0.65,
        "subject_focus_requested": focus_subject,
        "subject_focus_applied": focus_applied,
        **load_stats,
        **export_stats,
    }
    del splats, optimizers
    torch.cuda.empty_cache()
    return gaussians, stats
