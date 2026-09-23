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
    1200: {"image_side": 720, "steps": 3_500, "max_splats": 500_000},
    1600: {"image_side": 900, "steps": 5_500, "max_splats": 750_000},
    2000: {"image_side": 1080, "steps": 7_500, "max_splats": 1_000_000},
}


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


def _load_training_views(workspace: Path, image_side: int, focus_subject: bool):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(workspace / "sparse")
    registered = sorted(
        (image for image in reconstruction.images.values() if image.has_pose),
        key=lambda image: image.name,
    )
    if len(registered) < 8:
        raise RuntimeError("True 3DGS training needs at least eight registered COLMAP cameras.")

    views = []
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
        if focus_subject:
            crop_width, crop_height = max(32, round(width * 0.94)), max(32, round(height * 0.94))
            left, top = (width - crop_width) // 2, (height - crop_height) // 2
            frame = frame.crop((left, top, left + crop_width, top + crop_height))
            K[0, 2] -= left
            K[1, 2] -= top
            width, height = frame.size
        resize = min(1.0, image_side / max(width, height))
        output_size = (max(16, round(width * resize)), max(16, round(height * resize)))
        if output_size != frame.size:
            frame = frame.resize(output_size, Image.Resampling.LANCZOS)
            K[0, :] *= output_size[0] / width
            K[1, :] *= output_size[1] / height
        views.append(
            {
                "name": image.name,
                "pixels": np.asarray(frame, dtype=np.uint8).copy(),
                "K": K,
                "viewmat": _image_pose_matrix(image),
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

    centers = np.asarray([image.projection_center() for image in registered], dtype=np.float32)
    scene_center = np.mean(centers, axis=0)
    scene_scale = max(float(np.linalg.norm(centers - scene_center, axis=1).max()), 1e-3)
    return views, xyz, colors, scene_scale


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


def _initial_parameters(xyz: np.ndarray, colors: np.ndarray, scene_scale: float, device: str):
    import torch

    neighbors = min(4, len(xyz))
    distances, _ = cKDTree(xyz).query(xyz, k=neighbors, workers=-1)
    local = np.mean(np.square(distances[:, 1:]), axis=1)
    fallback = max(float(np.median(local[local > 0])) if np.any(local > 0) else scene_scale * 1e-4, 1e-8)
    local = np.sqrt(np.maximum(local, fallback)).astype(np.float32)
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


def _export_arrays(splats, focus: dict | None, maximum: int = 1_250_000):
    import torch

    means = splats["means"].detach().cpu().numpy().astype(np.float32)
    scales = torch.exp(splats["scales"]).detach().cpu().numpy().astype(np.float32)
    quats = splats["quats"].detach().cpu().numpy().astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    opacities = torch.sigmoid(splats["opacities"]).detach().cpu().numpy().astype(np.float32)
    colors = torch.sigmoid(splats["colors"]).detach().cpu().numpy().astype(np.float32)
    finite = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacities)
        & np.isfinite(colors).all(axis=1)
    )
    keep = finite & (opacities >= 0.02) & (scales > 0).all(axis=1)
    focus_applied = False
    if focus is not None:
        target = np.asarray(focus["target"], dtype=np.float32)
        distance = np.linalg.norm(means - target, axis=1)
        focused = distance <= float(focus["camera_distance"] * 0.5)
        if np.count_nonzero(keep & focused) >= max(1_500, int(np.count_nonzero(keep) * 0.05)):
            keep &= focused
            focus_applied = True
    indices = np.flatnonzero(keep)
    if len(indices) > maximum:
        indices = indices[np.argpartition(opacities[indices], -maximum)[-maximum:]]
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
    return validate(gaussians), focus_applied


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
    views, xyz, seed_colors, scene_scale = _load_training_views(
        workspace, profile["image_side"], focus_subject
    )
    device = "cuda:0"
    torch.manual_seed(42)
    np.random.seed(42)
    splats, optimizers = _initial_parameters(xyz, seed_colors, scene_scale, device)
    steps = profile["steps"]
    strategy = DefaultStrategy(
        refine_start_iter=min(300, max(100, steps // 10)),
        refine_stop_iter=max(500, int(steps * 0.82)),
        reset_every=max(1_200, steps // 2),
        refine_every=100,
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
        rendered, _, info = rasterization(
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
            rasterize_mode="antialiased",
        )
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = F.l1_loss(rendered, pixels)
        loss = 0.8 * l1 + 0.2 * (1.0 - _ssim(rendered, pixels))
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=True)
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
    gaussians, focus_applied = _export_arrays(splats, focus if focus_subject else None)
    version = getattr(gsplat, "__version__", "1.5.3")
    stats = {
        "representation": "trained-3dgs",
        "trainer": f"gsplat {version}",
        "training_steps": steps,
        "training_loss": round(last_loss or 0.0, 6),
        "seed_points": len(xyz),
        "trained_gaussians": len(gaussians),
        "training_image_side": profile["image_side"],
        "training_splat_budget": profile["max_splats"],
        "subject_focus_requested": focus_subject,
        "subject_focus_applied": focus_applied,
    }
    del splats, optimizers
    torch.cuda.empty_cache()
    return gaussians, stats
