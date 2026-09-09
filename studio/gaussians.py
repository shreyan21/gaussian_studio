"""Gaussian geometry and portable exports. Quaternions are scalar-first (w,x,y,z).

Array layout: position[3], opacity[1], standard deviations[3], reserved[1],
quaternion[4], sRGB[3], reserved[1]. Viewer coordinates: +X right, +Y up, -Z forward.
"""
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814

VIEW_QUATERNIONS = {
    "front": np.array([1, 0, 0, 0], dtype=np.float32),
    "right": np.array([np.sqrt(.5), 0, np.sqrt(.5), 0], dtype=np.float32),
    "back": np.array([0, 0, 1, 0], dtype=np.float32),
    "left": np.array([np.sqrt(.5), 0, -np.sqrt(.5), 0], dtype=np.float32),
    "top": np.array([np.sqrt(.5), -np.sqrt(.5), 0, 0], dtype=np.float32),
    "bottom": np.array([np.sqrt(.5), np.sqrt(.5), 0, 0], dtype=np.float32),
}


def quaternion_multiply(a, b):
    """Hamilton product for scalar-first quaternions, with NumPy broadcasting."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ), axis=-1)


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float32)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)


def align_gaussian_view(g, view, target_radius=1.0):
    """Normalize an independent SHARP result and rotate its labelled camera view.

    This is deterministic geometric fusion, not native multi-view conditioning.
    Robust percentiles reduce the influence of distant floaters/background splats.
    """
    if view not in VIEW_QUATERNIONS:
        raise ValueError(f"Unsupported camera view: {view}")
    g = validate(g)
    xyz = g[:, :3]
    center = np.median(xyz, axis=0)
    low, high = np.percentile(xyz, [10, 90], axis=0)
    radius = max(float(np.linalg.norm(high-low) / 2), 1e-6)
    scale = target_radius / radius
    rotation_q = VIEW_QUATERNIONS[view]
    rotation = quaternion_matrix(rotation_q)
    g[:, :3] = ((xyz-center) * scale) @ rotation.T
    g[:, 4:7] *= scale
    g[:, 8:12] = quaternion_multiply(rotation_q, g[:, 8:12])
    return validate(g)


def fuse_gaussian_views(views, max_gaussians=2_000_000):
    """Align and combine ``[(gaussians, direction), ...]`` with equal coverage."""
    if not views:
        raise ValueError("At least one Gaussian view is required.")
    budget = max(1, max_gaussians // len(views))
    aligned = []
    for index, (g, direction) in enumerate(views):
        g = align_gaussian_view(g, direction)
        if len(g) > budget:
            chosen = np.sort(np.random.default_rng(4100+index).choice(len(g), budget, replace=False))
            g = g[chosen].copy()
            # Preserve approximate projected coverage after uniform thinning.
            g[:, 4:6] *= np.sqrt(len(views[index][0]) / budget)
        aligned.append(g)
    return validate(np.concatenate(aligned, axis=0))


def validate(g):
    g = np.asarray(g, dtype="<f4")
    if g.ndim != 2 or g.shape[1] != 16 or not len(g):
        raise ValueError("No valid 3D Gaussians were produced.")
    good = np.isfinite(g).all(axis=1) & (g[:, 4:7] > 0).all(axis=1)
    good &= (g[:, 3] > 0.005) & (np.linalg.norm(g[:, 8:12], axis=1) > 1e-8)
    g = g[good].copy()
    if not len(g):
        raise ValueError("The model produced only invalid or transparent Gaussians.")
    g[:, 3] = np.clip(g[:, 3], 0.001, 0.999)
    g[:, 8:12] /= np.linalg.norm(g[:, 8:12], axis=1, keepdims=True)
    g[:, 12:15] = np.clip(g[:, 12:15], 0, 1)
    return g


def from_depth(image: Image.Image, disparity, max_side=512, depth_strength=1.0):
    """Lift relative inverse depth to anisotropic Gaussian surfels (not a learned 3DGS model).

    One splat per sampled pixel. Tangential scales cover neighboring pixels and the
    smaller normal scale makes a thin volume. Depth discontinuities shrink splats
    to reduce foreground/background bridges. Scale is relative, not measured metres.
    """
    w, h = image.size
    ratio = min(1, max_side / max(w, h))
    w, h = max(16, round(w * ratio)), max(16, round(h * ratio))
    rgb = np.asarray(image.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    d = np.asarray(Image.fromarray(np.asarray(disparity, dtype=np.float32)).resize((w, h), Image.Resampling.BILINEAR))
    if not np.isfinite(d).all():
        raise ValueError("Depth prediction contains non-finite values.")
    lo, hi = np.percentile(d, [2, 98])
    d = np.clip((d - lo) / max(float(hi - lo), 1e-6), 0, 1)
    z = 1 / (0.35 + 0.65 * d)
    z = 2 + (z - np.median(z)) * depth_strength
    z = np.maximum(z, 0.3)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    focal = max(w, h) * 0.85
    xyz = np.stack(((xx - (w - 1) / 2) * z / focal, -(yy - (h - 1) / 2) * z / focal, -z), axis=-1)
    dx, dy = np.gradient(xyz, axis=1), -np.gradient(xyz, axis=0)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    # Rotation taking +Z to the surface normal.
    q = np.stack((1 + normal[..., 2], -normal[..., 1], normal[..., 0], np.zeros_like(z)), axis=-1)
    qnorm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(qnorm > 1e-6, q / np.maximum(qnorm, 1e-6), np.array([0, 1, 0, 0]))
    edge = np.maximum(abs(np.gradient(z, axis=0)), abs(np.gradient(z, axis=1)))
    pixel = z / focal
    scale = pixel * 0.75 * np.where(edge > 5 * pixel, 0.55, 1)
    g = np.zeros((w * h, 16), dtype=np.float32)
    g[:, :3], g[:, 3] = xyz.reshape(-1, 3), 0.96
    g[:, 4:7] = np.stack((scale, scale, scale * 0.22), axis=-1).reshape(-1, 3)
    g[:, 8:12], g[:, 12:15] = q.reshape(-1, 4), rgb.reshape(-1, 3)
    return validate(g), Image.fromarray(np.uint8(d * 255)), {"fov_y": float(np.degrees(2 * np.arctan(h / (2 * focal)))), "image_size": [w, h]}


def write_ply(path, g):
    fields = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.zeros(len(g), dtype=[(k, "<f4") for k in fields])
    for i, name in enumerate(["x", "y", "z"]):
        data[name] = g[:, i]
    for i in range(3):
        data[f"f_dc_{i}"] = (g[:, 12+i] - 0.5) / SH_C0
        data[f"scale_{i}"] = np.log(g[:, 4+i])
    data["opacity"] = np.log(g[:, 3] / (1-g[:, 3]))
    for i in range(4):
        data[f"rot_{i}"] = g[:, 8+i]
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<", comments=["Gaussian Scene Studio; sRGB SH0; +X right +Y up -Z forward"]).write(str(path))


def read_ply(path):
    v = PlyData.read(str(path))["vertex"]
    g = np.zeros((len(v), 16), np.float32)
    for i, n in enumerate(["x", "y", "z"]):
        g[:, i] = v[n]
    g[:, 3] = 1 / (1 + np.exp(-np.clip(v["opacity"], -30, 30)))
    for i in range(3):
        g[:, 4+i] = np.exp(np.clip(v[f"scale_{i}"], -30, 20))
        g[:, 12+i] = v[f"f_dc_{i}"] * SH_C0 + 0.5
    for i in range(4):
        g[:, 8+i] = v[f"rot_{i}"]
    return validate(g)


def export_scene(directory: Path, g, metadata, preview_limit=1_500_000):
    g = validate(g)
    write_ply(directory / "scene.ply", g)
    # Spatially unbiased deterministic preview; full PLY retains every valid Gaussian.
    if len(g) > preview_limit:
        indices = np.sort(np.random.default_rng(42).choice(len(g), preview_limit, replace=False))
        preview = g[indices].copy()
        preview[:, 4:6] *= np.sqrt(len(g) / preview_limit)
    else:
        preview = g
    with (directory / "scene.gsb").open("wb") as f:
        f.write(struct.pack("<4sIII", b"GSS1", len(preview), 16, 0))
        f.write(preview.astype("<f4", copy=False).tobytes())
    center = np.median(g[:, :3], axis=0)
    low, high = np.percentile(g[:, :3], [2, 98], axis=0)
    metadata.update({"gaussians": len(g), "preview_gaussians": len(preview), "target": center.tolist(), "radius": max(float(np.linalg.norm(high-low) / 2), 0.1), "coordinate_system": "+X right, +Y up, -Z forward", "source_camera": [0, 0, 0], "format": "3DGS binary PLY, SH degree 0"})
    (directory / "scene.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def make_demo():
    """Procedural, fully 3D calibration sculpture. Never presented as AI output."""
    rng = np.random.default_rng(7)
    n = 18000
    u, v = rng.uniform(0, 2*np.pi, (2, n))
    r = 0.23
    g = np.zeros((n, 16), np.float32)
    g[:, :3] = np.stack(((0.8+r*np.cos(v))*np.cos(u), (0.8+r*np.cos(v))*np.sin(u), r*np.sin(v)-3), axis=-1)
    g[:, 3], g[:, 4:7], g[:, 8] = 0.88, 0.016, 1
    g[:, 12:15] = np.stack((0.24+0.25*(1+np.cos(u)), 0.45+0.25*np.sin(v), 0.7+0.22*np.sin(u)), axis=-1)
    g[:, 1] *= 0.8
    g[:, 2] += g[:, 0] * 0.4
    return validate(g)
