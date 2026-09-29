"""Optional single-image generation through the official TripoSplat release.

The upstream demo keeps every network resident on CUDA.  This adapter loads the
five stages sequentially so an 8-16 GB GPU has a substantially lower peak.  No
model code or checkpoint is vendored in this repository; setup pins both.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from studio.config import ROOT
from studio.gaussians import read_ply, validate

TRIPOSPLAT_CODE_REVISION = "d8db9e018b413dd9c4a9fe22463781bf98e8e68d"
TRIPOSPLAT_WEIGHTS_REVISION = "56a96e603204ec410c4da60c13ea4fa09a2169a9"
CHECKPOINT_FILES = {
    "flow": "diffusion_models/triposplat_fp16.safetensors",
    "decoder": "vae/triposplat_vae_decoder_fp16.safetensors",
    "dinov3": "clip_vision/dino_v3_vit_h.safetensors",
    "vae": "vae/flux2-vae.safetensors",
    "rmbg": "background_removal/birefnet.safetensors",
}

# TripoSplat's exported PLY still needs the two rotations used by its official
# Spark viewer (Y +90 degrees, then X 180 degrees).  Folding those rotations
# into the export gives this app's +Y-up, camera-on-+Z viewer an upright source
# view instead of displaying the generated object inverted.
TRIPOSPLAT_VIEWER_TRANSFORM = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
]


def _aligned_source_backdrop(
    source: Image.Image,
    isolated: Image.Image,
    size: int = 1024,
    erode_radius: int = 1,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Apply TripoSplat's subject crop to the original photo and its mask."""
    width, height = isolated.size
    scale = size / min(width, height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    source = source.convert("RGB").resize(resized_size, Image.Resampling.LANCZOS)
    isolated = isolated.convert("RGBA").resize(resized_size, Image.Resampling.LANCZOS)
    alpha = isolated.getchannel("A")
    if erode_radius > 0:
        alpha = alpha.filter(ImageFilter.MinFilter(2 * erode_radius + 1))
    alpha_array = np.asarray(alpha)
    rows, columns = np.nonzero(alpha_array)
    if not len(columns):
        raise RuntimeError("TripoSplat background removal could not find a foreground object.")
    left, top, right, bottom = columns.min(), rows.min(), columns.max(), rows.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * 1.2
    box = [int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)]
    coverage = Image.new("L", resized_size, 255)
    backdrop = source.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    backdrop = backdrop.filter(ImageFilter.GaussianBlur(radius=2))
    foreground = alpha.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    coverage = coverage.crop(box).resize((size, size), Image.Resampling.BILINEAR)
    return backdrop, foreground, coverage


def _source_background_gaussians(
    backdrop: Image.Image,
    foreground: Image.Image,
    coverage: Image.Image,
    subject: np.ndarray,
    grid_size: int = 112,
) -> np.ndarray:
    """Make a thin, masked photo backdrop behind a generated foreground object."""
    colors = np.asarray(backdrop.resize((grid_size, grid_size), Image.Resampling.LANCZOS), np.float32) / 255
    alpha = np.asarray(
        foreground.filter(ImageFilter.MaxFilter(17)).resize((grid_size, grid_size), Image.Resampling.BILINEAR),
        np.float32,
    ) / 255
    valid = np.asarray(coverage.resize((grid_size, grid_size), Image.Resampling.BILINEAR), np.float32) > 192
    keep = (alpha < 0.08) & valid
    rows, columns = np.nonzero(keep)
    if len(columns) < 100:
        return np.empty((0, 16), dtype=np.float32)

    low, high = np.percentile(subject[:, :3], [2, 98], axis=0)
    center = np.median(subject[:, :3], axis=0)
    subject_size = max(float(high[0] - low[0]), float(high[1] - low[1]), 0.1)
    plane_size = subject_size * 1.45
    step = plane_size / max(grid_size - 1, 1)
    behind = float(low[2] - max(0.06 * subject_size, 2 * step))

    result = np.zeros((len(columns), 16), dtype=np.float32)
    result[:, 0] = center[0] + (columns / (grid_size - 1) - 0.5) * plane_size
    result[:, 1] = center[1] - (rows / (grid_size - 1) - 0.5) * plane_size
    result[:, 2] = behind
    result[:, 3] = 0.72
    result[:, 4:7] = (step * 0.62, step * 0.62, max(step * 0.08, 1e-5))
    result[:, 8] = 1.0
    result[:, 12:15] = colors[rows, columns]
    return validate(result)


def triposplat_root() -> Path:
    return Path(os.environ.get("GSS_TRIPOSPLAT_ROOT", ROOT / "tools" / "TripoSplat")).resolve()


def triposplat_status() -> tuple[bool, str]:
    root = triposplat_root()
    required = [root / "triposplat.py", root / "model.py"]
    required.extend(root / "ckpts" / relative for relative in CHECKPOINT_FILES.values())
    if missing := [path for path in required if not path.is_file()]:
        return False, f"TripoSplat is not installed ({missing[0].name} is missing)."
    for package in ("torch", "torchvision", "safetensors"):
        if importlib.util.find_spec(package) is None:
            return False, f"TripoSplat dependency '{package}' is not installed."
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "TripoSplat requires an NVIDIA CUDA GPU."
    except (ImportError, OSError) as exc:
        return False, f"CUDA PyTorch could not be loaded: {exc}"
    return True, "Official TripoSplat code and checkpoints are ready."


def triposplat_ready() -> bool:
    return triposplat_status()[0]


def _upstream_module():
    """Import the pinned checkout without allowing an unrelated `model` module."""
    root = triposplat_root()
    root_text = str(root)
    for name in ("triposplat", "model"):
        loaded = sys.modules.get(name)
        location = Path(getattr(loaded, "__file__", "")).resolve() if loaded else None
        if location and root not in location.parents:
            del sys.modules[name]
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("triposplat")


def _release_cuda(torch) -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _gaussian_budget(torch) -> int:
    override = os.environ.get("GSS_TRIPOSPLAT_GAUSSIANS", "").strip()
    if override:
        try:
            requested = int(override)
        except ValueError as exc:
            raise RuntimeError("GSS_TRIPOSPLAT_GAUSSIANS must be an integer.") from exc
        return max(32_768, min(262_144, round(requested / 32) * 32))
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gb < 10:
        return 65_536
    if memory_gb < 20:
        return 131_072
    return 262_144


def generate_single_image(directory: Path, image_path: Path, progress) -> tuple[np.ndarray, dict]:
    ready, reason = triposplat_status()
    if not ready:
        raise RuntimeError(reason + " Run the TripoSplat setup, restart Studio, and try again.")

    import torch

    upstream = _upstream_module()
    checkpoint_root = triposplat_root() / "ckpts"
    paths = {name: str(checkpoint_root / relative) for name, relative in CHECKPOINT_FILES.items()}
    device = torch.device("cuda:0")
    encoder_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    count = _gaussian_budget(torch)
    steps = max(10, min(30, int(os.environ.get("GSS_TRIPOSPLAT_STEPS", "20"))))
    raw_ply = directory / "triposplat-raw.ply"
    generator = torch.Generator(device=device).manual_seed(42)

    try:
        progress(5, "Preparing the subject with TripoSplat background removal")
        rmbg = upstream.load_rmbg(paths["rmbg"], device=device, dtype=torch.float16)
        with Image.open(image_path) as source_file:
            source = source_file.convert("RGBA")
        has_alpha = source.getchannel("A").getextrema()[0] < 255
        isolated = source if has_alpha else rmbg.remove_background(source.convert("RGB"))
        prepared = upstream.preprocess_image(isolated, rmbg, erode_radius=1)
        backdrop, foreground, coverage = _aligned_source_backdrop(source, isolated)
        prepared.save(directory / "triposplat-input.png")
        backdrop.save(directory / "triposplat-backdrop.jpg", quality=90)
        del rmbg
        _release_cuda(torch)

        progress(18, "Encoding the source photograph")
        dinov3 = upstream.load_dinov3(paths["dinov3"], device=device, dtype=encoder_dtype)
        vae_encoder = upstream.load_vae_encoder(paths["vae"], device=device, dtype=encoder_dtype)
        condition = upstream.encode_image(prepared, dinov3, vae_encoder, generator=generator)
        del dinov3, vae_encoder, prepared
        _release_cuda(torch)

        progress(32, f"Generating hidden 3D structure in {steps} AI steps")
        flow_model = upstream.load_flow_model(paths["flow"], device=device, dtype=torch.float16)

        def sampling_progress(step, total):
            progress(32 + round(45 * step / total), f"Generating hidden 3D structure ({step}/{total})")

        sampled = upstream.sample_latent(
            flow_model,
            condition,
            steps=steps,
            guidance_scale=3.0,
            shift=3.0,
            generator=generator,
            callback=sampling_progress,
        )
        latent = sampled["latent"].detach().cpu()
        del sampled, condition, flow_model
        _release_cuda(torch)

        progress(80, f"Decoding {count:,} learned 3D Gaussians")
        decoder = upstream.load_decoder(paths["decoder"], device=device, dtype=torch.float16)
        gaussian = decoder.decode(latent.to(device), num_gaussians=count)
        gaussian.save_ply(raw_ply, transform=TRIPOSPLAT_VIEWER_TRANSFORM)
        del gaussian, decoder, latent
        _release_cuda(torch)

        progress(91, "Converting TripoSplat output for the local viewer")
        subject_gaussians = read_ply(raw_ply)
        background_gaussians = _source_background_gaussians(
            backdrop, foreground, coverage, subject_gaussians
        )
        gaussians = validate(np.concatenate((subject_gaussians, background_gaussians)))
        raw_ply.unlink(missing_ok=True)
        return gaussians, {
            "engine": "TripoSplat",
            "method": "triposplat",
            "representation": "generated-3dgs",
            "device": "cuda",
            "input_count": 1,
            "registered_images": 1,
            "image_size": [1024, 1024],
            "source_camera": [0.0, 0.0, 2.0],
            "fov_y": 45.0,
            "recommended_splat_scale": 1.0,
            "view_limits": {"yaw_degrees": 55, "pitch_degrees": 35},
            "background_mode": "masked-source-photo-backdrop" if len(background_gaussians) else "none",
            "object_gaussians": len(subject_gaussians),
            "background_gaussians": len(background_gaussians),
            "generated_gaussians": len(gaussians),
            "triposplat_steps": steps,
            "triposplat_code_revision": TRIPOSPLAT_CODE_REVISION,
            "triposplat_weights_revision": TRIPOSPLAT_WEIGHTS_REVISION,
            "licence": "TripoSplat release: MIT; DINOv3 encoder: DINOv3 License. See docs/MODEL-LICENSES.md.",
            "limitation": (
                "Generated from one photograph. Visible object details follow the upload, but hidden surfaces are AI predictions. "
                + (
                    "The surrounding source photo is retained as a shallow masked backdrop, not reconstructed 3D geometry."
                    if len(background_gaussians)
                    else "No usable source background was present, so only the generated object is shown."
                )
            ),
        }
    except torch.OutOfMemoryError as exc:
        _release_cuda(torch)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        raise RuntimeError(
            f"TripoSplat ran out of GPU memory on this {memory_gb:.1f} GB GPU. Close other GPU apps, "
            "set GSS_TRIPOSPLAT_GAUSSIANS=32768, restart Studio, and retry."
        ) from exc

