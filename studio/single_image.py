"""Commercial-friendly limited-angle 3D preview from one photograph."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from studio.gaussians import from_depth
from studio.gsplat_runtime import AI_DEPTH_MODEL_ID, ai_depth_cache_dir

Progress = Callable[[int, str], None]


def predict_relative_inverse_depth(image: Image.Image, device: str) -> np.ndarray:
    """Run the Apache-2.0 Depth Anything V2 Small checkpoint."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    cache_dir = ai_depth_cache_dir()
    processor = AutoImageProcessor.from_pretrained(
        AI_DEPTH_MODEL_ID,
        cache_dir=cache_dir,
        use_fast=False,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        AI_DEPTH_MODEL_ID,
        cache_dir=cache_dir,
    ).to(device).eval()
    inputs = {
        name: value.to(device)
        for name, value in processor(images=image, return_tensors="pt").items()
    }
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth[0].float().cpu().numpy()
    del inputs, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if prediction.ndim != 2 or not np.isfinite(prediction).all():
        raise RuntimeError("The single-photo AI produced an invalid depth prediction.")
    return prediction


def reconstruct_single_image(
    image_path: Path,
    max_side: int,
    use_gpu: bool,
    depth_strength: float,
    progress: Progress,
) -> tuple[np.ndarray, dict]:
    """Create a perspective Gaussian depth relief and safe viewer limits."""
    import torch

    if not use_gpu or not torch.cuda.is_available():
        raise RuntimeError("Single-photo 3D preview requires an NVIDIA CUDA GPU.")
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    progress(18, "Predicting single-photo AI depth")
    disparity = predict_relative_inverse_depth(image, "cuda:0")
    progress(72, "Building the limited-angle Gaussian scene")
    gaussians, depth_preview, camera = from_depth(
        image,
        disparity,
        max_side=max_side,
        depth_strength=depth_strength,
    )
    depth_preview.save(image_path.parent / "depth.png")
    metadata = {
        "engine": "Depth Anything V2 Small",
        "method": "single-image-depth",
        "representation": "single-image-depth-splat",
        "device": "cuda",
        "input_count": 1,
        "registered_images": 1,
        "ai_depth_prior": AI_DEPTH_MODEL_ID,
        "single_image_preview": True,
        "recommended_splat_scale": 1.0,
        "view_limits": {"yaw_degrees": 22, "pitch_degrees": 14},
        "source_camera": [0, 0, 0],
        "subject_focus_requested": False,
        "subject_focus_applied": False,
        "limitation": (
            "AI-estimated limited-angle 2.5D preview from one photograph. "
            "Hidden sides are not measured; rotation is locked before stretched or missing areas become visible."
        ),
        **camera,
    }
    return gaussians, metadata
