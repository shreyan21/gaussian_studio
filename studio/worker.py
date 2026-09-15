"""One inference process per job: cancellation and GPU release are deterministic."""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from studio.config import DEPTH_DIR, SHARP_SOURCE, SHARP_WEIGHTS
from studio.gaussians import export_scene, from_depth

# Reduce fragmentation in the short-lived CUDA worker. This must be set before
# PyTorch is imported; PyTorch 2.8 supports this allocator option on Windows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def progress(directory, percent, message):
    temporary = directory / "progress.tmp"
    temporary.write_text(json.dumps({"progress": percent, "message": message}), encoding="utf-8")
    os.replace(temporary, directory / "progress.json")
    print(message, flush=True)


def choose_device(requested, engine):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Run Setup NVIDIA Workstation.cmd on the workstation, or select CPU for the depth model.")
    if engine == "anysplat":
        if not torch.cuda.is_available():
            raise RuntimeError("AnySplat requires an NVIDIA CUDA GPU in this app. Run it on the office workstation after CUDA setup; Depth Anything remains available on CPU here.")
        return "cuda"
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"


def depth_predict(image, directory, options, device):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    if not (DEPTH_DIR / "model.safetensors").is_file():
        raise RuntimeError("Depth model is not installed. Run .venv\\Scripts\\python.exe scripts\\download_models.py --model depth, then retry.")
    progress(directory, 20, "Loading Depth Anything V2 Small on " + device.upper())
    processor = AutoImageProcessor.from_pretrained(str(DEPTH_DIR), local_files_only=True, use_fast=False)
    model = AutoModelForDepthEstimation.from_pretrained(str(DEPTH_DIR), local_files_only=True, use_safetensors=True).eval().to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    progress(directory, 40, "Estimating relative depth from your image")
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth[0].float().cpu().numpy()
    del model, inputs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    progress(directory, 70, "Lifting predicted depth into oriented 3D Gaussians")
    g, depth, camera = from_depth(image, prediction, options["resolution"], options["depth_strength"])
    depth.save(directory / "depth.png")
    return g, {
        **camera,
        "engine": "Depth Anything V2 Small + Gaussian lifting",
        "method": "depth",
        "limitation": "Relative-depth reconstruction of visible surfaces. Hidden sides are not generated; scale is not metric.",
        "licence": "Apache-2.0 model",
        "device": device,
    }


def filter_pixel_gaussians(gaussians, foreground_mask, layer_count=2):
    """Resize a source mask to SHARP's output grid and keep every depth layer."""
    if layer_count < 1 or len(gaussians) % layer_count:
        raise RuntimeError("SHARP output does not match its configured depth layers.")
    pixel_count = len(gaussians) // layer_count
    output_side = int(round(np.sqrt(pixel_count)))
    if output_side * output_side != pixel_count:
        raise RuntimeError("SHARP output is not a square pixel grid.")
    mask_image = Image.fromarray(np.uint8(np.asarray(foreground_mask, dtype=bool)) * 255)
    mask = np.asarray(
        mask_image.resize((output_side, output_side), Image.Resampling.BOX), dtype=np.uint8
    ).reshape(-1) > 0
    return gaussians[np.tile(mask, layer_count)], layer_count


def sharp_predict(image, directory, options, device):
    import psutil
    import torch
    import torch.nn.functional as F

    if not SHARP_WEIGHTS.is_file() or not (SHARP_SOURCE / "sharp" / "models").is_dir():
        raise RuntimeError("SHARP is not installed. Install requirements-sharp.txt and run scripts/download_models.py --model sharp.")
    if device == "cpu" and psutil.virtual_memory().total < 14 * 1024**3:
        raise RuntimeError("SHARP CPU mode needs at least 14 GB RAM. Use CUDA or the lightweight depth model.")
    sys.path.insert(0, str(SHARP_SOURCE))
    from sharp.models import PredictorParams, create_predictor
    from sharp.utils.gaussians import Gaussians3D, unproject_gaussians

    source_size = list(image.size)
    object_only = bool(options.get("object_only", True))
    foreground_mask = None
    foreground_coverage = None
    if object_only:
        from studio.foreground import focus_object, load_session, predict_mask

        progress(directory, 8, "Detecting and tightly framing the main subject")
        session = load_session()
        image, foreground_mask, foreground_coverage = focus_object(
            image, predict_mask(image, session), size=1536
        )
        Image.fromarray(np.uint8(foreground_mask) * 255).save(directory / "object-mask-0.png")
        del session

    progress(directory, 15, "Loading Apple SHARP weights on " + device.upper())
    state = torch.load(str(SHARP_WEIGHTS), map_location="cpu", weights_only=True, mmap=True)
    predictor_params = PredictorParams()
    with torch.device("meta"):
        predictor = create_predictor(predictor_params)
    predictor.load_state_dict(state, strict=True, assign=True)
    del state
    predictor.eval().to(device)
    width, height = image.size
    focal = max(width, height) * 0.85
    input_tensor = torch.from_numpy(np.asarray(image).copy()).to(device).float().permute(2, 0, 1)[None] / 255
    input_tensor = F.interpolate(input_tensor, (1536, 1536), mode="bilinear", align_corners=True)
    factor = torch.tensor([focal / width], dtype=torch.float32, device=device)
    progress(directory, 40, "Predicting 3D Gaussians from the source photograph")
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        raw = predictor(input_tensor, factor)
    raw = Gaussians3D(*(tensor.detach().float().cpu() for tensor in raw))
    del predictor, input_tensor, factor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    progress(directory, 70, "Unprojecting Gaussians into the 3D camera view")
    intrinsics = torch.tensor([[focal, 0, width / 2, 0], [0, focal, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    intrinsics[0] *= 1536 / width
    intrinsics[1] *= 1536 / height
    with torch.inference_mode():
        gaussian = unproject_gaussians(raw, torch.eye(4), intrinsics, (1536, 1536))
    means = gaussian.mean_vectors.reshape(-1, 3).numpy()
    result = np.zeros((len(means), 16), np.float32)
    result[:, :3] = means * [1, -1, -1]
    result[:, 3] = gaussian.opacities.reshape(-1).numpy()
    result[:, 4:7] = gaussian.singular_values.reshape(-1, 3).numpy()
    quaternion = gaussian.quaternions.reshape(-1, 4).numpy()
    result[:, 8:12] = np.stack((-quaternion[:, 1], quaternion[:, 0], -quaternion[:, 3], quaternion[:, 2]), axis=-1)
    linear = np.clip(gaussian.colors.reshape(-1, 3).numpy(), 0, 1)
    result[:, 12:15] = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    layer_count = None
    if foreground_mask is not None:
        result, layer_count = filter_pixel_gaussians(
            result, foreground_mask, predictor_params.initializer.num_layers
        )
    return result, {
        "fov_y": float(np.degrees(2 * np.arctan(height / (2 * focal)))),
        "image_size": [width, height], "source_image_size": source_size,
        "engine": "Apple SHARP", "method": "sharp", "device": device,
        "precision": "CUDA mixed float16; CPU float32 postprocessing" if device == "cuda" else "float32",
        "licence": "Apple ML Research Model: non-commercial scientific research only",
        "object_only": object_only,
        "foreground_coverage": round(foreground_coverage, 4) if foreground_coverage is not None else None,
        "gaussian_layers_per_pixel": layer_count,
        "view_limits": {"yaw_degrees": 30, "pitch_degrees": 18},
        "limitation": "Single-image prediction supports nearby novel views only. Rotation is intentionally clamped to ±30° horizontally and ±18° vertically because unseen sides are not reconstructed.",
    }


def anysplat_predict(directory, options, device):
    import torch
    from studio.anysplat_runtime import automatic_view_limit, cuda_inference_profile, load_model, preprocess_image, preprocess_object_image, select_inputs, to_viewer_gaussians

    inputs = options["inputs"]
    paths = [directory / item["file"] for item in inputs]
    requested_limit = options.get("view_limit", "auto")
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    vram_gb = total_bytes / 1024**3
    free_vram_gb = free_bytes / 1024**3
    if requested_limit == "all":
        limit = len(paths)
    elif requested_limit == "auto":
        # Account for memory already used by Windows, the browser and other apps.
        limit = automatic_view_limit(min(vram_gb, free_vram_gb), len(paths))
    else:
        limit = min(len(paths), int(requested_limit))
    selected = select_inputs(paths, limit)
    selected_names = [path.name for path in selected]

    if len(selected) < 2:
        raise RuntimeError(
            f"AnySplat needs at least 5.5 GB of free GPU memory; only {free_vram_gb:.1f} GB is free. "
            "Close browser tabs, QGIS and other GPU-heavy programs, then retry with Auto-safe."
        )
    low_vram, compute_dtype = cuda_inference_profile(torch, vram_gb)
    object_only = bool(options.get("object_only", True))
    foreground_mask = None
    mask_coverages = []
    progress(directory, 8, f"Preparing {len(selected)} of {len(paths)} uploaded views at 448 x 448")
    if object_only:
        from studio.foreground import load_session

        progress(directory, 10, "Detecting and framing the main object in every view")
        session = load_session()
        prepared, masks = [], []
        for index, path in enumerate(selected):
            tensor, mask, coverage = preprocess_object_image(path, session)
            prepared.append(tensor)
            masks.append(mask)
            mask_coverages.append(round(coverage, 4))
            Image.fromarray(np.uint8(mask.numpy()) * 255).save(directory / f"object-mask-{index}.png")
        images = torch.stack(prepared, dim=0).unsqueeze(0)
        foreground_mask = torch.stack(masks).numpy()
        del session, prepared, masks
    else:
        images = torch.stack([preprocess_image(path) for path in selected], dim=0).unsqueeze(0)
    images = images.to(device=device, dtype=compute_dtype if low_vram else torch.float32)
    profile = " · 8 GB low-memory mode" if low_vram else ""
    progress(directory, 20, f"Loading AnySplat pretrained weights on {torch.cuda.get_device_name(0)}{profile}")
    model = load_model(device, parameter_dtype=compute_dtype if low_vram else None)
    progress(directory, 42, f"Jointly reconstructing from {len(selected)} uncalibrated views")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=compute_dtype):
        gaussians, poses = model.inference(images)
    del images, model
    gc.collect()
    torch.cuda.empty_cache()
    progress(directory, 82, "Converting AnySplat output to portable Gaussian PLY")
    g = to_viewer_gaussians(gaussians, foreground_mask=foreground_mask, camera_poses=poses)
    predicted_count = int(gaussians.means.shape[1])
    del gaussians, poses
    gc.collect()
    torch.cuda.empty_cache()
    return g, {
        "fov_y": 50.0,
        "image_size": [448, 448],
        "engine": "AnySplat feed-forward multi-view reconstruction",
        "method": "anysplat",
        "device": "cuda",
        "precision": f"CUDA {str(compute_dtype).removeprefix('torch.')} autocast"
        + (" and model weights" if low_vram else "")
        + "; float32 PLY export",
        "licence": "AnySplat code and published model: MIT; see docs/MODEL-LICENSES.md for bundled third-party notices",
        "input_count": len(selected),
        "uploaded_count": len(paths),
        "selected_inputs": selected_names,
        "view_limit": requested_limit,
        "vram_gb": round(vram_gb, 1),
        "free_vram_gb_at_start": round(free_vram_gb, 1),
        "low_vram_mode": low_vram,
        "predicted_gaussians": predicted_count,
        "object_only": object_only,
        "foreground_coverage": mask_coverages,
        "foreground_pixels": int(np.count_nonzero(foreground_mask)) if foreground_mask is not None else None,
        "limitation": "Object-only mode segments the main subject in every view and excludes background Gaussians. Fine transparent parts or backgrounds similar to the object can still need cleaner source photos; unseen regions remain incomplete.",
    }


def main():
    directory = Path(sys.argv[1]).resolve()
    options = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        progress(directory, 5, "Checking inference environment")
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        device = choose_device(options["device"], options["engine"])
        try:
            if options["engine"] == "anysplat":
                g, meta = anysplat_predict(directory, options, device)
            else:
                first = options.get("inputs", [{"file": "input.png"}])[0]
                with Image.open(directory / first["file"]) as source:
                    image = source.convert("RGB")
                g, meta = sharp_predict(image, directory, options, device) if options["engine"] == "sharp" else depth_predict(image, directory, options, device)
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            if options["engine"] == "anysplat":
                message = "AnySplat ran out of GPU memory. On an RTX A1000, use Auto-safe or exactly 2 views and close QGIS/browser GPU-heavy work. No fallback output was substituted."
            elif options["engine"] == "sharp":
                message = "SHARP ran out of GPU memory. Close GPU-heavy programs or use a larger CUDA GPU. No fallback output was substituted."
            else:
                message = "The depth model ran out of GPU memory. Close GPU-heavy programs, reduce Detail, or select CPU. No fallback output was substituted."
            raise RuntimeError(message) from exc
        progress(directory, 88, "Writing Gaussian PLY and preparing the interactive scene")
        meta["seconds"] = round(time.monotonic() - started, 2)
        export_scene(directory, g, meta)
        progress(directory, 100, "Scene ready")
    except Exception as exc:
        (directory / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
