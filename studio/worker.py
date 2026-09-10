"""One inference process per job: cancellation and GPU release are deterministic."""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from studio.config import DEPTH_DIR
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


def anysplat_predict(directory, options, device):
    import torch
    from studio.anysplat_runtime import automatic_view_limit, load_model, preprocess_image, select_inputs, to_viewer_gaussians

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
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    low_vram = vram_gb <= 8.5
    progress(directory, 10, f"Preparing {len(selected)} of {len(paths)} uploaded views at 448 x 448")
    images = torch.stack([preprocess_image(path) for path in selected], dim=0).unsqueeze(0)
    images = images.to(device=device, dtype=compute_dtype if low_vram else torch.float32)
    profile = " · 8 GB low-memory mode" if low_vram else ""
    progress(directory, 20, f"Loading AnySplat pretrained weights on {torch.cuda.get_device_name(0)}{profile}")
    model = load_model(device, parameter_dtype=compute_dtype if low_vram else None)
    progress(directory, 42, f"Jointly reconstructing from {len(selected)} uncalibrated views")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=compute_dtype):
        gaussians, poses = model.inference(images)
    del images, poses, model
    gc.collect()
    torch.cuda.empty_cache()
    progress(directory, 82, "Converting AnySplat output to portable Gaussian PLY")
    g = to_viewer_gaussians(gaussians)
    predicted_count = int(gaussians.means.shape[1])
    del gaussians
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
        "limitation": "Feed-forward reconstruction from uncalibrated overlapping views. Regions never seen in any input can remain incomplete; output is capped at two million strongest Gaussians for portable viewing.",
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
                g, meta = depth_predict(image, directory, options, device)
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            if options["engine"] == "anysplat":
                message = "AnySplat ran out of GPU memory. On an RTX A1000, use Auto-safe or exactly 2 views and close QGIS/browser GPU-heavy work. No fallback output was substituted."
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
