"""One inference process per job: cancellation and GPU memory release are deterministic."""
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
from studio.gaussians import export_scene, from_depth, fuse_gaussian_views


def progress(directory, percent, message):
    p = directory / "progress.tmp"
    p.write_text(json.dumps({"progress": percent, "message": message}), encoding="utf-8")
    os.replace(p, directory / "progress.json")
    print(message, flush=True)


def choose_device(requested):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Run setup.ps1 -Device CUDA on the workstation, or select CPU.")
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
    return g, {**camera, "engine": "Depth Anything V2 Small + Gaussian lifting", "method": "depth", "limitation": "Relative-depth reconstruction of visible surfaces. Hidden sides are not generated; scale is not metric.", "licence": "Apache-2.0 model", "device": device}


def load_sharp_predictor(directory, device):
    import torch
    import psutil
    if not SHARP_WEIGHTS.is_file() or not (SHARP_SOURCE / "sharp" / "models" / "__init__.py").is_file():
        raise RuntimeError("SHARP is not installed. Run setup.ps1 -Device CUDA -IncludeSharp, then retry.")
    if device == "cpu" and psutil.virtual_memory().total < 14 * 1024**3:
        raise RuntimeError("SHARP CPU mode is disabled on computers with less than 14 GB RAM to avoid exhausting memory. Use the lightweight depth model here, or SHARP on your workstation.")
    sys.path.insert(0, str(SHARP_SOURCE))
    from sharp.models import PredictorParams, create_predictor
    progress(directory, 12, "Loading SHARP pretrained weights on " + device.upper())
    state = torch.load(str(SHARP_WEIGHTS), map_location="cpu", weights_only=True, mmap=True)
    # Allocate model structure without a second 2.6 GB initialized parameter copy.
    # assign=True adopts checkpoint tensors, then .to(device) materializes CUDA data.
    with torch.device("meta"):
        predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state, strict=True, assign=True)
    del state
    predictor.eval().to(device)
    return predictor


def sharp_predict(image, directory, options, device, predictor=None, progress_values=(15, 40, 70), label="image"):
    import torch
    import torch.nn.functional as F
    from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
    owns_predictor = predictor is None
    if predictor is None:
        predictor = load_sharp_predictor(directory, device)
    width, height = image.size
    # SHARP's fixed internal resolution must not be changed to claim VRAM savings.
    # Match SHARP's official 30 mm full-frame fallback when upload EXIF is absent.
    focal_35mm = options.get("focal_35mm") or 30.0
    focal = focal_35mm * np.hypot(width, height) / np.hypot(36, 24)
    input_tensor = torch.from_numpy(np.asarray(image).copy()).to(device).float().permute(2, 0, 1)[None] / 255
    input_tensor = F.interpolate(input_tensor, (1536, 1536), mode="bilinear", align_corners=True)
    factor = torch.tensor([focal / width], dtype=torch.float32, device=device)
    progress(directory, progress_values[1], f"Predicting Gaussians from {label}")
    # Autocast reduces activation memory on 8 GB GPUs; SVD/unprojection stays float32 on CPU.
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        raw = predictor(input_tensor, factor)
    raw = Gaussians3D(*(t.detach().float().cpu() for t in raw))
    del input_tensor, factor
    if owns_predictor:
        del predictor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    progress(directory, progress_values[2], f"Unprojecting and aligning {label}")
    k = torch.tensor([[focal,0,width/2,0],[0,focal,height/2,0],[0,0,1,0],[0,0,0,1]], dtype=torch.float32)
    k[0] *= 1536/width
    k[1] *= 1536/height
    with torch.inference_mode():
        gaussian = unproject_gaussians(raw, torch.eye(4), k, (1536,1536))
    means = gaussian.mean_vectors.reshape(-1,3).numpy()
    g = np.zeros((len(means),16), np.float32)
    g[:,:3] = means * [1,-1,-1]
    g[:,3] = gaussian.opacities.reshape(-1).numpy()
    g[:,4:7] = gaussian.singular_values.reshape(-1,3).numpy()
    # OpenCV -> viewer rotation: q' = quaternion(180deg about X) * q.
    q = gaussian.quaternions.reshape(-1,4).numpy()
    g[:,8:12] = np.stack((-q[:,1], q[:,0], -q[:,3], q[:,2]), axis=-1)
    linear = np.clip(gaussian.colors.reshape(-1,3).numpy(), 0, 1)
    g[:,12:15] = np.where(linear <= 0.0031308, linear*12.92, 1.055*linear**(1/2.4)-0.055)
    return g, {"fov_y": float(np.degrees(2*np.arctan(height/(2*focal)))), "image_size": [width,height], "engine": "Apple SHARP", "method": "sharp", "device": device, "precision": "CUDA mixed float16; CPU float32 postprocessing" if device == "cuda" else "float32", "licence": "Apple ML Research Model: non-commercial scientific research only", "limitation": "Single-image Gaussian prediction for nearby views. Unseen sides are not a complete 360-degree reconstruction; focal length is estimated."}


def main():
    directory = Path(sys.argv[1]).resolve()
    options = json.loads((directory / "request.json").read_text())
    started = time.monotonic()
    try:
        progress(directory, 5, "Checking inference environment")
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        device = choose_device(options["device"])
        try:
            inputs = options.get("inputs") or [{"file": "input.png", "view": "front", "focal_35mm": options.get("focal_35mm", 30.0)}]
            if options["engine"] == "sharp":
                predictor = load_sharp_predictor(directory, device)
                predicted = []
                first_meta = None
                count = len(inputs)
                for index, spec in enumerate(inputs):
                    start = 15 + round(index * 68 / count)
                    end = 15 + round((index + 1) * 68 / count)
                    middle = start + max(1, (end-start) // 2)
                    with Image.open(directory / spec["file"]) as source:
                        image = source.convert("RGB")
                    per_view_options = {**options, "focal_35mm": spec.get("focal_35mm", 30.0)}
                    label = f"{spec['view']} view ({index+1}/{count})"
                    one, one_meta = sharp_predict(image, directory, per_view_options, device, predictor, (start, middle, end), label)
                    predicted.append((one, spec["view"]))
                    first_meta = first_meta or one_meta
                del predictor
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                if count > 1:
                    progress(directory, 85, f"Fusing {count} labelled SHARP predictions")
                    g = fuse_gaussian_views(predicted)
                else:
                    g = predicted[0][0]
                meta = {
                    **first_meta,
                    "engine": "Apple SHARP multi-view fusion",
                    "method": "sharp-multiview" if count > 1 else "sharp",
                    "input_views": [spec["view"] for spec in inputs],
                    "input_count": count,
                    "limitation": (
                        "Geometric fusion of independent SHARP predictions aligned from labelled views. "
                        "SHARP does not jointly condition on multiple images, so unobserved areas are predictions and seams may remain."
                        if count > 1 else first_meta["limitation"]
                    ),
                }
            else:
                with Image.open(directory / inputs[0]["file"]) as source:
                    image = source.convert("RGB")
                g, meta = depth_predict(image, directory, options, device)
        except torch.cuda.OutOfMemoryError as exc:
            if options["engine"] == "sharp":
                message = "SHARP ran out of GPU memory. Close GPU-heavy programs and retry; multi-view images are processed one at a time, so adding views should not increase peak model VRAM. No output was substituted."
            else:
                message = "The depth model ran out of GPU memory. Close GPU-heavy programs, reduce Detail, or select CPU. No output was substituted."
            raise RuntimeError(message) from exc
        progress(directory, 88, "Writing Gaussian PLY and preparing the interactive scene")
        meta["seconds"] = round(time.monotonic()-started,2)
        export_scene(directory, g, meta)
        progress(directory, 100, "Scene ready")
    except Exception as exc:
        (directory / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
