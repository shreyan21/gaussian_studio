"""Verify bundled source, checkpoint identity, and optional import contract."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.anysplat_runtime import _install_inference_shims, cuda_inference_profile, model_ready
from studio.config import ANYSPLAT_COMMIT, ANYSPLAT_DIR, ANYSPLAT_ROOT, ANYSPLAT_SHA256

REQUIRED = (
    "src/model/model/anysplat.py",
    "src/model/encoder/anysplat.py",
    "src/model/encoder/vggt/models/aggregator.py",
    "src/model/encoder/vggt/models/vggt.py",
    "src/model/types.py",
    "LICENSE",
)


def source_status():
    marker_path = ANYSPLAT_ROOT / "PINNED_COMMIT"
    marker = marker_path.read_text(encoding="utf-8").strip() if marker_path.is_file() else ""
    missing = [name for name in REQUIRED if not (ANYSPLAT_ROOT / name).is_file()]
    return {"commit": marker, "expected_commit": ANYSPLAT_COMMIT, "missing": missing, "ready": marker == ANYSPLAT_COMMIT and not missing}


def checkpoint_sha256():
    path = ANYSPLAT_DIR / "model.safetensors"
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_status():
    import torch
    if str(ANYSPLAT_ROOT) not in sys.path:
        sys.path.insert(0, str(ANYSPLAT_ROOT))
    _install_inference_shims(torch)
    from src.model.encoder.vggt.models.vggt import VGGT
    VGGT.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls())
    from src.model.model.anysplat import AnySplat
    return AnySplat.__name__ == "AnySplat"


def inference_status():
    """Run one real, two-view CUDA forward pass."""
    import torch
    from studio.anysplat_runtime import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_vram_gb = free_bytes / 1024**3
    total_vram_gb = total_bytes / 1024**3
    if free_vram_gb < 5.5:
        raise RuntimeError(
            f"AnySplat needs at least 5.5 GB of free GPU memory; only {free_vram_gb:.1f} GB is free. "
            "Close QGIS, browser GPU tabs and other GPU-heavy programs, then rerun setup."
        )
    low_vram, compute_dtype = cuda_inference_profile(torch, total_vram_gb)
    grid = torch.linspace(0, 1, 448, device="cuda")
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    first = torch.stack((xx, yy, (xx + yy) / 2))
    second = torch.roll(first, shifts=8, dims=2)
    images = torch.stack((first, second), dim=0).unsqueeze(0)
    images = images.to(dtype=compute_dtype if low_vram else torch.float32)
    model = load_model("cuda", parameter_dtype=compute_dtype if low_vram else None)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=compute_dtype):
        gaussians, poses = model.inference(images)
    count = int(gaussians.means.shape[1])
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    del gaussians, poses, images, model
    torch.cuda.empty_cache()
    if count <= 0:
        raise RuntimeError("AnySplat returned no Gaussians")
    return {
        "gaussians": count,
        "peak_vram_gb": peak,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(total_vram_gb, 1),
        "free_vram_gb_at_start": round(free_vram_gb, 1),
        "precision": str(compute_dtype).removeprefix("torch.") if low_vram else "float32 weights",
        "low_vram_mode": low_vram,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hash", action="store_true")
    parser.add_argument("--check-import", action="store_true")
    parser.add_argument("--check-inference", action="store_true")
    args = parser.parse_args()
    result = {"source": source_status(), "model_ready": model_ready()}
    if args.check_hash:
        result["checkpoint_sha256"] = checkpoint_sha256()
        result["checkpoint_hash_ok"] = result["checkpoint_sha256"] == ANYSPLAT_SHA256
    if args.check_import:
        try:
            result["import_ok"] = import_status()
        except Exception as exc:
            result["import_ok"] = False
            result["import_error"] = str(exc)
    if args.check_inference:
        try:
            result["inference"] = inference_status()
            result["inference_ok"] = True
        except Exception as exc:
            result["inference_ok"] = False
            result["inference_error"] = str(exc)
    print(json.dumps(result, indent=2))
    ok = result["source"]["ready"]
    if args.check_hash:
        ok = ok and result["checkpoint_hash_ok"]
    if args.check_import:
        ok = ok and result["import_ok"]
    if args.check_inference:
        ok = ok and result["inference_ok"]
    raise SystemExit(0 if ok else 1)
