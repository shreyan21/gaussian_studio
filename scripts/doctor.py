import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.config import DEPTH_DIR, SHARP_WEIGHTS


def check():
    import psutil
    result = {"status": "ready", "python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(), "ram_gb": round(psutil.virtual_memory().total/1024**3,1), "available_ram_gb": round(psutil.virtual_memory().available/1024**3,1), "cuda": False, "depth_model": (DEPTH_DIR / "model.safetensors").exists(), "sharp_model": SHARP_WEIGHTS.exists()}
    try:
        import torch
        result.update(torch=torch.__version__, torch_cuda_runtime=torch.version.cuda, cuda=torch.cuda.is_available())
        if result["cuda"]:
            gpu = torch.cuda.get_device_properties(0)
            # Test actual CUDA compute; nvidia-smi alone is not sufficient.
            test = torch.tensor([2.,3.], device="cuda").square().sum().item()
            result.update(gpu=gpu.name, vram_gb=round(gpu.total_memory/1024**3,1), cuda_compute_test=test == 13.)
        else:
            result["gpu"] = "CPU inference / browser GPU rendering"
    except Exception as exc:
        result.update(status="attention", error=str(exc))
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if p.returncode == 0:
            result["nvidia_driver"] = p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    info = check()
    print(json.dumps(info, indent=2))
    if args.require_cuda and not (info.get("cuda") and info.get("cuda_compute_test")):
        print("CUDA compute failed. Check the driver and install the CUDA PyTorch build with setup.ps1 -Device CUDA.", file=sys.stderr)
        raise SystemExit(1)
