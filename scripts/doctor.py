import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.custom_sfm import engine_ready, find_colmap


def check():
    import psutil
    executable = find_colmap()
    result = {
        "status": "ready",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "available_ram_gb": round(psutil.virtual_memory().available / 1024**3, 1),
        "custom_engine": engine_ready(),
        "colmap": str(executable) if executable else None,
        "cuda": False,
    }
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if probe.returncode == 0:
            result["cuda"] = True
            result["nvidia_driver"] = probe.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not result["custom_engine"] or not result["cuda"]:
        result["status"] = "attention"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-custom", action="store_true")
    args = parser.parse_args()
    info = check()
    print(json.dumps(info, indent=2))
    if args.require_custom and not (info["custom_engine"] and info["cuda"]):
        print("CUDA COLMAP is not ready. Rerun Setup NVIDIA Workstation.cmd and verify nvidia-smi.", file=sys.stderr)
        raise SystemExit(1)
