# Gaussian Scene Studio — AnySplat

Local Windows application for reconstructing an interactive 3D Gaussian scene from
multiple photographs. AnySplat jointly estimates camera poses and Gaussians from
uncalibrated, overlapping views. The Python backend exports a binary 3DGS `.ply`;
the existing WebGL2 viewer provides orbit, pan, zoom, presets, screenshots, and export.

The prior SHARP inference and labelled-view fusion path has been removed. The existing
Depth Anything V2 one-image fallback, isolated jobs, cancellation, history, upload
validation, procedural viewer test, PLY writer, and WebGL renderer remain available.

## NVIDIA workstation setup

These instructions are for Windows 10/11 with an NVIDIA RTX A1000 (8 GB VRAM).
Use **64-bit Python 3.11**. Do not use Python 3.12 or 3.13 for this project.

### Before running setup

1. Install the current NVIDIA workstation driver, restart Windows, and confirm that
   the RTX A1000 appears in Device Manager.
2. Install 64-bit Python 3.11 from python.org. Enable **Python Launcher** during the
   installer. Adding Python to `PATH` is also recommended.
3. Copy or clone this repository to `D:\gaussian_studio`. Copy the source code and,
   optionally, the `models` folder. **Do not copy `.venv` from another computer**;
   compiled packages in a virtual environment are machine-specific.
4. Keep at least 15 GB of disk space free. The first setup needs an internet connection
   and downloads CUDA-enabled PyTorch, the 2.94 GB AnySplat checkpoint, and the depth
   fallback model.

Open **PowerShell** and run these two read-only checks:

```powershell
nvidia-smi
py -3.11 -c "import sys; print(sys.version); print(sys.executable); print('64-bit:', sys.maxsize > 2**32)"
```

`nvidia-smi` confirms that Windows can communicate with the NVIDIA driver. The second
command confirms that the Python launcher can find a 64-bit Python 3.11 installation.
Stop here and fix the driver or Python installation if either command fails.

### Copy/paste setup and launch

Run this complete block from PowerShell:

```powershell
Set-Location -LiteralPath 'D:\gaussian_studio'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA
.\.venv\Scripts\python.exe scripts\doctor.py --require-cuda --require-anysplat
.\.venv\Scripts\python.exe scripts\verify_anysplat.py --check-hash --check-import
.\.venv\Scripts\python.exe run.py
```

What each command does:

| Command | Purpose |
|---|---|
| `Set-Location ...` | Changes PowerShell to the repository folder so every relative path below is correct. |
| `setup.ps1 -Device CUDA` | Creates `.venv` with Python 3.11, upgrades pip, installs the app dependencies, installs the pinned CUDA-enabled PyTorch build, installs AnySplat dependencies, downloads both models, verifies the AnySplat checkpoint and import, checks package compatibility, and performs a real CUDA calculation. |
| `doctor.py ...` | Rechecks Python, RAM, NVIDIA driver visibility, CUDA-enabled PyTorch, a real tensor calculation on the GPU, GPU name/VRAM, and the local AnySplat files. It exits with an error instead of silently continuing if CUDA or AnySplat is unavailable. |
| `verify_anysplat.py ...` | Verifies the pinned AnySplat source, imports it, and recalculates the downloaded checkpoint's SHA-256 hash. |
| `run.py` | Starts the local web application. Keep this PowerShell window open while using the app; press `Ctrl+C` to stop it. |

When startup completes, use `http://127.0.0.1:7860`. The setup script supplies the
PyTorch CUDA 12.8 runtime, so the full NVIDIA CUDA Toolkit, `nvcc`, Linux, WSL, Docker,
and a local gsplat build are not required for this app's PLY-only inference path.

If Python 3.11 is installed but the launcher cannot find it, give setup the exact path:

```powershell
Set-Location -LiteralPath 'D:\gaussian_studio'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA -PythonExe 'C:\Path\To\Python311\python.exe'
```

Replace the example Python path with the real `python.exe` path. Setup rejects the
wrong Python version or a 32-bit interpreter before installing anything.

### Safe reruns and repairs

It is safe to rerun the CUDA setup command after an interrupted download or failed
installation. It reuses the existing `.venv`, checks the installed PyTorch runtime,
repairs mismatched packages, resumes/rechecks model files, and repeats diagnostics.
To repair only a model, run:

```powershell
Set-Location -LiteralPath 'D:\gaussian_studio'
.\.venv\Scripts\python.exe scripts\download_models.py --model anysplat
.\.venv\Scripts\python.exe scripts\download_models.py --model depth
.\.venv\Scripts\python.exe scripts\verify_anysplat.py --check-hash --check-import
```

If a verified `models` folder was copied from another workstation, setup can avoid the
large model download. It still recreates and validates the local Python environment:

```powershell
Set-Location -LiteralPath 'D:\gaussian_studio'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA -SkipModelDownload
.\.venv\Scripts\python.exe scripts\verify_anysplat.py --check-hash --check-import
```

The double-click alternative performs the same main setup: run
**Setup NVIDIA Workstation.cmd**, then **Start Studio.cmd**. The explicit PowerShell
commands above are preferable when setting up a new workstation because failures remain
visible and the two verification commands give a clear pass/fail result.

For the first reconstruction on an RTX A1000, close other GPU-heavy programs, choose
**Auto-safe for GPU**, and start with two sharp, overlapping photographs of a rigid,
textured object. The worker refuses to start AnySplat below 5.5 GB of currently free
VRAM rather than crashing partway through. Uploaded images, job metadata, and generated
PLY files are stored under `data/` and are not committed to Git.

## Capture and reconstruction

- Upload 2–16 images for AnySplat. Walk around one stationary subject and keep adjacent
  views overlapping. Keep lighting, focus, orientation, and subject scale consistent.
- Upload order should follow the camera path. If a view budget selects fewer images,
  the app samples them evenly in that order.
- **Auto-safe for GPU** uses two views when at least 5.5 GB is free on an 8 GB GPU,
  four on 12 GB, six on 16 GB,
  ten on 24 GB, and up to sixteen above 24 GB. This is a conservative heuristic,
  not a memory guarantee. **Use every upload** attempts all selected files together.
- More images help only when they show useful, overlapping evidence. Blurry, unrelated,
  or inconsistent views can reduce reconstruction quality.
- AnySplat requires CUDA in this application. For a CPU-only computer, choose
  **Depth Anything V2 · one image** to keep using the earlier visible-surface workflow.

The model always receives 448×448 centre crops, matching the official AnySplat demo.
On GPUs with 8.5 GB VRAM or less, including the RTX A1000, the worker transfers model
weights to CUDA directly in BF16 (FP16 on GPUs without BF16 support). This avoids a
temporary float32 GPU copy and saves roughly 1.5 GB. Auto-safe also considers currently
free VRAM, so close other GPU-heavy applications if it reports less than 5.5 GB free.
The export keeps at most two million strongest valid Gaussians. The interactive preview
keeps at most 1.5 million. Metadata records uploads, processed inputs, view budget,
GPU, VRAM, compute time, model name, counts, and known limitations.

## Architecture

```text
2–16 photographs
  -> FastAPI upload validation and ordered local job files
  -> isolated CUDA worker
  -> AnySplat native joint multi-view inference
  -> camera/coordinate and quaternion conversion
  -> portable 3DGS PLY + bounded WebGL preview + metadata
  -> local anisotropic Gaussian splatting viewer

1 photograph
  -> Depth Anything V2 Small
  -> relative-depth Gaussian lifting
  -> same PLY and viewer
```

The AnySplat adapter uses the official source at pinned commit
`5f5e208a7dd57d52e43ea0d553a95eab526e8775` and the Hugging Face model at pinned
revision `d2e8c343672646041ad4ea518184968f94362f01`. The 2.94 GB checkpoint SHA-256 is
`1c4de2ba5a29c540b899af901bf02107395b5f0617655d347e262f814b4c0c7c`.

The upstream demo depends on Linux-focused gsplat, xFormers, and torch-scatter builds.
This app does not use the upstream CUDA video decoder. It uses PyTorch scaled-dot-product
attention and scatter operations for the same inference roles, then renders the result
with the existing WebGL2 splatter. The full AnySplat model still runs; this is not a
mock, point cloud substitute, or per-image fusion path.

## Commands

```powershell
# CPU setup with the one-image depth fallback only
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CPU

# Download or repair models manually
.\.venv\Scripts\python.exe scripts\download_models.py --model anysplat
.\.venv\Scripts\python.exe scripts\download_models.py --model depth

# Verify source/import/checkpoint and local tests
.\.venv\Scripts\python.exe scripts\verify_anysplat.py --check-import --check-hash
.\.venv\Scripts\python.exe -m pytest -q
```

## Troubleshooting

| Symptom | Action |
|---|---|
| AnySplat model missing | Run `Setup NVIDIA Workstation.cmd`, or download `--model anysplat`. |
| CUDA is false | Rerun NVIDIA setup. `nvidia-smi` alone does not prove CUDA-enabled PyTorch works. |
| CUDA out of memory | On the RTX A1000, close QGIS/browser GPU-heavy apps and use Auto-safe or exactly 2 views. |
| Poor/fragmented scene | Use sharper adjacent views with more overlap and consistent framing. |
| Blank viewport | Enable browser graphics acceleration, restart Chrome/Edge, and load the calibration scene. |
| Download blocked | Use an approved network/proxy, or copy `models/anysplat` from a completed setup. Do not copy `.venv`. |

See `docs/VALIDATION.md` for tested and untested boundaries. See
`docs/MODEL-LICENSES.md` before commercial distribution.

## Main files

- `studio/server.py`: local API, ordered multi-upload validation, jobs, history, cancellation.
- `studio/worker.py`: AnySplat and depth inference orchestration.
- `studio/anysplat_runtime.py`: Windows inference shims, model loading, preprocessing, conversion.
- `studio/gaussians.py`: validation, binary PLY, preview buffer, depth lifting.
- `static/`: UI, WebGL2 renderer, and depth-sort worker.
- `vendor/anysplat/`: pinned upstream source and licence.
- `models/`: downloaded local weights; ignored by Git and omitted from source ZIPs.

Sources: [AnySplat repository](https://github.com/OpenRobotLab/AnySplat),
[AnySplat model](https://huggingface.co/lhjiang/anysplat),
[AnySplat paper](https://arxiv.org/abs/2505.23716), and
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf).
