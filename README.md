# Gaussian Scene Studio — AnySplat

Local Windows application for reconstructing an interactive 3D Gaussian scene from
multiple photographs. AnySplat jointly estimates camera poses and Gaussians from
uncalibrated, overlapping views. The Python backend exports a binary 3DGS `.ply`;
the existing WebGL2 viewer provides orbit, pan, zoom, presets, screenshots, and export.

The prior SHARP inference and labelled-view fusion path has been removed. The existing
Depth Anything V2 one-image fallback, isolated jobs, cancellation, history, upload
validation, procedural viewer test, PLY writer, and WebGL renderer remain available.

## NVIDIA workstation setup

The supplied workstation screenshot shows an NVIDIA RTX A1000 with 8188 MiB VRAM,
driver 580.97, and a CUDA 13.0 driver capability. Use 64-bit Python 3.11.

1. Place this folder at `D:\GaussianSceneStudio`.
2. Double-click **Setup NVIDIA Workstation.cmd**. Internet is needed on first setup.
3. Double-click **Start Studio.cmd**.
4. Open `http://127.0.0.1:7860` if the browser does not open automatically.

Setup installs PyTorch 2.8 with its CUDA 12.8 runtime, downloads the pinned AnySplat
checkpoint (about 2.94 GB), verifies its SHA-256, downloads Depth Anything V2 Small,
checks the vendored AnySplat import, and performs a real CUDA tensor calculation.
The NVIDIA CUDA Toolkit, `nvcc`, Linux, WSL, Docker, and a local gsplat build are not
required for this app's PLY-only inference path.

PowerShell equivalent:

```powershell
cd D:\GaussianSceneStudio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA
.\.venv\Scripts\python.exe scripts\doctor.py --require-cuda --require-anysplat
.\.venv\Scripts\python.exe run.py
```

## Capture and reconstruction

- Upload 2–16 images for AnySplat. Walk around one stationary subject and keep adjacent
  views overlapping. Keep lighting, focus, orientation, and subject scale consistent.
- Upload order should follow the camera path. If a view budget selects fewer images,
  the app samples them evenly in that order.
- **Auto-safe for GPU** uses two views on an 8 GB GPU, four on 12 GB, six on 16 GB,
  ten on 24 GB, and up to sixteen above 24 GB. This is a conservative heuristic,
  not a memory guarantee. **Use every upload** attempts all selected files together.
- More images help only when they show useful, overlapping evidence. Blurry, unrelated,
  or inconsistent views can reduce reconstruction quality.
- AnySplat requires CUDA in this application. For a CPU-only computer, choose
  **Depth Anything V2 · one image** to keep using the earlier visible-surface workflow.

The model always receives 448×448 centre crops, matching the official AnySplat demo.
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
| CUDA out of memory | Close QGIS and GPU-heavy apps. Use Auto-safe or fewer views. |
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
