# Gaussian Scene Studio

A local Windows app with a Python backend and an interactive Gaussian-splatting
viewer. Upload JPEG, PNG or WebP, reconstruct, orbit/pan/zoom, export a real 3DGS
`.ply`, and save snapshots of the current view. No paid API or cloud inference.

**Single-image limitation:** camera navigation works through 360 degrees, but a
single photograph does not contain the true back or occluded sides. Neither
included pipeline reconstructs a complete, reliable all-sides scene. Nearby
viewpoints look best; use multiple real viewpoints for complete reconstruction.
This is not a mesh generator, a multi-view 3DGS trainer, or a hidden-side generator.

## Start on this computer

Open **Start Studio.cmd** in `D:\GaussianSceneStudio`.

The app opens in Chrome/Edge/your default browser at `http://127.0.0.1:7860`.
If that port is occupied it reserves the next free port and prints the address.
Keep its terminal open; press Ctrl+C to stop the server and active inference.

1. The initial blue sculpture is a **procedural renderer test**, not AI output.
2. Upload a photo, select **Depth Anything V2 · lightweight**, leave Compute on
   **Auto detect**, and choose **Create 3D scene**.
3. Drag to orbit; right-drag or Shift-drag to pan; scroll to zoom. Arrow keys
   rotate, + / - zoom, R resets. Source / Left / Right / Back / Top are camera presets.
4. **Export .ply** downloads every valid Gaussian. **Save image** saves the viewport.

The first install downloads Python dependencies and weights. Once installed,
inference and rendering work offline. Images remain in `data/jobs/` on this PC.

## Tomorrow: NVIDIA RTX A1000 workstation

The supplied photos show **NVIDIA RTX A1000, 8188 MiB VRAM, driver 580.97,
and nvidia-smi CUDA Version 13.0**. That CUDA number is the driver's supported
CUDA level, not proof that the development toolkit or CUDA-enabled PyTorch is
installed. This app uses PyTorch 2.8.0 + CUDA 12.8 prebuilt wheels. The newer
driver can support that runtime. No CUDA Toolkit, `nvcc`, Visual Studio compiler,
gsplat extension build, Node.js, Docker or WSL is required for these app paths.

1. Transfer the application ZIP, extract it as `D:\GaussianSceneStudio`.
   Do **not** copy `.venv` from another computer. Virtual environments are not portable.
   The main ZIP includes the lightweight model. The optional `SHARP-weights.zip`
   can be extracted into the same parent directory to add the already-downloaded
   SHARP checkpoint without downloading it again. Both ZIPs contain the same
   `GaussianSceneStudio` top-level folder. You can also copy `models` directly
   from this computer's `D:\GaussianSceneStudio`.
2. Install **64-bit Python 3.11** from https://www.python.org/downloads/windows/.
   Include the Python launcher. If Python is already installed, check `py -3.11 --version`.
3. Double-click **Setup NVIDIA Workstation.cmd**. It installs CUDA-enabled
   PyTorch, dependencies and the small depth model; it checks actual CUDA tensor
   computation before reporting success. Internet is needed for package downloads.
4. Open **Start Studio.cmd** and try Depth Anything with **Auto detect** first.
   The device card should show the RTX A1000. Scene details record the actual device.

PowerShell equivalent, including an explicit Python path if needed:

```powershell
cd D:\GaussianSceneStudio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA
# Or add: -PythonExe 'C:\Path\To\Python311\python.exe'
.\.venv\Scripts\python.exe scripts\doctor.py --require-cuda
.\.venv\Scripts\python.exe run.py
```

The lightweight model is the safer starting point for an 8 GB GPU. SHARP uses
considerably more VRAM/RAM; its CUDA path has not been validated on this A1000.
Close QGIS and other GPU-heavy applications before testing SHARP. For SHARP,
16 GB system RAM is a practical minimum and 32 GB gives more headroom.

## Optional direct image-to-Gaussian model: Apple SHARP

**Read `docs/MODEL-LICENSES.md` first. SHARP's model licence is research-only,
and excludes commercial product development.** The default depth model is Apache
licensed. Both are free to download under their respective terms.

For eligible non-commercial scientific research, install SHARP:

```powershell
cd D:\GaussianSceneStudio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA -IncludeSharp
```

This also downloads the approximately 2.6 GB SHARP checkpoint. Then select
**Apple SHARP · direct Gaussians** in the app and acknowledge its permitted use.
The first model load may take time. CUDA inference uses mixed precision to
reduce memory; covariance unprojection runs in float32 on CPU. The fixed 1536×1536
model input is preserved. SHARP does not use the depth reconstruction detail slider.

No claim is made that SHARP always fits in 8 GB VRAM. A failed CUDA allocation
produces an actionable error and does not silently substitute a different model.
You can choose CPU for SHARP if the workstation has sufficient RAM; CPU mode is
disabled below 14 GB total RAM to avoid exhausting this laptop's memory.

The supplied SHARP checkpoint is SHA-256 verified. Its 1,038 tensor names/shapes
match the pinned 702,305,169-parameter model. This compatibility check does not
establish CUDA inference correctness or runtime memory requirements.

## What is implemented

```text
Photograph -> Python/FastAPI -> isolated inference worker
  A: Depth Anything V2 Small -> relative inverse depth -> Gaussian lifting
  B: Apple SHARP             -> pretrained Gaussian parameter prediction
             -> 3DGS binary PLY + scene metadata + preview buffer
             -> local WebGL2 Gaussian projection + sorted alpha compositing
             -> mouse/keyboard-controlled 3D view
```

- **Depth path:** official 24.8M-parameter pretrained small model. Inverse depth is
  normalized, converted to relative distance, and unprojected through an estimated
  pinhole camera. Local depth derivatives determine surface orientation. Each sample
  becomes an anisotropic Gaussian with RGB colour, opacity, three scales and rotation.
  Surface discontinuities shrink splats to reduce bridges. There is no training or
  multi-view optimization, and distance/scale is not metrically calibrated.
- **SHARP path:** pinned official model code, weights-only checkpoint loading, fixed
  model resolution, official Gaussian unprojection, explicit OpenCV-to-viewer rotation
  and linear-RGB-to-sRGB conversion. Focal length is estimated, so absolute scale may
  differ from the original scene. No compilation-dependent gsplat rendering import.
- **Renderer:** custom WebGL2 code in `static/renderer.js`. Computes
  `Sigma3 = R diag(s^2) R^T`, projects with `Sigma2 = J V Sigma3 V^T J^T + 0.3 I`,
  draws 3-sigma elliptical quads, evaluates `alpha = opacity * exp(-r^2/2)`, and
  composites in back-to-front order. A worker performs 16-bit depth-bin sorting.
  This is real anisotropic Gaussian rasterization, not a point cloud or textured plane.
- **Preview budget:** at most 260,000 Gaussians. Larger scenes use deterministic
  uniform sampling and enlarged tangential footprints; full PLY retains all valid
  Gaussians. The UI reports preview and full counts. Preview appearance is approximate.
- **Format:** binary little-endian PLY, SH degree 0 colour, log-scale, logit-opacity,
  scalar-first quaternion. +X right, +Y up, -Z forward. Colours use the common sRGB
  PLY convention; blending is in display RGB, not physically exact linear light.
- **Operations:** single inference at a time, process cancellation, 30-minute timeout,
  restart recovery, per-job logs, upload checks, local-only HTTP server.

## Troubleshooting

| Symptom | Action |
|---|---|
| Python not found | Install Python 3.11 x64, or provide `-PythonExe` to setup.ps1. |
| CUDA false but nvidia-smi works | Run CUDA setup; the old environment may contain CPU-only PyTorch. |
| Model missing | Run `.venv\Scripts\python.exe scripts\download_models.py --model depth` (or `sharp`). |
| Download blocked at office | Use a permitted network/proxy or copy the already-downloaded `models` and `vendor` folders; do not copy `.venv`. |
| CUDA out of memory | Close GPU-heavy programs; use Depth Anything, or explicitly select SHARP CPU with adequate RAM. |
| SHARP source or import error | Rerun `setup.ps1 -Device CUDA -IncludeSharp`; read the job log. |
| Blank 3D viewport | Enable browser graphics acceleration, restart Chrome/Edge, click Source or Reset view. |
| Holes/streaks at the back | Expected single-image limitation. Move nearer the source view; complete geometry needs multiple views. |
| Image has shallow depth | Try another photo with perspective, or adjust Depth amount in the depth model settings. |
| App says disconnected | Keep the Python terminal running; restart Start Studio.cmd and reload its printed address. |

Manual models-only installation and CPU setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CPU
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

See `docs/VALIDATION.md` for measured local checks and the remaining GPU checks.

## Files

- `studio/server.py`: API, upload validation, subprocess jobs, history and cancellation.
- `studio/worker.py`: real model inference and SHARP adapter.
- `studio/gaussians.py`: Gaussian construction, PLY and preview export.
- `static/`: local interface, WebGL shader renderer and depth-sort worker.
- `scripts/`: diagnostics and pinned model/source downloads.
- `data/jobs/<id>/`: input, thumbnail, progress, output PLY, buffer, metadata and logs.
- `models/`: local weights. `vendor/ml-sharp/`: pinned optional upstream source/licences.

Sources: [SHARP](https://github.com/apple/ml-sharp),
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
[PyTorch 2.8 installation](https://pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/).
