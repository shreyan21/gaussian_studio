# Gaussian Scene Studio

Local Python app for reconstructing an interactive 3D Gaussian scene. Apple SHARP is the clearest single-image research mode; AnySplat handles overlapping multi-view photographs, and Depth Anything remains a lightweight fallback.

SHARP scenes intentionally stop at ±30° horizontal and ±18° vertical rotation. A single photograph cannot reveal the back of an object, so the viewer blocks movement beyond the useful predicted novel-view range instead of displaying broken geometry.

AnySplat enables **Isolate main object** by default: each view is segmented and cropped around the focused subject, neutral background pixels are withheld from the exported Gaussians, and rotations in the viewer show the reconstructed object instead of the surrounding scene.

## Office workstation: quickest setup

The photographed workstation has an NVIDIA RTX A1000 with 8 GB VRAM. Its displayed driver (580.97) is new enough for the CUDA 12.8 PyTorch build used here. Keep the driver; install Python 3.11 and Git if missing.

Before setup, confirm Windows sees both GPU and Python:

```powershell
nvidia-smi
py -3.11 -c "import sys; print(sys.version); print('64-bit:', sys.maxsize > 2**32)"
```

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
```

Open a new PowerShell window, then run:

```powershell
git clone https://github.com/shreyan21/gaussian_studio.git D:\GaussianSceneStudio
Set-Location D:\GaussianSceneStudio
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA
```

Setup creates `.venv`, installs pinned dependencies, downloads the AnySplat weights, verifies their SHA-256 checksum, and runs a real two-view CUDA inference smoke test. First setup needs internet and several GB of free disk space.

To additionally install SHARP, explicitly accept its research-only use and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA -IncludeSharp
```

If `D:\GaussianSceneStudio` already exists, update instead:

```powershell
Set-Location D:\GaussianSceneStudio
git pull --ff-only
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA
```

Start local app:

```powershell
.\Start Studio.cmd
```

Open `http://127.0.0.1:7860`.

## Temporary public link

Close any already-running Studio window, then run:

```powershell
.\Start Public Link.cmd
```

On first run it downloads the official Cloudflare tunnel client into the ignored local `tools` folder. It starts Studio, prints a temporary `trycloudflare.com` URL containing a random access token, and opens it in the browser.

- Keep the terminal window open.
- Share the full protected URL only with trusted people.
- Closing the window stops the app and tunnel.
- Each run creates a new URL and token.
- Quick Tunnels are for testing, not production hosting.

No public access is enabled by `Start Studio.cmd`; normal mode remains loopback-only.

## Manual validation commands

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py --require-cuda --require-anysplat
.\.venv\Scripts\python.exe scripts\verify_anysplat.py --check-import --check-hash --check-inference
.\.venv\Scripts\python.exe -m pytest -q tests
```

## Best settings for RTX A1000 8 GB

- Close QGIS, browsers using WebGL, and other GPU-heavy programs before reconstruction.
- Use **Auto-safe** view count. It measures free VRAM, not only total VRAM.
- Start with 2 adjacent, high-overlap images. Increase only after a successful run.
- Keep input images at the default processing resolution.
- If CUDA reports out-of-memory, restart Studio and use fewer views.
- The worker refuses AnySplat below 5.5 GB free VRAM instead of beginning a likely crash.
- On GPUs up to 8.5 GB, model weights and inputs load directly in BF16, or FP16 when BF16 is unavailable, avoiding a temporary float32 GPU copy.

The app can accept more photographs than fit into one 8 GB inference pass, but more input images do not create missing viewpoints automatically.

## Photograph capture rules

3D quality depends more on capture consistency than GPU size:

1. Keep the object and background completely still; move only the camera.
2. Capture a smooth arc or full circle in order, with 60-80% overlap.
3. Keep distance, zoom, exposure, focus, and lighting consistent.
4. Avoid motion blur, reflections, transparent objects, and featureless backgrounds.
5. For a standalone object, capture 20-40 views around it, then another slightly higher ring.

The four shoe photographs previously tested moved the shoe relative to the background. That violates multi-view geometry and produces separated fragments even when inference itself succeeds.

## Backend modes

- **Apple SHARP:** one image; clear nearby novel views, with hard camera limits. The model is restricted to non-commercial scientific research.
- **AnySplat:** two or more overlapping views; produces a Gaussian PLY and interactive viewer.
- **Depth Anything:** one-image geometric preview; cannot reconstruct unseen sides.

## Commercial-use warning

The AnySplat repository and released model are published under MIT, but this project also vendors or depends on third-party components with their own notices. Some current vendored source files contain non-commercial Creative Commons notices. Therefore, this package is **not yet represented as commercially cleared**. Review `docs/MODEL-LICENSES.md`, replace or obtain permission for restricted components, and obtain legal review before commercial distribution.

## Technical integration

The AnySplat adapter pins source commit `5f5e208a7dd57d52e43ea0d553a95eab526e8775` and model revision `d2e8c343672646041ad4ea518184968f94362f01`. It uses PyTorch scaled-dot-product attention and scatter compatibility paths, exports portable PLY, and renders with the bundled WebGL2 viewer. The app does not require Linux, WSL, Docker, `nvcc`, or a local gsplat build for this inference path.

## Useful scripts

```powershell
# CPU-only setup for UI development
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CPU -SkipInferenceTest

# Skip only the real CUDA smoke test
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Device CUDA -SkipInferenceTest

# Direct launch
.\.venv\Scripts\python.exe run.py
```

See `docs/VALIDATION.md` for verified and still-pending evidence.
