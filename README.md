# Gaussian Scene Studio 4

Pretrained-free multi-view reconstruction for local NVIDIA workstations. Upload one ordinary slow orbit video or ordered overlapping photographs; the app selects sharp keyframes, estimates camera poses, trains a true 3D Gaussian scene from the registered views, and opens an interactive browser viewer.

No SHARP, AnySplat, Depth Anything, downloaded reconstruction checkpoint, or cloud inference remains in this version.

Dense fusion now retries weak results automatically: strict geometric consistency first, relaxed geometric fusion second, and a photometric recovery pass only when necessary. This avoids discarding otherwise usable small-object captures at the very end of a run.

## What "custom model" means here

This is per-scene optimization, not a pretrained general-purpose neural foundation model:

1. SIFT features and geometric verification match uploaded photographs.
2. Incremental structure-from-motion solves camera intrinsics, poses, and sparse geometry.
3. `gsplat` initializes Gaussians from COLMAP's sparse geometry.
4. CUDA optimizes their positions, colors, opacity, scale, and orientation against the registered photographs.
5. The trained result is focused around the central subject and exported as portable 3DGS PLY/GSB files.

If CUDA `gsplat` is unavailable, the previous PatchMatch pipeline remains as a fallback. It uses strict geometric fusion first, relaxed geometric fusion second, and photometric recovery only when necessary.

This avoids pretrained-model licence and hallucination issues. It cannot invent sides absent from all photographs.

## Office workstation: three commands

Requirement: Windows 10/11, NVIDIA driver visible in `nvidia-smi`, approximately 8 GB free VRAM, 64-bit Python 3.10, and Git. Python 3.10 matches gsplat's official precompiled Windows wheel and avoids requiring a local Visual Studio/CUDA compilation toolchain.

```powershell
git clone https://github.com/shreyan21/gaussian_studio.git D:\GaussianSceneStudio
Set-Location D:\GaussianSceneStudio
.\Setup NVIDIA Workstation.cmd
```

Setup creates `.venv`, installs CUDA PyTorch 2.4.1 plus gsplat's official `pt24cu124` Windows wheel, downloads official COLMAP 4.2.0 CUDA binaries, verifies SHA-256, and runs automated tests. If this repository already has a Python 3.11 `.venv` from Studio 3, rename or remove only that `.venv` folder before running setup again.

Start app:

```powershell
.\Start Studio.cmd
```

Open `http://127.0.0.1:7860`.

Existing checkout:

```powershell
Set-Location D:\GaussianSceneStudio
git pull --ff-only
.\Setup NVIDIA Workstation.cmd
.\Start Studio.cmd
```

## Temporary protected public link

```powershell
.\Start Public Link.cmd
```

Keep terminal open. Share complete tokenized URL only with trusted testers. Quick Tunnel is temporary, not production hosting.

## Capture behavior

- Use one 20-60 second video, or 12-80 ordered photos (20-40 photos recommended).
- Keep object and background completely stationary. Move only camera.
- Walk one slow, smooth circle, then an optional slightly higher ring. Do not rotate the object.
- Keep the subject roughly centered and make one continuous orbit.
- Ordinary exposure and focus variation is tolerated, but moving petals and unseen surfaces cannot be reconstructed reliably.

The video path extracts 32, 40, or 48 sharp evenly spaced frames for Quick, Balanced, or High quality. The corresponding 3DGS profiles use 6,000, 9,000, or 12,000 optimization steps and enforce GPU-memory splat budgets. Four photographs are normally insufficient for a 360-degree reconstruction. Rotating an object while its background stays fixed violates camera geometry and causes shattered output.

The pipeline rejects captures when fewer than eight cameras register, less than 55% of the inputs align, or the sparse model has fewer than 500 points. This is intentional: the viewer should not present disconnected noise as a successful 3D scene.

For a flower or single object, leave **Focus central subject** enabled. The app triangulates the subject from recovered camera rays, follows its projected position in every training view, excludes distant background seeds before optimization, and filters oversized streak splats during export. Disable it only when the surrounding environment is intentionally part of the scene.

## RTX A1000 8 GB settings

- Start with **Balanced - 1600 px**.
- Close QGIS, games, WebGL-heavy tabs, and other GPU work.
- Try 20-30 photographs first.
- Use **Quick - 1200 px** after CUDA out-of-memory.
- High mode is optional and may exceed 8 GB VRAM.
- Default job timeout is 120 minutes. Override with `GSS_JOB_TIMEOUT_MINUTES` if needed.
- True 3DGS uses GPU 0. Dense fallback automatically uses every GPU reported by `nvidia-smi`; set `GSS_GPU_INDEX=0` to restrict fallback to one GPU.
- Set `GSS_GSPLAT_STEPS` to override the quality profile's training-step count.

## Kaggle GPU test

Import `notebooks/Gaussian_Studio_Custom_Kaggle.ipynb` into Kaggle, enable a GPU and Internet in Notebook options, then run its single code cell. The notebook pulls the latest `master`, installs `pycolmap-cuda12==4.2.0` and `gsplat==1.5.3`, starts the app, and prints a protected Cloudflare link. Open the complete URL including `?token=...`; an unprotected URL correctly shows **Access denied**.

The Kaggle public-link video limit is 90 MB because Cloudflare Free accepts request bodies up to 100 MB and the multipart request adds overhead. Trim or compress a longer recording before upload. Local workstation mode defaults to 750 MB.

Keep the code cell running while using the site. Kaggle storage is temporary. Download `scene.ply` immediately after each successful run. `dense.ply` is a support/debug cloud for trained scenes and the fused COLMAP cloud for fallback scenes.

## Manual checks

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py --require-custom
.\.venv\Scripts\python.exe -m pytest -q tests
```

## Output

- `scene.ply`: full Gaussian scene.
- `scene.gsb`: bounded browser preview.
- `dense.ply`: trained Gaussian-center support cloud, or raw COLMAP fusion when fallback was used.
- `scene.json`: method, runtime, input count, limitations, and scene framing.
- `worker.log`: reconstruction evidence and failure details.

Output is Gaussian splats, not a watertight CAD/Blender mesh.

## Commercial-use position

Application code is MIT, COLMAP is BSD-3-Clause, and gsplat is Apache-2.0. No pretrained model weights are used. Preserve third-party notices and obtain organisation-specific legal review before commercial distribution. See [docs/MODEL-LICENSES.md](docs/MODEL-LICENSES.md).

## Evidence boundary

CPU tests validate API, security, export, camera-data conversion, image preparation, and dense fallback. Full 3DGS training must still be acceptance-tested on Kaggle T4 and the actual RTX A1000; see [docs/VALIDATION.md](docs/VALIDATION.md).
