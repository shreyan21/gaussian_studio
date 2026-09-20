# Gaussian Scene Studio 3

Pretrained-free multi-view reconstruction for local NVIDIA workstations. Upload ordered overlapping photographs; app estimates camera poses, reconstructs dense surfaces, converts them to adaptive oriented 3D Gaussians, and opens an interactive browser viewer.

No SHARP, AnySplat, Depth Anything, downloaded reconstruction checkpoint, or cloud inference remains in this version.

## What "custom model" means here

This is a per-scene reconstruction model, not a newly trained general-purpose neural foundation model:

1. SIFT features and geometric verification match uploaded photographs.
2. Incremental structure-from-motion solves camera intrinsics, poses, and sparse geometry.
3. CUDA PatchMatch estimates dense multi-view depth.
4. Geometrically consistent samples are fused into an oriented point cloud.
5. Gaussian Scene Studio computes robust local spacing, filters outliers, orients each anisotropic Gaussian to its surface normal, and exports portable 3DGS PLY/GSB files.

This avoids pretrained-model licence and hallucination issues. It cannot invent sides absent from all photographs.

## Office workstation: three commands

Requirement: Windows 10/11, NVIDIA driver visible in `nvidia-smi`, approximately 8 GB free VRAM, 64-bit Python 3.11, and Git.

```powershell
git clone https://github.com/shreyan21/gaussian_studio.git D:\GaussianSceneStudio
Set-Location D:\GaussianSceneStudio
.\Setup NVIDIA Workstation.cmd
```

Setup creates `.venv`, installs pinned Python packages, downloads official COLMAP 4.2.0 CUDA binaries, verifies SHA-256, and runs automated tests.

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

## Capture requirements

- Six images minimum; 20-40 recommended.
- Keep object and background completely stationary. Move only camera.
- Capture one ordered smooth circle, then optional slightly higher ring.
- Keep 70-85% overlap between neighbors.
- Lock zoom, focus, exposure, white balance, lighting, and distance.
- Avoid motion blur, mirrors, glossy highlights, glass, thin foliage, and featureless surfaces.

Four photographs are normally insufficient for dense 360-degree reconstruction. Rotating a shoe while background stays fixed violates camera geometry and causes shattered output.

## RTX A1000 8 GB settings

- Start with **Balanced - 1600 px**.
- Close QGIS, games, WebGL-heavy tabs, and other GPU work.
- Try 20-30 photographs first.
- Use **Quick - 1200 px** after CUDA out-of-memory.
- High mode is optional and may exceed 8 GB VRAM.
- Default job timeout is 120 minutes. Override with `GSS_JOB_TIMEOUT_MINUTES` if needed.

## Kaggle GPU test

Open `notebooks/Gaussian_Studio_Custom_Kaggle.ipynb`, enable a GPU and internet, then run its single code cell. Notebook installs `pycolmap-cuda12==4.2.0`, starts app, and prints a protected Cloudflare link.

Kaggle storage is temporary. Download `scene.ply` immediately after each successful run.

## Manual checks

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py --require-custom
.\.venv\Scripts\python.exe -m pytest -q tests
```

## Output

- `scene.ply`: full Gaussian scene.
- `scene.gsb`: bounded browser preview.
- `scene.json`: method, runtime, input count, limitations, and scene framing.
- `worker.log`: reconstruction evidence and failure details.

Output is Gaussian splats, not a watertight CAD/Blender mesh.

## Commercial-use position

Application code is MIT. COLMAP library is BSD-3-Clause, but its official binary includes dependencies with their own notices. No pretrained model weights are used. Preserve third-party notices and obtain organisation-specific legal review before commercial distribution. See [docs/MODEL-LICENSES.md](docs/MODEL-LICENSES.md).

## Evidence boundary

CPU tests validate API, security, export, image preparation, and dense-cloud-to-Gaussian conversion. Full reconstruction must still be acceptance-tested on actual RTX A1000 hardware with a valid 20-40 image capture; see [docs/VALIDATION.md](docs/VALIDATION.md).
