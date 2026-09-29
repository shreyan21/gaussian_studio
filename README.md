# Gaussian Scene Studio 4

Hybrid photo-to-3D reconstruction for local NVIDIA workstations. One photograph creates an explicitly labelled, limited-angle AI depth preview. One ordinary slow orbit video or 12-80 ordered overlapping photographs creates a measured multi-view scene: the app selects sharp keyframes, estimates camera poses, predicts a lightweight depth prior for smooth surfaces, trains a true 3D Gaussian scene, and opens an interactive browser viewer.

## Single-photo mode

Upload exactly one photograph to create a 768 px perspective Gaussian depth relief with Depth Anything V2 Small. Edge-aware depth smoothing removes small noisy ripples, camera-facing boundary splats prevent black tears around petals and leaves, and the viewer background follows the photograph's border colour. Rotation limits adapt from ±8-18 degrees horizontally and ±6-12 degrees vertically based on depth complexity; auto-orbit reverses at those limits and side/back/top presets remain disabled. The splat-size slider is restricted to 0.8-1.25× in this mode to prevent excessive blur. It is a 2.5D AI preview, not a measured complete reconstruction; video or 12-80 overlapping photographs still use the full 2000 px COLMAP + 3DGS pipeline.

No SHARP, AnySplat, or cloud inference is used. In multi-view mode, Depth Anything V2 Small supplies only a soft relative-depth constraint while COLMAP cameras and uploaded pixels remain authoritative. In single-photo mode, its estimated depth creates the explicitly labelled 2.5D relief, so that result is predictive rather than measured.

Dense fusion now retries weak results automatically: strict geometric consistency first, relaxed geometric fusion second, and a photometric recovery pass only when necessary. This avoids discarding otherwise usable small-object captures at the very end of a run.

## What "custom model" means here

The final representation is still optimized independently for each capture:

1. SIFT features and geometric verification match uploaded photographs.
2. Incremental structure-from-motion solves camera intrinsics, poses, and sparse geometry.
3. `gsplat` initializes Gaussians from COLMAP's sparse geometry.
4. Apache-2.0 Depth Anything V2 Small predicts per-view relative inverse depth. Per-view scale/shift normalization prevents it from inventing metric camera geometry.
5. CUDA optimizes Gaussian positions, colors, opacity, scale, and orientation against the registered photographs plus the low-weight depth constraint.
6. The complete trained scene is exported as portable 3DGS PLY/GSB files.

If CUDA `gsplat` is unavailable, the previous PatchMatch pipeline remains as a fallback. It uses strict geometric fusion first, relaxed geometric fusion second, and photometric recovery only when necessary.

The AI prior improves smooth surfaces such as pots, walls, and tabletops. Its value-and-gradient constraint encourages continuous surfaces, while robust video-pixel weighting and a conservative spatial-coherence export pass suppress ghost sheets and isolated floating splats. The viewer opens at a fuller 0.85 splat size to hide tiny sampling gaps; the slider remains adjustable. It still cannot reliably invent sides absent from all photographs.

## Office workstation: three commands

Requirement: Windows 10/11, NVIDIA driver visible in `nvidia-smi`, approximately 8 GB free VRAM, 64-bit Python 3.10, and Git. Python 3.10 matches gsplat's official precompiled Windows wheel and avoids requiring a local Visual Studio/CUDA compilation toolchain.

```powershell
git clone https://github.com/shreyan21/gaussian_studio.git D:\GaussianSceneStudio
Set-Location D:\GaussianSceneStudio
.\Setup NVIDIA Workstation.cmd
```

Setup creates `.venv`, installs CUDA PyTorch 2.4.1 plus gsplat's official `pt24cu124` Windows wheel, caches the approximately 100 MB Depth Anything V2 Small checkpoint, downloads official COLMAP 4.2.0 CUDA binaries, verifies SHA-256, and runs automated tests. If this repository already has a Python 3.11 `.venv` from Studio 3, rename or remove only that `.venv` folder before running setup again.

The first setup/Kaggle run needs Internet access to download the exact AI checkpoint. It is reused from `data/models` afterward. For an offline run without AI guidance, set `GSS_DISABLE_AI_DEPTH=1` before starting the server; the result then uses the original COLMAP + gsplat path.

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

- Use one photograph for a limited-angle AI preview, or use one 20-60 second video / 12-80 ordered photos for measured multi-view 3D (20-40 photos recommended).
- Keep object and background completely stationary. Move only camera.
- Walk one slow, smooth circle, then an optional slightly higher ring. Do not rotate the object.
- Keep the subject roughly centered and make one continuous orbit.
- Ordinary exposure and focus variation is tolerated, but moving petals and unseen surfaces cannot be reconstructed reliably.

The application uses the fixed **High - 2000 px** full-scene profile: up to 44 sharp evenly spaced video frames, 9,500 optimization steps, a 750,000-splat GPU budget, and an Apache-2.0 AI depth prior for smoother weak-texture geometry. Four photographs are normally insufficient for a 360-degree reconstruction. Rotating an object while its background stays fixed violates camera geometry and causes shattered output.

The pipeline rejects captures when fewer than eight cameras register, less than 55% of the inputs align, or the sparse model has fewer than 500 points. This is intentional: the viewer should not present disconnected noise as a successful 3D scene.

Every reconstruction runs in full-scene mode. There is no central crop, foreground mask, or object-isolation filter: the complete registered views train the geometry, and foreground plus background Gaussians are exported together. Only invalid, oversized, highly elongated, or extremely transparent splats are filtered. Thin moving leaves can still duplicate because a dynamic subject violates multi-view geometry; use a windless capture whenever possible.

## RTX A1000 8 GB settings

- The interface always uses **High - 2000 px**.
- Close QGIS, games, WebGL-heavy tabs, and other GPU work.
- Try 20-30 photographs first.
- This profile can take substantially longer and may approach the 8 GB VRAM limit.
- Default job timeout is 120 minutes. Override with `GSS_JOB_TIMEOUT_MINUTES` if needed.
- True 3DGS uses GPU 0. Dense fallback automatically uses every GPU reported by `nvidia-smi`; set `GSS_GPU_INDEX=0` to restrict fallback to one GPU.
- Set `GSS_GSPLAT_STEPS` to override the quality profile's training-step count.

## Kaggle GPU test

Import `notebooks/Gaussian_Studio_Custom_Kaggle.ipynb` into Kaggle, enable a GPU and Internet in Notebook options, then run its single code cell. The notebook pulls the latest `master`, installs `pycolmap-cuda12==4.2.0` and `gsplat==1.5.3`, starts the app, and prints a protected Cloudflare link. Open the complete URL including `?token=...`; an unprotected URL correctly shows **Access denied**.

Kaggle accepts videos up to 500 MB. The browser sends videos in protected 8 MB chunks and the server reassembles them before reconstruction, avoiding Cloudflare's per-request body limit. Local workstation mode defaults to 750 MB. Keep enough free Kaggle working-storage space for the original video, extracted frames, and reconstruction outputs.

Keep the code cell running while using the site. Kaggle storage is temporary. Download `scene.ply` immediately after each successful run. `dense.ply` is a support/debug cloud for trained scenes and the fused COLMAP cloud for fallback scenes.

For a focused result made with older code, open it from **History** and press **Remove floating fragments**. This creates a separate cleaned scene from the saved Gaussian PLY in seconds; it preserves the original result and does not repeat camera registration or 3DGS training.

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

Application code is MIT, COLMAP is BSD-3-Clause, and both gsplat and the Depth Anything V2 Small checkpoint are Apache-2.0. Preserve third-party notices and obtain organisation-specific legal review before commercial distribution. See [docs/MODEL-LICENSES.md](docs/MODEL-LICENSES.md).

## Evidence boundary

CPU tests validate API, security, export, camera-data conversion, image preparation, and dense fallback. Full 3DGS training must still be acceptance-tested on Kaggle T4 and the actual RTX A1000; see [docs/VALIDATION.md](docs/VALIDATION.md).
