# Gaussian Scene Studio 5

Single-image AI generation plus hybrid multi-view reconstruction for local NVIDIA workstations. Upload one clear object photograph for TripoSplat generation, or upload one ordinary slow orbit video / 2-80 ordered overlapping photographs for measured COLMAP reconstruction (12+ photos recommended). Both paths produce portable Gaussian PLY output for the same interactive browser viewer.

No SHARP, AnySplat, or hosted inference API is used. The single-image path runs the official TripoSplat weights locally and labels its unseen surfaces as AI-generated. In the multi-view path, Depth Anything V2 Small supplies only a soft relative-depth constraint; COLMAP cameras and the uploaded pixels remain authoritative.

## One photograph: TripoSplat

- A single JPEG, PNG, or WebP automatically selects TripoSplat.
- The model removes the background for object generation and produces up to 65,536 object Gaussians on an 8 GB GPU, 131,072 on a 10-19 GB GPU, or 262,144 on a 20+ GB GPU. The app then adds a masked, slightly blurred source-photo backdrop behind the object for visual context.
- Because a single photograph cannot reconstruct the background in 3D, the backdrop is explicitly labelled as shallow and the viewer limits rotation to useful angles.
- The app loads TripoSplat stages sequentially and clears CUDA memory between them to reduce peak VRAM use.
- Rotation is limited to a broad 145-degree yaw and 55-degree pitch around the source view. This exposes useful 3D while avoiding an unrestricted underside view.
- Hidden sides are plausible AI predictions. They are not measurements and may be wrong.

Set `GSS_TRIPOSPLAT_GAUSSIANS=32768` before startup if an 8 GB workstation still runs out of CUDA memory. The first setup downloads approximately 3.8 GB of official checkpoints.

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

Setup creates `.venv`, installs CUDA PyTorch 2.4.1 plus gsplat's official `pt24cu124` Windows wheel, installs the pinned official TripoSplat source/checkpoints, caches the approximately 100 MB Depth Anything V2 Small checkpoint, downloads official COLMAP 4.2.0 CUDA binaries, verifies SHA-256, and runs automated tests. If this repository already has a Python 3.11 `.venv` from Studio 3, rename or remove only that `.venv` folder before running setup again.

The first setup/Kaggle run needs Internet access to download the exact AI checkpoints. They are reused afterward. For an offline multi-view run without AI depth guidance, set `GSS_DISABLE_AI_DEPTH=1` before starting the server; the result then uses the original COLMAP + gsplat path. TripoSplat itself requires its downloaded checkpoints.

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

- Use one 20-60 second video, or 2-80 ordered photos (12-40 photos recommended).
- Keep object and background completely stationary. Move only camera.
- Walk one slow, smooth circle, then an optional slightly higher ring. Do not rotate the object.
- Keep the subject roughly centered and make one continuous orbit.
- Ordinary exposure and focus variation is tolerated, but moving petals and unseen surfaces cannot be reconstructed reliably.

The application uses the fixed **High - 2000 px** full-scene profile: up to 44 sharp evenly spaced video frames, 9,500 optimization steps, a 750,000-splat GPU budget, and an Apache-2.0 AI depth prior for smoother weak-texture geometry. Four photographs are normally insufficient for a 360-degree reconstruction. Rotating an object while its background stays fixed violates camera geometry and causes shattered output.

The pipeline rejects captures when fewer than eight cameras register, less than 55% of the inputs align, or the sparse model has fewer than 500 points. This is intentional: the viewer should not present disconnected noise as a successful 3D scene.

Every multi-view reconstruction runs in full-scene mode. There is no central crop, foreground mask, or object-isolation filter: the complete registered views train the geometry, and foreground plus background Gaussians are exported together. Only invalid, oversized, highly elongated, or extremely transparent splats are filtered. Thin moving leaves can still duplicate because a dynamic subject violates multi-view geometry; use a windless capture whenever possible. The separate one-photo TripoSplat path generates an isolated 3D foreground object and places a masked 2D source-photo backdrop behind it; that backdrop provides context but is not reconstructed geometry.

## RTX A1000 8 GB settings

- The interface always uses **High - 2000 px**.
- Close QGIS, games, WebGL-heavy tabs, and other GPU work.
- Try 20-30 photographs first.
- This profile can take substantially longer and may approach the 8 GB VRAM limit.
- Default job timeout is 120 minutes. Override with `GSS_JOB_TIMEOUT_MINUTES` if needed.
- True 3DGS uses GPU 0. Dense fallback automatically uses every GPU reported by `nvidia-smi`; set `GSS_GPU_INDEX=0` to restrict fallback to one GPU.
- Set `GSS_GSPLAT_STEPS` to override the quality profile's training-step count.

## Kaggle GPU test

Import `notebooks/Gaussian_Studio_Custom_Kaggle.ipynb` into Kaggle, enable a GPU and Internet in Notebook options, then run its single code cell. The notebook pulls the latest `master`, installs the multi-view stack, downloads the pinned TripoSplat source/checkpoints, starts the app, and prints a protected Cloudflare link. Open the complete URL including `?token=...`; an unprotected URL correctly shows **Access denied**. The initial TripoSplat download is approximately 3.8 GB and therefore makes the first notebook startup slower.

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

Application code is MIT, COLMAP is BSD-3-Clause, and both gsplat and the Depth Anything V2 Small checkpoint are Apache-2.0. TripoSplat's repository labels its code and released weights MIT, while its DINOv3 encoder lineage is governed by Meta's DINOv3 terms. Preserve all third-party notices and obtain organisation-specific legal review before commercial distribution. See [docs/MODEL-LICENSES.md](docs/MODEL-LICENSES.md).

## Evidence boundary

CPU tests validate API, security, export, camera-data conversion, image preparation, and dense fallback. Full 3DGS training must still be acceptance-tested on Kaggle T4 and the actual RTX A1000; see [docs/VALIDATION.md](docs/VALIDATION.md).
