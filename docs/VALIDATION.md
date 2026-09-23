# Validation and acceptance

## Automated checks

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe scripts\doctor.py --require-custom
```

Tests cover photo and streamed-video upload rules, access-token protection, path restrictions, job persistence, video keyframe selection, registration quality gates, equal-size ordered frame preparation, adaptive Gaussian construction, PLY/GSB export, and viewer assets.

## RTX A1000 acceptance test

Do not mark hardware validation complete until every item passes on physical workstation:

1. `nvidia-smi` reports RTX A1000 and no heavy competing process.
2. Setup completes checksum verification and doctor reports `custom_engine: true` and `cuda: true`.
3. Capture one slow 20-60 second video or 20-30 sharp ordered photos around one stationary textured scene.
4. Balanced reconstruction completes without CUDA out-of-memory.
5. At least most images register; COLMAP log shows dense PatchMatch and stereo fusion completion.
6. With central-subject focus enabled, viewer loads a compact coherent object without distant floating background islands; disable it to validate a deliberately wide scene.
7. `scene.ply`, raw `dense.ply`, `scene.gsb`, `scene.json`, and `worker.log` download successfully.
8. Restart Studio; completed scene still appears in Recent scenes.

## Quality failure diagnosis

- **Shattered copies:** object or background moved, uploads unordered, too little overlap, or mixed zoom.
- **Camera alignment failed:** too few features, blur, reflections, repeated texture, or insufficient baseline.
- **Holes:** surfaces unseen in inputs or rejected by geometric consistency.
- **CUDA out-of-memory:** close GPU apps, use Quick mode, or reduce photo count while keeping full coverage.
- **Sparse/noisy surface:** capture more neighboring views and improve lighting/texture.
- **Capture rejected:** read the registration counts in the error/log. Do not bypass the quality gate; recapture with slower motion, more texture, and greater overlap.

## Current evidence boundary

Automated non-GPU checks do not prove final RTX A1000 quality or runtime. Kaggle CUDA success does not substitute for workstation acceptance because GPU, drivers, and image sets differ.
