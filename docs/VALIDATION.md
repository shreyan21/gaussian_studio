# Validation record

Date: 2026-09-13

## Confirmed

### Kaggle NVIDIA T4

- Official AnySplat checkpoint loaded successfully.
- Real two-view CUDA forward inference completed.
- Real four-view shoe inference completed in about 12-13 seconds.
- The four-view run produced 714,609 raw Gaussians before optional filtering.
- The model and CUDA path worked; scene quality was limited by inconsistent captures, because the shoe moved relative to the background between photographs.

### Local source and API tests

- Python source compiles.
- API accepts ordered AnySplat uploads and stores up to 16 sanitized images.
- AnySplat requires at least two images; Depth Anything requires exactly one.
- CPU selection is rejected for AnySplat.
- Auto/all/explicit view budgets are validated and saved.
- Auto-safe uses currently free VRAM, not only total installed VRAM.
- RTX A1000 low-memory loading uses 16-bit model weights and refuses inference below 5.5 GB currently free VRAM.
- The setup smoke test now uses the same 16-bit model-loading path as normal inference on an 8 GB GPU; it no longer creates a full-precision-only setup failure.
- PLY round trips and AnySplat output conversion are covered.
- Multi-image assets beyond the old four-image limit are retrievable safely.
- Upload hardening, cross-origin blocking, jobs, cancellation, history, and recovery remain covered.
- JavaScript syntax checks pass.
- Pinned AnySplat marker, required source files, and model hash are verified.
- Public mode requires a random access token; normal local mode remains loopback-only.

## Fixes added after Kaggle inference

- Restored two AnySplat VGGT files missing from the repository:
  `aggregator.py` and `vggt.py`.
- Added the missing SciPy runtime dependency.
- Fixed PyTorch meta-device initialization in `vision_transformer.py`.
- Added a real two-view CUDA smoke test to default workstation setup.
- Added free-VRAM-aware view selection for the office RTX A1000 8 GB.
- Added a token-protected Cloudflare Quick Tunnel launcher.

## Not yet claimed

- The final Windows package has not yet completed its real CUDA smoke test on the photographed RTX A1000 workstation.
- The Cloudflare launcher is unit-tested at the application-auth layer, but its temporary public URL still requires a live end-to-end run.
- Kaggle T4 success does not prove RTX A1000 speed, maximum safe view count, or final scene quality.
- The dependency bundle is not represented as commercially cleared; see `MODEL-LICENSES.md`.

## Required RTX A1000 acceptance test

1. Run `Setup NVIDIA Workstation.cmd` and require a clean exit. The setup now performs real two-view inference.
2. Run `.venv\Scripts\python.exe scripts\doctor.py --require-cuda --require-anysplat`.
3. Run `.venv\Scripts\python.exe scripts\verify_anysplat.py --check-import --check-hash --check-inference`.
4. Start Studio. Load the procedural scene and verify orbit, pan, zoom, reset, screenshot, fullscreen, and PLY download.
5. Upload two adjacent, overlapping views. Use AnySplat and Auto-safe.
6. Confirm completion and metadata: `method: anysplat`, `input_count: 2`, and `device: cuda`.
7. Retry with more views only while free VRAM remains safe. Close QGIS and GPU-heavy browser tabs first.
8. Cancel one running job and confirm another job can start.
9. Restart Studio and confirm the completed scene remains in Recent scenes.
10. Run `Start Public Link.cmd`; verify the printed tokenized URL works and a URL without its token receives HTTP 401.

Do not report RTX A1000 CUDA execution as validated until steps 1-10 pass on that workstation.
