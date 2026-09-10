# Validation record

Date: 2026-09-10

## Confirmed on this CPU-only computer

- Python source compiles.
- API accepts ordered AnySplat uploads and stores up to 16 sanitized images.
- AnySplat requires 2 or more images; Depth Anything requires exactly 1.
- CPU selection is rejected for AnySplat.
- Auto/all/explicit view budgets are validated and saved.
- Multi-image assets beyond the old four-image limit are retrievable safely.
- PLY round trips preserve position, scale, rotation, opacity, and SH0 colour.
- AnySplat output conversion handles coordinate axes, xyzw-to-wxyz rotation,
  opacity, SH0 colour, invalid splats, and the two-million output cap.
- Auto-safe VRAM budgets and deterministic ordered sampling are unit tested.
- Existing procedural scene, WebGL files, job cancellation, history, restart recovery,
  upload hardening, cross-origin blocking, and error-log flow remain covered.
- JavaScript syntax check passes.
- Pinned AnySplat source marker and required files are verified.

## Not claimed on this computer

This computer has no usable NVIDIA CUDA runtime. The AnySplat 2.94 GB checkpoint is not
downloaded into the source package, and a real AnySplat forward pass cannot be run here.
Source/import/contract tests do not prove CUDA execution, VRAM fit, speed, or scene quality.

## Required NVIDIA acceptance test

1. Run `Setup NVIDIA Workstation.cmd` and require a clean exit.
2. Run `.venv\Scripts\python.exe scripts\doctor.py --require-cuda --require-anysplat`.
3. Run `.venv\Scripts\python.exe scripts\verify_anysplat.py --check-import --check-hash`.
4. Start the app. Load the procedural scene and verify orbit, pan, zoom, reset,
   screenshot, fullscreen, and PLY download.
5. Upload two overlapping 448-pixel-or-larger views. Use AnySplat + Auto-safe.
6. Confirm the job completes, scene renders, metadata says `method: anysplat`,
   `input_count: 2`, `device: cuda`, and export opens again in the viewer.
7. Retry with four views only if two-view inference has safe free VRAM. On the RTX A1000,
   close QGIS and other GPU-heavy programs first.
8. Cancel one running job and confirm the slot releases and another job can start.
9. Restart the app and confirm the completed scene remains in Recent scenes.

Do not report AnySplat CUDA as validated until steps 1–9 pass on the workstation.
