# Validation record — 8 September 2026

## Verified on this computer

- Windows 11 build 26200; Python 3.11.15; 7.7 GiB usable system RAM.
- PyTorch 2.8.0+cpu / Torchvision 0.23.0+cpu. CUDA unavailable on this computer.
- Dependencies installed in a dedicated `D:\GaussianSceneStudio\.venv`.
- Corrected PowerShell installer completed successfully from the D-drive folder;
  `pip check` reported no broken requirements. All resolved Python dependency
  versions are constrained by `constraints-windows-py311.txt`.
- **18 tests passed** from the D-drive installation. Coverage: Gaussian PLY
  round-trip, perspective/depth direction, anisotropy, finite outputs, preview/full
  export separation, valid/invalid uploads, research flag, cross-origin writes,
  streamed upload size, single active job, real process cancellation, failed CUDA
  worker reporting, stale-job recovery, and upstream SHARP covariance conversion.
- Test output has two dependency deprecation warnings from Starlette's HTTPX
  test-client integration. They did not fail tests or appear in browser runtime.
- Python compilation and JavaScript syntax checks passed.
- Actual CPU inference on the supplied `gpu-detail-1.jpeg`: **196,096 Gaussians**,
  **21.97 seconds** as recorded before final export, 512×383 sampling resolution.
  A repeat also completed (approximately 21 seconds). These are local observations,
  not general speed claims or GPU benchmarks.
- Independent final-installation check: `gpu-detail-2.jpeg` processed by
  `D:\GaussianSceneStudio\.venv\Scripts\python.exe` produced **110,208 Gaussians**
  at 384×287 sampling resolution in **11.44 seconds** on CPU. The installed
  service reported the D-drive interpreter, both model files available, and a
  completed job with valid scene metadata.
- Full workflow in a real Chromium browser: upload, create, progress, scene load,
  mouse orbit, source/back presets, keyboard rotate/zoom/reset, PLY download,
  PNG screenshot download, model switch and quick-guide dialog.
- Downloaded PLY was parsed again: **196,096 valid Gaussians**, with nonzero depth
  variation (Z range approximately 1.857 relative units). Screenshot was a valid
  1031×550 PNG with nonuniform colour content.
- Desktop view at 1440×1000 and mobile view at 390×844 were visually inspected.
  Mobile document scroll width equalled viewport width; no horizontal overflow.
- Browser console: **zero errors/warnings** during tested workflows. WebGL error
  value was 0 after camera orbit/back view and keyboard reset.
- Port fallback was observed: with the development server occupying 7860, the
  installed server reserved 7861. Reopening the final launcher reused the existing
  installation instead of starting a competing job manager.

## SHARP compatibility checks

- Official source pinned to `1eaa046834b81852261262b41b0919f5c1efdd2e`.
- The official 2,809,738,232-byte checkpoint is downloaded locally.
- SHA-256: `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d`.
- Inference imports work without gsplat or CUDA extension compilation.
- Meta-device architecture/checkpoint compatibility: **702,305,169 parameters**,
  **1,038 tensors**, no missing or unexpected keys, matching tensor shapes.
- Real upstream unprojection/covariance conversion was tested on tiny tensors.

**These checks do not constitute a SHARP forward-pass test.** Full SHARP inference,
mixed-precision output quality, CUDA driver/runtime execution and actual VRAM usage
remain unverified. An 8 GB A1000 is the target workstation, not a tested guarantee.

## Office acceptance checks

1. Run the NVIDIA setup and confirm `doctor.py --require-cuda` passes a real CUDA
   tensor computation and identifies the RTX A1000.
2. Generate a lightweight-model scene with Auto compute. Scene metadata must say
   `cuda`, and the PLY download must load in the interactive viewer.
3. For permitted research, install/enable SHARP, run a clear normal photograph and
   inspect source/nearby views and Gaussian metadata. Record time and memory use.
4. Check cancel/retry and a second image. Close other GPU-heavy programs if memory
   is insufficient; explicitly choose the lightweight model or SHARP CPU with
   adequate system RAM if needed.
5. Review the hidden-side limitation with the senior: 360-degree camera control
   does not mean unseen geometry has been accurately reconstructed.

No application can honestly be guaranteed bug-free across untested machines.
This record distinguishes executed checks from workstation checks still required.
