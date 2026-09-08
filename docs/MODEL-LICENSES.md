# Models and dependencies

The application code is MIT licensed. Model weights retain their own licences.

## Default: Depth Anything V2 Small

- Model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Licence: Apache-2.0, according to the official model card.
- Predicts **relative inverse depth**, not Gaussian parameters. The local Python
  algorithm lifts the depth and source colours into oriented anisotropic Gaussians.
- No paid API or hosted inference. The official weights are approximately 100 MB.

## Optional: Apple SHARP

- Source: https://github.com/apple/ml-sharp
- Pinned source commit: `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Checkpoint: https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
- Code licence: https://github.com/apple/ml-sharp/blob/main/LICENSE
- Model licence: https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL
- **Weights are restricted to non-commercial scientific research and academic
  development. The licence excludes commercial exploitation and product development.**
  Do not use this model for office production or commercial product development.
  Select the Apache-licensed depth model if that restriction does not fit your task.
- The optional installer preserves the upstream code and model licence notices in
  `vendor/ml-sharp`. No Apple weights are included in the base transfer package.
- Predicts Gaussian positions, scales, rotations, opacity and colour directly.
  It is intended for nearby-view synthesis, not complete unseen-side generation.

## Runtime dependencies

PyTorch / Torchvision: https://pytorch.org (BSD-style)

Transformers: https://github.com/huggingface/transformers (Apache-2.0)

FastAPI: https://github.com/fastapi/fastapi (MIT)

NumPy: https://numpy.org (BSD-3-Clause)

Pillow: https://python-pillow.github.io (HPND)

Other packages retain the licences distributed with their installations. There
are no third-party browser assets: the viewer, UI and shaders are local source.
