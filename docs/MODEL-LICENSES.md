# Model and source licences

## Apple SHARP

- Official source: https://github.com/apple/ml-sharp
- Pinned source commit: `1eaa046834b81852261262b41b0919f5c1efdd2e`
- Model licence: non-commercial scientific research only
- Preserved notices: `vendor/ml-sharp/LICENSE_MODEL`, `vendor/ml-sharp/LICENSE`, and `docs/Apple-Model-Attribution.txt`

The app requires an explicit acknowledgement before each SHARP job. Do not use the
SHARP checkpoint or its output for a commercial product unless Apple grants separate
permission. The camera clamp is a rendering safeguard, not a change to that licence.

## AnySplat

- Official source: https://github.com/InternRobotics/AnySplat
- Pinned source commit: `5f5e208a7dd57d52e43ea0d553a95eab526e8775`
- Published model: https://huggingface.co/lhjiang/anysplat
- Pinned model revision: `d2e8c343672646041ad4ea518184968f94362f01`
- Official repository and model-card licence: MIT
- Local upstream licence: `vendor/anysplat/LICENSE`

The official project and model card state that the code and models use MIT. However,
the upstream repository also contains individual files with third-party copyright and
licence headers, including some CC BY-NC-SA 4.0 notices inherited from Naver research
code. This application preserves those headers. Do not describe the entire dependency
bundle as commercially cleared without your organisation's legal/licence review.

Gaussian Scene Studio's adapter does not import AnySplat's image utility or CUDA video
renderer. Some upstream modules carrying third-party notices remain in the dependency
and import graph. The app's technical integration cannot resolve that legal question.

## Depth Anything V2 Small

- Model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Pinned revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Model card licence: Apache-2.0

## Silueta foreground model

- Model file: https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx
- Architecture/source: https://github.com/xuebinqin/U-2-Net
- Purpose: offline main-object segmentation before AnySplat reconstruction
- Source licence: Apache-2.0

The model is downloaded during setup and verified against the pinned SHA-256 in
`studio/config.py`. It runs through ONNX Runtime on CPU; uploaded images stay local.

## Application and dependencies

Gaussian Scene Studio's original code is MIT; see the root `LICENSE`. Upstream source,
model weights, Python packages, fonts, drivers, and runtime components retain their own
licences. Keep their notices when redistributing.
