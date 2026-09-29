# Source and licence inventory

## Gaussian Scene Studio

- Repository code: MIT (`LICENSE`).
- Custom adaptive Gaussian conversion: `studio/custom_sfm.py`.
- The final 3D scene is optimized from each user's registered images; no pretrained model directly emits the reconstruction.

## Depth Anything V2 Small

- Official model: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Official project: https://github.com/DepthAnything/Depth-Anything-V2
- Checkpoint: `depth-anything/Depth-Anything-V2-Small-hf` (24.8M parameters, approximately 100 MB).
- Licence: Apache-2.0. The Base, Large, and Giant variants are not used because their official licence is CC-BY-NC-4.0.
- Purpose in multi-view mode: predicts relative inverse depth for a soft, per-view scale/shift-invariant 3DGS training loss. It does not replace COLMAP cameras there.
- Purpose in single-photo mode: supplies the estimated relative depth used to build an explicitly labelled, limited-angle 2.5D Gaussian relief. Hidden geometry is not presented as measured reconstruction.
- The checkpoint is downloaded from Hugging Face during setup/first use and cached locally.
- Preserve Apache-2.0 licence and attribution notices when redistributing the checkpoint.

## gsplat 1.5.3

- Official project: https://github.com/nerfstudio-project/gsplat
- Package: `gsplat==1.5.3`
- Licence: Apache-2.0.
- Used to optimize a new Gaussian scene from each user's registered photographs. It does not provide or download pretrained reconstruction weights.
- Preserve the Apache-2.0 licence and attribution notices when redistributing the dependency.

## COLMAP 4.2.0

- Official project: https://github.com/colmap/colmap
- Pinned Windows CUDA archive: `colmap-x64-windows-cuda.zip`
- Pinned SHA-256: `991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e`
- COLMAP library licence: BSD-3-Clause.
- Official licence: https://github.com/colmap/colmap/blob/4.2.0/COPYING.txt

COLMAP states its own library licence is independent of dependency licences. Preserve notices shipped inside official binary archive and review them before redistribution.

## PyCOLMAP CUDA (Kaggle/Linux only)

- Package: `pycolmap-cuda12==4.2.0`
- Publisher/source: COLMAP project.
- Licence classifier: BSD-3-Clause.
- Used as CUDA backend when standalone `colmap` command is unavailable.

## Removed pretrained systems

Apple SHARP, AnySplat, Silueta, and their checkpoints, vendored sources, download scripts, and UI modes are removed from this branch. Depth Anything V2 Small was reintroduced solely as the Apache-2.0 soft depth prior documented above. Old Git history still contains earlier versions; do not redistribute old dependency bundles without reviewing their licences.

This inventory is technical documentation, not legal advice.
