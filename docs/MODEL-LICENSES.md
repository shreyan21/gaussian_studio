# Source and licence inventory

## Gaussian Scene Studio

- Repository code: MIT (`LICENSE`).
- Custom adaptive Gaussian conversion: `studio/custom_sfm.py`.
- No pretrained neural reconstruction weights are downloaded or executed.

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

Apple SHARP, AnySplat, Depth Anything, Silueta, their checkpoints, vendored sources, download scripts, and UI modes are removed from this branch. Old Git history still contains earlier versions; do not redistribute old dependency bundles without reviewing their licences.

This inventory is technical documentation, not legal advice.
