"""Isolated CUDA PatchMatch entry point so hangs can be timed out safely."""
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 7:
        raise SystemExit("usage: patchmatch_worker WORKSPACE MAX_SIDE ITERATIONS SAMPLES CACHE_GB GPU_INDEX")
    import pycolmap

    workspace = Path(sys.argv[1]).resolve()
    options = pycolmap.PatchMatchOptions()
    options.gpu_index = sys.argv[6]
    options.max_image_size = int(sys.argv[2])
    options.num_iterations = int(sys.argv[3])
    options.num_samples = int(sys.argv[4])
    options.cache_size = float(sys.argv[5])
    options.geom_consistency = True
    pycolmap.patch_match_stereo(workspace, options=options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
