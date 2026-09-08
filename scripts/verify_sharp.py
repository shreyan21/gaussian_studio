"""Compare official checkpoint tensor keys/shapes with the pinned model architecture.

This uses the meta device to avoid allocating the 702M-parameter network in RAM.
It validates checkpoint compatibility; it is NOT a forward-pass or CUDA test.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.config import SHARP_SOURCE, SHARP_WEIGHTS, SHARP_COMMIT


def verify():
    import torch
    sys.path.insert(0, str(SHARP_SOURCE))
    from sharp.models import PredictorParams, create_predictor
    with torch.device("meta"):
        model = create_predictor(PredictorParams())
    state = torch.load(str(SHARP_WEIGHTS), map_location="meta", weights_only=True, mmap=True)
    result = model.load_state_dict(state, strict=True, assign=True)
    info = {"status": "compatible", "commit": SHARP_COMMIT, "parameters": sum(p.numel() for p in model.parameters()), "tensors": len(state), "missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys, "inference_tested": False}
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    verify()
