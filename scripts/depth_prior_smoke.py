"""Download and validate the commercial-friendly AI geometry prior."""
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.gsplat_runtime import AI_DEPTH_MODEL_ID, ai_depth_cache_dir


def main():
    cache_dir = ai_depth_cache_dir()
    processor = AutoImageProcessor.from_pretrained(
        AI_DEPTH_MODEL_ID,
        cache_dir=cache_dir,
        use_fast=False,
    )
    model = AutoModelForDepthEstimation.from_pretrained(AI_DEPTH_MODEL_ID, cache_dir=cache_dir)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters < 20_000_000 or processor is None:
        raise RuntimeError("Depth Anything V2 Small did not load correctly.")
    inputs = processor(images=Image.new("RGB", (64, 64), "gray"), return_tensors="pt")
    with torch.inference_mode():
        prediction = model(**inputs).predicted_depth
    if prediction.ndim != 3 or not torch.isfinite(prediction).all():
        raise RuntimeError("Depth Anything V2 Small produced an invalid depth map.")
    print(json.dumps({
        "ai_depth_prior": AI_DEPTH_MODEL_ID,
        "parameters": parameters,
        "license": "Apache-2.0",
        "cache": str(cache_dir),
        "output_shape": list(prediction.shape),
    }))


if __name__ == "__main__":
    main()
