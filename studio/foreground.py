"""Offline main-subject segmentation and object-centric square framing."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from studio.config import FOREGROUND_MODEL

MODEL_SIZE = 320
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def model_ready() -> bool:
    return FOREGROUND_MODEL.is_file()


def load_session():
    if not model_ready():
        raise RuntimeError("Foreground isolation model is missing. Rerun Setup NVIDIA Workstation.cmd.")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(FOREGROUND_MODEL), sess_options=options, providers=["CPUExecutionProvider"]
    )


def predict_mask(image: Image.Image, session) -> Image.Image:
    """Return a soft 8-bit foreground mask at the original image size."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    sample = image.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(sample, dtype=np.float32) / 255.0
    array = ((array - MEAN) / STD).transpose(2, 0, 1)[None]
    input_name = session.get_inputs()[0].name
    output = np.asarray(session.run(None, {input_name: array})[0])[0, 0]
    low, high = float(output.min()), float(output.max())
    output = (output - low) / max(high - low, 1e-6)
    mask = Image.fromarray(np.uint8(np.clip(output, 0, 1) * 255))
    return mask.resize(image.size, Image.Resampling.LANCZOS)


def focus_object(image: Image.Image, mask: Image.Image, size: int = 448):
    """Crop around the main subject, neutralize its background, and return its pixel mask."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.LANCZOS)
    binary = np.asarray(mask) >= 48
    coverage = float(binary.mean())
    if coverage < 0.005:
        raise ValueError("No clear foreground object was detected. Use a closer view with a distinct background.")
    if coverage > 0.96:
        raise ValueError("The object cannot be separated from the background. Use a cleaner, contrasting background.")

    ys, xs = np.nonzero(binary)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0) * 1.24
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = int(round(cx + side / 2)), int(round(cy + side / 2))
    square_side = max(1, right - left, bottom - top)

    square_image = Image.new("RGB", (square_side, square_side), (127, 127, 127))
    square_mask = Image.new("L", (square_side, square_side), 0)
    source_box = (max(0, left), max(0, top), min(image.width, left + square_side), min(image.height, top + square_side))
    destination = (source_box[0] - left, source_box[1] - top)
    square_image.paste(image.crop(source_box), destination)
    square_mask.paste(mask.crop(source_box), destination)

    square_image = square_image.resize((size, size), Image.Resampling.LANCZOS)
    square_mask = square_mask.resize((size, size), Image.Resampling.LANCZOS)
    focused = Image.composite(square_image, Image.new("RGB", (size, size), (127, 127, 127)), square_mask)
    export_mask = square_mask.point(lambda value: 255 if value >= 32 else 0).filter(ImageFilter.MaxFilter(5))
    return focused, np.asarray(export_mask, dtype=np.uint8) > 0, coverage
