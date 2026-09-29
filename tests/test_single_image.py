import numpy as np
import torch
from PIL import Image

from studio import single_image


def test_single_photo_builds_limited_angle_depth_scene(tmp_path, monkeypatch):
    path = tmp_path / "input.png"
    Image.new("RGB", (80, 60), "coral").save(path)
    disparity = np.tile(np.linspace(0.1, 1.0, 80, dtype=np.float32), (60, 1))
    monkeypatch.setattr(single_image, "predict_relative_inverse_depth", lambda image, device: disparity)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    messages = []

    gaussians, metadata = single_image.reconstruct_single_image(
        path,
        max_side=64,
        use_gpu=True,
        depth_strength=1.0,
        progress=lambda percent, message: messages.append((percent, message)),
    )

    assert len(gaussians) == 64 * 48
    assert metadata["representation"] == "single-image-depth-splat"
    assert 8 <= metadata["view_limits"]["yaw_degrees"] <= 18
    assert 6 <= metadata["view_limits"]["pitch_degrees"] <= 12
    assert len(metadata["background_color"]) == 3
    assert metadata["single_image_preview"] is True
    assert (tmp_path / "depth.png").is_file()
    assert messages[-1][0] == 72
