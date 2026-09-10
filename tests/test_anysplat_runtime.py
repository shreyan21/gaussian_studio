from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from studio.anysplat_runtime import automatic_view_limit, preprocess_image, select_inputs, to_viewer_gaussians


def fake_gaussians(count=4):
    means = torch.tensor([[[1.0, 2.0, 3.0]] * count])
    scales = torch.full((1, count, 3), 0.02)
    rotations = torch.tensor([[[0.0, 0.0, 0.0, 1.0]] * count])  # xyzw
    harmonics = torch.zeros((1, count, 3, 1))
    opacities = torch.linspace(0.1, 0.9, count).unsqueeze(0)
    return SimpleNamespace(means=means, scales=scales, rotations=rotations, harmonics=harmonics, opacities=opacities)


def test_anysplat_tensor_conversion_axes_rotation_and_sh0():
    converted = to_viewer_gaussians(fake_gaussians())
    np.testing.assert_allclose(converted[0, :3], [1, -2, -3])
    np.testing.assert_allclose(converted[0, 8:12], [0, 1, 0, 0])
    np.testing.assert_allclose(converted[:, 12:15], 0.5)
    np.testing.assert_allclose(np.linalg.norm(converted[:, 8:12], axis=1), 1)


def test_anysplat_conversion_keeps_strongest_with_bounded_output():
    converted = to_viewer_gaussians(fake_gaussians(6), max_gaussians=2)
    assert len(converted) == 2
    assert np.all(converted[:, 3] >= 0.7)


def test_auto_budget_and_even_input_sampling():
    paths = list(range(16))
    assert automatic_view_limit(5.0, 16) == 1
    assert automatic_view_limit(8.0, 16) == 2
    assert automatic_view_limit(16.0, 16) == 6
    assert automatic_view_limit(48.0, 16) == 16
    assert select_inputs(paths, 4) == [0, 5, 10, 15]


def test_preprocess_is_rgb_float_448_square(tmp_path):
    path = tmp_path / "portrait.png"
    Image.new("RGBA", (200, 500), (20, 40, 60, 128)).save(path)
    tensor = preprocess_image(path)
    assert tensor.shape == (3, 448, 448)
    assert tensor.dtype == torch.float32
    assert 0 <= tensor.min() <= tensor.max() <= 1
