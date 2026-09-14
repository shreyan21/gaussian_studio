from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from studio.anysplat_runtime import automatic_view_limit, preprocess_image, select_inputs, to_viewer_gaussians
from studio.foreground import focus_object


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


def test_anysplat_conversion_removes_background_pixels():
    converted = to_viewer_gaussians(fake_gaussians(), foreground_mask=[True, False, True, False])
    assert len(converted) == 2


def test_anysplat_conversion_rejects_misaligned_foreground_mask():
    with pytest.raises(ValueError, match="Foreground mask"):
        to_viewer_gaussians(fake_gaussians(), foreground_mask=[True])


def test_anysplat_conversion_projects_image_masks_through_predicted_cameras():
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[:, 2, 2] = True
    poses = {
        "extrinsic": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "intrinsic": torch.tensor([[[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]]).repeat(1, 2, 1, 1),
    }
    gaussians = fake_gaussians(2)
    gaussians.means = torch.tensor([[[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]])
    converted = to_viewer_gaussians(gaussians, foreground_mask=masks, camera_poses=poses)
    assert len(converted) == 1


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


def test_object_focus_crops_subject_and_neutralizes_background():
    image = Image.new("RGB", (240, 120), "white")
    mask = Image.new("L", image.size, 0)
    for x in range(80, 160):
        for y in range(35, 85):
            image.putpixel((x, y), (180, 30, 20))
            mask.putpixel((x, y), 255)
    focused, foreground, coverage = focus_object(image, mask)
    assert focused.size == (448, 448)
    assert foreground.shape == (448, 448)
    assert 0.12 < coverage < 0.15
    assert focused.getpixel((0, 0)) == (127, 127, 127)
