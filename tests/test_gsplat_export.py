import numpy as np

from studio.gsplat_runtime import _select_export_indices


def _scene(count: int):
    return (
        np.zeros((count, 3), np.float32),
        np.full((count, 3), np.exp(-4), np.float32),
    )


def test_density_floor_preserves_low_opacity_trained_surfaces():
    means, scales = _scene(100_000)
    opacities = np.full(100_000, 0.018, np.float32)
    indices, focused, stats = _select_export_indices(
        means, scales, opacities, None, scene_scale=1.0, maximum=1_250_000
    )
    assert focused is False
    assert len(indices) == 60_000
    assert stats["export_retention_floor"] == 60_000
    assert stats["export_confident_gaussians"] == 0


def test_subject_volume_keeps_wide_three_dimensional_extent():
    means, scales = _scene(2_000)
    means[:, 0] = np.linspace(-0.7, 0.7, len(means))
    opacities = np.full(len(means), 0.8, np.float32)
    indices, focused, stats = _select_export_indices(
        means,
        scales,
        opacities,
        {"target": np.zeros(3), "camera_distance": 1.0},
        scene_scale=1.0,
        maximum=1_250_000,
    )
    assert focused is True
    assert len(indices) > int(len(means) * 0.95)
    assert stats["export_focus_radius_ratio"] == 0.72


def test_subject_mask_support_removes_background_inside_focus_volume():
    means, scales = _scene(4_000)
    means[:, 0] = np.linspace(-0.5, 0.5, len(means))
    opacities = np.full(len(means), 0.8, np.float32)
    support = np.zeros(len(means), np.float32)
    support[1_000:3_000] = 0.9
    indices, focused, stats = _select_export_indices(
        means,
        scales,
        opacities,
        {"target": np.zeros(3), "camera_distance": 1.0},
        scene_scale=1.0,
        maximum=1_250_000,
        subject_support=support,
    )
    assert focused is True
    assert len(indices) == 2_000
    assert indices.min() == 1_000
    assert indices.max() == 2_999
    assert stats["export_mask_support_threshold"] == 0.55
    assert stats["export_mask_supported_gaussians"] == 2_000


def test_export_still_removes_oversized_and_elongated_splats():
    means, scales = _scene(2_002)
    scales[-2] = [2.0, 0.01, 0.01]
    scales[-1] = [0.05, 0.0001, 0.05]
    opacities = np.full(len(means), 0.8, np.float32)
    indices, _, stats = _select_export_indices(
        means, scales, opacities, None, scene_scale=1.0, maximum=1_250_000
    )
    assert len(indices) == len(means) - 2
    assert stats["export_anisotropy_limit"] == 6.0
