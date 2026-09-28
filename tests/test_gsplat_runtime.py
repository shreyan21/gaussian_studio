import numpy as np
import pytest
from plyfile import PlyData

from studio.gsplat_runtime import (
    _adaptive_subject_crop_ratio,
    _clamp_log_scale_anisotropy_,
    _export_arrays,
    _scale_shift_invariant_depth_loss,
    _select_export_indices,
    _spatial_coherence_mask,
    _subject_crop_bounds,
    _training_profile,
    _viewer_quaternions,
    write_support_cloud,
)

torch = pytest.importorskip("torch")


def test_training_profiles_and_step_override(monkeypatch):
    monkeypatch.delenv("GSS_GSPLAT_STEPS", raising=False)
    assert _training_profile(1200) == {
        "image_side": 640,
        "steps": 4800,
        "max_splats": 350_000,
    }
    monkeypatch.setenv("GSS_GSPLAT_STEPS", "900")
    assert _training_profile(1600)["steps"] == 900


def test_viewer_coordinate_export_and_focus_crop(tmp_path):
    count = 2_000
    means = np.zeros((count, 3), np.float32)
    means[:, 0] = np.linspace(-0.45, 0.45, count)
    means[:, 1] = 0.25
    means[:, 2] = 1.0
    splats = {
        "means": torch.from_numpy(means),
        "scales": torch.full((count, 3), -4.0),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count,), 2.0),
        "colors": torch.zeros((count, 3)),
    }
    gaussians, focused = _export_arrays(
        splats,
        {"target": np.array([0.0, 0.25, 1.0]), "camera_distance": 1.0},
    )
    assert focused is True
    assert len(gaussians) > int(count * 0.95)
    np.testing.assert_allclose(gaussians[0, 1:3], [-0.25, -1.0])
    np.testing.assert_allclose(gaussians[0, 8:12], [0, 1, 0, 0])
    path = tmp_path / "support.ply"
    write_support_cloud(path, gaussians)
    assert len(PlyData.read(path)["vertex"]) == count


def test_viewer_quaternion_composes_x_flip():
    result = _viewer_quaternions(np.array([[1, 0, 0, 0]], np.float32))
    np.testing.assert_allclose(result, [[0, 1, 0, 0]])


def test_subject_crop_follows_projected_focus_and_stays_in_frame():
    K = np.array([[100, 0, 100], [0, 100, 50], [0, 0, 1]], np.float32)
    viewmat = np.eye(4, dtype=np.float32)
    bounds = _subject_crop_bounds(200, 100, K, viewmat, np.array([1.5, 0, 3]))
    assert bounds == (64, 16, 200, 84)


def test_subject_crop_adapts_to_camera_distance_with_safe_limits():
    focus = {"target": np.zeros(3), "camera_distance": 10.0}
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 10]), focus) == pytest.approx(0.68)
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 20]), focus) == pytest.approx(0.48)
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 5]), focus) == pytest.approx(0.86)


def test_training_clamps_anisotropy_without_changing_geometric_midpoint():
    scales = torch.log(torch.tensor([[0.001, 0.02, 0.2], [0.02, 0.02, 0.02]]))
    midpoints_before = (scales.amin(dim=1) + scales.amax(dim=1)) * 0.5
    _clamp_log_scale_anisotropy_(scales)
    ratios = torch.exp(scales.amax(dim=1) - scales.amin(dim=1))
    midpoints_after = (scales.amin(dim=1) + scales.amax(dim=1)) * 0.5
    assert torch.all(ratios <= 6.00001)
    torch.testing.assert_close(midpoints_after, midpoints_before)


def test_export_removes_oversized_streak_splats():
    count = 2_001
    scales = torch.full((count, 3), -4.0)
    scales[-1] = torch.log(torch.tensor([2.0, 0.01, 0.01]))
    splats = {
        "means": torch.zeros((count, 3)),
        "scales": scales,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count,), 2.0),
        "colors": torch.zeros((count, 3)),
    }
    stats = {}
    gaussians, _ = _export_arrays(splats, None, scene_scale=1.0, export_stats=stats)
    assert len(gaussians) == count - 1
    assert stats["export_removed_gaussians"] == 1


def test_export_removes_highly_anisotropic_shards():
    count = 2_001
    scales = torch.full((count, 3), -4.0)
    scales[-1] = torch.log(torch.tensor([0.05, 0.0001, 0.05]))
    splats = {
        "means": torch.zeros((count, 3)),
        "scales": scales,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count,), 2.0),
        "colors": torch.zeros((count, 3)),
    }
    stats = {}
    gaussians, _ = _export_arrays(splats, None, scene_scale=1.0, export_stats=stats)
    assert len(gaussians) == count - 1
    assert stats["export_anisotropy_limit"] == 6.0
    assert stats["export_opacity_minimum"] == 0.04


def test_export_retention_floor_preserves_trained_subject_density():
    count = 100_000
    splats = {
        "means": torch.zeros((count, 3)),
        "scales": torch.full((count, 3), -4.0),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        "opacities": torch.full((count,), -4.0),
        "colors": torch.zeros((count, 3)),
    }
    stats = {}
    gaussians, _ = _export_arrays(splats, None, scene_scale=1.0, export_stats=stats)
    assert len(gaussians) == 60_000
    assert stats["export_retention_floor"] == 60_000
    assert stats["export_confident_gaussians"] == 0
    assert stats["export_valid_gaussians"] == count


def test_focused_export_keeps_limited_nearby_context():
    count = 8_000
    means = np.zeros((count, 3), np.float32)
    means[4_000:7_000, 0] = 0.55
    means[7_000:, 0] = 0.80
    scales = np.full((count, 3), 0.01, np.float32)
    opacities = np.full(count, 0.80, np.float32)
    support = np.zeros(count, np.float32)
    support[:4_000] = 1.0
    stats = {}

    indices, focused, stats = _select_export_indices(
        means,
        scales,
        opacities,
        {"target": np.zeros(3), "camera_distance": 1.0},
        scene_scale=1.0,
        maximum=count,
        subject_support=support,
    )

    assert focused is True
    assert np.count_nonzero(indices < 4_000) == 4_000
    assert np.count_nonzero((indices >= 4_000) & (indices < 7_000)) == 1_000
    assert np.count_nonzero(indices >= 7_000) == 0
    assert stats["export_subject_gaussians"] == 4_000
    assert stats["export_context_gaussians"] == 1_000
    assert stats["export_context_radius_ratio"] == 0.62


def test_ai_depth_loss_ignores_monocular_scale_and_shift():
    inverse_depth = torch.linspace(0.2, 1.2, 100).reshape(1, 10, 10, 1)
    rendered_depth = inverse_depth.reciprocal().requires_grad_(True)
    predicted_inverse = inverse_depth[0, :, :, 0] * 3.5 + 7.0
    alpha = torch.ones_like(rendered_depth)

    loss = _scale_shift_invariant_depth_loss(
        rendered_depth,
        predicted_inverse,
        alpha,
    )

    assert loss.item() < 1e-6
    loss.backward()
    assert rendered_depth.grad is not None


def test_ai_depth_loss_penalizes_wrong_surface_order():
    inverse_depth = torch.linspace(0.2, 1.2, 100).reshape(1, 10, 10, 1)
    rendered_depth = inverse_depth.reciprocal()
    predicted_inverse = torch.flip(inverse_depth[0, :, :, 0], dims=(0, 1))
    alpha = torch.ones_like(rendered_depth)

    loss = _scale_shift_invariant_depth_loss(
        rendered_depth,
        predicted_inverse,
        alpha,
    )

    assert loss.item() > 0.5


def test_spatial_coherence_removes_only_extreme_isolated_splats():
    rng = np.random.default_rng(7)
    surface = rng.normal(0, 0.01, (2_000, 3)).astype(np.float32)
    outliers = np.array([[10, 10, 10], [-10, -10, -10]], np.float32)
    means = np.vstack((surface, outliers))
    scales = np.full((len(means), 3), 0.003, np.float32)

    keep, stats = _spatial_coherence_mask(
        means,
        scales,
        np.ones(len(means), dtype=bool),
    )

    assert np.all(keep[:2_000])
    assert not np.any(keep[2_000:])
    assert stats["export_coherence_evaluated"] is True
    assert stats["export_coherence_removed_gaussians"] == 2


def test_spatial_coherence_guard_preserves_unusually_sparse_scene():
    means = np.arange(2_000 * 3, dtype=np.float32).reshape(2_000, 3)
    means[::2] *= 100
    scales = np.full((len(means), 3), 0.001, np.float32)

    keep, stats = _spatial_coherence_mask(
        means,
        scales,
        np.ones(len(means), dtype=bool),
    )

    assert np.all(keep)
    assert stats["export_coherence_evaluated"] is False
