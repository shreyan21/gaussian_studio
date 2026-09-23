import numpy as np
import pytest
from plyfile import PlyData

from studio.gsplat_runtime import (
    _export_arrays,
    _subject_crop_bounds,
    _training_profile,
    _viewer_quaternions,
    write_support_cloud,
)

torch = pytest.importorskip("torch")


def test_training_profiles_and_step_override(monkeypatch):
    monkeypatch.delenv("GSS_GSPLAT_STEPS", raising=False)
    assert _training_profile(1200) == {
        "image_side": 720,
        "steps": 6000,
        "max_splats": 500_000,
    }
    monkeypatch.setenv("GSS_GSPLAT_STEPS", "900")
    assert _training_profile(1600)["steps"] == 900


def test_viewer_coordinate_export_and_focus_crop(tmp_path):
    count = 2_000
    means = np.zeros((count, 3), np.float32)
    means[:, 0] = np.linspace(-0.2, 0.2, count)
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
    assert len(gaussians) == count
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
    assert bounds == (56, 14, 200, 86)


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
