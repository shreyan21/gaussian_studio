import numpy as np

from studio.gsplat_runtime import (
    _adaptive_subject_crop_ratio,
    _automatic_subject_mask,
    _project_mask_support,
)


def test_adaptive_crop_ratio_tracks_camera_distance():
    focus = {"target": np.zeros(3), "camera_distance": 10.0}
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 10]), focus) == 0.68
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 20]), focus) == 0.48
    assert _adaptive_subject_crop_ratio(np.array([0, 0, 5]), focus) == 0.86


def test_automatic_subject_mask_separates_center_from_border():
    image = np.full((120, 160, 3), [35, 80, 150], dtype=np.uint8)
    image[25:108, 62:98] = [185, 55, 35]
    image[10:80, 77:83] = [35, 170, 60]
    mask, backend = _automatic_subject_mask(image)
    assert backend in {"grabcut-central-prior", "geometric-central-prior"}
    assert mask.shape == image.shape[:2]
    assert mask[60, 80] == 1
    assert mask[0, 0] == 0
    assert 0.04 <= float(mask.mean()) <= 0.82


def test_project_mask_support_requires_multi_view_agreement():
    means = np.array([[0.0, 0.0, 2.0], [0.8, 0.0, 2.0]], dtype=np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[7:14, 7:14] = 1
    K = np.array([[10, 0, 10], [0, 10, 10], [0, 0, 1]], dtype=np.float32)
    views = [
        {"mask": mask, "K": K, "viewmat": np.eye(4, dtype=np.float32)},
        {"mask": mask, "K": K, "viewmat": np.eye(4, dtype=np.float32)},
    ]
    support, hits, seen = _project_mask_support(means, views)
    np.testing.assert_array_equal(seen, [2, 2])
    np.testing.assert_array_equal(hits, [2, 0])
    np.testing.assert_allclose(support, [1.0, 0.0])
