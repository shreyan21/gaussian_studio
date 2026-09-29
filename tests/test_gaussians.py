import struct

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from studio.gaussians import export_scene, from_depth, isolate_largest_subject, make_demo, read_ply, validate, write_ply


def test_ply_preserves_real_gaussian_parameters(tmp_path):
    g = make_demo()[:37]
    path = tmp_path / "scene.ply"
    write_ply(path, g)
    np.testing.assert_allclose(read_ply(path), g, rtol=1e-5, atol=1e-6)
    v = PlyData.read(path)["vertex"]
    assert {"x", "scale_2", "rot_3", "opacity", "f_dc_2"} <= set(v.data.dtype.names)
    np.testing.assert_allclose(np.exp(v["scale_0"]), 0.016, rtol=1e-5)
    assert path.read_bytes().startswith(b"ply\nformat binary_little_endian")


def test_depth_is_perspective_3d_with_correct_axes():
    image = Image.new("RGB", (64,48), (200,100,50))
    disparity = np.tile(np.linspace(0,1,64), (48,1))
    g, _, camera = from_depth(image, disparity)
    grid = g.reshape(48,64,16)
    assert grid[24,0,0] < grid[24,-1,0]
    assert grid[0,32,1] > grid[-1,32,1]
    assert grid[24,0,2] < grid[24,-1,2] < 0  # low disparity is farther away
    assert np.ptp(g[:,2]) > 1
    assert (g[:,6] < g[:,4]).all()  # anisotropic, not point sprites
    np.testing.assert_allclose(np.linalg.norm(g[:,8:12],axis=1),1,atol=1e-6)
    assert 10 < camera["fov_y"] < 100
    assert 8 <= camera["safe_yaw_degrees"] <= 18
    assert 6 <= camera["safe_pitch_degrees"] <= 12
    assert len(camera["background_color"]) == 3


def test_depth_discontinuity_splats_face_camera_without_shrinking():
    image = Image.new("RGB", (64, 48), (120, 180, 220))
    disparity = np.full((48, 64), 0.2, np.float32)
    disparity[:, 32:] = 1.0

    g, _, camera = from_depth(image, disparity)
    grid = g.reshape(48, 64, 16)

    # Quaternion identity means the surfel normal remains +Z at the sharp
    # foreground/background boundary instead of becoming an edge-on black gap.
    boundary = grid[:, 30:34]
    np.testing.assert_allclose(boundary[:, :, 8], 1.0, atol=1e-5)
    np.testing.assert_allclose(boundary[:, :, 9:12], 0.0, atol=1e-5)
    boundary_coverage = boundary[:, :, 4] / np.abs(boundary[:, :, 2])
    surface_coverage = grid[:, :20, 4] / np.abs(grid[:, :20, 2])
    assert np.median(boundary_coverage) > np.median(surface_coverage)
    assert camera["depth_edge_ratio"] > 0


def test_constant_depth_and_degenerate_normals_are_finite():
    g, _, _ = from_depth(Image.new("RGB",(32,32)), np.ones((32,32)))
    assert np.isfinite(g).all()
    assert len(g) == 1024


def test_preview_does_not_reduce_full_ply(tmp_path):
    g = make_demo()
    meta = export_scene(tmp_path,g,{"engine":"test"},preview_limit=100)
    assert meta["gaussians"] == len(g)
    assert meta["preview_gaussians"] == 100
    assert len(PlyData.read(tmp_path / "scene.ply")["vertex"]) == len(g)
    raw = (tmp_path / "scene.gsb").read_bytes()
    assert struct.unpack("<4sIII",raw[:16]) == (b"GSS1",100,16,0)
    assert len(raw) == 16+100*64


def test_export_preserves_reconstructed_source_camera(tmp_path):
    meta = export_scene(tmp_path, make_demo()[:100], {"engine": "test", "source_camera": [1, 2, 3]})
    assert meta["source_camera"] == [1, 2, 3]


def test_isolated_subject_filter_removes_detached_gaussian_island():
    rng = np.random.default_rng(8)
    main = np.zeros((2_000, 16), np.float32)
    main[:, :3] = rng.normal(0, 0.08, (len(main), 3))
    island = np.zeros((400, 16), np.float32)
    island[:, :3] = rng.normal((2, 2, 2), 0.05, (len(island), 3))
    g = np.concatenate((main, island))
    g[:, 3], g[:, 4:7], g[:, 8], g[:, 12:15] = 0.8, 0.01, 1, 0.5
    cleaned, stats = isolate_largest_subject(g)
    assert stats["component_filter_applied"] is True
    assert len(cleaned) == 2_000
    assert stats["component_removed_gaussians"] == 400


def test_export_cleans_only_focused_subject_scenes(tmp_path):
    rng = np.random.default_rng(9)
    g = np.zeros((2_200, 16), np.float32)
    g[:2_000, :3] = rng.normal(0, 0.08, (2_000, 3))
    g[2_000:, :3] = rng.normal(2, 0.04, (200, 3))
    g[:, 3], g[:, 4:7], g[:, 8], g[:, 12:15] = 0.8, 0.01, 1, 0.5
    meta = export_scene(tmp_path, g, {"engine": "test", "subject_focus_applied": True})
    assert meta["component_filter_applied"] is True
    assert meta["gaussians"] == 2_000


def test_invalid_model_outputs_rejected():
    with pytest.raises(ValueError):
        validate(np.zeros((5,16)))
    with pytest.raises(ValueError):
        validate(np.full((5,16),np.nan))
    with pytest.raises(ValueError):
        from_depth(Image.new("RGB",(32,32)), np.full((32,32),np.nan))


def test_sanitizing_removes_nonfinite_and_transparent_splats():
    g = make_demo()[:10]
    g[0,0] = np.nan
    g[1,3] = 0
    g[2,4] = -1
    assert len(validate(g)) == 7


