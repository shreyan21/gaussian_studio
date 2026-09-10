import struct

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from studio.gaussians import export_scene, from_depth, make_demo, read_ply, validate, write_ply


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


