import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from studio.custom_sfm import _prepare_images, dense_cloud_to_gaussians, find_colmap


def test_prepare_images_makes_ordered_equal_square_frames(tmp_path):
    inputs = []
    for index, size in enumerate(((80, 40), (40, 80))):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", size, (20, 40, 60)).save(path)
        inputs.append(path)
    output = _prepare_images(inputs, tmp_path / "prepared", 128)
    assert [path.name for path in output] == ["0000.jpg", "0001.jpg"]
    assert [Image.open(path).size for path in output] == [(128, 128), (128, 128)]


def test_dense_cloud_becomes_valid_oriented_gaussians(tmp_path):
    side = 20
    yy, xx = np.mgrid[:side, :side]
    count = side * side
    fields = (("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"))
    data = np.zeros(count, dtype=list(fields))
    data["x"], data["y"], data["z"] = xx.ravel() / side, yy.ravel() / side, np.sin(xx.ravel()) * 0.01
    data["nz"], data["red"], data["green"], data["blue"] = 1, 180, 80, 30
    path = tmp_path / "fused.ply"
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)
    result = dense_cloud_to_gaussians(path)
    assert 300 <= len(result) <= count
    assert result.shape[1] == 16
    assert np.all(result[:, 4:7] > 0)
    np.testing.assert_allclose(np.linalg.norm(result[:, 8:12], axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(result[:, 12], 180 / 255, atol=1e-5)


def test_colmap_environment_override(tmp_path, monkeypatch):
    executable = tmp_path / "COLMAP.bat"
    executable.write_text("@echo off\n")
    monkeypatch.setenv("GSS_COLMAP", str(executable))
    assert find_colmap() == executable.resolve()
