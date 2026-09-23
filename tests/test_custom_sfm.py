import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from studio.custom_sfm import _focus_from_camera_rays, _gpu_indices, _limit_patch_match_sources, _patch_match_profile, _prepare_images, _validate_registration, dense_cloud_to_gaussians, extract_video_frames, find_colmap


def test_prepare_images_makes_ordered_equal_square_frames(tmp_path):
    inputs = []
    for index, size in enumerate(((80, 40), (40, 80))):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", size, (20, 40, 60)).save(path)
        inputs.append(path)
    output = _prepare_images(inputs, tmp_path / "prepared", 128)
    assert [path.name for path in output] == ["0000.jpg", "0001.jpg"]
    assert [Image.open(path).size for path in output] == [(128, 128), (128, 128)]


def test_video_frames_are_evenly_selected_and_renamed(tmp_path, monkeypatch):
    def fake_ffmpeg(command, **kwargs):
        raw = tmp_path / "video-frames-raw"
        raw.mkdir(exist_ok=True)
        for index in range(24):
            image = np.full((48, 64, 3), index * 5, dtype=np.uint8)
            image[:, index % 64] = 255
            Image.fromarray(image).save(raw / f"candidate_{index + 1:04d}.jpg")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("studio.custom_sfm.subprocess.run", fake_ffmpeg)
    output = extract_video_frames(tmp_path / "capture.mp4", tmp_path / "selected", max_frames=12)
    assert [path.name for path in output] == [f"input_{index:03d}.jpg" for index in range(12)]
    assert all(path.is_file() for path in output)


def test_patch_match_profiles_bound_dense_work():
    assert _patch_match_profile(1200) == {"iterations": 3, "samples": 10, "cache_gb": 4, "max_sources": 8}
    assert _patch_match_profile(1600)["iterations"] == 4
    assert _patch_match_profile(2000)["iterations"] == 5


def test_all_visible_gpus_are_used(monkeypatch):
    result = type("Result", (), {"returncode": 0, "stdout": "GPU 0: T4\nGPU 1: T4\n"})()
    monkeypatch.delenv("GSS_GPU_INDEX", raising=False)
    monkeypatch.setattr("studio.custom_sfm.subprocess.run", lambda *args, **kwargs: result)
    assert _gpu_indices() == "0,1"


def test_patch_match_source_views_are_bounded(tmp_path):
    stereo = tmp_path / "stereo"
    stereo.mkdir()
    config = stereo / "patch-match.cfg"
    config.write_text("0000.jpg\n__auto__, 30\n0001.jpg\n__auto__\n", encoding="utf-8")
    assert _limit_patch_match_sources(tmp_path, 10) == 2
    assert config.read_text(encoding="utf-8") == "0000.jpg\n__auto__, 10\n0001.jpg\n__auto__, 10\n"


def test_camera_rays_recover_central_subject():
    target = np.array([0.0, 0.0, -3.0])
    centers = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=float)
    directions = target - centers
    focus = _focus_from_camera_rays(centers, directions)
    np.testing.assert_allclose(focus["target"], target, atol=1e-6)
    np.testing.assert_allclose(focus["source_camera"], centers[0])


def test_dense_cloud_becomes_valid_oriented_gaussians(tmp_path):
    side = 110
    yy, xx = np.mgrid[:side, :side]
    count = side * side
    fields = (("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"))
    data = np.zeros(count, dtype=list(fields))
    data["x"], data["y"], data["z"] = xx.ravel() / side, yy.ravel() / side, np.sin(xx.ravel()) * 0.01
    data["nz"], data["red"], data["green"], data["blue"] = 1, 180, 80, 30
    path = tmp_path / "fused.ply"
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)
    result = dense_cloud_to_gaussians(path)
    assert 10_000 <= len(result) <= count
    assert result.shape[1] == 16
    assert np.all(result[:, 4:7] > 0)
    np.testing.assert_allclose(np.linalg.norm(result[:, 8:12], axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(result[:, 12], 180 / 255, atol=1e-5)


def test_dense_cloud_focus_crops_distant_background(tmp_path):
    side = 170
    yy, xx = np.mgrid[:side, :side]
    count = side * side
    fields = (("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"))
    data = np.zeros(count, dtype=list(fields))
    data["x"], data["y"] = xx.ravel() / side, yy.ravel() / side
    data["nz"], data["red"], data["green"], data["blue"] = 1, 220, 60, 120
    path = tmp_path / "focused.ply"
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)
    stats = {}
    result = dense_cloud_to_gaussians(
        path,
        focus={"target": np.array([0.5, 0.5, 0]), "camera_distance": 0.95, "source_camera": np.zeros(3)},
        focus_stats=stats,
    )
    assert stats["subject_focus_applied"] is True
    assert 10_000 <= len(result) < count


def test_colmap_environment_override(tmp_path, monkeypatch):
    executable = tmp_path / "COLMAP.bat"
    executable.write_text("@echo off\n")
    monkeypatch.setenv("GSS_COLMAP", str(executable))
    assert find_colmap() == executable.resolve()


def test_registration_quality_gate_rejects_scattered_capture():
    with np.testing.assert_raises_regex(RuntimeError, "Capture rejected"):
        _validate_registration(5, 20, 200)
    assert _validate_registration(15, 20, 2000)["registration_ratio"] == 0.75
