import json

import numpy as np
from PIL import Image

from studio import worker
from studio.triposplat_runtime import CHECKPOINT_FILES, triposplat_status


def test_triposplat_status_explains_missing_install(monkeypatch, tmp_path):
    monkeypatch.setenv("GSS_TRIPOSPLAT_ROOT", str(tmp_path / "not-installed"))
    ready, message = triposplat_status()
    assert ready is False
    assert "not installed" in message


def test_checkpoint_inventory_is_complete():
    assert set(CHECKPOINT_FILES) == {"flow", "decoder", "dinov3", "vae", "rmbg"}
    assert all(name.endswith(".safetensors") for name in CHECKPOINT_FILES.values())


def test_worker_exports_mocked_single_image_generation(monkeypatch, tmp_path):
    Image.new("RGB", (64, 64), "coral").save(tmp_path / "input.png")
    (tmp_path / "request.json").write_text(json.dumps({
        "engine": "triposplat",
        "inputs": [{"file": "input.png", "original_name": "rose.png"}],
        "video": None,
    }), encoding="utf-8")
    gaussians = np.zeros((64, 16), np.float32)
    gaussians[:, :3] = np.random.default_rng(3).normal(0, 0.1, (64, 3))
    gaussians[:, 3] = 0.8
    gaussians[:, 4:7] = 0.02
    gaussians[:, 8] = 1
    gaussians[:, 12:15] = 0.5

    monkeypatch.setattr(worker, "generate_single_image", lambda *args: (
        gaussians,
        {"method": "triposplat", "device": "cuda", "source_camera": [0, 0, 2]},
    ))
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(tmp_path)])

    assert worker.main() == 0
    assert (tmp_path / "scene.ply").is_file()
    assert (tmp_path / "scene.gsb").read_bytes()[:4] == b"GSS1"
    metadata = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    assert metadata["method"] == "triposplat"
    assert metadata["source_type"] == "image"
    assert metadata["gaussians"] == 64
