import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio.gaussians import read_ply, write_ply
from studio.server import Jobs, atomic_json, create_app


def photo(size=(64, 48), color="coral"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def views(count=12):
    return [("images", (f"view-{index:02d}.png", photo(color=(index * 20 % 255, 80, 120)), "image/png")) for index in range(count)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    with TestClient(create_app(tmp_path)) as test_client:
        yield test_client


def test_home_and_procedural_viewer_without_external_engine(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/renderer.js").status_code == 200
    health = client.get("/api/health").json()
    assert health["app"] == "Gaussian Scene Studio"
    assert health["version"] == "4.1.0"
    assert health["max_images"] == 80
    assert health["video_chunk_mb"] == 8
    assert set(health["engines"]) == {"custom", "gsplat", "single_photo"}
    assert health["single_photo_model"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert client.get("/api/demo/scene.gsb").content[:4] == b"GSS1"
    assert client.get("/api/demo/scene.json").json()["method"] == "demo"


def test_public_mode_requires_token_then_sets_session_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    monkeypatch.setenv("GSS_ACCESS_TOKEN", "correct-horse-battery-staple")
    with TestClient(create_app(tmp_path), base_url="https://studio.example") as public:
        assert public.get("/").status_code == 401
        assert public.get("/api/health").status_code == 401
        assert public.get("/?token=correct-horse-battery-staple").status_code == 200
        assert public.cookies.get("gss_access") == "correct-horse-battery-staple"


def test_public_mode_accepts_same_origin_forwarded_by_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    monkeypatch.setenv("GSS_ACCESS_TOKEN", "correct-horse-battery-staple")
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1:7860") as public:
        public.cookies.set("gss_access", "correct-horse-battery-staple")
        response = public.post("/api/jobs", headers={"origin": "https://studio.example", "x-forwarded-host": "studio.example", "x-forwarded-proto": "https"}, files=views())
        assert response.status_code == 202


def test_local_mode_does_not_trust_forwarded_host(client):
    response = client.post("/api/jobs", headers={"origin": "https://unrelated.example", "x-forwarded-host": "unrelated.example"}, files=views())
    assert response.status_code == 403


@pytest.mark.parametrize("contents,code", [(b"not an image", 415), (photo((16, 16)), 422), (photo((600, 32)), 422)])
def test_invalid_uploads(client, contents, code):
    files = [("images", ("bad.png", contents, "image/png"))] + views(11)
    assert client.post("/api/jobs", files=files).status_code == code


def test_invalid_options(client):
    for data in ({"engine": "unknown"}, {"device": "invalid"}, {"device": "cpu"}, {"view_limit": "99"}):
        assert client.post("/api/jobs", files=views(), data=data).status_code == 422


def test_cross_origin_and_invalid_paths_blocked(client):
    assert client.post("/api/jobs", headers={"origin": "https://unrelated.example"}, files=views()).status_code == 403
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/demo/request.json").status_code == 404
    assert client.get("/api/jobs").json() == []


def test_custom_multiview_saved_in_order(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    response = client.post("/api/jobs", files=views(20), data={"engine": "custom", "resolution": "512"})
    assert response.status_code == 202
    job = response.json()
    folder = tmp_path / "jobs" / job["id"]
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert job["image_count"] == 20
    assert request["engine"] == "custom"
    assert request["resolution"] == 768
    assert request["focus_subject"] is False
    assert [item["original_name"] for item in request["inputs"]] == [f"view-{i:02d}.png" for i in range(20)]


def test_custom_accepts_one_photo_or_requires_twelve_multiview_photos(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    response = client.post(
        "/api/jobs",
        files={"images": ("portrait.png", photo(), "image/png")},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["source_type"] == "single-photo"
    request = json.loads((tmp_path / "jobs" / job["id"] / "request.json").read_text(encoding="utf-8"))
    assert len(request["inputs"]) == 1

    client.app.state.jobs.active = None
    assert client.post("/api/jobs", files=views(11)).status_code == 422
    assert client.post("/api/jobs", files=views(12), data={"device": "cpu"}).status_code == 422


def test_custom_accepts_video_as_single_source(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"0" * 2048
    response = client.post("/api/jobs", files={"images": ("walk.mp4", payload, "video/mp4")}, data={"focus_subject": "false"})
    assert response.status_code == 202
    folder = tmp_path / "jobs" / response.json()["id"]
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert request["video"]["file"] == "input-video.mp4"
    assert request["inputs"] == []
    assert request["focus_subject"] is False
    assert (folder / "input-video.mp4").read_bytes() == payload


def test_chunked_video_upload_reassembles_before_starting_job(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"chunked-video" * 400
    response = client.post(
        "/api/video-uploads",
        data={"filename": "original-flower.mp4", "size": str(len(payload))},
    )
    assert response.status_code == 201
    upload = response.json()
    split = 2_000

    first = client.put(
        f"/api/video-uploads/{upload['id']}",
        params={"offset": 0},
        content=payload[:split],
    )
    second = client.put(
        f"/api/video-uploads/{upload['id']}",
        params={"offset": split},
        content=payload[split:],
    )
    assert first.json()["received"] == split
    assert second.json()["received"] == len(payload)

    response = client.post(
        f"/api/video-uploads/{upload['id']}/finish",
        data={"device": "auto"},
    )
    assert response.status_code == 202
    folder = tmp_path / "jobs" / response.json()["id"]
    assert (folder / "input-video.mp4").read_bytes() == payload
    assert json.loads((folder / "request.json").read_text(encoding="utf-8"))["resolution"] == 768


def test_chunked_video_upload_rejects_wrong_offset_and_incomplete_finish(client):
    payload = b"0" * 2_048
    upload = client.post(
        "/api/video-uploads",
        data={"filename": "flower.mp4", "size": str(len(payload))},
    ).json()

    assert client.put(
        f"/api/video-uploads/{upload['id']}",
        params={"offset": 1},
        content=payload,
    ).status_code == 409
    assert client.post(f"/api/video-uploads/{upload['id']}/finish").status_code == 409
    assert client.delete(f"/api/video-uploads/{upload['id']}").status_code == 204


def test_saved_video_can_be_rerun_with_current_code(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"0" * 2048
    original = client.post(
        "/api/jobs",
        files={"images": ("flower-pot.mp4", payload, "video/mp4")},
    ).json()
    jobs = client.app.state.jobs
    jobs.active = None
    response = client.post(f"/api/jobs/{original['id']}/retry")
    assert response.status_code == 202
    rerun = response.json()
    folder = tmp_path / "jobs" / rerun["id"]
    assert rerun["name"] == "flower-pot.mp4 - rerun"
    assert (folder / "input-video.mp4").read_bytes() == payload
    assert json.loads((folder / "request.json").read_text(encoding="utf-8"))["focus_subject"] is False


def test_saved_video_rerun_uses_fixed_high_quality_settings(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"0" * 2048
    original = client.post(
        "/api/jobs",
        files={"images": ("flower-pot.mp4", payload, "video/mp4")},
        data={"resolution": "512", "focus_subject": "false"},
    ).json()
    client.app.state.jobs.active = None

    response = client.post(
        f"/api/jobs/{original['id']}/retry",
        data={"resolution": "384", "focus_subject": "true"},
    )

    assert response.status_code == 202
    folder = tmp_path / "jobs" / response.json()["id"]
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert request["resolution"] == 768
    assert request["focus_subject"] is False


def test_completed_focused_scene_can_be_cleaned_without_retraining(client, tmp_path):
    job_id = "c" * 32
    folder = tmp_path / "jobs" / job_id
    folder.mkdir()
    rng = np.random.default_rng(11)
    g = np.zeros((2_400, 16), np.float32)
    g[:2_000, :3] = rng.normal(0, 0.08, (2_000, 3))
    g[2_000:, :3] = rng.normal((2, 2, 2), 0.05, (400, 3))
    g[:, 3], g[:, 4:7], g[:, 8], g[:, 12:15] = 0.8, 0.01, 1, 0.5
    write_ply(folder / "scene.ply", g)
    atomic_json(folder / "scene.json", {"engine": "test", "subject_focus_applied": True})
    atomic_json(folder / "request.json", {"engine": "custom", "inputs": [], "video": None})
    atomic_json(folder / "job.json", {
        "id": job_id,
        "name": "plant.mp4",
        "status": "completed",
        "engine": "custom",
        "image_count": 1,
        "source_type": "video",
    })

    response = client.post(f"/api/jobs/{job_id}/clean")
    assert response.status_code == 201
    cleaned = response.json()
    cleaned_folder = tmp_path / "jobs" / cleaned["id"]
    assert cleaned["name"] == "plant.mp4 - cleaned"
    assert len(read_ply(folder / "scene.ply")) == 2_400
    assert len(read_ply(cleaned_folder / "scene.ply")) == 2_000
    metadata = json.loads((cleaned_folder / "scene.json").read_text(encoding="utf-8"))
    assert metadata["component_removed_gaussians"] == 400
    assert metadata["cleaned_from_job"] == job_id


def test_upload_names_are_canonicalized(client, monkeypatch):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    files = [("images", ("../../unsafe.png", photo(), "image/png"))] + views(11)
    response = client.post("/api/jobs", files=files)
    assert response.status_code == 202
    job = response.json()
    assert job["name"] == "unsafe.png + 11 views"
    assert client.get(f"/api/jobs/{job['id']}/files/input_11.png").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/request.json").status_code == 404


def test_stale_running_jobs_recovered(tmp_path):
    path = tmp_path / ("a" * 32)
    path.mkdir()
    atomic_json(path / "job.json", {"status": "running", "id": "a" * 32})
    jobs = Jobs(tmp_path)
    assert jobs.read("a" * 32)["status"] == "failed"
