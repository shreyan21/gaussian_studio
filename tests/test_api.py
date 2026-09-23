import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
    assert health["version"] == "3.2.0"
    assert health["max_images"] == 80
    assert set(health["engines"]) == {"custom"}
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
    for data in ({"engine": "unknown"}, {"resolution": 12}, {"device": "invalid"}, {"device": "cpu"}, {"view_limit": "99"}):
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
    assert request["resolution"] == 512
    assert [item["original_name"] for item in request["inputs"]] == [f"view-{i:02d}.png" for i in range(20)]


def test_custom_requires_twelve_views(client):
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
