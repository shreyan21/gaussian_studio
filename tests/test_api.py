import io
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio.server import Jobs, atomic_json, create_app


def photo(size=(64, 48)):
    stream = io.BytesIO()
    Image.new("RGB", size, "coral").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    with TestClient(create_app(tmp_path)) as test_client:
        yield test_client


def test_home_and_procedural_viewer_without_models(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/renderer.js").status_code == 200
    health = client.get("/api/health").json()
    assert health["app"] == "Gaussian Scene Studio"
    assert health["version"] == "2.3.0"
    assert health["max_images"] == 16
    assert set(health["models"]) == {"depth", "sharp", "anysplat", "foreground"}
    assert client.get("/api/demo/scene.gsb").content[:4] == b"GSS1"
    assert client.get("/api/demo/scene.json").json()["method"] == "demo"


def test_public_mode_requires_token_then_sets_session_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    monkeypatch.setenv("GSS_ACCESS_TOKEN", "correct-horse-battery-staple")
    with TestClient(create_app(tmp_path), base_url="https://studio.example") as public:
        assert public.get("/").status_code == 401
        assert public.get("/api/health").status_code == 401
        unlocked = public.get("/?token=correct-horse-battery-staple")
        assert unlocked.status_code == 200
        assert public.cookies.get("gss_access") == "correct-horse-battery-staple"
        health = public.get("/api/health").json()
        assert health["remote_access"] is True


def test_public_mode_accepts_same_origin_forwarded_by_proxy(tmp_path, monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    monkeypatch.setenv("GSS_ACCESS_TOKEN", "correct-horse-battery-staple")
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1:7860") as public:
        public.cookies.set("gss_access", "correct-horse-battery-staple")
        response = public.post(
            "/api/jobs",
            headers={
                "origin": "https://studio.example",
                "x-forwarded-host": "studio.example",
                "x-forwarded-proto": "https",
            },
            files={"image": ("input.png", photo(), "image/png")},
        )
        assert response.status_code == 202


def test_local_mode_does_not_trust_forwarded_host(client):
    response = client.post(
        "/api/jobs",
        headers={
            "origin": "https://unrelated.example",
            "x-forwarded-host": "unrelated.example",
        },
        files={"image": ("input.png", photo(), "image/png")},
    )
    assert response.status_code == 403


@pytest.mark.parametrize("contents,code", [(b"not an image", 415), (photo((16, 16)), 422), (photo((600, 32)), 422)])
def test_invalid_uploads(client, contents, code):
    response = client.post("/api/jobs", files={"image": ("input.png", contents, "image/png")})
    assert response.status_code == code


def test_invalid_options(client):
    cases = (
        {"engine": "unknown"},
        {"resolution": 12},
        {"device": "invalid"},
        {"depth_strength": "NaN"},
        {"view_limit": "99"},
        {"engine": "anysplat", "device": "cpu"},
    )
    for data in cases:
        response = client.post("/api/jobs", files={"image": ("image.png", photo(), "image/png")}, data=data)
        assert response.status_code == 422


def test_cross_origin_and_invalid_paths_blocked(client):
    assert client.post("/api/jobs", headers={"origin": "https://unrelated.example"}, files={"image": ("x.png", photo())}).status_code == 403
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/demo/request.json").status_code == 404
    assert client.get("/api/jobs").json() == []


def test_streamed_oversized_upload_rejected_before_parsing(tmp_path, monkeypatch):
    import studio.server as server
    monkeypatch.setenv("GSS_SKIP_PROBE", "1")
    monkeypatch.setattr(server, "MAX_UPLOAD", 1024)
    monkeypatch.setattr(server, "MAX_IMAGES", 1)
    with TestClient(server.create_app(tmp_path)) as local:
        response = local.post("/api/jobs", content=(b"x" * 1024 for _ in range(1027)), headers={"content-type": "multipart/form-data; boundary=test"})
        assert response.status_code == 413


def test_cancel_stops_real_inference_process_and_releases_slot(client):
    response = client.post("/api/jobs", files={"image": ("cancel.png", photo())})
    job_id = response.json()["id"]
    time.sleep(0.15)
    assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and client.get("/api/health").json()["active_job"]:
        time.sleep(0.05)
    assert client.get("/api/health").json()["active_job"] is None
    assert client.get(f"/api/jobs/{job_id}/files/scene.ply").status_code == 409


def test_upload_canonicalized_and_single_job_enforced(client, monkeypatch):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    response = client.post("/api/jobs", files={"image": ("../../unsafe.png", photo(), "image/png")})
    assert response.status_code == 202
    job = response.json()
    assert job["name"] == "unsafe.png"
    assert client.post("/api/jobs", files={"image": ("second.png", photo())}).status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/files/input.png").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/scene.ply").status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/files/request.json").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"


def test_anysplat_multiview_saved_in_order_without_direction_labels(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    files = [("images", (f"view-{index}.png", photo(), "image/png")) for index in range(1, 7)]
    response = client.post("/api/jobs", files=files, data={"engine": "anysplat", "view_limit": "4"})
    assert response.status_code == 202
    job = response.json()
    folder = tmp_path / "jobs" / job["id"]
    assert job["image_count"] == 6
    assert job["name"] == "view-1.png + 5 views"
    assert all((folder / ("input.png" if i == 0 else f"input_{i}.png")).is_file() for i in range(6))
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    assert request["engine"] == "anysplat"
    assert request["view_limit"] == "4"
    assert request["object_only"] is True
    assert [item["original_name"] for item in request["inputs"]] == [f"view-{i}.png" for i in range(1, 7)]
    assert client.get(f"/api/jobs/{job['id']}/files/input_5.png").status_code == 200


def test_engine_image_count_contract(client):
    one = [("images", ("one.png", photo(), "image/png"))]
    two = one + [("images", ("two.png", photo(), "image/png"))]
    assert client.post("/api/jobs", files=one, data={"engine": "anysplat"}).status_code == 422
    assert client.post("/api/jobs", files=two, data={"engine": "depth"}).status_code == 422
    assert client.post("/api/jobs", files=two, data={"engine": "sharp", "research_use": "true"}).status_code == 422


def test_sharp_requires_research_acknowledgement(client, monkeypatch, tmp_path):
    monkeypatch.setattr(Jobs, "run", lambda *args: None)
    denied = client.post("/api/jobs", files={"image": ("rose.png", photo(), "image/png")}, data={"engine": "sharp"})
    assert denied.status_code == 422
    accepted = client.post("/api/jobs", files={"image": ("rose.png", photo(), "image/png")}, data={"engine": "sharp", "research_use": "true"})
    assert accepted.status_code == 202
    request = json.loads((tmp_path / "jobs" / accepted.json()["id"] / "request.json").read_text(encoding="utf-8"))
    assert request["research_use"] is True


def test_sharp_viewer_has_hard_camera_limits():
    renderer = (Path(__file__).parents[1] / "static" / "renderer.js").read_text(encoding="utf-8")
    assert "clampView()" in renderer
    assert "yaw_degrees" in renderer
    assert "pitch_degrees" in renderer


def test_stale_running_jobs_recovered(tmp_path):
    path = tmp_path / ("a" * 32)
    path.mkdir()
    atomic_json(path / "job.json", {"status": "running", "id": "a" * 32})
    jobs = Jobs(tmp_path)
    assert jobs.read("a" * 32)["status"] == "failed"
    assert "stopped" in jobs.read("a" * 32)["message"]


def test_real_failed_worker_surfaces_message(client):
    import torch
    if torch.cuda.is_available():
        pytest.skip("This failure-path test applies to a CPU-only environment")
    response = client.post("/api/jobs", files={"image": ("test.png", photo())}, data={"device": "cuda"})
    job_id = response.json()["id"]
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] != "running":
            break
        time.sleep(0.2)
    assert job["status"] == "failed"
    assert "CUDA" in job["message"]
    assert client.get(f"/api/jobs/{job_id}/files/worker.log").status_code == 200
