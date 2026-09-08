import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from studio.server import Jobs, atomic_json, create_app


def photo(size=(64,48)):
    stream=io.BytesIO()
    Image.new("RGB",size,"coral").save(stream,format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv("GSS_SKIP_PROBE","1")
    with TestClient(create_app(tmp_path)) as client:
        yield client


def test_home_and_procedural_viewer_without_models(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/renderer.js').status_code == 200
    assert client.get('/api/health').json()['app'] == 'Gaussian Scene Studio'
    assert client.get('/api/demo/scene.gsb').content[:4] == b'GSS1'
    assert client.get('/api/demo/scene.json').json()['method'] == 'demo'


@pytest.mark.parametrize('contents,code',[(b'not an image',415),(photo((16,16)),422),(photo((600,32)),422)])
def test_invalid_uploads(client,contents,code):
    response=client.post('/api/jobs',files={'image':('input.png',contents,'image/png')})
    assert response.status_code == code


def test_invalid_options_and_research_requirement(client):
    for data in ({'engine':'unknown'},{'resolution':12},{'device':'invalid'},{'depth_strength':'NaN'},{'engine':'sharp'}):
        response=client.post('/api/jobs',files={'image':('image.png',photo(),'image/png')},data=data)
        assert response.status_code == 422


def test_cross_origin_and_invalid_paths_blocked(client):
    assert client.post('/api/jobs',headers={'origin':'https://unrelated.example'},files={'image':('x.png',photo())}).status_code == 403
    assert client.get('/api/jobs/not-a-job').status_code == 404
    assert client.get('/api/demo/request.json').status_code == 404
    assert client.get('/api/jobs').json() == []


def test_streamed_oversized_upload_rejected_before_parsing(client):
    def body():
        for _ in range(23):
            yield b'x'*(1024*1024)
    assert client.post('/api/jobs',content=body(),headers={'content-type':'multipart/form-data; boundary=test'}).status_code == 413


def test_cancel_stops_real_inference_process_and_releases_slot(client):
    response=client.post('/api/jobs',files={'image':('cancel.png',photo())})
    job_id=response.json()['id']
    time.sleep(.15)
    assert client.post(f'/api/jobs/{job_id}/cancel').json()['status'] == 'cancelled'
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and client.get('/api/health').json()['active_job']:
        time.sleep(.05)
    assert client.get('/api/health').json()['active_job'] is None
    assert client.get(f'/api/jobs/{job_id}/files/scene.ply').status_code == 409


def test_upload_canonicalized_and_single_job_enforced(client,monkeypatch):
    monkeypatch.setattr(Jobs,'run',lambda *args:None)
    response=client.post('/api/jobs',files={'image':('../../unsafe.png',photo(),'image/png')})
    assert response.status_code == 202
    job=response.json()
    assert job['name'] == 'unsafe.png'
    assert client.post('/api/jobs',files={'image':('second.png',photo())}).status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/files/input.png").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/files/scene.ply").status_code == 409
    assert client.get(f"/api/jobs/{job['id']}/files/request.json").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()['status'] == 'cancelled'


def test_stale_running_jobs_recovered(tmp_path):
    path=tmp_path/('a'*32)
    path.mkdir()
    atomic_json(path/'job.json',{'status':'running','id':'a'*32})
    jobs=Jobs(tmp_path)
    assert jobs.read('a'*32)['status'] == 'failed'
    assert 'stopped' in jobs.read('a'*32)['message']


def test_real_failed_worker_surfaces_message(client):
    # CUDA is absent in the local CPU test environment. Exercises the actual subprocess
    # failure path instead of manufacturing a completed reconstruction.
    import torch
    if torch.cuda.is_available():
        pytest.skip('This failure-path test applies to a CPU-only environment')
    r=client.post('/api/jobs',files={'image':('test.png',photo())},data={'device':'cuda'})
    job_id=r.json()['id']
    deadline=time.monotonic()+40
    while time.monotonic()<deadline:
        job=client.get('/api/jobs/'+job_id).json()
        if job['status']!='running':
            break
        time.sleep(.2)
    assert job['status'] == 'failed'
    assert 'CUDA' in job['message']
    assert client.get(f'/api/jobs/{job_id}/files/worker.log').status_code == 200
