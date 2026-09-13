"""Launch a local server; choose a free loopback port and open it after readiness."""
import argparse
import json
import os
import socket
import threading
import time
import urllib.request
import urllib.parse
import webbrowser
import uuid

import uvicorn
from filelock import FileLock, Timeout
from studio.config import DATA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    access_token = os.environ.get("GSS_ACCESS_TOKEN", "").strip()
    DATA.mkdir(parents=True, exist_ok=True)
    instance_lock = FileLock(str(DATA / "server.lock"))
    try:
        instance_lock.acquire(timeout=0)
    except Timeout:
        # Repeated double-clicks reuse this installation's server and do not let
        # competing job managers mutate the same jobs directory.
        for _ in range(30):
            try:
                info = json.loads((DATA / "server.json").read_text())
                health_url = info["url"] + "/api/health"
                if access_token:
                    health_url += "?token=" + urllib.parse.quote(access_token, safe="")
                with urllib.request.urlopen(health_url, timeout=1) as r:
                    status = json.load(r)
                if status.get("instance_id") == info["instance_id"]:
                    print("Studio is already running:", info["url"])
                    if not args.no_browser:
                        webbrowser.open(info["url"])
                    raise SystemExit(0)
            except (OSError, ValueError, KeyError):
                time.sleep(.5)
        raise SystemExit("This installation is already starting or running. Use its open browser window.")
    bound = None
    for port in range(args.port, min(args.port+20,65536)):
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            candidate.bind(("127.0.0.1",port))
            candidate.listen(128)
            bound = candidate
            break
        except OSError:
            candidate.close()
    if bound is None:
        raise SystemExit("No free local port. Try python run.py --port 9000")
    url = f"http://127.0.0.1:{port}"
    instance_id = uuid.uuid4().hex
    os.environ["GSS_INSTANCE_ID"] = instance_id
    (DATA / "server.json").write_text(json.dumps({"url":url,"pid":os.getpid(),"instance_id":instance_id}))
    browser_url = url
    if access_token:
        browser_url += "/?token=" + urllib.parse.quote(access_token, safe="")
    print(f"Gaussian Scene Studio: {browser_url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        def open_browser():
            for _ in range(60):
                try:
                    health_url = url + "/api/health"
                    if access_token:
                        health_url += "?token=" + urllib.parse.quote(access_token, safe="")
                    with urllib.request.urlopen(health_url, timeout=1) as r:
                        if json.load(r).get("app") == "Gaussian Scene Studio":
                            webbrowser.open(browser_url)
                            return
                except Exception:
                    time.sleep(0.5)
        threading.Thread(target=open_browser, daemon=True).start()
    try:
        uvicorn.Server(uvicorn.Config("studio.server:app", host="127.0.0.1",port=port,log_level="info")).run(sockets=[bound])
    finally:
        bound.close()
        instance_lock.release()
