"""Download official weights once. Inference then runs fully offline."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.config import DEPTH_DIR, DEPTH_ID, DEPTH_REVISION, MODELS, SHARP_URL, SHARP_WEIGHTS, SHARP_SHA256


def download_depth():
    from huggingface_hub import snapshot_download
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=DEPTH_ID, revision=DEPTH_REVISION, local_dir=DEPTH_DIR, allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md", "LICENSE*"])
    print("Depth model ready:", DEPTH_DIR)


def download_sharp():
    import requests
    MODELS.mkdir(exist_ok=True)
    if not SHARP_WEIGHTS.is_file():
        temp = SHARP_WEIGHTS.with_suffix(".part")
        print("Downloading SHARP (~2.6 GB). This may take several minutes.", flush=True)
        offset = temp.stat().st_size if temp.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(SHARP_URL, headers=headers, stream=True, timeout=(20,60)) as response:
            response.raise_for_status()
            if response.status_code == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Unexpected partial download response; please retry.")
            if response.status_code != 206:
                offset = 0
            total = offset + int(response.headers.get("Content-Length", 0))
            downloaded, last_report = offset, -1
            with temp.open("ab" if offset else "wb") as output:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    percent = int(downloaded*100/max(total,1))
                    if percent//5 != last_report:
                        last_report = percent//5
                        print(f"SHARP download: {percent}% ({downloaded//1024**2} MiB)", flush=True)
        # Official checkpoint length verified from its origin Content-Length.
        if temp.stat().st_size != 2809738232:
            raise RuntimeError("SHARP checkpoint is incomplete or differs from the pinned release. Rerun to resume the download.")
        os.replace(temp, SHARP_WEIGHTS)
    with SHARP_WEIGHTS.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if digest != SHARP_SHA256:
        raise RuntimeError("SHARP checkpoint SHA-256 does not match the verified release. Remove the corrupted checkpoint and rerun the download.")
    (MODELS / "sharp-download.json").write_text(json.dumps({"url": SHARP_URL, "sha256": digest, "bytes": SHARP_WEIGHTS.stat().st_size}, indent=2))
    print("SHARP weights ready:", SHARP_WEIGHTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["depth", "sharp", "all"], default="depth")
    args = parser.parse_args()
    if args.model in ("depth", "all"):
        download_depth()
    if args.model in ("sharp", "all"):
        download_sharp()
