"""Download pinned official weights once; inference is offline afterward."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.config import ANYSPLAT_DIR, ANYSPLAT_ID, ANYSPLAT_REVISION, ANYSPLAT_SHA256, DEPTH_DIR, DEPTH_ID, DEPTH_REVISION


def download_depth():
    from huggingface_hub import snapshot_download
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DEPTH_ID,
        revision=DEPTH_REVISION,
        local_dir=DEPTH_DIR,
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "README.md", "LICENSE*"],
    )
    print("Depth model ready:", DEPTH_DIR)


def download_anysplat():
    from huggingface_hub import snapshot_download
    ANYSPLAT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading AnySplat checkpoint (~2.94 GB). This can take several minutes.", flush=True)
    snapshot_download(
        repo_id=ANYSPLAT_ID,
        revision=ANYSPLAT_REVISION,
        local_dir=ANYSPLAT_DIR,
        allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE*"],
    )
    checkpoint = ANYSPLAT_DIR / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("AnySplat checkpoint download did not complete.")
    print("Verifying AnySplat checkpoint SHA-256...", flush=True)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ANYSPLAT_SHA256:
        raise RuntimeError("AnySplat checkpoint SHA-256 mismatch. Remove models\\anysplat and retry setup.")
    (ANYSPLAT_DIR / "download.json").write_text(json.dumps({
        "repo": ANYSPLAT_ID,
        "revision": ANYSPLAT_REVISION,
        "sha256": digest,
        "bytes": checkpoint.stat().st_size,
    }, indent=2), encoding="utf-8")
    print("AnySplat model ready:", ANYSPLAT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["depth", "anysplat", "all"], default="depth")
    args = parser.parse_args()
    if args.model in ("depth", "all"):
        download_depth()
    if args.model in ("anysplat", "all"):
        download_anysplat()
