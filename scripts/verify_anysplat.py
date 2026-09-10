"""Verify bundled source, checkpoint identity, and optional import contract."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.anysplat_runtime import _install_inference_shims, model_ready
from studio.config import ANYSPLAT_COMMIT, ANYSPLAT_DIR, ANYSPLAT_ROOT, ANYSPLAT_SHA256

REQUIRED = ("src/model/model/anysplat.py", "src/model/encoder/anysplat.py", "src/model/types.py", "LICENSE")


def source_status():
    marker_path = ANYSPLAT_ROOT / "PINNED_COMMIT"
    marker = marker_path.read_text(encoding="utf-8").strip() if marker_path.is_file() else ""
    missing = [name for name in REQUIRED if not (ANYSPLAT_ROOT / name).is_file()]
    return {"commit": marker, "expected_commit": ANYSPLAT_COMMIT, "missing": missing, "ready": marker == ANYSPLAT_COMMIT and not missing}


def checkpoint_sha256():
    path = ANYSPLAT_DIR / "model.safetensors"
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_status():
    import torch
    if str(ANYSPLAT_ROOT) not in sys.path:
        sys.path.insert(0, str(ANYSPLAT_ROOT))
    _install_inference_shims(torch)
    from src.model.encoder.vggt.models.vggt import VGGT
    VGGT.from_pretrained = classmethod(lambda cls, *_args, **_kwargs: cls())
    from src.model.model.anysplat import AnySplat
    return AnySplat.__name__ == "AnySplat"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hash", action="store_true")
    parser.add_argument("--check-import", action="store_true")
    args = parser.parse_args()
    result = {"source": source_status(), "model_ready": model_ready()}
    if args.check_hash:
        result["checkpoint_sha256"] = checkpoint_sha256()
        result["checkpoint_hash_ok"] = result["checkpoint_sha256"] == ANYSPLAT_SHA256
    if args.check_import:
        try:
            result["import_ok"] = import_status()
        except Exception as exc:
            result["import_ok"] = False
            result["import_error"] = str(exc)
    print(json.dumps(result, indent=2))
    ok = result["source"]["ready"]
    if args.check_hash:
        ok = ok and result["checkpoint_hash_ok"]
    if args.check_import:
        ok = ok and result["import_ok"]
    raise SystemExit(0 if ok else 1)
