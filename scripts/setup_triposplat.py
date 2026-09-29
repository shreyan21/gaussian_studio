"""Install the pinned official TripoSplat source and checkpoint snapshot."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.triposplat_runtime import (
    CHECKPOINT_FILES,
    TRIPOSPLAT_CODE_REVISION,
    TRIPOSPLAT_WEIGHTS_REVISION,
    triposplat_root,
)

UPSTREAM = "https://github.com/VAST-AI-Research/TripoSplat.git"


def checked(*arguments: str) -> None:
    subprocess.run(arguments, check=True)


def install_code(root: Path) -> None:
    if root.exists():
        if not (root / ".git").is_dir():
            raise RuntimeError(f"Refusing to replace non-Git directory: {root}")
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"Refusing to overwrite modified TripoSplat checkout: {root}")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        checked("git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(root))
    checked("git", "-C", str(root), "fetch", "--depth=1", "origin", TRIPOSPLAT_CODE_REVISION)
    checked("git", "-C", str(root), "checkout", "--detach", "--force", TRIPOSPLAT_CODE_REVISION)


def install_weights(root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="VAST-AI/TripoSplat",
        revision=TRIPOSPLAT_WEIGHTS_REVISION,
        local_dir=str(root / "ckpts"),
        allow_patterns=list(CHECKPOINT_FILES.values()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    root = triposplat_root()
    install_code(root)
    if not args.code_only:
        install_weights(root)
    missing = [relative for relative in CHECKPOINT_FILES.values() if not (root / "ckpts" / relative).is_file()]
    if missing and not args.code_only:
        raise RuntimeError(f"Checkpoint download is incomplete: {missing[0]}")
    print(f"TripoSplat source pinned at {TRIPOSPLAT_CODE_REVISION}")
    if not args.code_only:
        print(f"TripoSplat checkpoints pinned at {TRIPOSPLAT_WEIGHTS_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

