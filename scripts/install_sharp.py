"""Install a pinned copy of official SHARP inference sources; preserve licences."""
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.config import ROOT, SHARP_COMMIT


def main():
    target = ROOT / "vendor" / "ml-sharp"
    marker = target / "PINNED_COMMIT"
    if marker.is_file() and marker.read_text().strip() == SHARP_COMMIT:
        print("Pinned SHARP source is already installed.")
        return
    url = f"https://codeload.github.com/apple/ml-sharp/zip/{SHARP_COMMIT}"
    print("Downloading official SHARP source:", SHARP_COMMIT, flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read(30_000_000)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts[1:]
            if not parts or member.is_dir():
                continue
            if parts[0] != "src" and parts[0] not in ("LICENSE", "LICENSE_MODEL", "ACKNOWLEDGEMENTS", "README.md"):
                continue
            if any(part in ("..", ".") or ":" in part or "\\" in part for part in parts):
                raise RuntimeError("Unsafe source archive path")
            destination = (target / Path(*parts)).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise RuntimeError("Source path leaves vendor directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    if not (target / "src/sharp/models/__init__.py").is_file():
        raise RuntimeError("SHARP source archive did not contain the expected model")
    marker.write_text(SHARP_COMMIT + "\n")
    print("SHARP source ready:", target)


if __name__ == "__main__":
    main()
