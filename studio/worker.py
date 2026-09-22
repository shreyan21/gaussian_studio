"""One pretrained-free reconstruction process per job."""
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from studio.custom_sfm import extract_video_frames, reconstruct
from studio.gaussians import export_scene


def progress(directory, percent, message):
    temporary = directory / "progress.tmp"
    temporary.write_text(json.dumps({"progress": percent, "message": message}), encoding="utf-8")
    os.replace(temporary, directory / "progress.json")
    print(message, flush=True)


def main():
    directory = Path(sys.argv[1]).resolve()
    options = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        if options["engine"] != "custom":
            raise RuntimeError("This build contains only the pretrained-free custom reconstruction engine.")
        video = options.get("video")
        if video:
            progress(directory, 3, "Extracting sharp, evenly spaced keyframes from video")
            paths = extract_video_frames(directory / video["file"], directory / "video-keyframes")
            with Image.open(paths[0]) as source:
                thumb = source.copy()
                thumb.thumbnail((480, 360))
                thumb.save(directory / "thumbnail.jpg", quality=85)
            source_type = "video"
        else:
            paths = [directory / item["file"] for item in options["inputs"]]
            source_type = "photos"
        gaussians, meta = reconstruct(
            directory,
            paths,
            options["resolution"],
            options["device"] != "cpu",
            lambda percent, message: progress(directory, percent, message),
        )
        meta["source_type"] = source_type
        meta["uploaded_count"] = 1 if video else len(paths)
        progress(directory, 94, "Writing Gaussian PLY and browser scene")
        meta["seconds"] = round(time.monotonic() - started, 2)
        export_scene(directory, gaussians, meta)
        del gaussians
        gc.collect()
        progress(directory, 100, "Scene ready")
    except Exception as exc:
        (directory / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
