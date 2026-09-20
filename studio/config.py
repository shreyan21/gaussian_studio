import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GSS_DATA_DIR", ROOT / "data")).resolve()
COLMAP_VERSION = "4.2.0"
COLMAP_ROOT = ROOT / "tools" / "colmap"
COLMAP_WINDOWS_URL = "https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-cuda.zip"
COLMAP_WINDOWS_SHA256 = "991e0bae403a496fcc4de0c1f1f428619bf12f8000978f77bc6799d9bfeac23e"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_IMAGES = 80
MAX_PIXELS = 24_000_000
