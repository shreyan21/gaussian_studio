import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GSS_DATA_DIR", ROOT / "data")).resolve()
MODELS = ROOT / "models"
ANYSPLAT_ROOT = ROOT / "vendor" / "anysplat"
ANYSPLAT_SOURCE = ANYSPLAT_ROOT / "src"
ANYSPLAT_DIR = MODELS / "anysplat"
ANYSPLAT_ID = "lhjiang/anysplat"
ANYSPLAT_REVISION = "d2e8c343672646041ad4ea518184968f94362f01"
ANYSPLAT_COMMIT = "5f5e208a7dd57d52e43ea0d553a95eab526e8775"
ANYSPLAT_SHA256 = "1c4de2ba5a29c540b899af901bf02107395b5f0617655d347e262f814b4c0c7c"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
DEPTH_DIR = MODELS / "depth-anything-v2-small"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_IMAGES = 16
MAX_PIXELS = 24_000_000

# Keep model caches with the app, never upload user images to a model service.
os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
