import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GSS_DATA_DIR", ROOT / "data")).resolve()
MODELS = ROOT / "models"
SHARP_SOURCE = ROOT / "vendor" / "ml-sharp" / "src"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
SHARP_COMMIT = "1eaa046834b81852261262b41b0919f5c1efdd2e"
SHARP_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
SHARP_SHA256 = "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
SHARP_WEIGHTS = MODELS / "sharp_2572gikvuh.pt"
DEPTH_DIR = MODELS / "depth-anything-v2-small"
MAX_UPLOAD = 20 * 1024 * 1024
MAX_PIXELS = 24_000_000

# Keep model caches with the app, never upload user images to a model service.
os.environ.setdefault("HF_HOME", str(MODELS / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
