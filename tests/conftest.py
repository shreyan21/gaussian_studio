"""Keep test scratch files inside the application, independent of Windows TEMP ACLs."""
import uuid
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        root = Path(__file__).resolve().parent.parent / "work"
        root.mkdir(exist_ok=True)
        target = (root / ("pytest-" + uuid.uuid4().hex)).resolve()
        if not target.is_relative_to(root.resolve()) or target.exists():
            raise RuntimeError("Unsafe test scratch path")
        config.option.basetemp = str(target)
