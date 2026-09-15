from scripts.install_sharp import REQUIRED_SOURCE_FILES, installation_is_complete
from studio.config import SHARP_COMMIT


def test_sharp_install_requires_sources_in_addition_to_marker(tmp_path):
    (tmp_path / "PINNED_COMMIT").write_text(SHARP_COMMIT, encoding="utf-8")
    assert not installation_is_complete(tmp_path)

    for relative in REQUIRED_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    assert installation_is_complete(tmp_path)
