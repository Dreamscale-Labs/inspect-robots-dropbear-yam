from __future__ import annotations

from pathlib import Path


def test_bootstrap_is_locked_and_has_one_explicit_sudo_confirmation() -> None:
    root = Path(__file__).resolve().parents[1]
    setup = (root / "setup.sh").read_text(encoding="utf-8")
    wrapper = (root / "dropbear-yam").read_text(encoding="utf-8")

    assert setup.count("read -r -p") == 1
    assert "sudo apt-get install" in setup
    assert '"$uv_bin" sync' in setup and "--locked --extra hardware" in setup
    assert "dropbear-yam setup" in setup
    assert "--locked --extra hardware" in wrapper
    assert "api_key" not in setup + wrapper
