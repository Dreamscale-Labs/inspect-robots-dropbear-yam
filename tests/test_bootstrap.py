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
    assert "Error:" in setup and "Next:" in setup
    assert "Error:" in wrapper and "Next:" in wrapper


def test_linux_hardware_extra_installs_realsense_discovery_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'pyrealsense2>=2.50; sys_platform == "linux"' in project


def test_readme_uses_the_customer_facing_stable_branch_without_a_rig_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "git clone --branch stable --depth 1" in readme
    assert './dropbear-yam doctor\n' in readme
    assert './dropbear-yam run "Pack container"' in readme
    assert "--max-steps 3600" in readme
    assert "[Y/n]" in readme
    assert "elapsed seconds" in readme
    assert "paid shadow inference" not in readme
