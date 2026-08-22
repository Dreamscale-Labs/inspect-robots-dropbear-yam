from __future__ import annotations

from pathlib import Path

import pytest

from dropbear_yam.config import RigConfig


@pytest.fixture
def rig() -> RigConfig:
    return RigConfig(
        top_camera="/dev/v4l/by-id/top-video-index0",
        left_camera="/dev/v4l/by-id/left-video-index0",
        right_camera="/dev/v4l/by-id/right-video-index0",
        left_channel="can0",
        right_channel="can1",
        collision_left_base_pos=(-0.25, 0.0, 0.0),
        collision_right_base_pos=(0.25, 0.0, 0.0),
        collision_left_base_yaw=0.0,
        collision_right_base_yaw=3.14159,
        collision_table_height=0.0,
    )


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config"
    state = tmp_path / "state"
    monkeypatch.setenv("DROPBEAR_YAM_CONFIG_HOME", str(config))
    monkeypatch.setenv("DROPBEAR_YAM_STATE_HOME", str(state))
    return tmp_path
