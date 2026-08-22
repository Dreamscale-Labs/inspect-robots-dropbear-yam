from __future__ import annotations

from pathlib import Path

import pytest

import dropbear_yam.config as config
from dropbear_yam.config import (
    I2RT_JOINT_HIGH,
    I2RT_JOINT_LOW,
    RigConfig,
    load_rig,
    save_rig,
)


def test_rig_round_trip_is_fixed_attended_strict_30_hz(
    rig: RigConfig, isolated_paths: Path
) -> None:
    path = save_rig(rig)

    loaded = load_rig(path)

    assert loaded == rig
    assert loaded.control_hz == 30
    assert loaded.auto_start is False
    assert loaded.unattended is False
    assert loaded.keep_warm == 0
    assert loaded.strict_policy_actions is True
    assert loaded.joints_are_delta is False
    assert loaded.joint_low == I2RT_JOINT_LOW
    assert loaded.joint_high == I2RT_JOINT_HIGH


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"control_hz": 15}, "exactly 30 Hz"),
        ({"auto_start": True}, "auto_start=false"),
        ({"unattended": True}, "unattended=false"),
        ({"keep_warm": 10}, "keep_warm=0"),
        ({"strict_policy_actions": False}, "strict abort"),
        ({"collision_table_height": None}, "all five collision measurements"),
        (
            {"collision_guardrail": False, "collision_table": False},
            "remove the collision measurements",
        ),
        ({"collision_table": False}, "table collision checking"),
        ({"left_camera": "/dev/video4"}, "stable camera"),
        ({"left_camera": "realsense:"}, "serial"),
    ],
)
def test_rig_refuses_unsafe_product_boundary(
    rig: RigConfig, changes: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RigConfig(**{**rig.as_dict(), **changes})


def test_collision_geometry_can_be_deliberately_omitted(
    rig: RigConfig, isolated_paths: Path
) -> None:
    without_geometry = RigConfig(
        **{
            **rig.as_dict(),
            "collision_guardrail": False,
            "collision_table": False,
            "collision_left_base_pos": None,
            "collision_right_base_pos": None,
            "collision_left_base_yaw": None,
            "collision_right_base_yaw": None,
            "collision_table_height": None,
        }
    )

    path = save_rig(without_geometry)
    saved = path.read_text(encoding="utf-8")

    assert load_rig(path) == without_geometry
    assert "collision_left_base_pos" not in saved
    assert without_geometry.yam_kwargs()["collision_guardrail"] is False
    assert without_geometry.yam_kwargs()["collision_table"] is False
    assert "collision_table_height" not in without_geometry.yam_kwargs()


def test_save_rig_does_not_replace_confirmed_values_without_force(
    rig: RigConfig, isolated_paths: Path
) -> None:
    path = save_rig(rig)
    changed = RigConfig(**{**rig.as_dict(), "left_channel": "can9"})

    with pytest.raises(FileExistsError, match="--reconfigure"):
        save_rig(changed)

    assert load_rig(path) == rig
    save_rig(changed, replace=True)
    assert load_rig(path).left_channel == "can9"


def test_mixed_stable_camera_sources_map_to_the_yam_backends(rig: RigConfig) -> None:
    mixed = RigConfig(
        **{
            **rig.as_dict(),
            "top_camera": "/dev/v4l/by-id/d435-video-index0",
            "left_camera": "realsense:LEFT-D405",
            "right_camera": "/dev/v4l/by-path/pci-usb-right-video-index4",
        }
    )

    kwargs = mixed.yam_kwargs()

    assert kwargs["top_cam_device"] == "/dev/v4l/by-id/d435-video-index0"
    assert kwargs["left_depth_serial"] == "LEFT-D405"
    assert kwargs["right_cam_device"] == "/dev/v4l/by-path/pci-usb-right-video-index4"
    assert "left_cam_device" not in kwargs
    assert "right_depth_serial" not in kwargs
    assert kwargs["realsense_capture"] == "process"
    assert kwargs["depth_fps"] == 30


def test_named_profiles_are_isolated_and_ambiguous_implicit_selection_fails(
    rig: RigConfig, isolated_paths: Path
) -> None:
    left_path = save_rig(rig, profile="jay-left")
    right_rig = RigConfig(**{**rig.as_dict(), "left_channel": "can2", "right_channel": "can3"})
    right_path = save_rig(right_rig, profile="jay-right")

    assert left_path == config.rig_path("jay-left")
    assert right_path == config.rig_path("jay-right")
    assert load_rig(profile="jay-left") == rig
    assert load_rig(profile="jay-right") == right_rig
    assert config.resolve_rig_path("jay-left") == left_path
    with pytest.raises(ValueError, match="multiple rig profiles"):
        config.resolve_rig_path()


@pytest.mark.parametrize("profile", ["", "../escape", "a/b", "with space"])
def test_named_profile_cannot_escape_config_home(profile: str, isolated_paths: Path) -> None:
    with pytest.raises(ValueError, match="rig profile"):
        config.rig_path(profile)
