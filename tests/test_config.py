from __future__ import annotations

from pathlib import Path

import pytest

from dropbear_yam.config import I2RT_JOINT_HIGH, I2RT_JOINT_LOW, RigConfig, load_rig, save_rig


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
        ({"collision_table_height": None}, "collision geometry"),
    ],
)
def test_rig_refuses_unsafe_product_boundary(
    rig: RigConfig, changes: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        RigConfig(**{**rig.as_dict(), **changes})


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
