"""Fail-closed, rig-local configuration for the YAM composition."""

from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w

# Raw ArmType.YAM limits from i2rt's pinned robot_models/arm/yam/yam.xml.
# Gripper slots use the adapter's normalized [0, 1] policy units.
_ARM_LOW = (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944, 0.0)
_ARM_HIGH = (3.05433, 3.65, 3.66519, 1.5708, 1.5708, 2.0944, 1.0)
I2RT_JOINT_LOW: tuple[float, ...] = _ARM_LOW * 2
I2RT_JOINT_HIGH: tuple[float, ...] = _ARM_HIGH * 2
STRICT_STEP_LIMITS: tuple[float, ...] = ((0.2,) * 6 + (1.0,)) * 2


def config_home() -> Path:
    """Return the secret-free per-user config directory."""
    override = os.environ.get("DROPBEAR_YAM_CONFIG_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "dropbear-yam"


def state_home() -> Path:
    """Return the per-user mutable state directory."""
    override = os.environ.get("DROPBEAR_YAM_STATE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "state" / "dropbear-yam"


def rig_path() -> Path:
    return config_home() / "rig.toml"


@dataclasses.dataclass(frozen=True)
class RigConfig:
    """One confirmed physical rig, with non-negotiable attended defaults."""

    top_camera: str
    left_camera: str
    right_camera: str
    left_channel: str
    right_channel: str
    collision_left_base_pos: tuple[float, float, float]
    collision_right_base_pos: tuple[float, float, float]
    collision_left_base_yaw: float
    collision_right_base_yaw: float
    collision_table_height: float | None
    schema_version: int = 1
    model_target: str = "dreamzero-yam"
    cam_width: int = 640
    cam_height: int = 360
    control_hz: int = 30
    control_interface: str = "joints"
    joints_are_delta: bool = False
    gripper_type: str = "LINEAR_4310"
    joint_low: tuple[float, ...] = I2RT_JOINT_LOW
    joint_high: tuple[float, ...] = I2RT_JOINT_HIGH
    step_limits: tuple[float, ...] = STRICT_STEP_LIMITS
    collision_guardrail: bool = True
    collision_table: bool = True
    auto_start: bool = False
    unattended: bool = False
    keep_warm: int = 0
    strict_policy_actions: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported rig schema_version")
        if self.model_target != "dreamzero-yam":
            raise ValueError("model_target must be dreamzero-yam")
        if self.control_hz != 30:
            raise ValueError("DreamZero-YAM must run at exactly 30 Hz")
        if (self.cam_width, self.cam_height) != (640, 360):
            raise ValueError("DreamZero-YAM cameras must be 640x360")
        if self.control_interface != "joints" or self.joints_are_delta:
            raise ValueError("the live rig requires absolute joint control")
        if self.auto_start:
            raise ValueError("attended mode requires auto_start=false")
        if self.unattended:
            raise ValueError("attended mode requires unattended=false")
        if self.keep_warm != 0:
            raise ValueError("billing-safe cleanup requires keep_warm=0")
        if not self.strict_policy_actions:
            raise ValueError("live policy execution requires strict abort behavior")
        if not self.collision_guardrail or not self.collision_table:
            raise ValueError("predictive collision and table checks are mandatory")
        if self.collision_table_height is None:
            raise ValueError("measured collision geometry is mandatory")
        if len(self.collision_left_base_pos) != 3 or len(self.collision_right_base_pos) != 3:
            raise ValueError("collision base positions must contain x y z")
        if self.joint_low != I2RT_JOINT_LOW or self.joint_high != I2RT_JOINT_HIGH:
            raise ValueError("joint bounds must match the pinned I2RT YAM model")
        if self.step_limits != STRICT_STEP_LIMITS:
            raise ValueError("strict step limits must be 0.2 rad and one gripper stroke")
        devices = (self.top_camera, self.left_camera, self.right_camera)
        if any(not value for value in devices) or len(set(devices)) != 3:
            raise ValueError("three distinct camera role assignments are required")
        channels = (self.left_channel, self.right_channel)
        if any(not value for value in channels) or len(set(channels)) != 2:
            raise ValueError("two distinct CAN role assignments are required")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def yam_kwargs(self) -> dict[str, Any]:
        """Return only arguments owned by the Dreamscale YAM fork."""
        return {
            "top_cam_device": self.top_camera,
            "left_cam_device": self.left_camera,
            "right_cam_device": self.right_camera,
            "left_channel": self.left_channel,
            "right_channel": self.right_channel,
            "cam_width": self.cam_width,
            "cam_height": self.cam_height,
            "control_hz": self.control_hz,
            "control_interface": self.control_interface,
            "joints_are_delta": self.joints_are_delta,
            "gripper_type": self.gripper_type,
            "joint_low": self.joint_low,
            "joint_high": self.joint_high,
            "step_limits": self.step_limits,
            "collision_guardrail": self.collision_guardrail,
            "collision_left_base_pos": self.collision_left_base_pos,
            "collision_right_base_pos": self.collision_right_base_pos,
            "collision_left_base_yaw": self.collision_left_base_yaw,
            "collision_right_base_yaw": self.collision_right_base_yaw,
            "collision_table": self.collision_table,
            "collision_table_height": self.collision_table_height,
            "auto_start": self.auto_start,
            "unattended": self.unattended,
            "strict_policy_actions": self.strict_policy_actions,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RigConfig:
        values = dict(raw)
        for name in (
            "collision_left_base_pos",
            "collision_right_base_pos",
            "joint_low",
            "joint_high",
            "step_limits",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)


def load_rig(path: Path | None = None) -> RigConfig:
    resolved = path or rig_path()
    try:
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"rig config missing: {resolved}; run dropbear-yam setup") from exc
    raw = payload.get("rig")
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved} has no [rig] table")
    return RigConfig.from_dict(raw)


def save_rig(
    rig: RigConfig,
    path: Path | None = None,
    *,
    replace: bool = False,
) -> Path:
    resolved = path or rig_path()
    if resolved.exists():
        current = load_rig(resolved)
        if current == rig:
            return resolved
        if not replace:
            raise FileExistsError(
                f"confirmed rig already exists at {resolved}; use --reconfigure to replace it"
            )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(".toml.tmp")
    temporary.write_text(tomli_w.dumps({"rig": rig.as_dict()}), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, resolved)
    return resolved
