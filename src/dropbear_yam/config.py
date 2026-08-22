"""Fail-closed, rig-local configuration for the YAM composition."""

from __future__ import annotations

import dataclasses
import os
import re
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
REALSENSE_PREFIX = "realsense:"
_RIG_PROFILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


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


def rig_path(profile: str | None = None) -> Path:
    """Return the legacy path or one traversal-safe named rig path."""
    if profile is None:
        return config_home() / "rig.toml"
    if not _RIG_PROFILE.fullmatch(profile):
        raise ValueError(
            "rig profile must start with a letter or number and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return config_home() / "rigs" / f"{profile}.toml"


def rig_profiles() -> dict[str, Path]:
    """Return every configured rig without guessing among multiple physical rigs."""
    profiles: dict[str, Path] = {}
    legacy = rig_path()
    if legacy.exists():
        profiles["legacy"] = legacy
    directory = config_home() / "rigs"
    try:
        entries = sorted(directory.glob("*.toml"))
    except OSError:
        entries = []
    for entry in entries:
        profiles[entry.stem] = entry
    return profiles


def resolve_rig_path(profile: str | None = None) -> Path:
    """Resolve an explicit rig, or the only configured rig on the host."""
    if profile is not None:
        return rig_path(profile)
    configured = rig_profiles()
    if not configured:
        return rig_path()
    if len(configured) == 1:
        return next(iter(configured.values()))
    names = ", ".join(sorted(configured))
    raise ValueError(f"multiple rig profiles are configured ({names}); pass --rig NAME")


def camera_source(source: str) -> tuple[str, str]:
    """Split one stable source into the YAM backend name and backend identity."""
    if source.startswith(REALSENSE_PREFIX):
        serial = source.removeprefix(REALSENSE_PREFIX).strip()
        if not serial:
            raise ValueError("RealSense camera source requires a serial")
        return "realsense", serial
    return "v4l2", source


def stable_camera_source(source: str) -> bool:
    """Whether a source survives Linux device-number changes."""
    kind, value = camera_source(source)
    if kind == "realsense":
        return bool(value)
    return value.startswith(("/dev/v4l/by-id/", "/dev/v4l/by-path/"))


@dataclasses.dataclass(frozen=True)
class RigConfig:
    """One confirmed physical rig, with non-negotiable attended defaults."""

    top_camera: str
    left_camera: str
    right_camera: str
    left_channel: str
    right_channel: str
    collision_left_base_pos: tuple[float, float, float] | None = None
    collision_right_base_pos: tuple[float, float, float] | None = None
    collision_left_base_yaw: float | None = None
    collision_right_base_yaw: float | None = None
    collision_table_height: float | None = None
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
        collision_values = (
            self.collision_left_base_pos,
            self.collision_right_base_pos,
            self.collision_left_base_yaw,
            self.collision_right_base_yaw,
            self.collision_table_height,
        )
        if self.collision_guardrail:
            if not self.collision_table:
                raise ValueError(
                    "table collision checking must be on when predictive collision checking is on"
                )
            if any(value is None for value in collision_values):
                raise ValueError(
                    "all five collision measurements are required when predictive collision "
                    "checking is on"
                )
            if (
                len(self.collision_left_base_pos or ()) != 3
                or len(self.collision_right_base_pos or ()) != 3
            ):
                raise ValueError("each arm-base collision position must contain x y z")
        elif self.collision_table:
            raise ValueError(
                "table collision checking cannot be on when predictive collision checking is off"
            )
        elif any(value is not None for value in collision_values):
            raise ValueError(
                "remove the collision measurements when predictive collision checking is off"
            )
        if self.joint_low != I2RT_JOINT_LOW or self.joint_high != I2RT_JOINT_HIGH:
            raise ValueError("joint bounds must match the pinned I2RT YAM model")
        if self.step_limits != STRICT_STEP_LIMITS:
            raise ValueError("strict step limits must be 0.2 rad and one gripper stroke")
        devices = (self.top_camera, self.left_camera, self.right_camera)
        if any(not value for value in devices) or len(set(devices)) != 3:
            raise ValueError("three distinct camera role assignments are required")
        if any(not stable_camera_source(source) for source in devices):
            raise ValueError(
                "each camera must use a stable camera source: RealSense serial, "
                "/dev/v4l/by-id, or /dev/v4l/by-path"
            )
        channels = (self.left_channel, self.right_channel)
        if any(not value for value in channels) or len(set(channels)) != 2:
            raise ValueError("two distinct CAN role assignments are required")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def yam_kwargs(self) -> dict[str, Any]:
        """Return only arguments owned by the Dreamscale YAM fork."""
        kwargs: dict[str, Any] = {
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
            "collision_table": self.collision_table,
            "auto_start": self.auto_start,
            "unattended": self.unattended,
            "strict_policy_actions": self.strict_policy_actions,
        }
        if self.collision_guardrail:
            kwargs.update(
                {
                    "collision_left_base_pos": self.collision_left_base_pos,
                    "collision_right_base_pos": self.collision_right_base_pos,
                    "collision_left_base_yaw": self.collision_left_base_yaw,
                    "collision_right_base_yaw": self.collision_right_base_yaw,
                    "collision_table_height": self.collision_table_height,
                }
            )
        for slot, source in (
            ("top", self.top_camera),
            ("left", self.left_camera),
            ("right", self.right_camera),
        ):
            kind, value = camera_source(source)
            key = f"{slot}_depth_serial" if kind == "realsense" else f"{slot}_cam_device"
            kwargs[key] = value
        sources = (self.top_camera, self.left_camera, self.right_camera)
        if any(camera_source(source)[0] == "realsense" for source in sources):
            kwargs["realsense_capture"] = "process"
            kwargs["depth_fps"] = self.control_hz
        return kwargs

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
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return cls(**values)


def load_rig(path: Path | None = None, *, profile: str | None = None) -> RigConfig:
    if path is not None and profile is not None:
        raise ValueError("pass either a rig path or profile, not both")
    resolved = path or resolve_rig_path(profile)
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
    profile: str | None = None,
    replace: bool = False,
) -> Path:
    if path is not None and profile is not None:
        raise ValueError("pass either a rig path or profile, not both")
    resolved = path or rig_path(profile)
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
    serialized = {key: value for key, value in rig.as_dict().items() if value is not None}
    temporary.write_text(tomli_w.dumps({"rig": serialized}), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, resolved)
    return resolved
