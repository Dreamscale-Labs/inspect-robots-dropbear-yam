"""Idempotent interactive setup for facts software cannot infer safely."""

from __future__ import annotations

import glob
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dropbear.config import load_config

from dropbear_yam.config import RigConfig, load_rig, rig_path, save_rig


def discover_cameras() -> list[str]:
    """Return stable V4L2 colour-node candidates, never /dev/videoN aliases."""
    candidates = glob.glob("/dev/v4l/by-id/*-video-index0")
    return sorted(str(Path(candidate)) for candidate in candidates)


def discover_can_interfaces() -> list[str]:
    """Return SocketCAN interfaces reported by Linux sysfs."""
    result: list[str] = []
    for entry in sorted(Path("/sys/class/net").glob("*")):
        try:
            if (entry / "type").read_text(encoding="utf-8").strip() == "280":
                result.append(entry.name)
        except OSError:
            continue
    return result


def _authenticated() -> bool:
    return bool(load_config().api_key)


def _login() -> None:
    subprocess.run(["dropbear", "login"], check=True)


@dataclass
class SetupDependencies:
    discover_cameras: Callable[[], list[str]] = discover_cameras
    discover_can: Callable[[], list[str]] = discover_can_interfaces
    authenticated: Callable[[], bool] = _authenticated
    login: Callable[[], None] = _login
    input: Callable[[str], str] = input
    output: Callable[[str], None] = print


def _select(
    label: str,
    candidates: list[str],
    used: set[str],
    *,
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> str:
    available = [candidate for candidate in candidates if candidate not in used]
    if not available:
        raise RuntimeError(f"no unassigned candidates remain for {label}")
    output(f"Assign {label}:")
    for index, candidate in enumerate(candidates, 1):
        suffix = " (already assigned)" if candidate in used else ""
        output(f"  {index}. {candidate}{suffix}")
    while True:
        answer = input_fn(f"{label} [1-{len(candidates)}]: ").strip()
        try:
            selected = candidates[int(answer) - 1]
        except (ValueError, IndexError):
            output("Enter one listed number.")
            continue
        if selected in used:
            output("That device is already assigned; choose another.")
            continue
        used.add(selected)
        return selected


def _float(prompt: str, input_fn: Callable[[str], str], output: Callable[[str], None]) -> float:
    while True:
        try:
            return float(input_fn(prompt).strip())
        except ValueError:
            output("Enter one number in metres or radians as labelled.")


def _xyz(
    prompt: str,
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> tuple[float, float, float]:
    while True:
        parts = input_fn(prompt).replace(",", " ").split()
        try:
            values = tuple(float(part) for part in parts)
        except ValueError:
            values = ()
        if len(values) == 3:
            return values[0], values[1], values[2]
        output("Enter exactly three numbers: x y z (metres).")


def setup(*, reconfigure: bool = False, deps: SetupDependencies | None = None) -> Path:
    deps = deps or SetupDependencies()
    path = rig_path()
    if path.exists() and not reconfigure:
        load_rig(path)
        if not deps.authenticated():
            deps.output("Dropbear credentials are absent; opening login.")
            deps.login()
        deps.output(f"Rig already confirmed: {path}")
        return path

    cameras = deps.discover_cameras()
    if len(cameras) < 3:
        raise RuntimeError("need at least three stable /dev/v4l/by-id/*-video-index0 camera paths")
    channels = deps.discover_can()
    if len(channels) < 2:
        raise RuntimeError("need two SocketCAN interfaces; bring up both CAN adapters first")

    used_cameras: set[str] = set()
    top = _select("top camera", cameras, used_cameras, input_fn=deps.input, output=deps.output)
    left_camera = _select(
        "left camera", cameras, used_cameras, input_fn=deps.input, output=deps.output
    )
    right_camera = _select(
        "right camera", cameras, used_cameras, input_fn=deps.input, output=deps.output
    )
    used_channels: set[str] = set()
    left_channel = _select(
        "left arm CAN", channels, used_channels, input_fn=deps.input, output=deps.output
    )
    right_channel = _select(
        "right arm CAN", channels, used_channels, input_fn=deps.input, output=deps.output
    )

    deps.output("Enter measured collision geometry in the shared rig coordinate frame.")
    left_pos = _xyz("Left arm base x y z (m): ", deps.input, deps.output)
    right_pos = _xyz("Right arm base x y z (m): ", deps.input, deps.output)
    left_yaw = _float("Left arm base yaw (rad): ", deps.input, deps.output)
    right_yaw = _float("Right arm base yaw (rad): ", deps.input, deps.output)
    table_height = _float("Table top height z (m): ", deps.input, deps.output)

    rig = RigConfig(
        top_camera=top,
        left_camera=left_camera,
        right_camera=right_camera,
        left_channel=left_channel,
        right_channel=right_channel,
        collision_left_base_pos=left_pos,
        collision_right_base_pos=right_pos,
        collision_left_base_yaw=left_yaw,
        collision_right_base_yaw=right_yaw,
        collision_table_height=table_height,
    )
    saved = save_rig(rig, replace=reconfigure)
    if not deps.authenticated():
        deps.output("Dropbear credentials are absent; opening login.")
        deps.login()
    deps.output(f"Confirmed rig written to {saved}")
    return saved
