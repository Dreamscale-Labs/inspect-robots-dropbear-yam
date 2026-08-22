"""Idempotent interactive setup for facts software cannot infer safely."""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dropbear.config import load_config

from dropbear_yam.config import RigConfig, load_rig, rig_path, rig_profiles, save_rig

_USB_PORT = re.compile(r"(?:^|/)(\d+-\d+(?:\.\d+)*)(?=[:/]|$)")


def _physical_id(value: str | None, fallback: str) -> str:
    if value:
        matches = _USB_PORT.findall(value)
        if matches:
            return f"usb:{matches[-1]}"
    return fallback


def _discover_v4l_devices() -> list[dict[str, str]]:
    """Use Inspect Robots' color-capability probe and stable-name trust ladder."""
    from inspect_robots._setup import (  # pyright: ignore[reportPrivateUsage]
        V4L_BY_ID,
        V4L_BY_PATH,
        _ambiguous_identities,
        _camera_inventory,
        _preferred_name,
    )

    inventory = _camera_inventory(V4L_BY_ID, V4L_BY_PATH, Path("/sys/class/video4linux"))
    ambiguous = _ambiguous_identities(inventory)
    records: list[dict[str, str]] = []
    for record in inventory:
        source = _preferred_name([record], ambiguous, prefer_by_id=True)
        if source.startswith("/dev/video"):
            # A raw kernel index is not stable enough for a physical rig config.
            continue
        serial_key = f"serial:{record.serial}" if record.serial else ""
        records.append(
            {
                "source": source,
                "physical_id": _physical_id(
                    record.camera,
                    serial_key or f"node:{record.node}",
                ),
                "model": record.model or "V4L2 camera",
                "serial": record.serial or "",
            }
        )
    return records


def _rs_info(device: Any, rs: Any, name: str) -> str:
    info = getattr(rs.camera_info, name, None)
    if info is None:
        return ""
    try:
        if hasattr(device, "supports") and not device.supports(info):
            return ""
        return str(device.get_info(info)).strip()
    except Exception:
        return ""


def _discover_realsense_devices() -> list[dict[str, str]]:
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError:
        return []
    records: list[dict[str, str]] = []
    for device in rs.context().query_devices():
        serial = _rs_info(device, rs, "serial_number")
        asic_serial = _rs_info(device, rs, "asic_serial_number")
        selected_serial = serial or asic_serial
        if not selected_serial:
            continue
        port = _rs_info(device, rs, "physical_port")
        model = _rs_info(device, rs, "name") or "Intel RealSense"
        records.append(
            {
                "source": f"realsense:{selected_serial}",
                "physical_id": _physical_id(port, f"serial:{selected_serial}"),
                "model": model,
                "serial": selected_serial,
                "asic_serial": asic_serial,
            }
        )
    return records


def _source_key(record: Mapping[str, str]) -> str:
    return record.get("source", "")


def _prefer_camera_sources(
    v4l_devices: Sequence[Mapping[str, str]],
    realsense_devices: Sequence[Mapping[str, str]],
) -> list[str]:
    """Select one stable backend per physical camera.

    Robocurve's documented primary layout uses a D435 through V4L2 and D405
    wrist cameras through isolated RealSense processes. Physical USB identity
    joins the two Linux representations before that backend preference is
    applied, so one camera cannot be offered twice when the kernel exposes it
    through both APIs.
    """
    records = [*v4l_devices, *realsense_devices]
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    serial_records: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for record in records:
        physical = record.get("physical_id", "")
        if not physical:
            continue
        find(physical)
        is_realsense = _source_key(record).startswith("realsense:")
        for key in ("serial", "asic_serial"):
            serial = record.get(key, "")
            if not serial:
                continue
            serial_records[serial].append((physical, is_realsense))
    for aliases in serial_records.values():
        v4l_physical = {physical for physical, is_rs in aliases if not is_rs}
        rs_physical = {physical for physical, is_rs in aliases if is_rs}
        # Join only a unique cross-backend match. A duplicated serial within
        # one backend is ambiguous and must never collapse physical cameras.
        if len(v4l_physical) == 1 and len(rs_physical) == 1:
            union(next(iter(v4l_physical)), next(iter(rs_physical)))

    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        source = _source_key(record)
        physical = record.get("physical_id", "")
        if source and physical:
            grouped[find(physical)].append(record)

    selected: list[tuple[str, str]] = []
    for physical, records in grouped.items():
        v4l = [record for record in records if not _source_key(record).startswith("realsense:")]
        realsense = [
            record for record in records if _source_key(record).startswith("realsense:")
        ]
        model = " ".join(record.get("model", "") for record in records).lower()
        if "d405" in model and realsense:
            chosen = realsense[0]
        elif "d435" in model and v4l:
            chosen = v4l[0]
        elif realsense:
            chosen = realsense[0]
        elif v4l:
            chosen = v4l[0]
        else:  # pragma: no cover - records are filtered before grouping
            continue
        selected.append((physical, _source_key(chosen)))
    return [source for _physical, source in sorted(selected)]


def discover_cameras(
    *,
    v4l_devices: Sequence[Mapping[str, str]] | None = None,
    realsense_devices: Sequence[Mapping[str, str]] | None = None,
) -> list[str]:
    """Return one stable, color-capable source per physical camera."""
    v4l = list(v4l_devices) if v4l_devices is not None else _discover_v4l_devices()
    realsense = (
        list(realsense_devices)
        if realsense_devices is not None
        else _discover_realsense_devices()
    )
    return _prefer_camera_sources(v4l, realsense)


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


def _setup_path(
    rig_name: str | None,
    *,
    input_fn: Callable[[str], str],
    output: Callable[[str], None],
) -> Path:
    if rig_name is not None:
        return rig_path(rig_name)
    configured = rig_profiles()
    if len(configured) == 1:
        return next(iter(configured.values()))
    if configured:
        output(f"Configured rig profiles: {', '.join(sorted(configured))}")
    while True:
        answer = input_fn("Rig profile name (for example jay-rig-1): ").strip()
        if answer in configured:
            return configured[answer]
        try:
            return rig_path(answer)
        except ValueError as exc:
            output(str(exc))


def setup(
    *,
    rig_name: str | None = None,
    reconfigure: bool = False,
    deps: SetupDependencies | None = None,
) -> Path:
    deps = deps or SetupDependencies()
    path = _setup_path(rig_name, input_fn=deps.input, output=deps.output)
    if path.exists() and not reconfigure:
        load_rig(path)
        if not deps.authenticated():
            deps.output("Dropbear credentials are absent; opening login.")
            deps.login()
        deps.output(f"Rig already confirmed: {path}")
        return path

    cameras = deps.discover_cameras()
    if len(cameras) < 3:
        raise RuntimeError(
            "need three distinct color cameras with stable RealSense serial, "
            "/dev/v4l/by-id, or /dev/v4l/by-path identities; detected "
            f"{len(cameras)}"
        )
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
    saved = save_rig(rig, path=path, replace=reconfigure)
    if not deps.authenticated():
        deps.output("Dropbear credentials are absent; opening login.")
        deps.login()
    deps.output(f"Confirmed rig written to {saved}")
    return saved
