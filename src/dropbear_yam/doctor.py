"""Motion-free, session-free diagnostics with stable machine-readable codes."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from dropbear.config import load_config
from dropbear.control import ControlPlaneClient

from dropbear_yam.config import (
    I2RT_JOINT_HIGH,
    I2RT_JOINT_LOW,
    RigConfig,
    stable_camera_source,
)

Status = Literal["pass", "fail", "warn"]
CAMERA_NAMES = ("top_cam", "left_cam", "right_cam")
MAX_IMAGE_AGE_S = 5.0
MAX_IMAGE_FUTURE_S = 1.0
CAMERA_SKEW_WARN_S = 0.05


@dataclass(frozen=True)
class Diagnostic:
    code: str
    status: Status
    summary: str
    remediation: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[Diagnostic, ...]
    forced_ok: bool | None = None
    schema_version: int = 1
    generated_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        if self.forced_ok is not None:
            return self.forced_ok
        return all(check.status != "fail" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CameraProbe:
    shapes: dict[str, tuple[int, ...]]
    image_times: dict[str, float]


@dataclass(frozen=True)
class CloudProbe:
    authenticated: bool
    entitled: bool
    target_available: bool
    sessions: tuple[str, ...] = ()
    parked_sessions: tuple[str, ...] = ()
    detail: str = ""


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _clock_synchronized() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"timedatectl unavailable: {exc}"
    value = result.stdout.strip().lower()
    return result.returncode == 0 and value == "yes", value or result.stderr.strip()


def _camera_probe(rig: RigConfig) -> CameraProbe:
    from inspect_robots_yam.config import YamConfig
    from inspect_robots_yam.embodiment import YAMEmbodiment

    config = YamConfig(**rig.yam_kwargs())
    # Construction chooses and owns only the configured camera readers. The
    # I2RT driver factory is stored but never called unless prepare/reset runs;
    # doctor invokes neither method.
    embodiment = YAMEmbodiment(config)
    try:
        captured: Any = embodiment._camera_reader(  # pyright: ignore[reportPrivateUsage]
            config
        )
        return CameraProbe(
            shapes={name: tuple(image.shape) for name, image in captured.items()},
            image_times=dict(captured.image_times),
        )
    finally:
        embodiment.close()


def _cadence_probe(rig: RigConfig) -> tuple[float, float, float]:
    """Resolve all three public cadence declarations without connecting remotely."""
    from inspect_robots_dropbear.policy import DropbearPolicy
    from inspect_robots_yam.config import YamConfig

    yam = YamConfig(**rig.yam_kwargs())
    policy = DropbearPolicy(control_hz=rig.control_hz, keep_warm_s=0)
    try:
        return float(rig.control_hz), float(yam.control_hz), float(policy.info.control_hz)
    finally:
        policy.close()


def _can_probe(channel: str) -> tuple[bool, str]:
    type_path = Path("/sys/class/net") / channel / "type"
    try:
        if type_path.read_text(encoding="utf-8").strip() != "280":
            return False, f"{channel} is not a SocketCAN interface"
    except OSError as exc:
        return False, f"{channel} is unavailable: {exc}"
    try:
        result = subprocess.run(
            ["ip", "-details", "link", "show", channel],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    text = f"{result.stdout}\n{result.stderr}".strip()
    return result.returncode == 0 and "state UP" in result.stdout, text


async def _cloud_probe_async() -> CloudProbe:
    config = load_config()
    if not config.api_key:
        return CloudProbe(False, False, False, detail="Dropbear API key is absent")
    client = ControlPlaneClient(config.control_plane_url, config.api_key)
    try:
        candidates = await client.probe_candidates("dreamzero-yam")
        session_rows = await client.list_sessions()
        parked_session_ids = tuple(
            str(row["session_id"])
            for row in session_rows
            if row.get("session_id")
            and row.get("model") == "dreamzero-yam"
            and row.get("status") == "parked"
        )
        session_ids = tuple(
            str(row["session_id"])
            for row in session_rows
            if row.get("session_id") and str(row["session_id"]) not in parked_session_ids
        )
        return CloudProbe(
            authenticated=True,
            entitled=bool(candidates),
            target_available=bool(candidates),
            sessions=session_ids,
            parked_sessions=parked_session_ids,
            detail=f"{len(candidates)} candidate target(s)",
        )
    except Exception as exc:
        text = str(exc)
        lowered = text.lower()
        invalid_auth = "api key" in lowered or "unauthorized" in lowered
        not_entitled = "entitl" in lowered or "model_not_entitled" in lowered
        return CloudProbe(
            authenticated=not invalid_auth,
            entitled=not (invalid_auth or not_entitled),
            target_available=False,
            detail=text,
        )
    finally:
        await client.close()


def _cloud_probe() -> CloudProbe:
    return asyncio.run(_cloud_probe_async())


def lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "composition.lock.toml"


def _installed_direct_url(package: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(package)
    raw = distribution.read_text("direct_url.json")
    return json.loads(raw) if raw else {}


def _provenance() -> list[Diagnostic]:
    payload = tomllib.loads(lock_path().read_text(encoding="utf-8"))
    components = payload["components"]
    failures: list[str] = []
    installed: dict[str, dict[str, str]] = {}
    for name, expected in components.items():
        package = str(expected["package"])
        try:
            version = importlib.metadata.version(package)
            direct = _installed_direct_url(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{package} is not installed")
            continue
        actual_commit = str(direct.get("vcs_info", {}).get("commit_id", ""))
        installed[name] = {"package": package, "version": version, "commit": actual_commit}
        expected_version = expected.get("version")
        expected_commit = expected.get("commit")
        if expected_version and version != expected_version:
            failures.append(f"{package} version {version} != {expected_version}")
        if expected_commit and actual_commit != expected_commit:
            failures.append(f"{package} commit {actual_commit or 'unknown'} != {expected_commit}")
    if failures:
        return [
            Diagnostic(
                "DBY-PROVENANCE",
                "fail",
                "The installed Python packages do not match this release: " + "; ".join(failures),
                "From this cloned repository, run `uv sync --locked --extra hardware`, then "
                "rerun doctor",
                installed,
            )
        ]
    return [
        Diagnostic(
            "DBY-PROVENANCE",
            "pass",
            "all packages match composition.lock.toml",
            details=installed,
        )
    ]


@dataclass
class DoctorDependencies:
    system_name: Callable[[], str] = platform.system
    command_exists: Callable[[str], bool] = _command_exists
    provenance: Callable[[], list[Diagnostic]] = _provenance
    clock_synchronized: Callable[[], tuple[bool, str]] = _clock_synchronized
    camera_probe: Callable[[RigConfig], CameraProbe] = _camera_probe
    cadence_probe: Callable[[RigConfig], tuple[float, float, float]] = _cadence_probe
    can_probe: Callable[[str], tuple[bool, str]] = _can_probe
    cloud_probe: Callable[[], CloudProbe] = _cloud_probe
    now: Callable[[], float] = time.time


def _pass(code: str, summary: str, details: dict[str, Any] | None = None) -> Diagnostic:
    return Diagnostic(code, "pass", summary, details=details or {})


def _fail(code: str, summary: str, remediation: str) -> Diagnostic:
    return Diagnostic(code, "fail", summary, remediation)


def _warn(
    code: str,
    summary: str,
    remediation: str,
    details: dict[str, Any] | None = None,
) -> Diagnostic:
    return Diagnostic(code, "warn", summary, remediation, details or {})


def _camera_checks(rig: RigConfig, deps: DoctorDependencies) -> list[Diagnostic]:
    checks: list[Diagnostic] = []
    devices = (rig.top_camera, rig.left_camera, rig.right_camera)
    try:
        stable = all(stable_camera_source(device) for device in devices)
    except ValueError:
        stable = False
    if len(set(devices)) == 3 and all(device for device in devices) and stable:
        checks.append(
            _pass(
                "DBY-CAMERA-ROLES",
                "three distinct stable camera roles are assigned",
                {
                    "top_cam": rig.top_camera,
                    "left_cam": rig.left_camera,
                    "right_cam": rig.right_camera,
                },
            )
        )
    else:
        checks.append(
            _fail(
                "DBY-CAMERA-ROLES",
                "The top, left and right camera assignments are not three stable devices",
                "Run ./dropbear-yam setup --reconfigure and assign a different detected camera "
                "to each role",
            )
        )
        return checks
    try:
        probe = deps.camera_probe(rig)
    except Exception as exc:
        checks.append(
            _fail(
                "DBY-CAMERA-FRAMES",
                f"A configured camera could not provide an image: {exc}",
                "Check camera USB power and permissions, close other camera programs, then "
                "rerun ./dropbear-yam doctor",
            )
        )
        return checks
    expected_shapes = {name: (rig.cam_height, rig.cam_width, 3) for name in CAMERA_NAMES}
    if probe.shapes == expected_shapes:
        checks.append(_pass("DBY-CAMERA-FRAMES", "all cameras produce RGB 640x360 frames"))
    else:
        checks.append(
            _fail(
                "DBY-CAMERA-FRAMES",
                f"The cameras did not all provide 640x360 color images: {probe.shapes}",
                "Run ./dropbear-yam setup --reconfigure to check the three roles, then "
                "rerun doctor",
            )
        )
    times = probe.image_times
    now = deps.now()
    reason = ""
    if set(times) != set(CAMERA_NAMES):
        reason = "At least one camera did not report when its image was captured"
    elif any(not math.isfinite(value) for value in times.values()):
        reason = "At least one camera reported an invalid capture time"
    elif any(now - value > MAX_IMAGE_AGE_S for value in times.values()):
        reason = "At least one camera image is more than 5 seconds old"
    elif any(value - now > MAX_IMAGE_FUTURE_S for value in times.values()):
        reason = "At least one camera capture time is incorrectly in the future"
    if reason:
        checks.append(
            _fail(
                "DBY-CAMERA-TIMESTAMPS",
                reason,
                "Synchronize the Ubuntu clock, reconnect the cameras, close other camera "
                "programs, then rerun ./dropbear-yam doctor",
            )
        )
    else:
        skew_ms = (max(times.values()) - min(times.values())) * 1_000
        summary = (
            f"Unix-epoch source timestamps are fresh; observed camera skew is "
            f"{skew_ms:.1f} ms"
        )
        details = {"image_times": times, "skew_ms": skew_ms}
        if skew_ms > CAMERA_SKEW_WARN_S * 1_000:
            checks.append(
                _warn(
                "DBY-CAMERA-TIMESTAMPS",
                    summary,
                    "No action is required to continue. If this is much larger than usual, "
                    "close other camera programs, reconnect the cameras, and rerun doctor",
                    details,
                )
            )
        else:
            checks.append(
                _pass(
                    "DBY-CAMERA-TIMESTAMPS",
                    summary,
                    details,
                )
            )
    return checks


def doctor(rig: RigConfig, *, deps: DoctorDependencies | None = None) -> DoctorReport:
    deps = deps or DoctorDependencies()
    checks: list[Diagnostic] = []
    if deps.system_name() == "Linux":
        checks.append(_pass("DBY-HOST-LINUX", "Linux host detected"))
    else:
        checks.append(
            _fail(
                "DBY-HOST-LINUX",
                "This is not the Linux computer connected to the YAM rig",
                "Run these commands on the Ubuntu computer connected to both arms and all cameras",
            )
        )

    required = ("git", "uv", "ip", "v4l2-ctl", "cmake", "pkg-config")
    missing = [name for name in required if not deps.command_exists(name)]
    checks.append(
        _fail(
            "DBY-HOST-BUILD",
            f"Required system tools are missing: {', '.join(missing)}",
            "Run ./setup.sh from this cloned repository and approve its one system-package step",
        )
        if missing
        else _pass("DBY-HOST-BUILD", "host and build prerequisites are present")
    )
    provenance = deps.provenance()
    checks.extend(provenance or [_pass("DBY-PROVENANCE", "provenance supplied by caller")])

    synchronized, clock_detail = deps.clock_synchronized()
    checks.append(
        _pass("DBY-CLOCK", clock_detail)
        if synchronized
        else _fail(
            "DBY-CLOCK",
            f"The computer clock is not synchronized: {clock_detail}",
            "Enable Ubuntu time synchronization with systemd-timesyncd or chrony, wait until it "
            "reports synchronized, then rerun doctor",
        )
    )
    checks.extend(_camera_checks(rig, deps))

    can_details: dict[str, str] = {}
    can_ok = True
    for channel in (rig.left_channel, rig.right_channel):
        ok, detail = deps.can_probe(channel)
        can_ok &= ok
        can_details[channel] = detail
    checks.append(
        _pass("DBY-CAN", "both SocketCAN interfaces are UP", can_details)
        if can_ok
        else _fail(
            "DBY-CAN",
            f"At least one arm CAN connection is not UP: {can_details}",
            "Connect both CAN adapters, bring the configured interfaces UP, confirm them with "
            "`ip -details link show type can`, then rerun doctor",
        )
    )

    bounds_ok = rig.joint_low == I2RT_JOINT_LOW and rig.joint_high == I2RT_JOINT_HIGH
    checks.append(
        _pass("DBY-JOINT-LIMITS", "joint bounds match pinned I2RT ArmType.YAM")
        if bounds_ok
        else _fail(
            "DBY-JOINT-LIMITS",
            "The configured joint limits do not match this YAM software release",
            "Run ./dropbear-yam setup --reconfigure from this checkout",
        )
    )
    if rig.collision_guardrail:
        checks.append(
            _pass(
                "DBY-GEOMETRY",
                "Predictive arm and table collision checking is configured",
            )
        )
    else:
        checks.append(
            _warn(
                "DBY-GEOMETRY",
                "Predictive collision checking is turned off by configuration",
                "To add it later, measure both arm-base positions and yaws plus table height, "
                "then run ./dropbear-yam setup --reconfigure",
            )
        )
    try:
        cadences = deps.cadence_probe(rig)
    except Exception as exc:
        cadences = (float(rig.control_hz), math.nan, math.nan)
        cadence_detail = str(exc)
    else:
        cadence_detail = f"rig/YAM/policy={cadences}"
    checks.append(
        _pass("DBY-CADENCE", "rig, YAM and DreamZero-YAM agree on exactly 30 Hz")
        if cadences == (30.0, 30.0, 30.0)
        else _fail(
            "DBY-CADENCE",
            f"The rig, YAM driver and model do not all use 30 Hz: {cadence_detail}",
            "Run `uv sync --locked --extra hardware`, then run "
            "./dropbear-yam setup --reconfigure from this checkout",
        )
    )

    try:
        cloud = deps.cloud_probe()
    except Exception as exc:
        cloud = CloudProbe(False, False, False, detail=str(exc))
    checks.append(
        _pass("DBY-AUTH", "Dropbear authentication succeeded")
        if cloud.authenticated
        else _fail(
            "DBY-AUTH",
            "Dropbear login could not be verified"
            + (f": {cloud.detail}" if cloud.detail else ""),
            "Run `dropbear login`, complete sign-in, then rerun doctor",
        )
    )
    checks.append(
        _pass("DBY-ENTITLEMENT", "dreamzero-yam entitlement is present")
        if cloud.entitled
        else _fail(
            "DBY-ENTITLEMENT",
            "This Dropbear account cannot use dreamzero-yam"
            + (f": {cloud.detail}" if cloud.detail else ""),
            "Ask Dreamscale to grant this Dropbear account access to dreamzero-yam, then "
            "rerun doctor",
        )
    )
    checks.append(
        _pass("DBY-TARGET", "at least one dreamzero-yam target is available")
        if cloud.target_available
        else _fail(
            "DBY-TARGET",
            "No dreamzero-yam compute target is available"
            + (f": {cloud.detail}" if cloud.detail else ""),
            "Wait a few minutes and rerun doctor; if it still fails, send Dreamscale the output "
            "of ./dropbear-yam doctor --json",
        )
    )
    if cloud.sessions:
        checks.append(
            _fail(
                "DBY-SESSION-CLEAR",
                f"A Dropbear session is already running: {', '.join(cloud.sessions)}",
                "Do not start another run. Stop or resolve the listed session first, then rerun "
                "doctor; doctor will not stop it automatically",
            )
        )
    elif len(cloud.parked_sessions) > 1:
        checks.append(
            _fail(
                "DBY-SESSION-CLEAR",
                "More than one warm DreamZero-YAM reservation exists: "
                f"{', '.join(cloud.parked_sessions)}",
                "Stop the extra exact sessions with `dropbear sessions stop <session-id>`, then "
                "rerun doctor",
            )
        )
    elif cloud.parked_sessions:
        checks.append(
            _pass(
                "DBY-SESSION-CLEAR",
                "one owned warm DreamZero-YAM reservation is parked and ready for reuse",
            )
        )
    else:
        checks.append(_pass("DBY-SESSION-CLEAR", "no existing Dropbear session"))
    return DoctorReport(tuple(checks))


_SECRET_FRAGMENTS = ("api_key", "token", "secret", "password", "authorization")


def _redact(value: Any, key: str = "") -> Any:
    if any(fragment in key.lower() for fragment in _SECRET_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(child): _redact(item, str(child)) for child, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def create_support_bundle(
    destination: Path,
    report: DoctorReport,
    rig: RigConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Create a credential-redacted bundle Jay or his agents can attach."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dropbear-yam-support-") as temporary:
        root = Path(temporary)
        (root / "doctor.json").write_text(report.to_json() + "\n", encoding="utf-8")
        (root / "rig.redacted.json").write_text(
            json.dumps(_redact(rig.as_dict()), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "environment.redacted.json").write_text(
            json.dumps(_redact(extra or {}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(destination, "w:gz") as archive:
            for path in sorted(root.iterdir()):
                archive.add(path, arcname=path.name)
    return destination
