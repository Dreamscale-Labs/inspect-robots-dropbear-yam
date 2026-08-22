"""Gated shadow and attended physical-run orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dropbear.config import load_config
from dropbear.control import ControlPlaneClient
from inspect_robots.approver import Approver, ChainApprover
from inspect_robots.scene import Scene
from inspect_robots.task import Task
from inspect_robots.types import Action, Observation

from dropbear_yam.config import RigConfig, state_home
from dropbear_yam.doctor import DoctorReport
from dropbear_yam.doctor import doctor as run_doctor
from dropbear_yam.errors import emit_error


def default_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "composition.lock.toml"


def configuration_digest(rig: RigConfig, lock_path: Path | None = None) -> str:
    """Hash every material deployment, device, cadence and safety input."""
    lock = lock_path or default_lock_path()
    payload = {
        "rig": rig.as_dict(),
        "composition_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def shadow_path(digest: str) -> Path:
    return state_home() / "shadow" / f"{digest}.json"


def _shadow_passed(digest: str) -> bool:
    path = shadow_path(digest)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        payload.get("schema_version") == 1 and payload.get("configuration_digest") == digest
    )


def _record_shadow(digest: str, session_id: str | None, action: Action) -> Path:
    path = shadow_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    action_hash = hashlib.sha256(
        json.dumps([float(value) for value in action.data], allow_nan=False).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "configuration_digest": digest,
        "validated_at": time.time(),
        "session_id": session_id,
        "action_sha256": action_hash,
        "action_source": action.meta.get("dropbear_action_source"),
        "executed": False,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


class NonRewritingApprover:
    """Fail if an abort-only guard ever substitutes a policy action."""

    def __init__(self, inner: Approver):
        self._inner = inner

    def review(self, action: Action, store: dict[str, Any]) -> Action:
        reviewed = self._inner.review(action, store)
        if reviewed is not action:
            from inspect_robots.errors import SafetyAbort

            raise SafetyAbort("strict guard attempted to rewrite a policy action")
        return action


@dataclass(frozen=True)
class CleanupResult:
    session_id: str | None
    disappeared: bool
    forced: bool
    detail: str = ""


async def _session_present(client: ControlPlaneClient, session_id: str) -> bool:
    sessions = await client.list_sessions()
    return any(str(row.get("session_id")) == session_id for row in sessions)


async def _cleanup_async(
    session_id: str | None,
    *,
    grace_s: float = 10.0,
    poll_s: float = 0.5,
) -> CleanupResult:
    if session_id is None:
        return CleanupResult(None, True, False, "no session was created")
    config = load_config()
    if not config.api_key:
        return CleanupResult(session_id, False, False, "credentials unavailable for verification")
    client = ControlPlaneClient(config.control_plane_url, config.api_key)
    try:
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if not await _session_present(client, session_id):
                return CleanupResult(session_id, True, False, "session disappeared after close")
            await asyncio.sleep(poll_s)
        await client.delete_session(session_id)
        forced_deadline = time.monotonic() + grace_s
        while time.monotonic() < forced_deadline:
            if not await _session_present(client, session_id):
                return CleanupResult(
                    session_id,
                    True,
                    True,
                    "normal close failed; exact session was explicitly stopped",
                )
            await asyncio.sleep(poll_s)
        return CleanupResult(
            session_id,
            False,
            True,
            "exact session remained after explicit stop",
        )
    except Exception as exc:
        return CleanupResult(session_id, False, False, f"cleanup verification failed: {exc}")
    finally:
        await client.close()


def cleanup_session(session_id: str | None) -> CleanupResult:
    return asyncio.run(_cleanup_async(session_id))


def _confirm(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt}\nContinue? [Y/n] ").strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


@contextlib.contextmanager
def _loading_status(message: str) -> Iterator[None]:
    """Show continuous startup progress so a slow compute allocation never looks frozen."""
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    started = time.monotonic()
    stopped = threading.Event()
    interactive = sys.stdout.isatty()

    sys.stdout.write(f"\r{frames[0]} {message} — 0s waiting")
    sys.stdout.flush()

    def render() -> None:
        frame = 1
        while not stopped.is_set():
            elapsed = int(time.monotonic() - started)
            sys.stdout.write(f"\r{frames[frame % len(frames)]} {message} — {elapsed}s waiting")
            sys.stdout.flush()
            frame += 1
            stopped.wait(0.1)

    thread = threading.Thread(target=render, name="dropbear-yam-loading", daemon=True)
    if interactive:
        thread.start()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        stopped.set()
        if interactive:
            thread.join(timeout=1.0)
        elapsed = int(time.monotonic() - started)
        outcome = "stopped" if failed else "ready"
        symbol = "!" if failed else "✓"
        clear_line = "\033[2K" if interactive else ""
        sys.stdout.write(f"\r{clear_line}{symbol} {message} {outcome} after {elapsed}s\n")
        sys.stdout.flush()


def _embodiment(rig: RigConfig) -> Any:
    from inspect_robots_yam.config import YamConfig
    from inspect_robots_yam.embodiment import YAMEmbodiment

    return YAMEmbodiment(YamConfig(**rig.yam_kwargs()))


def _policy(rig: RigConfig) -> Any:
    from inspect_robots_dropbear.policy import DropbearPolicy

    return DropbearPolicy(
        model=rig.model_target,
        control_hz=rig.control_hz,
        keep_warm_s=rig.keep_warm,
    )


def _evaluate(*args: Any, **kwargs: Any) -> Any:
    from inspect_robots.eval import eval as inspect_eval

    return inspect_eval(*args, **kwargs)


@dataclass
class RunDependencies:
    doctor: Callable[[RigConfig], DoctorReport] = run_doctor
    confirm: Callable[[str], bool] = _confirm
    embodiment: Callable[[RigConfig], Any] = _embodiment
    policy: Callable[[RigConfig], Any] = _policy
    evaluate: Callable[..., Any] = _evaluate
    cleanup: Callable[[str | None], CleanupResult] = cleanup_session
    output: Callable[[str], None] = print
    loading: Callable[[str], contextlib.AbstractContextManager[None]] = _loading_status


@contextlib.contextmanager
def _signal_stops() -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _shadow_observation(observation: Observation) -> Observation:
    return replace(observation, extra={**dict(observation.extra), "env_step": 0})


def _strict_approver(
    embodiment: Any,
    *,
    collision_guardrail_enabled: bool,
) -> ChainApprover:
    if not collision_guardrail_enabled:
        return ChainApprover()
    contribution = embodiment.contribute_guardrails(embodiment.info.action_space)
    if contribution.warnings:
        raise RuntimeError("; ".join(contribution.warnings))
    if not contribution.approvers:
        raise RuntimeError(
            "Predictive collision checking was enabled, but its collision checker did not start"
        )
    return ChainApprover(
        *(NonRewritingApprover(approver) for _name, approver in contribution.approvers)
    )


def _run_shadow(
    instruction: str,
    observation: Observation,
    embodiment: Any,
    policy: Any,
    digest: str,
) -> None:
    scene = Scene(id="shadow", instruction=instruction)
    policy.reset(scene)
    chunk = policy.act(_shadow_observation(observation))
    action = chunk.actions[0]
    if action.meta.get("dropbear_action_source") != "model":
        raise RuntimeError("shadow inference did not return a model action")
    reference = observation.state.get("joint_pos")
    if reference is None:
        raise RuntimeError("prepared observation has no joint_pos reference")
    embodiment.validate_policy_action(action, reference=reference)
    _record_shadow(digest, getattr(policy, "session_id", None), action)


def run(
    instruction: str,
    rig: RigConfig,
    *,
    deps: RunDependencies | None = None,
    lock_path: Path | None = None,
    max_steps: int = 3600,
    log_dir: Path | None = None,
) -> int:
    """Run doctor -> physical gate -> shadow -> attended eval -> exact cleanup."""
    deps = deps or RunDependencies()
    report = deps.doctor(rig)
    if not report.ok:
        deps.output("Error: Doctor found problems that must be fixed before hardware can connect.")
        for check in report.checks:
            if check.status == "fail":
                deps.output(f"  [{check.code}] {check.summary}")
                if check.remediation:
                    deps.output(f"    Next: {check.remediation}")
        deps.output("Next: Fix the failed checks above, then rerun ./dropbear-yam doctor.")
        return 2
    if not instruction.strip():
        emit_error(
            deps.output,
            "The task instruction is empty",
            "Repeat the command with a trained task, for example "
            './dropbear-yam run "Pack container"',
        )
        return 2
    if max_steps < 1:
        emit_error(
            deps.output,
            "--max-steps must be at least 1",
            "Repeat the command with --max-steps 3600 or another positive number",
        )
        return 2
    if not deps.confirm(
        "E-stop must be in hand and working. Connecting I2RT will enable control "
        "traffic and calibrate both LINEAR_4310 grippers; keep clear of them."
    ):
        deps.output("Cancelled: no YAM hardware connection was opened.")
        return 2

    embodiment: Any | None = None
    policy: Any | None = None
    session_id: str | None = None
    exit_code = 0
    try:
        embodiment = deps.embodiment(rig)
        policy = deps.policy(rig)
        prepared = embodiment.prepare_observation(instruction)
        approver = _strict_approver(
            embodiment,
            collision_guardrail_enabled=rig.collision_guardrail,
        )
        digest = configuration_digest(rig, lock_path)
        if not _shadow_passed(digest):
            deps.output("Running one non-commanding shadow inference for this configuration.")
            with deps.loading("Starting Dropbear compute"):
                _run_shadow(instruction, prepared, embodiment, policy, digest)
            deps.output(f"Shadow validation passed: {shadow_path(digest)}")
        else:
            deps.output("Shadow validation already passed for this exact configuration.")
            with deps.loading("Starting Dropbear compute"):
                policy.reset(Scene(id="startup", instruction=instruction))

        task = Task(
            name="jay-dreamzero-yam",
            scenes=[Scene(id="jay-attended", instruction=instruction)],
            scorer="operator",
            max_steps=max_steps,
        )
        with _signal_stops():
            logs = deps.evaluate(
                task,
                policy,
                embodiment,
                log_dir=str(log_dir or (state_home() / "logs")),
                approver=approver,
                grader="operator",
                store_frames=True,
                store_actions=True,
                fail_on_error=True,
            )
        if not logs or any(getattr(item, "status", "error") != "success" for item in logs):
            emit_error(
                deps.output,
                "Inspect Robots reported that the task did not finish successfully",
                "Keep the robot stopped, review the run log under "
                "~/.local/state/dropbear-yam/logs, then rerun doctor before another attempt",
            )
            exit_code = 1
    except KeyboardInterrupt:
        deps.output("Operator stop received. Closing the YAM hardware and Dropbear session.")
        exit_code = 130
    except BaseException as exc:
        lowered = str(exc).lower()
        if "collision" in lowered or "safety" in lowered or "joint" in lowered:
            next_step = (
                "Keep the robot stopped, check the scene and safety configuration, and only start "
                "a new run when the cause is understood"
            )
        elif "camera" in lowered or "image" in lowered:
            next_step = (
                "Check camera power and USB connections, close other camera programs, then rerun "
                "./dropbear-yam doctor"
            )
        else:
            next_step = (
                "Run ./dropbear-yam doctor; if it passes and this repeats, run "
                "./dropbear-yam doctor --support-bundle ~/dropbear-yam-support.tar.gz and send "
                "that file to Dreamscale"
            )
        emit_error(deps.output, f"The run stopped before completion: {exc}", next_step)
        exit_code = 1
    finally:
        if policy is not None:
            session_id = getattr(policy, "session_id", None)
        for label, resource in (("embodiment", embodiment), ("policy", policy)):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                name = "YAM hardware" if label == "embodiment" else "Dropbear connection"
                emit_error(
                    deps.output,
                    f"The {name} did not close cleanly: {exc}",
                    "Keep the e-stop ready and do not begin another run until doctor reports no "
                    "existing session",
                )
                exit_code = exit_code or 1
        try:
            cleanup = deps.cleanup(session_id)
        except BaseException as exc:
            emit_error(
                deps.output,
                f"Could not verify that Dropbear session {session_id or 'unknown'} ended: {exc}",
                "Do not start another run; rerun doctor and contact Dreamscale if it lists "
                "a session",
            )
            exit_code = exit_code or 1
        else:
            if not cleanup.disappeared:
                emit_error(
                    deps.output,
                    f"Dropbear session {session_id or 'unknown'} is still present: "
                    f"{cleanup.detail}",
                    "Do not start another run; rerun doctor and contact Dreamscale to stop this "
                    "exact session",
                )
                exit_code = exit_code or 1
            elif cleanup.forced:
                emit_error(
                    deps.output,
                    f"Dropbear session {session_id or 'unknown'} needed an explicit stop: "
                    f"{cleanup.detail}",
                    "Rerun doctor before another run and send a support bundle to Dreamscale if "
                    "this happens again",
                )
                exit_code = exit_code or 1
            else:
                deps.output(f"Dropbear cleanup verified for {session_id or 'no session'}")
    return exit_code
