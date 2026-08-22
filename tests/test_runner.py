from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from inspect_robots.approver import ChainApprover
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from dropbear_yam.doctor import DoctorReport
from dropbear_yam.runner import (
    CleanupResult,
    NonRewritingApprover,
    RunDependencies,
    _cleanup_async,
    configuration_digest,
    run,
    shadow_path,
)


class IdentityApprover:
    def review(self, action, _store):
        return action


class FakeEmbodiment:
    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []
        self.validations: list[tuple[np.ndarray, np.ndarray]] = []
        self.closed = False
        self.info = SimpleNamespace(action_space=object())

    def prepare_observation(self, instruction: str) -> Observation:
        now = 1_800_000_000.0
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        return Observation(
            images={name: image for name in ("top_cam", "left_cam", "right_cam")},
            state={"joint_pos": np.zeros(14)},
            instruction=instruction,
            image_times={name: now for name in ("top_cam", "left_cam", "right_cam")},
        )

    def validate_policy_action(self, action: Action, *, reference) -> np.ndarray:
        self.validations.append((np.asarray(action.data), np.asarray(reference)))
        return np.asarray(action.data)

    def contribute_guardrails(self, _space):
        return SimpleNamespace(approvers=(("yam-collision", IdentityApprover()),), warnings=())

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    def __init__(self) -> None:
        self.session_id = "session-owned"
        self.reset_calls: list[Scene] = []
        self.act_calls = 0
        self.closed = False

    def reset(self, scene: Scene) -> None:
        self.reset_calls.append(scene)

    def act(self, observation: Observation) -> ActionChunk:
        self.act_calls += 1
        assert observation.extra["env_step"] == 0
        return ActionChunk(
            actions=[Action(np.zeros(14), {"dropbear_action_source": "model"})],
            control_hz=30,
        )

    def close(self) -> None:
        self.closed = True


def _ok_report() -> DoctorReport:
    return DoctorReport(checks=())


def test_configuration_digest_invalidates_every_material_input(rig, tmp_path: Path) -> None:
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    original = configuration_digest(rig, lock)

    changed_rig = type(rig)(**{**rig.as_dict(), "left_channel": "can9"})
    assert configuration_digest(changed_rig, lock) != original
    lock.write_text("commit='two'\n")
    assert configuration_digest(rig, lock) != original


def test_shadow_inference_is_validated_never_executed_and_session_is_reused(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    eval_calls: list[dict[str, object]] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")

    def fake_eval(task, active_policy, active_embodiment, **kwargs):
        assert active_policy is policy
        assert active_embodiment is embodiment
        assert isinstance(kwargs["approver"], ChainApprover)
        eval_calls.append(kwargs)
        return [SimpleNamespace(status="success")]

    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig: policy,
        evaluate=fake_eval,
        cleanup=lambda session_id: CleanupResult(session_id, disappeared=True, forced=False),
    )

    result = run("move the blue cup", rig, deps=deps, lock_path=lock, max_steps=2)

    assert result == 0
    assert policy.act_calls == 1
    assert len(embodiment.validations) == 1
    assert embodiment.commands == []
    assert len(eval_calls) == 1
    assert embodiment.closed is True
    assert policy.closed is True
    assert shadow_path(configuration_digest(rig, lock)).exists()


def test_run_closes_both_objects_and_forced_cleanup_exits_nonzero(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")

    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig: policy,
        evaluate=lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        cleanup=lambda session_id: CleanupResult(session_id, disappeared=True, forced=True),
    )

    result = run("stop safely", rig, deps=deps, lock_path=lock, max_steps=1)

    assert result != 0
    assert embodiment.closed is True
    assert policy.closed is True


def test_run_blocks_before_hardware_when_doctor_fails(rig, tmp_path: Path) -> None:
    deps = RunDependencies(
        doctor=lambda _rig: DoctorReport(checks=(), forced_ok=False),
        confirm=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        embodiment=lambda _rig: (_ for _ in ()).throw(AssertionError("hardware opened")),
        policy=lambda _rig: (_ for _ in ()).throw(AssertionError("session opened")),
        evaluate=lambda *_args, **_kwargs: None,
        cleanup=lambda _session_id: CleanupResult(None, True, False),
    )

    assert run("blocked", rig, deps=deps, lock_path=tmp_path / "missing") != 0


def test_non_rewriting_wrapper_aborts_action_substitution() -> None:
    class Rewriter:
        def review(self, action, _store):
            return Action(np.asarray(action.data).copy())

    wrapped = NonRewritingApprover(Rewriter())

    with pytest.raises(Exception, match="rewrite"):
        wrapped.review(Action(np.zeros(14)), {})


def test_cleanup_deletes_only_the_exact_owned_session(monkeypatch) -> None:
    class Client:
        def __init__(self, *_args):
            self.owned_present = True
            self.deleted: list[str] = []

        async def list_sessions(self):
            sessions = [{"session_id": "unrelated"}]
            if self.owned_present:
                sessions.append({"session_id": "session-owned"})
            return sessions

        async def delete_session(self, session_id: str):
            self.deleted.append(session_id)
            if session_id == "session-owned":
                self.owned_present = False

        async def close(self):
            return None

    client = Client()
    monkeypatch.setattr(
        "dropbear_yam.runner.load_config",
        lambda: SimpleNamespace(api_key="hidden", control_plane_url="https://example.invalid"),
    )
    monkeypatch.setattr("dropbear_yam.runner.ControlPlaneClient", lambda *_args: client)

    result = asyncio.run(_cleanup_async("session-owned", grace_s=0.001, poll_s=0))

    assert result.disappeared is True
    assert result.forced is True
    assert client.deleted == ["session-owned"]
