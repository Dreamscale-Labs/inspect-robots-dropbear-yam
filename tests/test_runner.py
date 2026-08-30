from __future__ import annotations

import asyncio
import contextlib
import json
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
    _action_approver,
    _cleanup_async,
    _confirm,
    _loading_status,
    _shadow_passed,
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
        self.prepare_calls = 0
        self.closed = False
        self.info = SimpleNamespace(action_space=object())
        self.reference = np.zeros(14)

    def prepare_observation(self, instruction: str) -> Observation:
        self.prepare_calls += 1
        now = 1_800_000_000.0 + self.prepare_calls
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

    def policy_action_reference(self) -> np.ndarray:
        return self.reference.copy()

    def contribute_guardrails(self, _space):
        return SimpleNamespace(approvers=(("yam-collision", IdentityApprover()),), warnings=())

    def close(self) -> None:
        self.closed = True


class FakePolicy:
    def __init__(self) -> None:
        self.session_id = "session-owned"
        self.prepare_calls = 0
        self.reset_calls: list[Scene] = []
        self.predict_calls = 0
        self.predicted_observations: list[Observation] = []
        self.act_calls = 0
        self.closed = False
        self.predicted_action = np.zeros(14)

    def prepare(self) -> None:
        self.prepare_calls += 1

    def reset(self, scene: Scene) -> None:
        self.reset_calls.append(scene)

    def act(self, observation: Observation) -> ActionChunk:
        self.act_calls += 1
        assert observation.extra["env_step"] == 0
        return ActionChunk(
            actions=[Action(np.zeros(14), {"dropbear_action_source": "hold"})],
            control_hz=30,
        )

    def predict_model_action(self, observation: Observation, *, instruction: str) -> Action:
        self.predict_calls += 1
        self.predicted_observations.append(observation)
        assert instruction
        assert observation.extra["env_step"] == 0
        return Action(self.predicted_action.copy(), {"dropbear_action_source": "model"})

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
    policy.predicted_action[0] = 0.25
    eval_calls: list[dict[str, object]] = []
    loading: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")

    def fake_eval(task, active_policy, active_embodiment, **kwargs):
        assert active_policy is policy
        assert active_embodiment is embodiment
        assert isinstance(kwargs["approver"], ChainApprover)
        active_policy.reset(task.scenes[0])
        eval_calls.append(kwargs)
        return [SimpleNamespace(status="success")]

    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig, **_kwargs: policy,
        evaluate=fake_eval,
        cleanup=lambda session_id, **_kwargs: CleanupResult(
            session_id, disappeared=True, forced=False
        ),
        loading=lambda message: _record_context(loading, message),
    )

    result = run(
        "move the blue cup",
        rig,
        deps=deps,
        lock_path=lock,
        max_steps=2,
        warm_minutes=0,
    )

    assert result == 0
    assert policy.prepare_calls == 1
    assert policy.predict_calls == 1
    assert embodiment.prepare_calls == 2
    assert policy.predicted_observations[0].image_times["top_cam"] == 1_800_000_002.0
    assert policy.act_calls == 0
    assert [scene.id for scene in policy.reset_calls] == ["jay-attended"]
    assert len(embodiment.validations) == 1
    assert embodiment.validations[0][0][0] == pytest.approx(0.2)
    assert embodiment.commands == []
    assert len(eval_calls) == 1
    assert embodiment.closed is True
    assert policy.closed is True
    receipt = json.loads(
        shadow_path(configuration_digest(rig, lock)).read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == 2
    assert receipt["projected_dimensions"] == ["left_j0"]
    assert receipt["requested_action_sha256"] != receipt["applied_action_sha256"]
    assert receipt["executed"] is False
    assert loading == ["Starting Dropbear compute (a cold start can take a few minutes)"]


def test_v1_shadow_receipt_is_invalidated(isolated_paths: Path) -> None:
    path = shadow_path("digest")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 1, "configuration_digest": "digest"}),
        encoding="utf-8",
    )

    assert _shadow_passed("digest") is False


def test_live_projection_caps_and_continues_with_local_audit(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    without_collision = type(rig)(
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
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    logs = tmp_path / "logs"

    def fake_eval(_task, _policy, _embodiment, **kwargs):
        requested = np.zeros(14)
        requested[0] = 0.25
        first = kwargs["approver"].review(Action(requested), {})
        embodiment.commands.append(np.asarray(first.data).copy())
        embodiment.reference = np.asarray(first.data).copy()
        second = kwargs["approver"].review(Action(requested), {})
        embodiment.commands.append(np.asarray(second.data).copy())
        return [SimpleNamespace(status="success")]

    result = run(
        "move the blue cup",
        without_collision,
        deps=RunDependencies(
            doctor=lambda _rig: _ok_report(),
            confirm=lambda _prompt: True,
            embodiment=lambda _rig: embodiment,
            policy=lambda _rig, **_kwargs: policy,
            evaluate=fake_eval,
            cleanup=lambda session_id, **_kwargs: CleanupResult(session_id, True, False),
            output=output.append,
            loading=lambda message: _record_context([], message),
        ),
        lock_path=lock,
        max_steps=2,
        warm_minutes=0,
        log_dir=logs,
    )

    assert result == 0
    assert embodiment.commands[0][0] == pytest.approx(0.2)
    assert embodiment.commands[1][0] == pytest.approx(0.25)
    sidecars = list(logs.glob("yam-projections-*.jsonl"))
    assert len(sidecars) == 1
    assert len(sidecars[0].read_text(encoding="utf-8").splitlines()) == 1
    assert sum("capped" in line.lower() for line in output) == 1
    assert any("Projected 1 policy action" in line for line in output)


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
        policy=lambda _rig, **_kwargs: policy,
        evaluate=lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        cleanup=lambda session_id, **_kwargs: CleanupResult(
            session_id, disappeared=True, forced=True
        ),
    )

    result = run(
        "stop safely",
        rig,
        deps=deps,
        lock_path=lock,
        max_steps=1,
        warm_minutes=0,
    )

    assert result != 0
    assert embodiment.closed is True
    assert policy.closed is True


def test_run_blocks_before_hardware_when_doctor_fails(rig, tmp_path: Path) -> None:
    deps = RunDependencies(
        doctor=lambda _rig: DoctorReport(checks=(), forced_ok=False),
        confirm=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        embodiment=lambda _rig: (_ for _ in ()).throw(AssertionError("hardware opened")),
        policy=lambda _rig, **_kwargs: (_ for _ in ()).throw(AssertionError("session opened")),
        evaluate=lambda *_args, **_kwargs: None,
        cleanup=lambda _session_id, **_kwargs: CleanupResult(None, True, False),
    )

    assert run("blocked", rig, deps=deps, lock_path=tmp_path / "missing") != 0


def test_non_rewriting_wrapper_aborts_action_substitution() -> None:
    class Rewriter:
        def review(self, action, _store):
            return Action(np.asarray(action.data).copy())

    wrapped = NonRewritingApprover(Rewriter())

    with pytest.raises(Exception, match="rewrite"):
        wrapped.review(Action(np.zeros(14)), {})


@contextlib.contextmanager
def _record_context(records: list[str], message: str):
    records.append(message)
    yield


def test_connect_confirmation_is_yes_no_with_yes_as_default(monkeypatch, capsys) -> None:
    answers = iter(["maybe", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert _confirm("Connect hardware?") is True
    assert "Please answer y or n." in capsys.readouterr().out

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert _confirm("Connect hardware?") is False


def test_loading_status_immediately_shows_symbol_and_elapsed_seconds(capsys) -> None:
    with _loading_status("Starting Dropbear compute"):
        pass

    output = capsys.readouterr().out
    assert "⠋ Starting Dropbear compute — 0s waiting" in output
    assert "Starting Dropbear compute ready after 0s" in output


def test_collision_approver_is_optional_but_action_validation_remains_strict(rig) -> None:
    embodiment = FakeEmbodiment()
    embodiment.contribute_guardrails = lambda _space: (_ for _ in ()).throw(
        AssertionError("collision contribution should not be constructed")
    )

    without_collision = type(rig)(
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
    approver = _action_approver(embodiment, without_collision, audit=None)
    requested = np.zeros(14)
    requested[0] = 0.25
    action = Action(requested)

    applied = approver.review(action, {})
    assert applied.data[0] == pytest.approx(0.2)


def test_collision_reviews_projected_action_and_remains_abort_only(rig) -> None:
    seen: list[np.ndarray] = []

    class CollisionApprover:
        def review(self, action: Action, _store):
            seen.append(np.asarray(action.data).copy())
            raise RuntimeError("predicted collision")

    embodiment = FakeEmbodiment()
    embodiment.contribute_guardrails = lambda _space: SimpleNamespace(
        approvers=(("yam-collision", CollisionApprover()),),
        warnings=(),
    )
    requested = np.zeros(14)
    requested[0] = 0.25

    with pytest.raises(RuntimeError, match="predicted collision"):
        _action_approver(embodiment, rig, audit=None).review(Action(requested), {})

    assert seen[0][0] == pytest.approx(0.2)


def test_shadow_message_does_not_call_inference_paid(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig, **_kwargs: policy,
        evaluate=lambda *_args, **_kwargs: [SimpleNamespace(status="success")],
        cleanup=lambda session_id, **_kwargs: CleanupResult(session_id, True, False),
        output=output.append,
        loading=lambda message: _record_context([], message),
    )

    assert run(
        "move the blue cup",
        rig,
        deps=deps,
        lock_path=lock,
        max_steps=1,
        warm_minutes=0,
    ) == 0
    shadow_line = next(line for line in output if "shadow inference" in line.lower())
    assert "paid" not in shadow_line.lower()


def test_run_failure_is_plain_and_actionable(rig, isolated_paths: Path, tmp_path: Path) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig, **_kwargs: policy,
        evaluate=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("camera unplugged")),
        cleanup=lambda session_id, **_kwargs: CleanupResult(session_id, True, False),
        output=output.append,
        loading=lambda message: _record_context([], message),
    )

    assert run(
        "move the blue cup",
        rig,
        deps=deps,
        lock_path=lock,
        max_steps=1,
        warm_minutes=0,
    ) == 1
    text = "\n".join(output)
    assert "Error: The run stopped before completion: camera unplugged" in text
    assert "Next:" in text
    assert "RuntimeError" not in text


def test_run_surfaces_the_specific_inspect_failure(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    """Catch the regression where Inspect's exact error was replaced by a generic failure."""
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig, **_kwargs: policy,
        evaluate=lambda *_args, **_kwargs: [
            SimpleNamespace(
                status="error",
                error=(
                    "SafetyAbort: strict policy action jump exceeds configured limit: left_j1"
                ),
            )
        ],
        cleanup=lambda session_id, **_kwargs: CleanupResult(session_id, True, False),
        output=output.append,
        loading=lambda message: _record_context([], message),
    )

    assert (
        run(
            "Pack container",
            rig,
            deps=deps,
            lock_path=lock,
            max_steps=1,
            warm_minutes=0,
        )
        == 1
    )
    text = "\n".join(output)
    assert (
        "Error: The run stopped before completion: strict policy action jump exceeds "
        "configured limit: left_j1."
    ) in text
    assert "Inspect Robots reported" not in text
    assert "Keep the robot stopped" in text


def test_run_prefers_scene_error_over_inspect_failure_threshold(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")
    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=lambda _rig, **_kwargs: policy,
        evaluate=lambda *_args, **_kwargs: [
            SimpleNamespace(
                status="error",
                error="fail_on_error threshold exceeded (1 errors)",
                samples=(
                    SimpleNamespace(
                        error="PolicyError: camera frames were not accepted by the policy"
                    ),
                ),
            )
        ],
        cleanup=lambda session_id, **_kwargs: CleanupResult(session_id, True, False),
        output=output.append,
        loading=lambda message: _record_context([], message),
    )

    assert (
        run(
            "Pack container",
            rig,
            deps=deps,
            lock_path=lock,
            max_steps=1,
            warm_minutes=0,
        )
        == 1
    )
    text = "\n".join(output)
    assert "camera frames were not accepted by the policy" in text
    assert "threshold exceeded" not in text


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


def test_warm_cleanup_waits_for_exact_session_to_park_without_deleting(monkeypatch) -> None:
    class Client:
        def __init__(self, *_args):
            self.polls = 0
            self.deleted: list[str] = []

        async def list_sessions(self):
            self.polls += 1
            status = "parking" if self.polls == 1 else "parked"
            return [
                {"session_id": "unrelated", "status": "active"},
                {"session_id": "session-owned", "status": status},
            ]

        async def delete_session(self, session_id: str):
            self.deleted.append(session_id)

        async def close(self):
            return None

    client = Client()
    monkeypatch.setattr(
        "dropbear_yam.runner.load_config",
        lambda: SimpleNamespace(api_key="hidden", control_plane_url="https://example.invalid"),
    )
    monkeypatch.setattr("dropbear_yam.runner.ControlPlaneClient", lambda *_args: client)

    result = asyncio.run(_cleanup_async("session-owned", keep_warm_s=300, grace_s=1, poll_s=0))

    assert result.parked is True
    assert result.disappeared is False
    assert result.forced is False
    assert client.deleted == []


def test_run_passes_requested_warm_hold_to_policy_and_cleanup(
    rig, isolated_paths: Path, tmp_path: Path
) -> None:
    embodiment = FakeEmbodiment()
    policy = FakePolicy()
    requested: dict[str, int] = {}
    output: list[str] = []
    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='one'\n")

    def policy_factory(_rig, *, keep_warm_s: int):
        requested["policy"] = keep_warm_s
        return policy

    def cleanup(session_id: str | None, *, keep_warm_s: int):
        requested["cleanup"] = keep_warm_s
        return CleanupResult(session_id, disappeared=False, forced=False, parked=True)

    deps = RunDependencies(
        doctor=lambda _rig: _ok_report(),
        confirm=lambda _prompt: True,
        embodiment=lambda _rig: embodiment,
        policy=policy_factory,
        evaluate=lambda *_args, **_kwargs: [SimpleNamespace(status="success")],
        cleanup=cleanup,
        output=output.append,
        loading=lambda message: _record_context([], message),
    )

    assert run(
        "move the blue cup",
        rig,
        deps=deps,
        lock_path=lock,
        max_steps=1,
        warm_minutes=5,
    ) == 0
    assert requested == {"policy": 300, "cleanup": 300}
    assert any("5 minutes" in line and "warm" in line.lower() for line in output)
