from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import YAMEmbodiment
from inspect_robots_yam.operator import OperatorIO

from dropbear_yam.config import load_rig
from dropbear_yam.doctor import CameraProbe, CloudProbe, DoctorDependencies, doctor
from dropbear_yam.runner import CleanupResult, RunDependencies, run
from dropbear_yam.setup_command import SetupDependencies, setup


class FakeDriver:
    def __init__(self) -> None:
        self.state = np.zeros(14)
        self.commands: list[np.ndarray] = []
        self.closed = False

    def get_joint_pos(self):
        return self.state.copy()

    def command_joint_pos(self, target):
        self.state = np.asarray(target, dtype=np.float64).copy()
        self.commands.append(self.state.copy())

    def close(self):
        self.closed = True


class FakeDropbearPolicy:
    session_id = "fake-owned-session"

    def __init__(self) -> None:
        self.closed = False

    def reset(self, _scene: Scene) -> None:
        return None

    def act(self, observation) -> ActionChunk:
        return ActionChunk(
            [
                Action(
                    np.asarray(observation.state["joint_pos"]).copy(),
                    {"dropbear_action_source": "model"},
                )
            ],
            control_hz=30,
        )

    def close(self) -> None:
        self.closed = True


def test_fake_setup_doctor_shadow_gated_run_and_cleanup(
    isolated_paths: Path, tmp_path: Path
) -> None:
    answers = iter(
        ["1", "2", "3", "1", "2", "0 0.3 0", "0 -0.3 0", "0", "0", "0"]
    )
    setup(
        deps=SetupDependencies(
            discover_cameras=lambda: [
                "/dev/v4l/by-id/top-video-index0",
                "/dev/v4l/by-id/left-video-index0",
                "/dev/v4l/by-id/right-video-index0",
            ],
            discover_can=lambda: ["can0", "can1"],
            authenticated=lambda: True,
            login=lambda: None,
            input=lambda _prompt: next(answers),
            output=lambda _line: None,
        )
    )
    rig = load_rig()
    now = time.time()
    doctor_deps = DoctorDependencies(
        system_name=lambda: "Linux",
        command_exists=lambda _name: True,
        provenance=lambda: [],
        clock_synchronized=lambda: (True, "synchronized"),
        camera_probe=lambda _rig: CameraProbe(
            {name: (360, 640, 3) for name in ("top_cam", "left_cam", "right_cam")},
            {name: now for name in ("top_cam", "left_cam", "right_cam")},
        ),
        can_probe=lambda channel: (True, f"{channel} UP"),
        cloud_probe=lambda: CloudProbe(True, True, True),
        now=lambda: now,
    )
    assert doctor(rig, deps=doctor_deps).ok is True

    driver = FakeDriver()
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    embodiment = YAMEmbodiment(
        YamConfig(**{**rig.yam_kwargs(), "rest_secs": 0.01}),
        driver_factory=lambda _config: driver,
        camera_reader=lambda _config: {
            name: image for name in ("top_cam", "left_cam", "right_cam")
        },
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=lambda _line: None),
        poll_end=lambda: True,
        sleep_fn=lambda _delay: None,
    )
    policy = FakeDropbearPolicy()
    eval_started_without_shadow_motion: list[bool] = []
    output: list[str] = []

    def fake_eval(task, active_policy, active_embodiment, **kwargs):
        eval_started_without_shadow_motion.append(driver.commands == [])
        scene = task.scenes[0]
        observation = active_embodiment.reset(scene)
        active_policy.reset(scene)
        action = active_policy.act(replace(observation, extra={"env_step": 0})).actions[0]
        reviewed = kwargs["approver"].review(action, {})
        active_embodiment.step(reviewed)
        return [SimpleNamespace(status="success")]

    lock = tmp_path / "composition.lock.toml"
    lock.write_text("commit='simulated'\n", encoding="utf-8")
    result = run(
        "move the object",
        rig,
        lock_path=lock,
        max_steps=1,
        deps=RunDependencies(
            doctor=lambda _rig: doctor(rig, deps=doctor_deps),
            confirm=lambda _prompt: True,
            embodiment=lambda _rig: embodiment,
            policy=lambda _rig: policy,
            evaluate=fake_eval,
            cleanup=lambda session_id: CleanupResult(session_id, True, False),
            output=output.append,
        ),
    )

    assert result == 0, output
    assert eval_started_without_shadow_motion == [True]
    assert driver.commands
    assert driver.closed is True
    assert policy.closed is True
