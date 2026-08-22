from __future__ import annotations

from pathlib import Path

from dropbear_yam.config import load_rig
from dropbear_yam.setup_command import SetupDependencies, setup


def test_setup_prompts_only_for_unavoidable_assignments_and_geometry(isolated_paths: Path) -> None:
    answers = iter(
        [
            "1",
            "2",
            "3",  # top, left, right cameras
            "1",
            "2",  # left and right CAN
            "-0.25 0 0",
            "0.25 0 0",
            "0",
            "3.14159",
            "0",
        ]
    )
    prompts: list[str] = []
    login_calls: list[bool] = []
    deps = SetupDependencies(
        discover_cameras=lambda: ["/stable/cam-a", "/stable/cam-b", "/stable/cam-c"],
        discover_can=lambda: ["can0", "can1"],
        authenticated=lambda: False,
        login=lambda: login_calls.append(True),
        input=lambda prompt: (prompts.append(prompt), next(answers))[1],
        output=lambda _line: None,
    )

    path = setup(deps=deps)
    rig = load_rig(path)

    assert (rig.top_camera, rig.left_camera, rig.right_camera) == (
        "/stable/cam-a",
        "/stable/cam-b",
        "/stable/cam-c",
    )
    assert (rig.left_channel, rig.right_channel) == ("can0", "can1")
    assert login_calls == [True]
    assert len(prompts) == 10


def test_setup_is_idempotent_and_does_not_prompt_when_rig_exists(rig, isolated_paths: Path) -> None:
    from dropbear_yam.config import save_rig

    expected = save_rig(rig)
    deps = SetupDependencies(
        discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        discover_can=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        authenticated=lambda: True,
        login=lambda: (_ for _ in ()).throw(AssertionError("login called")),
        input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        output=lambda _line: None,
    )

    assert setup(deps=deps) == expected
    assert load_rig(expected) == rig
