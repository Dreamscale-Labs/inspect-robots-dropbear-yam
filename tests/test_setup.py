from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w
from dropbear import errors as dropbear_errors

from dropbear_yam.config import load_rig
from dropbear_yam.errors import UserFacingError
from dropbear_yam.setup_command import SetupDependencies, _login, discover_cameras, setup


def test_default_login_uses_the_locked_sdk_and_suppresses_generic_next_steps(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        "dropbear_yam.setup_command.run_login",
        lambda *, print_next_steps: calls.append(print_next_steps),
    )

    _login()

    assert calls == [False]


def test_setup_prompts_only_for_unavoidable_assignments_and_geometry(isolated_paths: Path) -> None:
    answers = iter(
        [
            "1",
            "2",
            "3",  # top, left, right cameras
            "1",
            "2",  # left and right CAN
            "y",  # opt in to predictive collision geometry
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
        discover_cameras=lambda: [
            "/dev/v4l/by-id/cam-a",
            "/dev/v4l/by-id/cam-b",
            "/dev/v4l/by-id/cam-c",
        ],
        discover_can=lambda: ["can0", "can1"],
        authenticated=lambda: False,
        login=lambda: login_calls.append(True),
        input=lambda prompt: (prompts.append(prompt), next(answers))[1],
        output=lambda _line: None,
    )

    path = setup(deps=deps)
    rig = load_rig(path)

    from dropbear_yam import config

    assert path == config.rig_path("default")
    assert (rig.top_camera, rig.left_camera, rig.right_camera) == (
        "/dev/v4l/by-id/cam-a",
        "/dev/v4l/by-id/cam-b",
        "/dev/v4l/by-id/cam-c",
    )
    assert (rig.left_channel, rig.right_channel) == ("can0", "can1")
    assert rig.collision_guardrail is True
    assert login_calls == [True]
    assert len(prompts) == 11


def test_setup_reports_the_saved_rig_before_login_failure(isolated_paths: Path) -> None:
    answers = iter(["1", "2", "3", "1", "2", "n"])
    output: list[str] = []
    failure = dropbear_errors.catalog("cli_login_start_failed", detail="HTTP 503")
    deps = SetupDependencies(
        discover_cameras=lambda: [
            "/dev/v4l/by-id/cam-a",
            "/dev/v4l/by-id/cam-b",
            "/dev/v4l/by-id/cam-c",
        ],
        discover_can=lambda: ["can0", "can1"],
        authenticated=lambda: False,
        login=lambda: (_ for _ in ()).throw(failure),
        input=lambda _prompt: next(answers),
        output=output.append,
    )

    with pytest.raises(dropbear_errors.DropbearError):
        setup(deps=deps)

    assert load_rig().top_camera == "/dev/v4l/by-id/cam-a"
    assert any(line.startswith("Confirmed rig written to ") for line in output)

    recovered = setup(
        deps=SetupDependencies(
            discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("rediscovered")),
            discover_can=lambda: (_ for _ in ()).throw(AssertionError("rediscovered")),
            authenticated=lambda: True,
            login=lambda: (_ for _ in ()).throw(AssertionError("login repeated")),
            input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompt repeated")),
            output=lambda _line: None,
        )
    )
    assert recovered.exists()


def test_setup_explains_and_allows_skipping_collision_geometry(isolated_paths: Path) -> None:
    answers = iter(["1", "2", "3", "1", "2", "n"])
    prompts: list[str] = []
    output: list[str] = []
    deps = SetupDependencies(
        discover_cameras=lambda: [
            "/dev/v4l/by-id/cam-a",
            "/dev/v4l/by-id/cam-b",
            "/dev/v4l/by-id/cam-c",
        ],
        discover_can=lambda: ["can0", "can1"],
        authenticated=lambda: True,
        login=lambda: None,
        input=lambda prompt: (prompts.append(prompt), next(answers))[1],
        output=output.append,
    )

    rig = load_rig(setup(deps=deps))

    assert rig.collision_guardrail is False
    assert rig.collision_table is False
    assert rig.collision_left_base_pos is None
    assert rig.collision_right_base_pos is None
    assert rig.collision_table_height is None
    assert len(prompts) == 6
    explanation = "\n".join(output)
    assert "optional" in explanation.lower()
    assert "left and right arm-base x y z" in explanation.lower()
    assert "predictive collision" in explanation.lower()


def test_setup_camera_failure_is_plain_and_actionable(isolated_paths: Path) -> None:
    deps = SetupDependencies(
        discover_cameras=lambda: [],
        discover_can=lambda: ["can0", "can1"],
        authenticated=lambda: True,
        login=lambda: None,
        input=lambda _prompt: "",
        output=lambda _line: None,
    )

    with pytest.raises(UserFacingError) as caught:
        setup(deps=deps)

    assert "found 0 usable color cameras" in str(caught.value)
    assert "Connect and power" in caught.value.next_step


def test_setup_is_idempotent_and_does_not_prompt_when_rig_exists(rig, isolated_paths: Path) -> None:
    from dropbear_yam.config import save_rig

    expected = save_rig(rig, profile="jay-rig-1")
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


def test_setup_preserves_advanced_step_limits_without_prompting(rig, isolated_paths: Path) -> None:
    from dropbear_yam.config import RigConfig, save_rig

    configured = RigConfig(**{**rig.as_dict(), "step_limits": (0.5,) * 14})
    expected = save_rig(configured, profile="default")
    deps = SetupDependencies(
        discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        discover_can=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        authenticated=lambda: True,
        login=lambda: (_ for _ in ()).throw(AssertionError("login called")),
        input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        output=lambda _line: None,
    )

    assert setup(deps=deps) == expected
    assert load_rig(expected).step_limits == (0.5,) * 14


def test_v0117_rig_parses_and_setup_does_not_rewrite_or_prompt(isolated_paths: Path) -> None:
    from dropbear_yam import config

    fixture = Path(__file__).parent / "fixtures" / "v0.1.17-rig.toml"
    expected = fixture.read_bytes()
    path = config.rig_path("default")
    path.parent.mkdir(parents=True)
    path.write_bytes(expected)
    deps = SetupDependencies(
        discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        discover_can=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        authenticated=lambda: True,
        login=lambda: (_ for _ in ()).throw(AssertionError("login called")),
        input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        output=lambda _line: None,
    )

    assert load_rig(path).step_limits == config.STRICT_STEP_LIMITS
    assert setup(deps=deps) == path
    assert path.read_bytes() == expected


def test_setup_migrates_generated_xml_bounds_without_reasking_for_rig_assignments(
    rig, isolated_paths: Path
) -> None:
    from dropbear_yam import config

    path = config.rig_path("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        **rig.as_dict(),
        "schema_version": 1,
        "joint_low": list(
            (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944, 0.0) * 2
        ),
        "joint_high": list(
            (3.05433, 3.65, 3.66519, 1.5708, 1.5708, 2.0944, 1.0) * 2
        ),
    }
    path.write_text(tomli_w.dumps({"rig": legacy}), encoding="utf-8")
    output: list[str] = []
    deps = SetupDependencies(
        discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        discover_can=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        authenticated=lambda: True,
        login=lambda: (_ for _ in ()).throw(AssertionError("login called")),
        input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        output=output.append,
    )

    assert setup(deps=deps) == path

    migrated = load_rig(path)
    assert migrated.schema_version == 2
    assert migrated.top_camera == rig.top_camera
    assert migrated.left_camera == rig.left_camera
    assert migrated.right_camera == rig.right_camera
    assert migrated.left_channel == rig.left_channel
    assert migrated.right_channel == rig.right_channel
    assert migrated.joint_low[1:3] == (-0.15, -0.15)
    assert "updated the generated i2rt joint bounds" in "\n".join(output).lower()


@pytest.mark.parametrize("customization", ["gripper", "rig-key", "top-level"])
def test_setup_refuses_to_silently_migrate_a_customized_v1_rig(
    rig, isolated_paths: Path, customization: str
) -> None:
    from dropbear_yam import config

    path = config.rig_path("default")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        **rig.as_dict(),
        "schema_version": 1,
        "joint_low": list(
            (-2.61799, 0.0, 0.0, -1.5708, -1.5708, -2.0944, 0.0) * 2
        ),
        "joint_high": list(
            (3.05433, 3.65, 3.66519, 1.5708, 1.5708, 2.0944, 1.0) * 2
        ),
    }
    payload: dict[str, object] = {"rig": legacy}
    if customization == "gripper":
        legacy["gripper_type"] = "LINEAR_3507"
    elif customization == "rig-key":
        legacy["custom_setting"] = True
    else:
        payload["custom"] = {"setting": True}
    original = tomli_w.dumps(payload)
    path.write_text(original, encoding="utf-8")
    deps = SetupDependencies(
        discover_cameras=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        discover_can=lambda: (_ for _ in ()).throw(AssertionError("discovery called")),
        authenticated=lambda: True,
        login=lambda: (_ for _ in ()).throw(AssertionError("login called")),
        input=lambda _prompt: (_ for _ in ()).throw(AssertionError("prompted")),
        output=lambda _line: None,
    )

    with pytest.raises(ValueError, match="unsupported rig format"):
        setup(deps=deps)

    assert path.read_text(encoding="utf-8") == original


def test_discovery_prefers_mixed_documented_realsense_backends_without_index0_assumption() -> None:
    cameras = discover_cameras(
        v4l_devices=[
            {
                "source": "/dev/v4l/by-id/d435-video-index0",
                "physical_id": "usb:1-1",
                "model": "Intel RealSense D435",
            },
            {
                "source": "/dev/v4l/by-path/pci-usb-1-2-video-index4",
                "physical_id": "usb:1-2",
                "model": "Intel RealSense D405",
            },
            {
                "source": "/dev/v4l/by-path/pci-usb-1-3-video-index4",
                "physical_id": "usb:1-3",
                "model": "Intel RealSense D405",
            },
        ],
        realsense_devices=[
            {
                "source": "realsense:D435-SERIAL",
                "physical_id": "usb:1-1",
                "model": "Intel RealSense D435",
            },
            {
                "source": "realsense:D405-LEFT",
                "physical_id": "usb:1-2",
                "model": "Intel RealSense D405",
            },
            {
                "source": "realsense:D405-RIGHT",
                "physical_id": "usb:1-3",
                "model": "Intel RealSense D405",
            },
        ],
    )

    assert cameras == [
        "/dev/v4l/by-id/d435-video-index0",
        "realsense:D405-LEFT",
        "realsense:D405-RIGHT",
    ]


def test_discovery_accepts_a_color_capable_by_path_node_when_by_id_is_ambiguous() -> None:
    cameras = discover_cameras(
        v4l_devices=[
            {
                "source": "/dev/v4l/by-path/pci-usb-1-2-video-index4",
                "physical_id": "usb:1-2",
                "model": "unknown",
            }
        ],
        realsense_devices=[],
    )

    assert cameras == ["/dev/v4l/by-path/pci-usb-1-2-video-index4"]


def test_discovery_joins_realsense_device_and_asic_serial_namespaces() -> None:
    cameras = discover_cameras(
        v4l_devices=[
            {
                "source": "/dev/v4l/by-id/usb-Intel_D405_ASIC-123-video-index4",
                "physical_id": "node:/dev/video8",
                "model": "8086:0b5b",
                "serial": "ASIC-123",
            }
        ],
        realsense_devices=[
            {
                "source": "realsense:DEVICE-456",
                "physical_id": "serial:DEVICE-456",
                "model": "Intel RealSense D405",
                "serial": "DEVICE-456",
                "asic_serial": "ASIC-123",
            }
        ],
    )

    assert cameras == ["realsense:DEVICE-456"]


def test_discovery_never_collapses_two_physical_cameras_with_a_duplicated_serial() -> None:
    cameras = discover_cameras(
        v4l_devices=[
            {
                "source": "/dev/v4l/by-path/pci-usb-1-2-video-index4",
                "physical_id": "usb:1-2",
                "model": "8086:0b5b",
                "serial": "DUPLICATED",
            },
            {
                "source": "/dev/v4l/by-path/pci-usb-1-3-video-index4",
                "physical_id": "usb:1-3",
                "model": "8086:0b5b",
                "serial": "DUPLICATED",
            },
        ],
        realsense_devices=[],
    )

    assert cameras == [
        "/dev/v4l/by-path/pci-usb-1-2-video-index4",
        "/dev/v4l/by-path/pci-usb-1-3-video-index4",
    ]
