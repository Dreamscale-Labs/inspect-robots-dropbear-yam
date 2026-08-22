from __future__ import annotations

from pathlib import Path

from dropbear_yam.config import load_rig
from dropbear_yam.setup_command import SetupDependencies, discover_cameras, setup


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
    assert login_calls == [True]
    assert len(prompts) == 10


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
