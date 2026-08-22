from __future__ import annotations

import json
import time
from pathlib import Path

from dropbear_yam.doctor import (
    CameraProbe,
    CloudProbe,
    DoctorDependencies,
    _camera_probe,
    create_support_bundle,
    doctor,
)


def _deps(rig, *, sessions=(), skew=0.01) -> DoctorDependencies:
    now = time.time()
    return DoctorDependencies(
        system_name=lambda: "Linux",
        command_exists=lambda _name: True,
        provenance=lambda: [],
        clock_synchronized=lambda: (True, "NTP synchronized"),
        camera_probe=lambda _rig: CameraProbe(
            shapes={name: (360, 640, 3) for name in ("top_cam", "left_cam", "right_cam")},
            image_times={
                "top_cam": now,
                "left_cam": now - skew / 2,
                "right_cam": now - skew,
            },
        ),
        can_probe=lambda channel: (True, f"{channel} is UP"),
        cloud_probe=lambda: CloudProbe(
            authenticated=True,
            entitled=True,
            target_available=True,
            sessions=tuple(sessions),
        ),
        now=lambda: now,
    )


def test_doctor_json_has_stable_schema_and_never_receives_motion_or_session_factory(rig) -> None:
    deps = _deps(rig)
    assert not hasattr(deps, "driver_factory")
    assert not hasattr(deps, "paid_session_factory")

    report = doctor(rig, deps=deps)
    payload = report.as_dict()

    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    required = {"code", "status", "summary", "remediation"}
    assert all(required <= check.keys() for check in payload["checks"])
    assert json.loads(report.to_json())["ok"] is True


def test_doctor_blocks_cross_camera_skew_and_existing_session(rig) -> None:
    report = doctor(rig, deps=_deps(rig, sessions=("session-other",), skew=0.2))
    by_code = {check.code: check for check in report.checks}

    assert report.ok is False
    assert by_code["DBY-CAMERA-TIMESTAMPS"].status == "fail"
    assert by_code["DBY-SESSION-CLEAR"].status == "fail"


def test_support_bundle_redacts_nested_secrets(rig, tmp_path: Path) -> None:
    destination = tmp_path / "support.tar.gz"
    report = doctor(rig, deps=_deps(rig))

    create_support_bundle(
        destination,
        report,
        rig,
        extra={"api_key": "db-secret", "nested": {"session_token": "token-secret"}},
    )

    import tarfile

    with tarfile.open(destination, "r:gz") as archive:
        text = "\n".join(
            archive.extractfile(member).read().decode()  # type: ignore[union-attr]
            for member in archive.getmembers()
            if member.isfile()
        )
    assert "db-secret" not in text
    assert "token-secret" not in text
    assert "[REDACTED]" in text


def test_doctor_rejects_stale_future_missing_and_wrong_shape_camera_data(rig) -> None:
    now = time.time()
    names = ("top_cam", "left_cam", "right_cam")
    correct_shapes = {name: (360, 640, 3) for name in names}
    cases = [
        CameraProbe({}, {}),
        CameraProbe(correct_shapes, {name: now - 6 for name in names}),
        CameraProbe(correct_shapes, {name: now + 2 for name in names}),
        CameraProbe({name: (480, 640, 3) for name in names}, {name: now for name in names}),
    ]
    for probe in cases:
        deps = _deps(rig)
        deps.camera_probe = lambda _rig, result=probe: result
        assert doctor(rig, deps=deps).ok is False


def test_doctor_accepts_distinct_stable_mixed_camera_sources(rig) -> None:
    from dropbear_yam.config import RigConfig

    mixed = RigConfig(
        **{
            **rig.as_dict(),
            "top_camera": "/dev/v4l/by-id/d435-video-index0",
            "left_camera": "realsense:LEFT-D405",
            "right_camera": "/dev/v4l/by-path/pci-usb-right-video-index4",
        }
    )

    report = doctor(mixed, deps=_deps(mixed))
    by_code = {check.code: check for check in report.checks}

    assert by_code["DBY-CAMERA-ROLES"].status == "pass"
    assert report.ok is True


def test_doctor_warns_but_does_not_block_when_collision_geometry_was_skipped(rig) -> None:
    from dropbear_yam.config import RigConfig

    without_geometry = RigConfig(
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

    report = doctor(without_geometry, deps=_deps(without_geometry))
    geometry = {check.code: check for check in report.checks}["DBY-GEOMETRY"]

    assert report.ok is True
    assert geometry.status == "warn"
    assert "turned off" in geometry.summary
    assert "setup --reconfigure" in geometry.remediation


def test_real_camera_probe_uses_mixed_yam_reader_without_preparing_driver(
    rig, monkeypatch
) -> None:
    import numpy as np

    from dropbear_yam.config import RigConfig

    calls: list[str] = []
    now = time.time()

    class Captured(dict):
        image_times = {name: now for name in ("top_cam", "left_cam", "right_cam")}

    class CameraOnlyEmbodiment:
        def __init__(self, config) -> None:
            assert config.left_depth_serial == "LEFT-D405"
            self._camera_reader = lambda _config: Captured(
                {
                    name: np.zeros((360, 640, 3), dtype=np.uint8)
                    for name in ("top_cam", "left_cam", "right_cam")
                }
            )
            calls.append("constructed")

        def close(self) -> None:
            calls.append("closed")

    mixed = RigConfig(
        **{
            **rig.as_dict(),
            "left_camera": "realsense:LEFT-D405",
        }
    )
    monkeypatch.setattr(
        "inspect_robots_yam.embodiment.YAMEmbodiment",
        CameraOnlyEmbodiment,
    )

    probe = _camera_probe(mixed)

    assert probe.shapes == {
        name: (360, 640, 3) for name in ("top_cam", "left_cam", "right_cam")
    }
    assert calls == ["constructed", "closed"]
