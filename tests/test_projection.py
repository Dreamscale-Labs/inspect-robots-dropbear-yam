from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from inspect_robots.errors import SafetyAbort
from inspect_robots.types import Action

from dropbear_yam.config import I2RT_JOINT_HIGH, I2RT_JOINT_LOW, STRICT_STEP_LIMITS
from dropbear_yam.projection import ProjectionAudit, YamProjectionApprover


def _approver(
    reference: np.ndarray | None = None,
    *,
    step_limits: tuple[float, ...] = STRICT_STEP_LIMITS,
    audit=None,
) -> YamProjectionApprover:
    resolved = np.zeros(14) if reference is None else reference
    return YamProjectionApprover(
        reference=lambda: resolved.copy(),
        joint_low=I2RT_JOINT_LOW,
        joint_high=I2RT_JOINT_HIGH,
        step_limits=step_limits,
        audit=audit,
    )


def test_arm_jump_is_capped_to_point_two_and_metadata_is_preserved() -> None:
    requested = np.zeros(14)
    requested[0] = 0.25
    original = Action(requested, {"dropbear_action_source": "model", "epoch": 4})

    applied = _approver().review(original, {})

    assert applied is not original
    assert applied.data[0] == pytest.approx(0.2)
    assert applied.meta["dropbear_action_source"] == "model"
    assert applied.meta["epoch"] == 4
    assert applied.meta["delta_clamped"] is True
    assert applied.meta["dropbear_yam_projected"] is True
    assert "clamped" not in applied.meta


@pytest.mark.parametrize("requested", [0.2, -0.2])
def test_exact_step_boundary_returns_original_action(requested: float) -> None:
    target = np.zeros(14)
    target[0] = requested
    action = Action(target, {"source": "model"})

    assert _approver().review(action, {}) is action


def test_absolute_and_delta_intersection_marks_both_projection_reasons() -> None:
    reference = np.asarray(I2RT_JOINT_HIGH, dtype=np.float64).copy()
    reference[0] -= 0.05
    requested = reference.copy()
    requested[0] = I2RT_JOINT_HIGH[0] + 1.0

    applied = _approver(reference).review(Action(requested), {})

    assert applied.data[0] == pytest.approx(I2RT_JOINT_HIGH[0])
    assert applied.meta["clamped"] is True
    assert applied.meta["delta_clamped"] is True


def test_gripper_projection_uses_one_normalized_stroke() -> None:
    reference = np.zeros(14)
    requested = np.zeros(14)
    requested[6] = requested[13] = 3.0

    applied = _approver(reference).review(Action(requested), {})

    assert applied.data[6] == pytest.approx(1.0)
    assert applied.data[13] == pytest.approx(1.0)
    assert applied.meta["clamped"] is True
    assert applied.meta["delta_clamped"] is True


@pytest.mark.parametrize("requested", [np.zeros(13), np.full(14, np.nan)])
def test_malformed_action_aborts_before_audit(requested: np.ndarray) -> None:
    events = []

    with pytest.raises(SafetyAbort, match="exactly 14 finite"):
        _approver(audit=events.append).review(Action(requested), {})

    assert events == []


@pytest.mark.parametrize("reference", [np.zeros(13), np.full(14, np.inf)])
def test_impossible_reference_aborts(reference: np.ndarray) -> None:
    with pytest.raises(SafetyAbort, match="reference.*exactly 14 finite"):
        _approver(reference).review(Action(np.zeros(14)), {})


def test_repeated_gross_finite_targets_staircase_without_stalling() -> None:
    reference = np.zeros(14)
    approver = _approver(reference)
    applied: list[float] = []

    for _ in range(5):
        result = approver.review(Action(np.full(14, 100.0)), {})
        reference[:] = result.data
        applied.append(float(result.data[0]))

    assert applied == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])


def test_custom_finite_positive_limits_above_defaults_are_honored() -> None:
    limits = (0.5,) * 14
    requested = np.full(14, 2.0)

    applied = _approver(step_limits=limits).review(Action(requested), {})

    np.testing.assert_allclose(applied.data, np.full(14, 0.5))


def test_projection_audit_is_redacted_and_summarizes_changes(tmp_path: Path) -> None:
    path = tmp_path / "projection.jsonl"
    output: list[str] = []
    audit = ProjectionAudit(path, output=output.append)
    requested = np.zeros(14)
    requested[0] = 0.25

    _approver(audit=audit.record).review(
        Action(requested, {"api_key": "secret", "instruction": "private task"}), {}
    )
    audit.summarize()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["requested"][0] == pytest.approx(0.25)
    assert payload["applied"][0] == pytest.approx(0.2)
    assert payload["dimensions"] == ["left_j0"]
    assert payload["reasons"] == ["step_limit"]
    assert "secret" not in path.read_text(encoding="utf-8")
    assert "private task" not in path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == 0o600
    assert sum("capped" in line.lower() for line in output) == 1
    assert any("1 policy action" in line and str(path) in line for line in output)
