"""Composition-owned cap-and-continue projection for absolute YAM actions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from inspect_robots.errors import SafetyAbort
from inspect_robots.types import Action
from inspect_robots_yam.packing import DIM_LABELS

YAM_ACTION_DIM = 14


def _finite_vector(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SafetyAbort(f"{name} must contain exactly 14 finite values") from exc
    if vector.shape != (YAM_ACTION_DIM,) or not bool(np.all(np.isfinite(vector))):
        raise SafetyAbort(f"{name} must contain exactly 14 finite values")
    return vector


def _action_hash(values: npt.ArrayLike) -> str:
    payload = json.dumps(
        [float(value) for value in np.asarray(values, dtype=np.float64)],
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ProjectionEvent:
    """Secret-free evidence for one action changed by the composition."""

    action_index: int
    reference: tuple[float, ...]
    requested: tuple[float, ...]
    applied: tuple[float, ...]
    dimensions: tuple[str, ...]
    absolute_dimensions: tuple[str, ...]
    delta_dimensions: tuple[str, ...]

    @property
    def largest_adjustment(self) -> float:
        return max(
            abs(before - after)
            for before, after in zip(self.requested, self.applied, strict=True)
        )

    def as_dict(self) -> dict[str, Any]:
        reasons = []
        if self.absolute_dimensions:
            reasons.append("joint_bound")
        if self.delta_dimensions:
            reasons.append("step_limit")
        return {
            "schema_version": 1,
            "recorded_at": time.time(),
            "action_index": self.action_index,
            "reference": list(self.reference),
            "requested": list(self.requested),
            "applied": list(self.applied),
            "requested_sha256": _action_hash(self.requested),
            "applied_sha256": _action_hash(self.applied),
            "dimensions": list(self.dimensions),
            "absolute_dimensions": list(self.absolute_dimensions),
            "delta_dimensions": list(self.delta_dimensions),
            "reasons": reasons,
        }


class ProjectionAudit:
    """Append changed actions to one private JSONL sidecar and summarize them."""

    def __init__(self, path: Path, *, output: Callable[[str], None] = print):
        self.path = path
        self._output = output
        self._count = 0
        self._largest_adjustment = 0.0
        self._notice_emitted = False

    @property
    def count(self) -> int:
        return self._count

    def record(self, event: ProjectionEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.as_dict(), sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)
        self._count += 1
        self._largest_adjustment = max(self._largest_adjustment, event.largest_adjustment)
        if not self._notice_emitted:
            self._output(
                "Policy action capped to the configured joint and per-step limits; continuing."
            )
            self._notice_emitted = True

    def summarize(self) -> None:
        if self._count:
            noun = "action" if self._count == 1 else "actions"
            self._output(
                f"Projected {self._count} policy {noun}; largest requested adjustment was "
                f"{self._largest_adjustment:.3f}. Details: {self.path}"
            )


class YamProjectionApprover:
    """Cap finite absolute targets to joint and per-step limits, then continue."""

    def __init__(
        self,
        *,
        reference: Callable[[], npt.ArrayLike],
        joint_low: npt.ArrayLike,
        joint_high: npt.ArrayLike,
        step_limits: npt.ArrayLike,
        audit: Callable[[ProjectionEvent], None] | None = None,
    ):
        self._reference = reference
        self._joint_low = _finite_vector(joint_low, name="joint_low")
        self._joint_high = _finite_vector(joint_high, name="joint_high")
        self._step_limits = _finite_vector(step_limits, name="step_limits")
        if bool(np.any(self._joint_low > self._joint_high)):
            raise ValueError("joint_low must not exceed joint_high")
        if bool(np.any(self._step_limits <= 0)):
            raise ValueError("step_limits must contain exactly 14 finite positive values")
        self._audit = audit
        self._review_index = 0

    def review(self, action: Action, _store: dict[str, Any]) -> Action:
        action_index = self._review_index
        self._review_index += 1
        return self.project(action, reference=self._reference(), action_index=action_index)

    def project(
        self,
        action: Action,
        *,
        reference: npt.ArrayLike,
        action_index: int = 0,
    ) -> Action:
        """Apply the same pure projection with an explicit reference, e.g. shadow."""
        requested = _finite_vector(action.data, name="policy action")
        resolved_reference = _finite_vector(reference, name="policy action reference")
        permitted_low = np.maximum(
            self._joint_low,
            resolved_reference - self._step_limits,
        )
        permitted_high = np.minimum(
            self._joint_high,
            resolved_reference + self._step_limits,
        )
        impossible = np.flatnonzero(permitted_low > permitted_high)
        if impossible.size:
            labels = ", ".join(DIM_LABELS[int(index)] for index in impossible)
            raise SafetyAbort(
                "policy action reference has no safe target inside configured limits: " + labels
            )
        applied = np.clip(requested, permitted_low, permitted_high)
        changed = np.flatnonzero(applied != requested)
        if not changed.size:
            return action

        absolute = np.flatnonzero(
            (requested < self._joint_low) | (requested > self._joint_high)
        )
        delta = np.flatnonzero(
            (requested < resolved_reference - self._step_limits)
            | (requested > resolved_reference + self._step_limits)
        )
        metadata = dict(action.meta)
        if absolute.size:
            metadata["clamped"] = True
        if delta.size:
            metadata["delta_clamped"] = True
        metadata["dropbear_yam_projected"] = True
        event = ProjectionEvent(
            action_index=action_index,
            reference=tuple(float(value) for value in resolved_reference),
            requested=tuple(float(value) for value in requested),
            applied=tuple(float(value) for value in applied),
            dimensions=tuple(DIM_LABELS[int(index)] for index in changed),
            absolute_dimensions=tuple(DIM_LABELS[int(index)] for index in absolute),
            delta_dimensions=tuple(DIM_LABELS[int(index)] for index in delta),
        )
        if self._audit is not None:
            self._audit(event)
        return Action(applied, metadata)
