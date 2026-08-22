"""Small user-facing error contract for actionable CLI failures."""

from __future__ import annotations

from collections.abc import Callable


class UserFacingError(RuntimeError):
    """An expected failure with one concrete recovery action."""

    def __init__(self, message: str, next_step: str) -> None:
        super().__init__(message)
        self.next_step = next_step


def emit_error(
    output: Callable[[str], None],
    message: str,
    next_step: str,
) -> None:
    """Render failures consistently without exposing internal exception names."""
    output(f"Error: {message.rstrip('.')}.")
    output(f"Next: {next_step.rstrip('.')}.")


def explain_exception(exc: BaseException) -> tuple[str, str]:
    """Translate expected CLI exceptions into plain language and a next action."""
    if isinstance(exc, UserFacingError):
        return str(exc), exc.next_step
    text = str(exc).strip()
    if isinstance(exc, FileNotFoundError) and "rig config" in text.lower():
        return (
            "No YAM rig has been configured yet",
            "Run ./setup.sh from this cloned repository",
        )
    if isinstance(exc, PermissionError):
        return (
            f"The command does not have permission to access the requested path: {text}",
            "Choose a path under your home directory or fix its ownership, then repeat the command",
        )
    if "rig profile" in text.lower():
        return (
            text,
            "Use a short rig name containing only letters, numbers, dots, underscores or dashes",
        )
    if "multiple rig profiles" in text.lower():
        return (
            text,
            "Repeat the command with --rig NAME using one of the names shown",
        )
    if isinstance(exc, FileExistsError) and "--reconfigure" in text:
        return text, "Rerun ./dropbear-yam setup --reconfigure only if you want to replace it"
    if any(
        phrase in text.lower()
        for phrase in (
            "rig schema",
            "model_target",
            "30 hz",
            "640x360",
            "joint control",
            "attended mode",
            "keep_warm",
            "strict abort",
            "collision",
            "joint bounds",
            "step limits",
            "camera role",
            "stable camera",
            "can role",
            "[rig] table",
            "realsense camera source",
        )
    ):
        return (
            text,
            "Run ./dropbear-yam setup --reconfigure from this checkout instead of editing the "
            "rig file by hand",
        )
    return (
        text or "The command could not be completed",
        "Run ./dropbear-yam doctor; if it is still unclear, run "
        "./dropbear-yam doctor --support-bundle ~/dropbear-yam-support.tar.gz "
        "and send that file to Dreamscale",
    )
