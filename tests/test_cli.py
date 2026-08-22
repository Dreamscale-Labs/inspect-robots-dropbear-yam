from __future__ import annotations

from pathlib import Path

from dropbear_yam.cli import _doctor_command, _parser, main
from dropbear_yam.doctor import Diagnostic, DoctorReport


def test_every_rig_command_accepts_an_explicit_named_profile() -> None:
    parser = _parser()

    assert parser.parse_args(["setup", "--rig", "jay-left"]).rig == "jay-left"
    assert parser.parse_args(["doctor", "--rig", "jay-left"]).rig == "jay-left"
    assert parser.parse_args(["run", "--rig", "jay-left", "Pack container"]).rig == "jay-left"


def test_run_defaults_to_two_minutes_at_30_hz() -> None:
    args = _parser().parse_args(["run", "Pack container"])

    assert args.max_steps == 3600


def test_cli_error_has_plain_message_and_next_step(monkeypatch, capsys) -> None:
    missing = Path("/tmp/example-rig.toml")
    monkeypatch.setattr(
        "dropbear_yam.cli.load_rig",
        lambda **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError(f"rig config missing: {missing}")
        ),
    )

    assert main(["doctor"]) == 2
    error = capsys.readouterr().err
    assert "Error: No YAM rig has been configured yet." in error
    assert "Next: Run ./setup.sh" in error


def test_doctor_prints_action_for_warning_as_well_as_failure(monkeypatch, capsys) -> None:
    report = DoctorReport(
        checks=(
            Diagnostic(
                "DBY-GEOMETRY",
                "warn",
                "Predictive collision checking is turned off.",
                "Run ./dropbear-yam setup --reconfigure to add it.",
            ),
        )
    )
    monkeypatch.setattr("dropbear_yam.cli.load_rig", lambda **_kwargs: object())
    monkeypatch.setattr("dropbear_yam.cli.doctor", lambda _rig: report)

    assert _doctor_command(False, None) == 0
    output = capsys.readouterr().out
    assert "[WARN] DBY-GEOMETRY" in output
    assert "next: Run ./dropbear-yam setup --reconfigure" in output
