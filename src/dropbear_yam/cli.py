"""The single user-facing dropbear-yam command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dropbear import errors as dropbear_errors
from dropbear.quickstart import run_login

from dropbear_yam.config import load_rig
from dropbear_yam.doctor import create_support_bundle, doctor
from dropbear_yam.errors import emit_error, explain_exception
from dropbear_yam.runner import run
from dropbear_yam.setup_command import setup


def _warm_minutes(value: str) -> int:
    try:
        minutes = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--warm must be a whole number from 0 to 60") from exc
    if not 0 <= minutes <= 60:
        raise argparse.ArgumentTypeError("--warm must be a whole number from 0 to 60")
    return minutes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dropbear-yam")
    subcommands = parser.add_subparsers(dest="command", required=True)
    setup_parser = subcommands.add_parser(
        "setup", help="confirm devices, optional collision geometry and login"
    )
    setup_parser.add_argument("--rig", help="named physical rig profile")
    setup_parser.add_argument("--reconfigure", action="store_true")
    subcommands.add_parser("login", help="sign in to Dropbear for this YAM checkout")
    doctor_parser = subcommands.add_parser("doctor", help="motion-free, session-free checks")
    doctor_parser.add_argument("--rig", help="named physical rig profile")
    doctor_parser.add_argument("--json", action="store_true", dest="json_output")
    doctor_parser.add_argument("--support-bundle", type=Path)
    run_parser = subcommands.add_parser("run", help="run one attended DreamZero-YAM task")
    run_parser.add_argument("--rig", help="named physical rig profile")
    run_parser.add_argument("instruction")
    run_parser.add_argument("--max-steps", type=int, default=3600)
    run_parser.add_argument(
        "--warm",
        type=_warm_minutes,
        default=5,
        metavar="MINUTES",
        help="keep this exact compute warm after the run (0-60; default: 5; billed)",
    )
    run_parser.add_argument("--log-dir", type=Path)
    return parser


def _doctor_command(
    json_output: bool,
    support_bundle: Path | None,
    rig_name: str | None = None,
) -> int:
    rig = load_rig(profile=rig_name)
    report = doctor(rig)
    if json_output:
        print(report.to_json())
    else:
        for check in report.checks:
            marker = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[check.status]
            print(f"[{marker}] {check.code}: {check.summary}")
            if check.status != "pass" and check.remediation:
                print(f"       next: {check.remediation}")
        print("READY" if report.ok else "BLOCKED")
    if support_bundle is not None:
        path = create_support_bundle(support_bundle, report, rig)
        print(f"Redacted support bundle: {path}", file=sys.stderr)
    return 0 if report.ok else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "setup":
            setup(rig_name=args.rig, reconfigure=args.reconfigure)
            return 0
        if args.command == "login":
            run_login(print_next_steps=False)
            print("Dropbear login complete. Next run:")
            print("  ./dropbear-yam doctor")
            return 0
        if args.command == "doctor":
            return _doctor_command(args.json_output, args.support_bundle, args.rig)
        if args.command == "run":
            rig = load_rig(profile=args.rig)
            return run(
                args.instruction,
                rig,
                max_steps=args.max_steps,
                warm_minutes=args.warm,
                log_dir=args.log_dir,
            )
    except dropbear_errors.DropbearError as exc:
        rendered = exc.render().replace("next:  dropbear login", "next:  ./dropbear-yam login")
        print(rendered, file=sys.stderr)
        if args.command == "setup":
            print(
                "YAM rig configuration is saved; you will not need to choose devices again.",
                file=sys.stderr,
            )
            print("Next: Run ./dropbear-yam login, then ./dropbear-yam doctor.", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        message, next_step = explain_exception(exc)
        emit_error(lambda line: print(line, file=sys.stderr), message, next_step)
        return 2
    except KeyboardInterrupt:
        print("Cancelled: no further setup or run steps will be started.", file=sys.stderr)
        return 130
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
