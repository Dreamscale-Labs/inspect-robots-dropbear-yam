"""The single user-facing dropbear-yam command."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dropbear_yam.config import load_rig
from dropbear_yam.doctor import create_support_bundle, doctor
from dropbear_yam.runner import run
from dropbear_yam.setup_command import setup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dropbear-yam")
    subcommands = parser.add_subparsers(dest="command", required=True)
    setup_parser = subcommands.add_parser("setup", help="confirm devices, geometry and login")
    setup_parser.add_argument("--rig", help="named physical rig profile")
    setup_parser.add_argument("--reconfigure", action="store_true")
    doctor_parser = subcommands.add_parser("doctor", help="motion-free, session-free checks")
    doctor_parser.add_argument("--rig", help="named physical rig profile")
    doctor_parser.add_argument("--json", action="store_true", dest="json_output")
    doctor_parser.add_argument("--support-bundle", type=Path)
    run_parser = subcommands.add_parser("run", help="run one attended DreamZero-YAM task")
    run_parser.add_argument("--rig", help="named physical rig profile")
    run_parser.add_argument("instruction")
    run_parser.add_argument("--max-steps", type=int, default=300)
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
            if check.status == "fail" and check.remediation:
                print(f"       fix: {check.remediation}")
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
        if args.command == "doctor":
            return _doctor_command(args.json_output, args.support_bundle, args.rig)
        if args.command == "run":
            rig = load_rig(profile=args.rig)
            return run(
                args.instruction,
                rig,
                max_steps=args.max_steps,
                log_dir=args.log_dir,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"dropbear-yam: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
