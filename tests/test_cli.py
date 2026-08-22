from __future__ import annotations

from dropbear_yam.cli import _parser


def test_every_rig_command_accepts_an_explicit_named_profile() -> None:
    parser = _parser()

    assert parser.parse_args(["setup", "--rig", "jay-left"]).rig == "jay-left"
    assert parser.parse_args(["doctor", "--rig", "jay-left"]).rig == "jay-left"
    assert parser.parse_args(["run", "--rig", "jay-left", "Pack container"]).rig == "jay-left"
