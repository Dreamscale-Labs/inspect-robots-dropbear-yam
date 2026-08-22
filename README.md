# DreamZero-YAM through Dropbear

This public composition is the attended, fail-closed path for running DreamZero-YAM through
Dropbear on a bimanual I2RT YAM rig. It is deliberately detachable from Dropbear core: YAM
hardware behavior lives in Dreamscale's YAM fork, the generic policy bridge remains in
`inspect-robots-dropbear`, and this repo owns only installation, rig configuration, diagnostics,
gates and cleanup.

No credentials or rig-specific configuration are stored in this repository.

## Copy, paste, run

On the Linux computer connected to both arms and all three cameras:

```bash
git clone --branch v0.1.1 --depth 1 \
  https://github.com/Dreamscale-Labs/inspect-robots-dropbear-yam.git
cd inspect-robots-dropbear-yam
./setup.sh
./dropbear-yam doctor
./dropbear-yam run "Pack container"
```

`"Pack container"` is an exact in-distribution task from
[`allenai/01122025-box-01`](https://huggingface.co/datasets/allenai/01122025-box-01),
one of the repositories recorded in the checkpoint's
[`experiment_cfg/conf.yaml`](https://huggingface.co/robocurve/dreamzero-yam-molmoact2/blob/f9b72b8dfa124f7283c5b1d467ce2ff9253c737a/experiment_cfg/conf.yaml)
training mixture. Use it only with a scene arranged for that task; replace the quoted text with
another trained task when the scene differs.

`setup.sh` installs `uv` when needed and creates the locked project environment. No manual virtual
environment activation is required.

`setup.sh` is safe to rerun. It installs missing Debian/Ubuntu build prerequisites only after one
explicit sudo confirmation, installs `uv` when absent, reproduces `uv.lock`, and launches the rig
interview. Existing confirmed values are kept. To deliberately replace them:

```bash
./dropbear-yam setup --reconfigure
```

The interview asks only for facts software cannot safely infer:

- which stable `/dev/v4l/by-id/...-video-index0` device is top, left and right;
- which SocketCAN interface controls the left and right arm;
- measured left/right arm-base `(x, y, z)` and yaw in one rig coordinate frame;
- measured table-top `z` in that same frame; and
- Dropbear login, but only when credentials are absent.

It writes `~/.config/dropbear-yam/rig.toml` with permissions `0600`. The file contains no API key.
Authentication remains in Dropbear's own config.

## What doctor proves

```bash
./dropbear-yam doctor
./dropbear-yam doctor --json
./dropbear-yam doctor --support-bundle ~/dropbear-yam-support.tar.gz
```

Doctor performs no robot motion, does not construct the I2RT motor driver, and creates no paid
Dropbear session. It checks the exact locked package commits, Linux/build prerequisites,
authentication, DreamZero-YAM entitlement and target availability, system clock synchronization,
camera roles/shapes/fresh Unix-epoch timestamps, cross-camera skew, CAN state, I2RT model limits,
measured collision geometry, end-to-end 30 Hz declarations, and the absence of any existing
Dropbear session.

Every failure has a stable `DBY-*` code and blocks `run`. The optional support archive contains the
doctor result and redacted configuration for self-guided debugging or sharing with Dreamscale.
Warnings are reserved for non-blocking observability; safety and configuration problems are
failures.

## What run does

`run` always repeats doctor first. It then requires one explicit `CONNECT` confirmation that the
e-stop is ready and that connecting will enable I2RT control traffic and calibrate both
`LINEAR_4310` grippers. Keep hands clear of the grippers.

After that confirmation the program:

1. opens cameras and I2RT once, performs required gripper calibration, and observes the real
   pre-home state without sending an arm pose;
2. for a new configuration digest, performs exactly one paid shadow inference, validates its first
   model action against the measured pre-home state, and never executes that action;
3. reuses the same hardware connection and, when shadow was needed, the same Dropbear session;
4. retains the YAM fork's stand-clear homing prompt and scene-ready prompt;
5. runs Inspect Robots programmatically with only abort-only strict action and collision guards—no
   clamp, interpolation, hold substitution or action rewriting; and
6. synchronously closes hardware and policy on success, abort, exception, signal or operator stop,
   then verifies that the exact owned Dropbear session disappeared.

If ordinary close does not remove that exact session, the program explicitly stops only that
session and exits nonzero. It never uses a stop-all operation.

Shadow evidence is stored under `~/.local/state/dropbear-yam/shadow/`. Any change to the locked
package commits, model target, camera/CAN mapping, rig geometry, cadence, joint bounds or step
limits changes the digest and requires a new shadow. Shadow validates integration only; it is not a
physical-safety or task-success claim. The strict abort chain remains active on every action.

## Cadence: exactly 30 Hz

This release supports DreamZero-YAM at exactly 30 Hz. That is the checkpoint/data action timebase;
it is not 30 cloud inference calls per second and it is not I2RT's internal motor servo frequency.
Arbitrary 5–30 Hz operation is intentionally unavailable until observation production, wire
metadata, temporal admission, inference and action execution consume one resolved rate end to end.

## Before the first physical task

Do not proceed without separate authorization for physical motion and paid inference. For the
first short trained task, an operator must stand clear with the e-stop in hand. Acceptance requires
correct 640x360 camera roles and source times, 30 Hz on both policy and YAM, at least one telemetry
row whose action source is `model`, no silently changed action, working gates/abort behavior, and
absence of the exact Dropbear session afterward.

Run logs, frames, actions and adapter telemetry are under `~/.local/state/dropbear-yam/logs/`.

## Locked components

`composition.lock.toml` is the human-readable identity contract and `uv.lock` is the complete
resolver lock. They pin the Dreamscale YAM fork, generic Dropbear adapter, Dropbear SDK, Inspect
Robots and I2RT. Do not hand-edit installed packages; update the locks and repeat doctor/shadow.
