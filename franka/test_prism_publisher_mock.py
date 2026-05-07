"""test_prism_publisher_mock.py — drive PrismPublisher with synthetic targets.

Three modes of increasing risk:

  --mode hold       (default, ZERO motion)
      Target stays at the robot's current EE pose. Verifies:
        - publish thread is alive (30 Hz on /equilibrium_pose)
        - state subscribe works (FrankaState received)
        - error_recovery action server is reachable
        - pose_lock and reflex_lock work without deadlock
      The robot SHOULD NOT MOVE in this mode (cartesian impedance pulls
      it toward where it already is).

  --mode hover      (small oscillation, ~5 mm peak-to-peak in y)
      Target oscillates ±2.5 mm in y around the initial pose. Robot
      will visibly track the target (small wobble). Use only if you
      can reach the E-stop. This is the first test that meaningfully
      exercises the cartesian impedance controller's tracking loop.

  --mode line       (slow ramp, ~5 cm total in y)
      Target slides ±2.5 cm in y around initial pose, slowly (over
      ~10 s). This is a cleaner check that rate-limit + workspace clip
      are not over-restricting. Use only with E-stop within reach.

Run AFTER:
  source ~/franka_ws/setup_franka.sh
  python ~/franka_ws/hello_robot.py     # confirm read-side works first

Usage:
  python ~/franka_ws/test_prism_publisher_mock.py                  # safe default
  python ~/franka_ws/test_prism_publisher_mock.py --mode hover     # confirm
  python ~/franka_ws/test_prism_publisher_mock.py --mode line      # confirm
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from prism_publisher import (
    PrismPublisher,
    RateLimit,
    WorkspaceBox,
    ROBOT_MODE_IDLE,
    ROBOT_MODE_MOVE,
    ROBOT_MODE_REFLEX,
)


MODE_DESCRIPTIONS = {
    "hold":  "ZERO motion — target = current EE pose, robot should not move",
    "hover": "±2.5 mm y oscillation — small wobble, robot tracks target",
    "line":  "±2.5 cm y ramp — slow slide, exercises rate-limit and workspace",
}


def confirm_motion_mode(mode: str) -> bool:
    print()
    print("=" * 64)
    print(f"⚠  --mode {mode}: this WILL command motion on the real robot.")
    print(f"   {MODE_DESCRIPTIONS[mode]}")
    print("=" * 64)
    print("Required:")
    print("  • You can physically reach the E-stop")
    print("  • Workspace around the robot is clear")
    print("  • Lab control PC's stack is up (franka_run.py running)")
    print("  • You verified hello_robot.py shows mode=2 (MOVE), no errors")
    print()
    ans = input("Type 'YES' to continue (anything else aborts): ").strip()
    return ans == "YES"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_DESCRIPTIONS.keys()),
                    default="hold",
                    help="Test mode (default: hold = no motion)")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="How long to run the test (seconds)")
    ap.add_argument("--update-hz", type=float, default=10.0,
                    help="How often to call set_target_pose (Hz). Publish thread is fixed at 30 Hz.")
    args = ap.parse_args()

    print(f"\n[test] mode={args.mode}  duration={args.duration:.1f}s  "
          f"set_target_pose at {args.update_hz} Hz  (publisher 30 Hz internally)")
    print(f"[test] {MODE_DESCRIPTIONS[args.mode]}")

    if args.mode != "hold":
        if not confirm_motion_mode(args.mode):
            print("Aborted.")
            return 1

    # ── construct publisher ──
    # Tight workspace + rate limit for these tests
    pub = PrismPublisher(
        workspace=WorkspaceBox(
            x_min=0.20, x_max=0.55,
            y_min=-0.20, y_max=0.20,
            z_min=0.20, z_max=0.65,
        ),
        rate_limit=RateLimit(
            max_translation_per_update=0.005,   # 5 mm per call (very tight)
            max_rotation_per_update=0.05,
        ),
        verbose=True,
    )

    # ── settle ──
    time.sleep(1.0)

    init_xyz, init_q = pub.get_current_target()
    init_state = pub.get_robot_state()

    print(f"\n[test] initial target  = ({init_xyz[0]:+.3f}, {init_xyz[1]:+.3f}, {init_xyz[2]:+.3f})")
    if init_state is None:
        print("[test] ⚠ no FrankaState received yet — is the lab stack running?")
        return 2
    print(f"[test] initial robot EE = ({init_state.O_T_EE[12]:+.3f}, "
          f"{init_state.O_T_EE[13]:+.3f}, {init_state.O_T_EE[14]:+.3f})")
    print(f"[test] robot mode      = {init_state.robot_mode}  "
          f"({'IDLE' if init_state.robot_mode == ROBOT_MODE_IDLE else 'MOVE' if init_state.robot_mode == ROBOT_MODE_MOVE else f'mode-{init_state.robot_mode}'})")

    if init_state.robot_mode == ROBOT_MODE_REFLEX:
        print("[test] ⚠ robot is in REFLEX mode at start. Letting auto-recovery run...")
        time.sleep(3.0)
        new_state = pub.get_robot_state()
        print(f"[test] after wait, mode = {new_state.robot_mode}")
        if new_state.robot_mode == ROBOT_MODE_REFLEX:
            print("[test] ❌ still in reflex; manual reset needed on lab control PC")
            pub.shutdown()
            return 3

    # ── main loop ──
    t0 = time.time()
    period = 1.0 / args.update_hz
    n_updates = 0
    n_log = 0

    print(f"\n[test] starting {args.duration:.1f}s loop at {args.update_hz} Hz update rate...\n")

    while time.time() - t0 < args.duration:
        t = time.time() - t0

        # Compute target xyz for current mode
        if args.mode == "hold":
            target_xyz = init_xyz.copy()
        elif args.mode == "hover":
            # ±2.5 mm in y, 1 Hz oscillation
            dy = 0.0025 * np.sin(2 * np.pi * 1.0 * t)
            target_xyz = init_xyz + np.array([0.0, dy, 0.0])
        elif args.mode == "line":
            # ±2.5 cm in y, period = 10 s (slow)
            dy = 0.025 * np.sin(2 * np.pi * 0.1 * t)
            target_xyz = init_xyz + np.array([0.0, dy, 0.0])
        else:
            target_xyz = init_xyz.copy()

        info = pub.set_target_pose(target_xyz, init_q)
        n_updates += 1

        # Log once per second
        if int(t) > n_log:
            n_log = int(t)
            cur_target = pub.get_current_target()[0]
            cur_state = pub.get_robot_state()
            cur_robot = (np.array([cur_state.O_T_EE[12], cur_state.O_T_EE[13], cur_state.O_T_EE[14]])
                         if cur_state is not None else np.array([np.nan]*3))
            err = float(np.linalg.norm(cur_target - cur_robot))
            mode_str = (
                "MOVE" if cur_state and cur_state.robot_mode == ROBOT_MODE_MOVE else
                "IDLE" if cur_state and cur_state.robot_mode == ROBOT_MODE_IDLE else
                "REFLEX" if cur_state and cur_state.robot_mode == ROBOT_MODE_REFLEX else
                f"mode-{cur_state.robot_mode if cur_state else '?'}"
            )
            cflags = []
            if info["workspace_clipped"]:
                cflags.append("ws_clip")
            if info["translation_clipped"]:
                cflags.append("tr_clip")
            if info["rotation_clipped"]:
                cflags.append("rot_clip")
            cflag_str = ",".join(cflags) if cflags else "-"
            print(f"  t={t:5.1f}s  "
                  f"target=({cur_target[0]:+.3f}, {cur_target[1]:+.3f}, {cur_target[2]:+.3f})  "
                  f"robot=({cur_robot[0]:+.3f}, {cur_robot[1]:+.3f}, {cur_robot[2]:+.3f})  "
                  f"err={err*1000:5.1f}mm  "
                  f"mode={mode_str}  "
                  f"clip={cflag_str}")

        time.sleep(period)

    # ── summary ──
    print(f"\n[test] done. {n_updates} target updates over {time.time() - t0:.1f}s "
          f"({n_updates / max(1.0, time.time() - t0):.1f} Hz effective).")

    final_state = pub.get_robot_state()
    if final_state is not None:
        print(f"[test] final robot mode: {final_state.robot_mode}, errors: "
              f"{any(getattr(final_state.current_errors, slot) for slot in final_state.current_errors.__slots__)}")

    pub.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
