"""test_prism_publisher_quest_simple.py — minimal Quest teleop.

Strips back to the canonical oculus_reader pattern:

  - Press GRIP and HOLD it    → robot follows the controller (1:1 in base frame)
  - Release GRIP              → robot freezes at current pose
  - Ctrl-C                    → exit cleanly

No frame remapping (identity: controller xyz delta = robot xyz delta in
panda_link0). Adjust by experimenting then editing FRAME_R below.

No A/B/RJ buttons. No deadman trigger. No auto-home. No origin reset.
PrismPublisher's underlying safety still applies: workspace clip, per-call
rate limit, ErrorRecoveryAction on reflex.

Run AFTER:
  source ~/franka_ws/setup_franka.sh
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from oculus_reader import OculusReader
from prism_publisher import PrismPublisher, RateLimit, WorkspaceBox, _quat_angle
from test_prism_publisher_keyboard import slew_to


# Quest-controller frame → Franka panda_link0 frame.
# (Empirically calibrated for THIS lab's operator standing position +
# headset orientation. Edit if your physical setup is different.)
#
# Quest local frame in headset's view, lens facing forward:
#   +x  = headset's RIGHT
#   +y  = headset's UP
#   +z  = headset's BACKWARD (behind user)
#
# Franka panda_link0 frame:
#   +x  = forward, away from base
#   +y  = sideways (depends on installation)
#   +z  = up
#
# Each row = Franka axis, each column = Quest axis.
FRAME_R = np.array([
    [ 0.0,  0.0,  1.0],   # Franka x =  +Quest z
    [ 1.0,  0.0,  0.0],   # Franka y =  +Quest x
    [ 0.0,  1.0,  0.0],   # Franka z =  +Quest y
], dtype=np.float64)

# Franka canonical home pose, captured directly from this lab's robot when
# the user manually verified position AND orientation were correct.
# Joint config (rad):
#   [+0.0016, -0.7838, -0.0001, -2.3620, +0.0067, +1.5664, +0.7957]
# Matches Franka's standard home [0, -π/4, 0, -3π/4, 0, π/2, π/4] up to
# small per-robot calibration offsets.
FRANKA_HOME_XYZ  = np.array(
    [+0.305492, +0.001887, +0.483411], dtype=np.float64)
FRANKA_HOME_QUAT = np.array(
    [+0.999967, -0.004357, -0.005934, -0.003338], dtype=np.float64)  # xyzw
# ≈ (1, 0, 0, 0) = 180° rotation about base x-axis = gripper pointing down


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Quest-to-robot scale (1.0 = 1cm controller → 1cm robot)")
    ap.add_argument("--home-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=None,
                    help=f"Override the canonical home pose. "
                         f"Default = {tuple(FRANKA_HOME_XYZ)}.")
    ap.add_argument("--no-pre-home", action="store_true",
                    help="Skip the slew-to-home step at startup.")
    args = ap.parse_args()

    print("[simple] starting publisher...")
    pub = PrismPublisher(
        workspace=WorkspaceBox(
            x_min=0.20, x_max=0.65,
            y_min=-0.30, y_max=0.30,
            z_min=0.10, z_max=0.70,
        ),
        rate_limit=RateLimit(
            max_translation_per_update=0.05,   # 5 cm/call — matches Quest teleop
            max_rotation_per_update=0.30,      # bumped: 0.30 rad ~ 17° per call
                                               # so home slews back fast even after
                                               # significant rotational drift
        ),
        verbose=True,
    )
    time.sleep(1.0)

    # The "home" pose has BOTH xyz and quat hardcoded (captured from this
    # lab's robot in canonical pose). We use FRANKA_HOME_QUAT for everything:
    #   - pre-home slew at startup
    #   - A button slew
    #   - the orientation we publish during teleop (so cartesian impedance
    #     keeps pulling the robot back to canonical orientation)
    home_xyz = (np.array(args.home_xyz, dtype=np.float64)
                if args.home_xyz is not None
                else FRANKA_HOME_XYZ.copy())
    init_q = FRANKA_HOME_QUAT.copy()   # not captured from current robot;
                                        # forces canonical orientation

    # ── Pre-home: slew EE to Franka canonical home before letting the user
    #    teleop. Reasons: (1) every session starts from a known clean state,
    #    (2) workspace constraints + impedance behavior are tuned around
    #    that pose, (3) makes demo collection reproducible across sessions.
    if not args.no_pre_home:
        cur_xyz, _ = pub.get_current_target()
        if np.linalg.norm(cur_xyz - home_xyz) > 0.005:
            print(f"[simple] pre-home: slewing EE from "
                  f"({cur_xyz[0]:+.3f}, {cur_xyz[1]:+.3f}, {cur_xyz[2]:+.3f}) "
                  f"to HOME ({home_xyz[0]:+.3f}, {home_xyz[1]:+.3f}, {home_xyz[2]:+.3f}) "
                  f"with canonical orientation...")
            outcome = slew_to(pub, home_xyz, init_q)
            print(f"[simple] pre-home: {outcome}")
        else:
            print(f"[simple] already at HOME xyz — slewing to canonical orientation...")
            outcome = slew_to(pub, home_xyz, init_q)
            print(f"[simple] pre-home (orient-only): {outcome}")

    print("[simple] starting OculusReader (will install/restart APK)...")
    import subprocess
    subprocess.run(["adb", "shell", "am", "force-stop", "com.rail.oculus.teleop"],
                   check=False, capture_output=True)
    subprocess.run(["adb", "shell", "am", "start", "-n",
                    "com.rail.oculus.teleop/.MainActivity"],
                   check=False, capture_output=True)
    time.sleep(1.0)

    reader = OculusReader()
    reader.run()
    time.sleep(2.0)

    print("\n[simple] HOLD grip on right controller to follow; release to stop.")
    print("         Press A (lower face button) to slew to HOME at any time.")
    print("         Ctrl-C to exit.\n")

    # State
    grip_was_held = False
    a_was_pressed = False
    quest_origin: np.ndarray | None = None      # controller pose at grip press
    franka_origin: np.ndarray | None = None     # robot target at grip press
    last_log = 0.0

    try:
        while True:
            transforms, buttons = reader.get_transformations_and_buttons()

            # Read right controller pose + grip + A button
            if 'r' in transforms:
                pose = transforms['r']
                grip_val = buttons.get("rightGrip", 0.0)
                if isinstance(grip_val, (list, tuple)):
                    grip_val = grip_val[0] if grip_val else 0.0
                grip_held = float(grip_val) > 0.5
                a_now = bool(buttons.get("A", False))
            else:
                pose = None
                grip_held = False
                a_now = False

            # ── A button (rising edge) → slew back to home ──
            if a_now and not a_was_pressed:
                print(f"\n[simple] A pressed — slewing back to HOME "
                      f"({home_xyz[0]:+.3f}, {home_xyz[1]:+.3f}, {home_xyz[2]:+.3f})...")
                outcome = slew_to(pub, home_xyz, init_q)
                print(f"[simple] home: {outcome}")
                # After homing, drop any existing origin so next grip press re-anchors
                quest_origin = None
                franka_origin = None
                grip_was_held = False
                a_was_pressed = a_now
                continue
            a_was_pressed = a_now

            # Rising edge of grip → snapshot origins
            if grip_held and not grip_was_held and pose is not None:
                quest_origin = pose[:3, 3].copy()
                franka_origin, _ = pub.get_current_target()
                print(f"\n[simple] grip pressed: anchored "
                      f"controller=({quest_origin[0]:+.3f},{quest_origin[1]:+.3f},{quest_origin[2]:+.3f}) "
                      f"to robot=({franka_origin[0]:+.3f},{franka_origin[1]:+.3f},{franka_origin[2]:+.3f})")

            # Falling edge → reset origins
            if not grip_held and grip_was_held:
                quest_origin = None
                franka_origin = None
                print("\n[simple] grip released — frozen")

            # While grip held + origins set → follow
            if grip_held and quest_origin is not None and pose is not None:
                quest_now = pose[:3, 3]
                delta_quest = quest_now - quest_origin
                delta_franka = FRAME_R @ (delta_quest * args.scale)
                target_xyz = franka_origin + delta_franka
                pub.set_target_pose(target_xyz, init_q)

            # Status line at 2 Hz — show Quest pose, robot target, AND rotation
            # error so we can diagnose orientation convergence
            t = time.time()
            if t - last_log >= 0.5:
                last_log = t
                cur_target_xyz, _ = pub.get_current_target()
                ee = pub.get_robot_ee()
                if ee is not None:
                    err_mm = float(np.linalg.norm(cur_target_xyz - ee[0]) * 1000)
                    # rotation error: robot's actual quat vs init_q (canonical home)
                    rot_err_deg = float(np.degrees(_quat_angle(ee[1], init_q)))
                else:
                    err_mm = float("nan")
                    rot_err_deg = float("nan")
                state_str = "FOLLOW" if grip_held else "frozen"

                if pose is not None:
                    qxyz = pose[:3, 3]
                    quest_str = f"quest=({qxyz[0]:+.3f},{qxyz[1]:+.3f},{qxyz[2]:+.3f})"
                    g_str = f"grip={float(grip_val):.2f}"
                else:
                    quest_str = "quest=NO_CTRL"
                    g_str = "grip=??"

                sys.stdout.write(
                    f"\r[simple] {state_str:6s}  "
                    f"{quest_str}  {g_str}  "
                    f"→ target=({cur_target_xyz[0]:+.3f},{cur_target_xyz[1]:+.3f},{cur_target_xyz[2]:+.3f}) "
                    f"err_xyz={err_mm:5.1f}mm  rot_err={rot_err_deg:5.1f}°     "
                )
                sys.stdout.flush()

            grip_was_held = grip_held
            time.sleep(1.0 / 60)
    except KeyboardInterrupt:
        print("\n[simple] Ctrl-C — exiting (publisher freezes target where it is)")
    finally:
        # reader.stop() and pub.shutdown() can hang on thread joins or
        # re-raise KeyboardInterrupt if user pressed Ctrl-C twice; swallow
        # defensively so we always reach the "done" print.
        try:
            reader.stop()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            pub.shutdown()
        except (Exception, KeyboardInterrupt):
            pass
        print("[simple] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
