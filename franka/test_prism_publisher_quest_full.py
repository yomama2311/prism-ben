"""test_prism_publisher_quest_full.py — 6-DoF Quest teleop + gripper.

Extends the minimal one-button version with full 6-DoF EE pose tracking
(translation + orientation) and a gripper open/close action.

Buttons:
  Right grip   (HOLD)        : follow controller — translation + orientation
  Right trigger (rising edge) : close gripper (grasp at 20 N)
  Right trigger (falling edge): open gripper to 8 cm
  A button (rising edge)      : slew to canonical home + open gripper
  Ctrl-C                      : exit cleanly

Required ROS action servers (from Xiao Hu's franka_run.py):
  /franka_gripper/grasp   — close to grasp
  /franka_gripper/move    — open to width
  (See `rostopic list | grep franka_gripper` to verify.)

Run AFTER:
  source ~/franka_ws/setup_franka.sh
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import numpy as np

import actionlib
import rospy
# franka_gripper_msgs is our minimal local package (no libfranka dep) — same
# .action / .msg definitions as franka_gripper, lets us talk to a remote
# franka_gripper action server without needing the C++ control library.
from franka_gripper_msgs.msg import (
    GraspAction, GraspGoal, GraspEpsilon,
    MoveAction, MoveGoal,
)
from oculus_reader import OculusReader
from prism_publisher import (
    PrismPublisher,
    RateLimit,
    WorkspaceBox,
    _quat_angle,
    _rotation_matrix_to_quat,
)
from test_prism_publisher_keyboard import slew_to


# =============================================================================
# Frame mapping  (Quest → Franka)  —  empirically calibrated for THIS lab.
# Edit if your operator standing position / headset orientation differs.
# =============================================================================
FRAME_R = np.array([
    [ 0.0,  0.0,  1.0],   # Franka x =  Quest z
    [ 1.0,  0.0,  0.0],   # Franka y =  Quest x
    [ 0.0,  1.0,  0.0],   # Franka z =  Quest y
], dtype=np.float64)


# =============================================================================
# Canonical Franka home, captured from this lab's robot.
# =============================================================================
FRANKA_HOME_XYZ  = np.array(
    [+0.305492, +0.001887, +0.483411], dtype=np.float64)
FRANKA_HOME_QUAT = np.array(
    [+0.999967, -0.004357, -0.005934, -0.003338], dtype=np.float64)  # xyzw


# =============================================================================
# Gripper params
# =============================================================================
GRIPPER_OPEN_WIDTH  = 0.080   # 8 cm — Panda gripper max width
GRIPPER_GRASP_FORCE = 20.0    # N
GRIPPER_GRASP_SPEED = 0.10    # m/s
GRIPPER_MOVE_SPEED  = 0.10    # m/s


# =============================================================================
# Helpers — quaternion <-> rotation matrix
# =============================================================================
def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3x3 rotation matrix."""
    x, y, z, w = q_xyzw
    n = float(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z),  s*(x*y - w*z),      s*(x*z + w*y)    ],
        [s*(x*y + w*z),      1 - s*(x*x + z*z),  s*(y*z - w*x)    ],
        [s*(x*z - w*y),      s*(y*z + w*x),      1 - s*(x*x + y*y)],
    ])


# =============================================================================
# Gripper action client (async — fire-and-forget on rising/falling edge)
# =============================================================================
class GripperClient:
    """Thin wrapper around franka_gripper actions; never blocks the teleop loop."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._grasp_ac = actionlib.SimpleActionClient(
            "/franka_gripper/grasp", GraspAction)
        self._move_ac = actionlib.SimpleActionClient(
            "/franka_gripper/move", MoveAction)
        if self.verbose:
            print("[gripper] waiting up to 5 s for action servers...")
        ok_grasp = self._grasp_ac.wait_for_server(rospy.Duration(5.0))
        ok_move  = self._move_ac.wait_for_server(rospy.Duration(5.0))
        self.available = bool(ok_grasp and ok_move)
        if self.verbose:
            print(f"[gripper] grasp_server={ok_grasp}  move_server={ok_move}  "
                  f"→ {'AVAILABLE' if self.available else 'DISABLED'}")
        # Track last-sent state to avoid spamming the action server
        self._state = "unknown"   # "open" | "closed" | "unknown"

    def open(self):
        """Open gripper to GRIPPER_OPEN_WIDTH (async)."""
        if not self.available:
            return
        if self._state == "open":
            return
        self._state = "open"
        goal = MoveGoal()
        goal.width = GRIPPER_OPEN_WIDTH
        goal.speed = GRIPPER_MOVE_SPEED
        self._move_ac.send_goal(goal)
        if self.verbose:
            print(f"\n[gripper] OPEN → {GRIPPER_OPEN_WIDTH*1000:.0f}mm")

    def close(self):
        """Close gripper (grasp at GRIPPER_GRASP_FORCE, async)."""
        if not self.available:
            return
        if self._state == "closed":
            return
        self._state = "closed"
        goal = GraspGoal()
        goal.width = 0.0
        goal.speed = GRIPPER_GRASP_SPEED
        goal.force = GRIPPER_GRASP_FORCE
        goal.epsilon = GraspEpsilon(inner=0.005, outer=0.080)
        self._grasp_ac.send_goal(goal)
        if self.verbose:
            print(f"\n[gripper] CLOSE → grasp at {GRIPPER_GRASP_FORCE:.0f}N")


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Quest-to-robot scale (1.0 = 1cm controller → 1cm robot)")
    ap.add_argument("--home-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=None, help="Override canonical home xyz.")
    ap.add_argument("--no-pre-home", action="store_true",
                    help="Skip the slew-to-home step at startup.")
    ap.add_argument("--no-gripper", action="store_true",
                    help="Disable gripper control (don't open/close on trigger).")
    ap.add_argument("--no-orientation", action="store_true",
                    help="Lock orientation to canonical home (act like the simple version).")
    args = ap.parse_args()

    print("[full] starting publisher...")
    pub = PrismPublisher(
        workspace=WorkspaceBox(
            x_min=0.20, x_max=0.65,
            y_min=-0.30, y_max=0.30,
            z_min=0.10, z_max=0.70,
        ),
        rate_limit=RateLimit(
            max_translation_per_update=0.05,
            max_rotation_per_update=0.30,
        ),
        verbose=True,
    )
    time.sleep(1.0)

    home_xyz = (np.array(args.home_xyz, dtype=np.float64)
                if args.home_xyz is not None
                else FRANKA_HOME_XYZ.copy())
    home_quat = FRANKA_HOME_QUAT.copy()

    # Pre-home slew
    if not args.no_pre_home:
        cur_xyz, _ = pub.get_current_target()
        print(f"[full] pre-home: slewing to canonical home "
              f"({home_xyz[0]:+.3f}, {home_xyz[1]:+.3f}, {home_xyz[2]:+.3f})...")
        outcome = slew_to(pub, home_xyz, home_quat)
        print(f"[full] pre-home: {outcome}")

    # Gripper client
    gripper = GripperClient(verbose=True) if not args.no_gripper else None
    if gripper is not None:
        gripper.open()    # always start in known state

    # Quest reader
    print("[full] starting OculusReader (will install/restart APK)...")
    subprocess.run(["adb", "shell", "am", "force-stop", "com.rail.oculus.teleop"],
                   check=False, capture_output=True)
    subprocess.run(["adb", "shell", "am", "start", "-n",
                    "com.rail.oculus.teleop/.MainActivity"],
                   check=False, capture_output=True)
    time.sleep(1.0)
    reader = OculusReader()
    reader.run()
    time.sleep(2.0)

    print("\n[full] HOLD grip on right controller to follow (6-DoF).")
    print("       Squeeze trigger to CLOSE gripper; release to OPEN.")
    print("       Press A (lower face button) to slew to HOME.")
    print("       Ctrl-C to exit.\n")

    # State
    grip_was_held = False
    a_was_pressed = False
    trig_was_held = False
    quest_origin_xyz: np.ndarray | None = None
    quest_origin_R:   np.ndarray | None = None
    franka_origin_xyz: np.ndarray | None = None
    franka_origin_R:   np.ndarray | None = None
    last_log = 0.0

    try:
        while True:
            transforms, buttons = reader.get_transformations_and_buttons()

            if 'r' in transforms:
                pose = transforms['r']
                grip_val = buttons.get("rightGrip", 0.0)
                if isinstance(grip_val, (list, tuple)):
                    grip_val = grip_val[0] if grip_val else 0.0
                grip_held = float(grip_val) > 0.5
                trig_val = buttons.get("rightTrig", 0.0)
                if isinstance(trig_val, (list, tuple)):
                    trig_val = trig_val[0] if trig_val else 0.0
                trig_held = float(trig_val) > 0.5
                a_now = bool(buttons.get("A", False))
            else:
                pose = None
                grip_held = trig_held = a_now = False
                grip_val = trig_val = 0.0

            # ── Gripper: trigger rising/falling edge ──
            if gripper is not None:
                if trig_held and not trig_was_held:
                    gripper.close()
                elif not trig_held and trig_was_held:
                    gripper.open()
            trig_was_held = trig_held

            # ── A: slew to home + open gripper ──
            if a_now and not a_was_pressed:
                print(f"\n[full] A pressed — slewing back to HOME...")
                outcome = slew_to(pub, home_xyz, home_quat)
                print(f"[full] home: {outcome}")
                if gripper is not None:
                    gripper.open()
                # After homing, drop origins so next grip press re-anchors
                quest_origin_xyz = quest_origin_R = None
                franka_origin_xyz = franka_origin_R = None
                grip_was_held = False
                a_was_pressed = a_now
                continue
            a_was_pressed = a_now

            # ── Grip rising edge → snapshot origins (xyz + rotation) ──
            if grip_held and not grip_was_held and pose is not None:
                quest_origin_xyz = pose[:3, 3].copy()
                quest_origin_R   = pose[:3, :3].copy()
                franka_origin_xyz, franka_origin_q = pub.get_current_target()
                franka_origin_R = quat_to_R(franka_origin_q)
                print(f"\n[full] grip pressed — anchored "
                      f"controller=({quest_origin_xyz[0]:+.3f},"
                      f"{quest_origin_xyz[1]:+.3f},{quest_origin_xyz[2]:+.3f}) "
                      f"to robot=({franka_origin_xyz[0]:+.3f},"
                      f"{franka_origin_xyz[1]:+.3f},{franka_origin_xyz[2]:+.3f})")

            # ── Grip falling edge → freeze ──
            if not grip_held and grip_was_held:
                quest_origin_xyz = quest_origin_R = None
                franka_origin_xyz = franka_origin_R = None
                print("\n[full] grip released — frozen")

            # ── Following: compute 6-DoF target ──
            if (grip_held and quest_origin_xyz is not None and pose is not None):
                quest_now_xyz = pose[:3, 3]
                # Translation: same as simple version
                delta_xyz_q = quest_now_xyz - quest_origin_xyz
                delta_xyz_f = FRAME_R @ (delta_xyz_q * args.scale)
                target_xyz = franka_origin_xyz + delta_xyz_f

                # Orientation: 6-DoF mapping (or lock to home if --no-orientation)
                if args.no_orientation:
                    target_quat = home_quat
                else:
                    quest_now_R = pose[:3, :3]
                    # Δ rotation in Quest frame (origin → now)
                    delta_R_quest = quest_now_R @ quest_origin_R.T
                    # Transform Δ rotation into Franka frame via similarity
                    delta_R_franka = FRAME_R @ delta_R_quest @ FRAME_R.T
                    # Apply Δ to franka origin rotation
                    target_R = delta_R_franka @ franka_origin_R
                    target_quat = _rotation_matrix_to_quat(target_R)

                pub.set_target_pose(target_xyz, target_quat)

            # ── Status line at 2 Hz ──
            t = time.time()
            if t - last_log >= 0.5:
                last_log = t
                cur_target_xyz, cur_target_q = pub.get_current_target()
                ee = pub.get_robot_ee()
                if ee is not None:
                    err_mm = float(np.linalg.norm(cur_target_xyz - ee[0]) * 1000)
                    rot_err_deg = float(np.degrees(_quat_angle(ee[1], cur_target_q)))
                else:
                    err_mm = float("nan")
                    rot_err_deg = float("nan")

                state_str = "FOLLOW" if grip_held else "frozen"
                if pose is not None:
                    qxyz = pose[:3, 3]
                    quest_str = f"quest=({qxyz[0]:+.3f},{qxyz[1]:+.3f},{qxyz[2]:+.3f})"
                else:
                    quest_str = "quest=NO_CTRL"
                grip_state = ("CLOSED" if gripper and gripper._state == "closed"
                              else "open" if gripper and gripper._state == "open"
                              else "—")
                sys.stdout.write(
                    f"\r[full] {state_str:6s}  "
                    f"{quest_str}  "
                    f"trig={float(trig_val):.2f}({grip_state})  "
                    f"→ target_xyz=({cur_target_xyz[0]:+.3f},{cur_target_xyz[1]:+.3f},{cur_target_xyz[2]:+.3f}) "
                    f"err={err_mm:5.1f}mm  rot_err={rot_err_deg:5.1f}°     "
                )
                sys.stdout.flush()

            grip_was_held = grip_held
            time.sleep(1.0 / 60)
    except KeyboardInterrupt:
        print("\n[full] Ctrl-C — exiting (publisher freezes target where it is)")
    finally:
        try:
            reader.stop()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            pub.shutdown()
        except (Exception, KeyboardInterrupt):
            pass
        print("[full] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
