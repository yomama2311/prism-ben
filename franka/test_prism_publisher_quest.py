"""test_prism_publisher_quest.py — Meta Quest controller teleop for PrismPublisher.

Stream Quest controller pose over USB-ADB (via the rail-berkeley `oculus_reader`
library), map it to an EE target in panda_link0 frame, and feed our existing
PrismPublisher.

Buttons (right hand controller, default):
  Trigger (held)  : DEADMAN SWITCH — publisher follows the controller only
                    while the trigger is held. Release → target frozen.
  Grip            : Set ORIGIN — current controller pose becomes the (0, 0, 0)
                    reference for subsequent motion. Press once at episode start.
  A (one-shot)    : Toggle demo recording (stub for now; logs only)
  B (one-shot)    : Slew home — return EE to the initial pose at startup
  Joystick click  : Force shutdown (we treat right-stick click like Ctrl-C)

Modes (--mode):
  quest      : real Quest controller input (default — needs ADB + oculus_reader)
  mock       : synthetic small-circle motion of the controller, useful for
               testing the mapping + publisher integration without hardware
  hold       : ignore controller, just publish current target (smoke test)

Setup requirement (one-time):
  1. Quest in dev mode, USB connected, "Allow USB debugging" approved on headset
  2. `sudo apt install android-tools-adb`
  3. `pip install git+https://github.com/rail-berkeley/oculus_reader.git`
  4. `adb devices` should list the Quest serial

Run AFTER:
  source ~/franka_ws/setup_franka.sh
  python ~/franka_ws/hello_robot.py                                  # verify read
  python ~/franka_ws/test_prism_publisher_mock.py --mode hold        # verify publish
  # then this script:
  python ~/franka_ws/test_prism_publisher_quest.py --mode mock       # no robot motion
  python ~/franka_ws/test_prism_publisher_quest.py --mode quest      # real Quest
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from prism_publisher import (
    PrismPublisher,
    RateLimit,
    WorkspaceBox,
    ROBOT_MODE_MOVE,
    ROBOT_MODE_REFLEX,
    _rotation_matrix_to_quat,
)


# =============================================================================
# Franka canonical home pose (from joint config [0, -π/4, 0, -3π/4, 0, π/2, π/4])
# Measured EE position from hello_robot.py on this lab's Franka. Override via
# --home-xyz CLI flag if your lab's robot has a different effector tool offset.
# =============================================================================
FRANKA_HOME_XYZ = np.array([0.306, 0.000, 0.484], dtype=np.float64)

# Reuse the slew helper from the keyboard teleop
from test_prism_publisher_keyboard import slew_to


# =============================================================================
# Coordinate-frame mapping
# =============================================================================

@dataclass
class FrameMap:
    """Mapping from Quest headset frame to Franka panda_link0 frame.

    Defaults assume the operator is STANDING IN FRONT OF the robot, holding
    the right controller, looking at the robot:

      Quest  +x (right of operator)   →  Franka -y (operator's right == robot's right side)
      Quest  +y (up)                   →  Franka +z (up)
      Quest  -z (away from operator,
                 toward robot)         →  Franka +x (forward, away from base)

    The matrix below implements that. If the operator stands DIFFERENTLY
    relative to the robot, swap signs / axes via --frame-map.
    """
    R: np.ndarray = field(default_factory=lambda: np.array([
        # row = Franka axis, column = Quest axis
        [ 0.0,  0.0, -1.0],   # Franka_x =      -Quest_z
        [-1.0,  0.0,  0.0],   # Franka_y = -Quest_x
        [ 0.0,  1.0,  0.0],   # Franka_z =  Quest_y
    ], dtype=np.float64))
    scale: float = 1.0   # 1.0 = 1 cm controller motion → 1 cm EE motion

    def quest_delta_to_franka_delta(self, delta_quest_xyz: np.ndarray) -> np.ndarray:
        """Map a 3-vec position delta in Quest frame to panda_link0 frame."""
        return self.R @ (np.asarray(delta_quest_xyz) * self.scale)


# =============================================================================
# Quest input sources (real + mock)
# =============================================================================

@dataclass
class ControllerState:
    """Snapshot of right-hand controller state."""
    pose: np.ndarray         # 4x4 transform (headset frame)
    trigger: float           # [0..1]
    grip: float              # [0..1]
    grip_just_pressed: bool       # rising edge of grip > 0.5
    button_a_just_pressed: bool   # rising edge
    button_b_just_pressed: bool   # rising edge
    rj_just_pressed: bool         # right joystick click rising edge — soft reset
    valid: bool              # False if controller not detected


class QuestSource:
    """Abstract source of right-hand controller state."""
    def read(self) -> ControllerState:
        raise NotImplementedError

    def shutdown(self):
        pass


class OculusReaderSource(QuestSource):
    """Real Quest controller via oculus_reader / ADB."""

    APK_PACKAGE = "com.rail.oculus.teleop"

    def __init__(self, hand: str = "r", force_restart_apk: bool = True):
        try:
            from oculus_reader import OculusReader
        except ImportError as e:
            raise RuntimeError(
                "oculus_reader not installed. Install with:\n"
                "  pip install git+https://github.com/rail-berkeley/oculus_reader.git"
            ) from e

        self.hand = hand

        # The streaming APK on the Quest occasionally gets stuck in a state
        # where it has a PID but isn't pushing any controller data
        # (transforms == {}, buttons == {}). Force-restart it on every session
        # init to dodge this — adds ~2 s of startup latency but is reliable.
        if force_restart_apk:
            self._restart_apk()

        self.reader = OculusReader()
        # OculusReader.run() launches an ADB-streaming thread internally
        self.reader.run()
        time.sleep(2.0)  # let the ADB connection + APK warm up

        self._prev_a = False
        self._prev_b = False
        self._prev_rj = False

    def _restart_apk(self):
        """Force-stop and re-launch the controller-streaming APK on the Quest."""
        import subprocess
        try:
            subprocess.run(
                ["adb", "shell", "am", "force-stop", self.APK_PACKAGE],
                check=False, timeout=5, capture_output=True)
            subprocess.run(
                ["adb", "shell", "am", "start", "-n",
                 f"{self.APK_PACKAGE}/.MainActivity"],
                check=False, timeout=5, capture_output=True)
            time.sleep(1.0)
        except Exception as e:
            print(f"[oculus_reader] APK restart raised {e!r} — continuing anyway")

    def read(self) -> ControllerState:
        # oculus_reader API: get_transformations_and_buttons() → (transforms, buttons)
        # transforms: {'r': 4x4, 'l': 4x4} or empty {}
        # buttons keys vary by firmware/version; we read defensively
        transforms, buttons = self.reader.get_transformations_and_buttons()

        if not transforms or self.hand not in transforms:
            return ControllerState(
                pose=np.eye(4), trigger=0.0, grip=0.0,
                grip_just_pressed=False,
                button_a_just_pressed=False, button_b_just_pressed=False,
                rj_just_pressed=False,
                valid=False)

        pose = np.asarray(transforms[self.hand], dtype=np.float64)

        # Read buttons defensively — different oculus_reader versions use
        # different key naming conventions
        if self.hand == "r":
            trigger_key = "rightTrig"
            grip_key    = "rightGrip"
            a_key, b_key = "A", "B"
        else:
            trigger_key = "leftTrig"
            grip_key    = "leftGrip"
            a_key, b_key = "X", "Y"

        # Trigger and grip are typically tuples like (0.0,) or floats
        trig_val = buttons.get(trigger_key, 0.0)
        grip_val = buttons.get(grip_key, 0.0)
        if isinstance(trig_val, (list, tuple)):
            trig_val = trig_val[0] if trig_val else 0.0
        if isinstance(grip_val, (list, tuple)):
            grip_val = grip_val[0] if grip_val else 0.0

        # Buttons (booleans) — some versions use lowercase, some bool-typed
        a_now = bool(buttons.get(a_key, False))
        b_now = bool(buttons.get(b_key, False))
        rj_now = bool(buttons.get("RJ" if self.hand == "r" else "LJ", False))
        grip_now = float(grip_val) > 0.5

        a_edge = a_now and not self._prev_a
        b_edge = b_now and not self._prev_b
        rj_edge = rj_now and not getattr(self, "_prev_rj", False)
        grip_edge = grip_now and not getattr(self, "_prev_grip", False)
        self._prev_a, self._prev_b = a_now, b_now
        self._prev_rj, self._prev_grip = rj_now, grip_now

        return ControllerState(
            pose=pose,
            trigger=float(trig_val),
            grip=float(grip_val),
            grip_just_pressed=grip_edge,
            button_a_just_pressed=a_edge,
            button_b_just_pressed=b_edge,
            rj_just_pressed=rj_edge,
            valid=True)

    def shutdown(self):
        try:
            self.reader.stop()
        except Exception:
            pass


class MockQuestSource(QuestSource):
    """Synthetic source — slow-circle motion for testing without hardware."""

    def __init__(self, hold_trigger: bool = True):
        self.t0 = time.time()
        # Pretend trigger is held down so the publisher follows the synthetic
        # motion (otherwise we'd freeze and nothing visible happens)
        self._trigger = 1.0 if hold_trigger else 0.0
        self._grip_armed = False

    def read(self) -> ControllerState:
        t = time.time() - self.t0
        # ±2 cm circle at 0.2 Hz in Quest's xy plane
        pose = np.eye(4)
        pose[0, 3] = 0.02 * np.cos(2 * np.pi * 0.2 * t)
        pose[1, 3] = 0.02 * np.sin(2 * np.pi * 0.2 * t)
        # First call: pretend grip just got pressed to set origin
        grip = 1.0 if not self._grip_armed and t < 0.2 else 0.0
        if t > 0.2:
            self._grip_armed = True
        return ControllerState(
            pose=pose, trigger=self._trigger, grip=grip,
            grip_just_pressed=False,
            button_a_just_pressed=False, button_b_just_pressed=False,
            rj_just_pressed=False,
            valid=True)


# =============================================================================
# Teleop loop
# =============================================================================

class QuestTeleop:
    def __init__(
        self,
        pub: PrismPublisher,
        source: QuestSource,
        frame_map: FrameMap,
        init_xyz: np.ndarray,
        init_quat: np.ndarray,
        loop_hz: float = 60.0,
        verbose: bool = True,
    ):
        self.pub = pub
        self.source = source
        self.frame_map = frame_map
        self.init_xyz = init_xyz.copy()
        self.init_quat = init_quat.copy()
        self.loop_hz = loop_hz
        self.verbose = verbose

        # State
        self._origin_quest_xyz: Optional[np.ndarray] = None  # set when grip pressed
        self._origin_franka_xyz: Optional[np.ndarray] = None
        self._recording: bool = False
        self._n_recorded_episodes: int = 0
        self._last_log_t: float = 0.0

    def run(self, duration: float = 600.0):
        period = 1.0 / self.loop_hz
        t0 = time.time()

        if self.verbose:
            print(f"\n[quest] teleop running (max {duration:.0f}s). "
                  f"Hold TRIGGER to follow; press GRIP to set origin.")

        try:
            while time.time() - t0 < duration:
                cs = self.source.read()
                if not cs.valid:
                    if self.verbose and time.time() - self._last_log_t > 1.0:
                        print(f"\r[quest] no controller detected — pick up the right hand controller", end="", flush=True)
                        self._last_log_t = time.time()
                    time.sleep(period)
                    continue

                # ── Origin calibration (only on grip rising-edge) ──
                if cs.grip_just_pressed:
                    self._set_origin_from(cs.pose)

                # ── A: re-set "home" to robot's CURRENT EE pose ──
                if cs.button_a_just_pressed:
                    self._set_home_to_current_ee()

                # ── B: home (full slew back to initial pose) — also auto-sets
                #    origin so trigger works immediately afterward
                if cs.button_b_just_pressed:
                    self._home(controller_pose=cs.pose)

                # ── RJ click: soft reset (target → current robot, clear origin) ──
                if cs.rj_just_pressed:
                    self._soft_reset()

                # ── Trigger: deadman; only follow when held ──
                if cs.trigger > 0.5 and self._origin_quest_xyz is not None:
                    self._update_target_from(cs.pose)

                # Periodic status line
                if time.time() - self._last_log_t >= 0.5:
                    self._last_log_t = time.time()
                    self._log_status(cs)

                time.sleep(period)
        except KeyboardInterrupt:
            print("\n[quest] Ctrl-C")

    def _set_origin_from(self, pose: np.ndarray):
        """Bind the controller's CURRENT pose to Franka's INITIAL home pose.

        Pressing grip = "controller-where-it-is-right-now is the new (0,0,0),
        mapped to Franka's default home pose (where the robot was when we
        launched this script)."

        This makes grip a clean reset operation regardless of where the robot
        has drifted to. The robot does NOT move here; it'll only move once
        trigger is held — at which point the target slews from the current
        position back to init_xyz (rate-limited), and then tracks controller
        delta from there.
        """
        new_origin = pose[:3, 3].copy()
        if self._origin_quest_xyz is None or \
           not np.allclose(new_origin, self._origin_quest_xyz, atol=1e-6):
            self._origin_quest_xyz = new_origin
            self._origin_franka_xyz = self.init_xyz.copy()  # ALWAYS Franka home
            if self.verbose:
                print(f"\n[quest] origin set: quest=({new_origin[0]:+.3f},"
                      f" {new_origin[1]:+.3f}, {new_origin[2]:+.3f})  "
                      f"→ franka HOME ({self.init_xyz[0]:+.3f}, "
                      f"{self.init_xyz[1]:+.3f}, {self.init_xyz[2]:+.3f})")

    def _update_target_from(self, controller_pose: np.ndarray):
        controller_xyz = controller_pose[:3, 3]
        delta_quest = controller_xyz - self._origin_quest_xyz
        delta_franka = self.frame_map.quest_delta_to_franka_delta(delta_quest)
        target_xyz = self._origin_franka_xyz + delta_franka
        # Quat fixed at initial pose (PushT doesn't need rotation)
        self.pub.set_target_pose(target_xyz, self.init_quat)

    def _toggle_recording(self):
        # NOTE: temporarily unbound — A button now does _set_home_to_current_ee
        # below. Once we add real h5 demo recording, re-bind this to a
        # different button (likely RThU or wire a left-hand button).
        self._recording = not self._recording
        if self._recording:
            self._n_recorded_episodes += 1
            print(f"\n[quest] ▶ START recording demo #{self._n_recorded_episodes}")
        else:
            print(f"\n[quest] ⏹ STOP  recording demo #{self._n_recorded_episodes}")

    def _set_home_to_current_ee(self):
        """Replace the 'home' pose with wherever the robot's EE is right now.

        Use this when init_xyz captured at script start was a bad home
        (e.g. robot was in some leftover pose from a previous test).
        Workflow: use trigger to move robot to a good PushT home pose →
        release trigger → press A. From then on, grip / B / auto-home all
        target this new pose.
        """
        ee = self.pub.get_robot_ee()
        if ee is None:
            print("\n[quest] set-home: no FrankaState — skipping")
            return
        new_xyz, new_q = ee
        self.init_xyz = new_xyz.copy()
        self.init_quat = new_q.copy()
        # Also clear origin so user must press grip again before next teleop
        # (otherwise the old origin would map back to old home).
        self._origin_quest_xyz = None
        self._origin_franka_xyz = None
        print(f"\n[quest] new HOME set to current EE: "
              f"({new_xyz[0]:+.3f}, {new_xyz[1]:+.3f}, {new_xyz[2]:+.3f}); "
              f"press GRIP to re-anchor controller")

    def _home(self, controller_pose: np.ndarray | None = None):
        """Slew the robot back to init_xyz, then auto-set origin to the
        controller's current pose (so user can press trigger immediately
        after B without needing to also press grip).
        """
        print("\n[quest] homing back to initial pose...")
        outcome = slew_to(self.pub, self.init_xyz, self.init_quat)
        print(f"[quest] home: {outcome}")
        # Auto-bind origin to current controller pose → init_xyz so trigger
        # works right after B without needing a grip press
        if controller_pose is not None:
            self._origin_quest_xyz = controller_pose[:3, 3].copy()
            self._origin_franka_xyz = self.init_xyz.copy()
            print(f"[quest] origin auto-set after home → controller-here ↔ HOME")
        else:
            self._origin_quest_xyz = None
            self._origin_franka_xyz = None

    def _soft_reset(self):
        """Snap published target onto the robot's current EE pose, clear origin.

        Use this when:
          - the target has drifted far from the robot (large err_mm in status)
          - WS_clip is firing repeatedly because target is past workspace edge
          - the controller / Quest tracking has glitched and you want a clean
            restart without slewing the robot all the way back to home.

        Does NOT move the robot — target snaps to current EE = controller "stays
        where it is".
        """
        ok = self.pub.reset_target_to_current()
        # Clear origin so user must press grip again before the next motion
        self._origin_quest_xyz = None
        self._origin_franka_xyz = None
        print(f"\n[quest] soft reset: target snapped to current robot EE "
              f"({'OK' if ok else 'FAILED — no robot state'}); "
              f"press GRIP to re-set origin before next teleop")

    def _log_status(self, cs: ControllerState):
        target_xyz, _ = self.pub.get_current_target()
        ee = self.pub.get_robot_ee()
        if ee is None:
            return
        robot_xyz, _ = ee
        err_mm = float(np.linalg.norm(target_xyz - robot_xyz)) * 1000

        flags = []
        flags.append("●REC" if self._recording else "    ")
        flags.append("◢DEAD" if cs.trigger > 0.5 else "◇safe")
        flags.append("ORG" if self._origin_quest_xyz is not None else "no-org")

        sys.stdout.write(
            f"\r[quest] target=({target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}) "
            f"err={err_mm:5.1f}mm  trig={cs.trigger:.2f} grip={cs.grip:.2f}  "
            f"{'  '.join(flags)}     "
        )
        sys.stdout.flush()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quest", "mock", "hold"], default="quest")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Controller-to-EE scale factor (1.0 = 1 cm → 1 cm)")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--max-translation", type=float, default=0.02,
                    help="Per-call rate limit (m); raise if scale > 1")
    ap.add_argument("--home-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=None,
                    help=f"Override home pose XYZ in panda_link0 (meters). "
                         f"Default = canonical Franka home {tuple(FRANKA_HOME_XYZ)}.")
    ap.add_argument("--init-from-current", action="store_true",
                    help="Use the robot's CURRENT EE pose as 'home' instead of canonical Franka home. "
                         "Backward-compatible behavior; B button homes to wherever robot was at script start.")
    args = ap.parse_args()

    print(f"[quest] mode={args.mode}  scale={args.scale}  "
          f"max_translation={args.max_translation*1000:.0f}mm/call")

    # Build the input source
    if args.mode == "quest":
        try:
            source = OculusReaderSource(hand="r")
        except RuntimeError as e:
            print(f"\n❌ {e}")
            print("\nFalling back to instructions:")
            print("  1. Connect Quest via USB; approve 'Allow USB debugging' on headset")
            print("  2. Verify with: adb devices")
            print("  3. Install: pip install git+https://github.com/rail-berkeley/oculus_reader.git")
            return 2
    elif args.mode == "mock":
        source = MockQuestSource(hold_trigger=True)
    else:  # hold
        source = MockQuestSource(hold_trigger=False)  # never follows

    pub = PrismPublisher(
        workspace=WorkspaceBox(
            x_min=0.20, x_max=0.65,
            y_min=-0.30, y_max=0.30,
            z_min=0.10, z_max=0.70,
        ),
        rate_limit=RateLimit(
            max_translation_per_update=args.max_translation,
            max_rotation_per_update=0.10,
        ),
        verbose=True,
    )
    time.sleep(1.0)

    cur_target_xyz, cur_target_q = pub.get_current_target()
    state = pub.get_robot_state()
    if state is None:
        print("\n[quest] ⚠ no FrankaState received yet — is the lab stack running?")
        pub.shutdown()
        source.shutdown()
        return 3

    # ── Pick the "home" pose that B / grip / Ctrl-C will target ──
    if args.home_xyz is not None:
        init_xyz = np.array(args.home_xyz, dtype=np.float64)
        home_src = "--home-xyz CLI"
    elif args.init_from_current:
        init_xyz = cur_target_xyz.copy()
        home_src = "current robot EE (--init-from-current)"
    else:
        init_xyz = FRANKA_HOME_XYZ.copy()
        home_src = "canonical Franka home (default)"

    # Quaternion: use the current robot EE quat (sensible default for
    # whatever orientation the robot is configured with — gripper down etc.)
    init_q = cur_target_q.copy()

    print(f"[quest] current robot EE = ({cur_target_xyz[0]:+.3f},"
          f" {cur_target_xyz[1]:+.3f}, {cur_target_xyz[2]:+.3f})")
    print(f"[quest] HOME pose         = ({init_xyz[0]:+.3f},"
          f" {init_xyz[1]:+.3f}, {init_xyz[2]:+.3f})  [{home_src}]")
    if not np.allclose(init_xyz, cur_target_xyz, atol=0.01):
        print(f"[quest] note: current robot EE differs from HOME — pressing B / Ctrl-C "
              f"will slew the robot to HOME.")

    teleop = QuestTeleop(
        pub=pub, source=source,
        frame_map=FrameMap(scale=args.scale),
        init_xyz=init_xyz, init_quat=init_q,
        loop_hz=60.0, verbose=True,
    )

    try:
        teleop.run(duration=args.duration)
    finally:
        # Auto-home before shutdown
        try:
            cur_xyz, _ = pub.get_current_target()
            if float(np.linalg.norm(cur_xyz - init_xyz)) > 0.005:
                print("\n[quest] auto-homing back to initial pose...")
                outcome = slew_to(pub, init_xyz, init_q)
                print(f"[quest] auto-home: {outcome}")
        except Exception as e:
            print(f"[quest] auto-home raised: {e!r}")
        pub.shutdown()
        source.shutdown()
        print("[quest] shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
