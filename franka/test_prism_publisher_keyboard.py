"""test_prism_publisher_keyboard.py — keyboard teleop driving PrismPublisher.

Move the EE target via arrow keys + h/j (no rotation in this script):

  ↑ Up arrow    : +x  (forward, away from robot base)
  ↓ Down arrow  : -x  (backward, toward robot base)
  ← Left arrow  : -y  (one side; flip if your setup needs the other)
  → Right arrow : +y  (other side)
  j             : +z  (raise)
  h             : -z  (lower)
  r             : reset target to current robot EE pose
  ?             : show help again
  Esc / Ctrl-C  : exit cleanly

Step size: 1 cm per key press (use --step to change).

The PrismPublisher under the hood:
  - Publishes the target at 30 Hz on /equilibrium_pose
  - Clips every target to the workspace box
  - Rate-limits each set_target_pose call (default 2 cm)
  - Sends ErrorRecoveryAction on reflex; resets target to current EE on recover

Run AFTER:
  source ~/franka_ws/setup_franka.sh
  python ~/franka_ws/hello_robot.py                                # verify read
  python ~/franka_ws/test_prism_publisher_mock.py --mode hold      # verify publish
  python ~/franka_ws/test_prism_publisher_keyboard.py              # ← this script

⚠  This script COMMANDS MOTION. Make sure E-stop is reachable.
"""
from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

from prism_publisher import (
    PrismPublisher,
    RateLimit,
    WorkspaceBox,
    ROBOT_MODE_MOVE,
    ROBOT_MODE_REFLEX,
    ROBOT_MODE_USER_STOPPED,
    ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY,
)


# Arrow keys: terminals send one of TWO sequences depending on whether
# the application keypad mode is on:
#   normal mode      : \x1b[A   \x1b[B   \x1b[C   \x1b[D
#   application mode : \x1bOA   \x1bOB   \x1bOC   \x1bOD
# VSCode / Cursor / some SSH clients use application mode. We accept both.
KEY_UP    = ("\x1b[A", "\x1bOA")
KEY_DOWN  = ("\x1b[B", "\x1bOB")
KEY_RIGHT = ("\x1b[C", "\x1bOC")
KEY_LEFT  = ("\x1b[D", "\x1bOD")
KEY_CTRLC = "\x03"

# Terminal capability disable / re-enable sequences. Send these before/after
# raw mode so the terminal stops sending us focus events, mouse reports,
# bracketed-paste markers, etc. — all of which start with \x1b and would
# otherwise be misread as Esc / arrow keys.
TERM_DISABLE = (
    "\x1b[?1l"      # cursor-key application mode OFF (force standard \x1b[A form)
    "\x1b[?2004l"   # bracketed paste off
    "\x1b[?1000l"   # mouse click reporting off
    "\x1b[?1002l"   # mouse motion reporting off
    "\x1b[?1003l"   # all-mouse-events reporting off
    "\x1b[?1004l"   # focus in/out reporting off
    "\x1b[?1006l"   # SGR mouse mode off
)
TERM_ENABLE_RESTORE = ""  # we deliberately don't re-enable; assume defaults


def get_key(timeout: float = 0.1, esc_grace: float = 0.2) -> str:
    """Non-blocking read of one keystroke (including arrow-key sequences).

    Uses os.read() rather than sys.stdin.read() because Python's TextIOWrapper
    aggressively buffers stdin even in raw mode — and that buffer is invisible
    to select(), causing multi-byte escape sequences (e.g. \\x1b[A) to be
    fragmented across multiple get_key() calls. os.read on the underlying fd
    is unbuffered, so a single press of an arrow key returns all 3 bytes.
    """
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ""
    try:
        chunk = os.read(fd, 16)
    except OSError:
        return ""
    if not chunk:
        return ""
    s = chunk.decode("latin-1")  # identity byte → str map
    # If a slow terminal split the escape sequence across packets, give it
    # one more chance to deliver follow-up bytes.
    if s.startswith("\x1b") and len(s) < 3:
        r2, _, _ = select.select([sys.stdin], [], [], esc_grace)
        if r2:
            try:
                more = os.read(fd, 16)
                if more:
                    s += more.decode("latin-1")
            except OSError:
                pass
    return s


def print_help(step: float):
    print("\r\n" + "=" * 60 + "\r")
    print("  PRISM PUBLISHER — KEYBOARD TELEOP\r")
    print("=" * 60 + "\r\n")
    print("  ↑ / ↓   : +/- x  (forward / backward)\r")
    print("  ← / →   : -/+ y  (right / left side of base frame)\r")
    print("  j / h   : +/- z  (raise / lower)\r")
    print("  b       : slew back to the initial pose (home)\r")
    print("  r       : reset target to current robot EE (after reflex)\r")
    print("  ?       : show this help\r")
    print("  q / Ctrl-C : exit (auto-homes back to initial pose first)\r\n")
    print(f"  Step size       : {step*1000:.1f} mm per key press\r")
    print(f"  Rate limit / clip enforced by PrismPublisher\r")
    print(f"  Cartesian impedance is a low-pass filter — robot tracks smoothly\r")
    print("=" * 60 + "\r\n")


def slew_to(pub, target_xyz: np.ndarray, target_quat: np.ndarray,
            slew_timeout: float = 8.0, settle_timeout: float = 5.0,
            tol_target_xyz: float = 0.001, tol_robot_xyz: float = 0.008,
            tol_target_rot: float = 0.05, tol_robot_rot: float = 0.10,
            update_period: float = 0.1) -> str:
    """Smoothly slew the published target back to (target_xyz, target_quat),
    waiting for BOTH translation and rotation convergence.

    PrismPublisher.set_target_pose enforces per-call rate-limiting on both
    xyz (max_translation_per_update) and quaternion (max_rotation_per_update,
    SLERP-clamped). We call it repeatedly with the goal until both converge.

    tol_target_rot / tol_robot_rot are in radians (~3° / ~6° default).

    Returns: 'reached', 'reflex', 'slew_timeout', 'settle_timeout'.
    """
    from prism_publisher import _quat_angle

    # Phase 1: slew the TARGET to the goal pose (both xyz AND rotation)
    t0 = time.time()
    while time.time() - t0 < slew_timeout:
        st = pub.get_robot_state()
        if st is not None and st.robot_mode == ROBOT_MODE_REFLEX:
            return "reflex"
        cur_target_xyz, cur_target_q = pub.get_current_target()
        d_xyz = float(np.linalg.norm(target_xyz - cur_target_xyz))
        d_rot = float(_quat_angle(cur_target_q, target_quat))
        if d_xyz < tol_target_xyz and d_rot < tol_target_rot:
            break
        pub.set_target_pose(target_xyz, target_quat)
        time.sleep(update_period)
    else:
        return "slew_timeout"

    # Phase 2: wait for the ROBOT itself to converge (xyz AND rotation)
    t1 = time.time()
    while time.time() - t1 < settle_timeout:
        # Keep republishing the target so cartesian impedance keeps pulling
        pub.set_target_pose(target_xyz, target_quat)
        ee = pub.get_robot_ee()
        if ee is None:
            time.sleep(update_period); continue
        robot_xyz, robot_q = ee
        d_xyz = float(np.linalg.norm(target_xyz - robot_xyz))
        d_rot = float(_quat_angle(robot_q, target_quat))
        if d_xyz < tol_robot_xyz and d_rot < tol_robot_rot:
            return "reached"
        time.sleep(update_period)
    return "settle_timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.01,
                    help="EE target step in meters per key press (default 0.01 = 1 cm)")
    ap.add_argument("--max-translation", type=float, default=0.02,
                    help="Per-call rate limit in meters (default 0.02 = 2 cm; "
                         "must be >= --step or each press will be clamped)")
    ap.add_argument("--debug", action="store_true",
                    help="Print every received keystroke (raw bytes + hex) for diagnosis.")
    ap.add_argument("--no-home-on-exit", action="store_true",
                    help="Skip the automatic slew back to initial pose on exit.")
    args = ap.parse_args()

    step = float(args.step)
    if args.max_translation < args.step:
        print(f"⚠  warning: --max-translation ({args.max_translation*1000:.1f}mm) "
              f"< --step ({step*1000:.1f}mm) → each press will be rate-limit clamped.")

    print(f"\n[teleop] starting publisher (step={step*1000:.1f}mm, "
          f"max_translation={args.max_translation*1000:.1f}mm)...")
    print(f"[teleop] make sure E-stop is reachable.")

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

    init_xyz, init_q = pub.get_current_target()
    state = pub.get_robot_state()
    if state is None:
        print("[teleop] ⚠ no FrankaState received yet — is the lab stack running?")
        pub.shutdown()
        return 2

    print(f"\n[teleop] initial target = ({init_xyz[0]:+.3f}, {init_xyz[1]:+.3f}, {init_xyz[2]:+.3f}) m")
    print(f"[teleop] initial robot  = ({state.O_T_EE[12]:+.3f}, {state.O_T_EE[13]:+.3f}, {state.O_T_EE[14]:+.3f}) m")
    print(f"[teleop] mode = {state.robot_mode}\n")

    print_help(step)

    # Save terminal state — restore it on every exit path
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Disable focus reports / mouse / bracketed paste BEFORE going raw,
        # otherwise their \x1b[... sequences get misread as arrow keys / Esc.
        sys.stdout.write(TERM_DISABLE)
        sys.stdout.flush()

        tty.setraw(fd)
        while True:
            key = get_key(timeout=0.1)
            if not key:
                continue

            if args.debug:
                hexed = " ".join(f"{ord(c):02x}" for c in key)
                sys.stdout.write(f"[debug] key={repr(key)}  hex={hexed}\r\n")
                sys.stdout.flush()

            dx = dy = dz = 0.0
            label = None
            special = None

            if key in KEY_UP:
                dx = +step; label = f"+x  {step*1000:.1f}mm"
            elif key in KEY_DOWN:
                dx = -step; label = f"-x  {step*1000:.1f}mm"
            elif key in KEY_RIGHT:
                dy = +step; label = f"+y  {step*1000:.1f}mm"
            elif key in KEY_LEFT:
                dy = -step; label = f"-y  {step*1000:.1f}mm"
            elif key == "j":
                dz = +step; label = f"+z  {step*1000:.1f}mm"
            elif key == "h":
                dz = -step; label = f"-z  {step*1000:.1f}mm"
            elif key == "b":
                special = "home"
            elif key == "r":
                special = "reset"
            elif key == "?":
                special = "help"
            elif key in ("q", KEY_CTRLC):
                break
            else:
                # Unknown key (or stray escape sequence) — silently ignore
                continue

            # Handle special keys (need to print multi-line; restore tty)
            if special == "reset":
                ok = pub.reset_target_to_current()
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print(f"[teleop] reset target to current EE: {'OK' if ok else 'FAILED (no state)'}")
                tty.setraw(fd)
                continue
            elif special == "home":
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print(f"[teleop] homing back to initial pose "
                      f"({init_xyz[0]:+.3f}, {init_xyz[1]:+.3f}, {init_xyz[2]:+.3f})...")
                outcome = slew_to(pub, init_xyz, init_q)
                print(f"[teleop] home: {outcome}")
                tty.setraw(fd)
                continue
            elif special == "help":
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print_help(step)
                tty.setraw(fd)
                continue

            # ── Translation key — apply delta ──
            cur_xyz, cur_q = pub.get_current_target()
            new_xyz = cur_xyz + np.array([dx, dy, dz])
            info = pub.set_target_pose(new_xyz, cur_q)
            applied = info["applied_xyz"]

            # Status line
            cur_state = pub.get_robot_state()
            if cur_state is not None:
                cur_robot = np.array([cur_state.O_T_EE[12],
                                      cur_state.O_T_EE[13],
                                      cur_state.O_T_EE[14]])
                err = float(np.linalg.norm(applied - cur_robot)) * 1000
                if cur_state.robot_mode == ROBOT_MODE_MOVE:
                    mode_str = "MOVE"
                elif cur_state.robot_mode == ROBOT_MODE_REFLEX:
                    mode_str = "REFLEX!"
                elif cur_state.robot_mode == ROBOT_MODE_USER_STOPPED:
                    mode_str = "USER_STOPPED"
                elif cur_state.robot_mode == ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY:
                    mode_str = "RECOVERING"
                else:
                    mode_str = f"mode-{cur_state.robot_mode}"
            else:
                err = float("nan")
                mode_str = "?"

            cflags = []
            if info["workspace_clipped"]:
                cflags.append("WS_clip")
            if info["translation_clipped"]:
                cflags.append("RL_clip")
            cstr = "  [" + ",".join(cflags) + "]" if cflags else ""

            sys.stdout.write(
                f"{label:11s}  →  target=({applied[0]:+.3f}, {applied[1]:+.3f}, {applied[2]:+.3f})"
                f"  robot_err={err:5.1f}mm  {mode_str}{cstr}\r\n"
            )
            sys.stdout.flush()
    finally:
        # Restore terminal first so subsequent prints are line-buffered normally
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # Auto-home: slew back to the initial pose before shutting down.
        # The publisher must still be alive for slew_to to work, so we do
        # this BEFORE pub.shutdown(). If the user wants to skip homing,
        # they can pass --no-home-on-exit.
        if not getattr(args, "no_home_on_exit", False):
            try:
                cur_xyz, _ = pub.get_current_target()
                dist = float(np.linalg.norm(cur_xyz - init_xyz))
                if dist > 0.005:
                    print(f"\n[teleop] auto-homing back to initial pose "
                          f"({dist*1000:.0f} mm away)...")
                    outcome = slew_to(pub, init_xyz, init_q)
                    print(f"[teleop] auto-home: {outcome}")
                else:
                    print(f"\n[teleop] already at initial pose ({dist*1000:.1f} mm), skipping home.")
            except Exception as e:
                print(f"[teleop] auto-home raised: {e!r}")

        pub.shutdown()
        print("[teleop] shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
