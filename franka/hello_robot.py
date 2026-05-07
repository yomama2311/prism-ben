"""hello_robot.py — read-only sanity check that we can talk to the Franka via ROS.

Run AFTER:
  1. `source ~/franka_ws/setup_franka.sh`
  2. connecting to frankaros_5g WiFi with the IP set to match $ROS_HOSTNAME
  3. confirming roscore is running on the control PC (`rostopic list` shows topics)

This script ONLY subscribes — it never publishes a command, so it cannot move
the robot or trigger any safety action. Safe to run as a first test.

Output: prints a few seconds of FrankaState fields (joint positions, EE pose,
errors, robot mode), then exits cleanly.
"""
from __future__ import annotations

import sys
import time

import rospy
from franka_msgs.msg import FrankaState


ROBOT_MODES = {
    0: "OTHER",
    1: "IDLE",
    2: "MOVE",
    3: "GUIDING",
    4: "REFLEX",
    5: "USER_STOPPED",
    6: "AUTOMATIC_ERROR_RECOVERY",
}


class FrankaStateMonitor:
    def __init__(self):
        self.last: FrankaState | None = None
        self.count = 0

    def cb(self, msg: FrankaState) -> None:
        self.last = msg
        self.count += 1

    def summary(self) -> str:
        if self.last is None:
            return "(no messages yet)"
        m = self.last
        # O_T_EE is the 4x4 column-major homogeneous transform from base (O) to EE.
        ee_xyz = (m.O_T_EE[12], m.O_T_EE[13], m.O_T_EE[14])
        return (
            f"  q (joint pos) = [{', '.join(f'{x:+.3f}' for x in m.q)}]\n"
            f"  EE position    = ({ee_xyz[0]:+.3f}, {ee_xyz[1]:+.3f}, {ee_xyz[2]:+.3f})  m\n"
            f"  robot_mode     = {m.robot_mode} ({ROBOT_MODES.get(m.robot_mode, '?')})\n"
            f"  cmd success rate = {m.control_command_success_rate:.3f}\n"
            f"  any current error? {any_error(m.current_errors)}"
        )


def any_error(errors) -> bool:
    """Errors is a long bool struct from franka_msgs/Errors. Return True if any field set."""
    for slot in errors.__slots__:
        if getattr(errors, slot):
            return True
    return False


def main() -> int:
    rospy.init_node("hello_robot", anonymous=True, disable_signals=True)
    mon = FrankaStateMonitor()
    sub = rospy.Subscriber(
        "/franka_state_controller/franka_states",
        FrankaState,
        mon.cb,
        queue_size=1,
    )

    print("[hello_robot] waiting up to 5 s for first FrankaState message...")
    t_start = time.time()
    while mon.last is None and time.time() - t_start < 5.0:
        time.sleep(0.1)

    if mon.last is None:
        print("[hello_robot] ERROR: no messages received in 5 s.")
        print("  Check:")
        print("  - You are on frankaros_5g WiFi.")
        print(f"  - $ROS_MASTER_URI = {rospy.get_master().getUri()[1] if False else '???'}")
        print("    (run `echo $ROS_MASTER_URI` to verify it points at 192.168.1.10:11311).")
        print("  - WiFi IPv4 is manually set to match $ROS_HOSTNAME.")
        print("  - Lab control PC is on, running franka_control + roscore.")
        sub.unregister()
        return 1

    # Got first message — sample for 3 more seconds and report summary.
    print(f"[hello_robot] first message received after {time.time() - t_start:.2f} s. Sampling for 3 s...")
    time.sleep(3.0)

    print(f"\n[hello_robot] received {mon.count} FrankaState messages over ~3 s "
          f"(~{mon.count / 3:.0f} Hz)\n")
    print(mon.summary())
    sub.unregister()
    return 0


if __name__ == "__main__":
    sys.exit(main())
