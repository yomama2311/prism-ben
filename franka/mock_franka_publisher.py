"""mock_franka_publisher.py — fake /franka_state_controller/franka_states publisher.

Used for local-loopback testing of subscribers (e.g. hello_robot.py) without
needing a real Franka or the lab's control PC. Publishes a synthetic
FrankaState at 30 Hz with sinusoidal joint motion.

Usage:
  # in one shell:
  roscore &
  # in another:
  python ~/franka_ws/mock_franka_publisher.py
  # in a third:
  python ~/franka_ws/hello_robot.py
"""
from __future__ import annotations

import math
import time

import rospy
from franka_msgs.msg import FrankaState, Errors


def make_state(t: float) -> FrankaState:
    """Build a plausible-looking FrankaState. t = seconds since start."""
    msg = FrankaState()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "panda_link0"

    # 7 joints, each oscillating with a different phase / amplitude
    msg.q     = [0.3 * math.sin(t + i * 0.4) for i in range(7)]
    msg.q_d   = [0.3 * math.sin(t + i * 0.4) for i in range(7)]
    msg.dq    = [0.3 * math.cos(t + i * 0.4) for i in range(7)]
    msg.dq_d  = [0.3 * math.cos(t + i * 0.4) for i in range(7)]
    msg.ddq_d = [0.0] * 7
    msg.theta  = list(msg.q)
    msg.dtheta = list(msg.dq)
    msg.tau_J     = [0.0] * 7
    msg.dtau_J    = [0.0] * 7
    msg.tau_J_d   = [0.0] * 7
    msg.joint_collision = [0.0] * 7
    msg.joint_contact   = [0.0] * 7
    msg.tau_ext_hat_filtered = [0.0] * 7

    # EE pose (column-major 4x4) — slowly oscillating around a nominal grasp pose
    x = 0.5 + 0.05 * math.sin(t)
    y = 0.0 + 0.05 * math.cos(t)
    z = 0.5
    # Identity rotation; column-major
    msg.O_T_EE   = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  x, y, z, 1]
    msg.O_T_EE_d = list(msg.O_T_EE)
    msg.O_T_EE_c = list(msg.O_T_EE)
    msg.F_T_EE   = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0.107, 1]
    msg.F_T_NE   = list(msg.F_T_EE)
    msg.NE_T_EE  = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    msg.EE_T_K   = list(msg.NE_T_EE)

    # Cartesian state — zero contact forces / collisions in the mock
    msg.cartesian_collision = [0.0] * 6
    msg.cartesian_contact   = [0.0] * 6
    msg.K_F_ext_hat_K       = [0.0] * 6
    msg.O_F_ext_hat_K       = [0.0] * 6
    msg.O_dP_EE_d = [0.0] * 6
    msg.O_dP_EE_c = [0.0] * 6
    msg.O_ddP_EE_c = [0.0] * 6
    msg.O_ddP_O   = [0.0] * 3
    msg.elbow      = [0.0, 0.0]
    msg.elbow_d    = [0.0, 0.0]
    msg.elbow_c    = [0.0, 0.0]
    msg.delbow_c   = [0.0, 0.0]
    msg.ddelbow_c  = [0.0, 0.0]

    # Mass / inertia — plausible Panda numbers
    msg.m_ee     = 0.73
    msg.F_x_Cee  = [0.0, 0.0, 0.05]
    msg.I_ee     = [0.001, 0, 0,  0, 0.0025, 0,  0, 0, 0.0017]
    msg.m_load   = 0.0
    msg.F_x_Cload = [0.0] * 3
    msg.I_load   = [0.0] * 9
    msg.m_total  = msg.m_ee
    msg.F_x_Ctotal = list(msg.F_x_Cee)
    msg.I_total  = list(msg.I_ee)

    msg.time = t
    msg.control_command_success_rate = 1.0
    msg.robot_mode = FrankaState.ROBOT_MODE_IDLE
    msg.current_errors    = Errors()
    msg.last_motion_errors = Errors()
    return msg


def main() -> None:
    rospy.init_node("mock_franka_publisher", anonymous=True)
    pub = rospy.Publisher(
        "/franka_state_controller/franka_states",
        FrankaState,
        queue_size=1,
    )
    rate = rospy.Rate(30)  # 30 Hz, similar to a real Franka's state rate
    print("[mock_franka_publisher] publishing FrankaState at 30 Hz on "
          "/franka_state_controller/franka_states. Ctrl-C to stop.")

    t0 = time.time()
    while not rospy.is_shutdown():
        msg = make_state(time.time() - t0)
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
