#!/usr/bin/env bash
# Source this file in any shell that needs to talk to the Franka via ROS.
#   $ source ~/franka_ws/setup_franka.sh
#
# Prereq: connect to the frankaros_5g WiFi network and manually set the
# IPv4 of that interface to match $ROS_HOSTNAME below (e.g., 192.168.1.30).

# 1. Activate the RoboStack ROS1 Noetic conda env
source /home/yuhai/miniforge3/etc/profile.d/conda.sh
conda activate franka

# 2. Source the catkin workspace (so franka_msgs etc. are importable)
source /home/yuhai/franka_ws/devel/setup.bash

# 3. ROS networking — control PC at 192.168.1.10 runs roscore; we are a client.
export ROS_MASTER_URI=http://192.168.1.10:11311
export ROS_HOSTNAME=192.168.1.30   # CHANGE to a free .30+ IP your WiFi adapter is set to

echo "[franka] ROS env ready."
echo "  ROS_MASTER_URI = $ROS_MASTER_URI"
echo "  ROS_HOSTNAME   = $ROS_HOSTNAME"
echo "  python rospy   = $(python -c 'import rospy; print(rospy.__file__)' 2>/dev/null || echo MISSING)"
echo
echo "Sanity checks once connected to frankaros_5g WiFi:"
echo "  rostopic list                    # should include /franka_state_controller/franka_states"
echo "  rostopic echo -n1 /franka_state_controller/franka_states"
echo "  python ~/franka_ws/hello_robot.py"
