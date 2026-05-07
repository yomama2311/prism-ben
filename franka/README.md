# Franka real-robot scripts

These files run from `~/franka_ws/` (the catkin workspace), but live here
under version control. To use, either:

```bash
# option A: copy into ~/franka_ws/ (the catkin workspace home)
cp franka/*.py franka/*.sh ~/franka_ws/

# option B: run directly from this directory after sourcing the env
source ~/franka_ws/setup_franka.sh
python franka/test_prism_publisher_mock.py
```

## Files

| File | Purpose | Risk level |
|---|---|---|
| `setup_franka.sh` | Sources conda env + catkin workspace + sets `ROS_MASTER_URI` and `ROS_HOSTNAME` | none |
| `hello_robot.py` | Read-only — subscribes to `/franka_state_controller/franka_states` and prints state. Cannot move the robot. | none |
| `mock_franka_publisher.py` | Local-loopback mock that publishes synthetic FrankaState messages on a local roscore. For testing subscribers without the lab. | none |
| `prism_publisher.py` | The real-robot publisher: 30 Hz target pose to `/cartesian_impedance_example_controller/equilibrium_pose`, ErrorRecoveryAction on reflex, workspace clip + per-call rate limit. The PRISM-MPPI loop will call `set_target_pose()` on this. | **CAN MOVE THE ROBOT** if `set_target_pose()` is invoked |
| `test_prism_publisher_mock.py` | Drives `PrismPublisher` with synthetic targets in three modes: `hold` (no motion), `hover` (±2.5 mm wobble), `line` (±2.5 cm slide). Default is `hold`. | `hold` is no-motion; `hover`/`line` move the robot |

## Recommended first-run order at the lab

```bash
source ~/franka_ws/setup_franka.sh
# 1. confirm read works (no risk)
python ~/franka_ws/hello_robot.py

# 2. spin up the publisher in 'hold' mode (no motion expected)
python ~/franka_ws/test_prism_publisher_mock.py --mode hold

# 3. only after #2 looks correct, with E-stop reachable:
python ~/franka_ws/test_prism_publisher_mock.py --mode hover
```

`hold` mode publishes `target = current EE pose` at 30 Hz for ~12 s. The robot
should not move (cartesian impedance pulls it toward where it already is).
This verifies:
- 30 Hz publish thread alive
- ErrorRecoveryAction action server reachable
- FrankaState subscriber working
- pose_lock / reflex_lock not deadlocking

If `hold` works, `hover` is the first test that meaningfully exercises the
controller's tracking loop (~5 mm peak-to-peak in y).

## Architectural notes

`PrismPublisher` is the real-robot equivalent of the simulator-side
`solver.set_prior(mu, sigma)` flow. Where the simulator just feeds
the planner output back to the env's `step()`, the real-robot side has to:

1. Maintain a target pose at 30 Hz (Xiao Hu's lab pattern; if we drop below
   this rate, the controller treats it as a stale target).
2. Detect `robot_mode == REFLEX (4)` and send `ErrorRecoveryAction` (Xiao Hu's
   keyboard script does NOT do this; we do).
3. Reset the published target to the robot's current EE pose **before**
   clearing the reflex flag, so the controller doesn't snap back to a stale
   pre-collision target.
4. Clip every target to a workspace box and rate-limit the per-call delta
   (cartesian impedance produces force spikes if the target jumps far past
   current EE pose).

The PRISM-MPPI integration (next step, not yet implemented) will call
`set_target_pose(xyz, quat_xyzw)` from a separate `PrismHeadPolicy`-equivalent
class that converts MPPI's normalized action space to base-frame EE targets.
