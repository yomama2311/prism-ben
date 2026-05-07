"""prism_publisher.py — Real-robot Franka publisher for PRISM-MPPI.

Subscribes to robot state, monitors for reflex/errors, sends ErrorRecoveryAction
when needed, and publishes a target EE pose at 30 Hz to the cartesian impedance
controller. The PRISM-MPPI loop calls `set_target_pose()` to update the published
target.

Key invariants:
  - Publish never stops (30 Hz, separate daemon thread).
  - Until first `set_target_pose()` call, target = current EE pose at startup.
  - Reflex (mode 4) → send ErrorRecoveryAction → reset target to current EE pose
    BEFORE clearing the reflex flag, so the controller doesn't snap back to a
    stale target after recovery.
  - All targets are clipped to a configurable workspace box.
  - All targets are rate-limited (per-call max Δxyz, max Δrotation) to avoid
    Cartesian-impedance spikes.

Topology:

    [user / PRISM-MPPI loop]
            │ set_target_pose(xyz, quat)
            ▼
       [pose_lock]                          [state_lock]
       ┌──────┐                              ┌──────┐
       │ self.│                              │ self.│
       │_target│                             │_state│
       └──┬───┘                              └──┬───┘
          │                                     │
          ▼                       ┌─────────────┘
   ┌─────────────┐                │
   │ publish     │                ▼
   │ thread      │       ┌────────────────┐
   │ (30 Hz)     │       │ reflex thread  │
   └─────┬───────┘       │  monitor mode, │
         │               │  send recovery │
         ▼               └────────────────┘
   /equilibrium_pose
   (panda_link0 frame)

Run AFTER:
  source ~/franka_ws/setup_franka.sh
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import actionlib
import numpy as np
import rospy
from franka_msgs.msg import (
    ErrorRecoveryAction,
    ErrorRecoveryGoal,
    FrankaState,
)
from geometry_msgs.msg import PoseStamped


# Franka robot mode enum (from FrankaState.msg)
ROBOT_MODE_OTHER                    = 0
ROBOT_MODE_IDLE                     = 1
ROBOT_MODE_MOVE                     = 2
ROBOT_MODE_GUIDING                  = 3
ROBOT_MODE_REFLEX                   = 4
ROBOT_MODE_USER_STOPPED             = 5
ROBOT_MODE_AUTOMATIC_ERROR_RECOVERY = 6


@dataclass
class WorkspaceBox:
    """Cartesian limits in panda_link0 (base) frame, meters.

    Defaults are conservative. Override for your specific lab setup.
    The Franka home pose is around (0.306, -0.001, 0.483) — these limits
    keep the EE within ±20 cm of that in y, ±20 cm vertically, and a
    range of about 35-65 cm forward of the base.
    """
    x_min: float = 0.20
    x_max: float = 0.65
    y_min: float = -0.30
    y_max: float = 0.30
    z_min: float = 0.10
    z_max: float = 0.70


@dataclass
class RateLimit:
    """Per-call clamp on how far the target can move from its current value.

    Cartesian impedance does NOT teleport; it tracks via a virtual spring +
    damper. A target jump too far past current EE pose causes a force spike
    that can trip a reflex. We bound the per-update step here.
    """
    max_translation_per_update: float = 0.02   # 2 cm  per set_target_pose call
    max_rotation_per_update:    float = 0.10   # 0.1 rad per set_target_pose call


def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion in (x, y, z, w) order.

    Uses the standard Shepperd-style branch on the largest diagonal element
    to avoid numerical instability when w is small.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / max(np.linalg.norm(q), 1e-12)


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Shortest-arc angle between two unit quaternions (xyzw), in radians."""
    dot = abs(float(np.clip(np.dot(q1, q2), -1.0, 1.0)))
    return 2.0 * np.arccos(dot)


def _quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """SLERP from q1 to q2 by fraction t ∈ [0, 1]. Both inputs unit (xyzw)."""
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        # Nearly aligned — linear interpolate + renormalize
        q = q1 + t * (q2 - q1)
        return q / max(np.linalg.norm(q), 1e-12)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)
    a = np.sin((1.0 - t) * omega) / sin_omega
    b = np.sin(t * omega) / sin_omega
    return a * q1 + b * q2


def _o_t_ee_to_xyz_quat(o_t_ee) -> tuple[np.ndarray, np.ndarray]:
    """FrankaState.O_T_EE (column-major 16-vec) → (xyz, quat_xyzw)."""
    arr = np.asarray(o_t_ee, dtype=np.float64)
    xyz = arr[[12, 13, 14]].copy()
    R = np.array([
        [arr[0], arr[4], arr[8]],
        [arr[1], arr[5], arr[9]],
        [arr[2], arr[6], arr[10]],
    ])
    quat = _rotation_matrix_to_quat(R)
    return xyz, quat


class PrismPublisher:
    """Real-robot publisher for PRISM-MPPI on Franka over cartesian impedance."""

    def __init__(
        self,
        publish_rate_hz: float = 30.0,
        frame_id: str = "panda_link0",
        workspace: WorkspaceBox | None = None,
        rate_limit: RateLimit | None = None,
        equilibrium_topic: str = "/cartesian_impedance_example_controller/equilibrium_pose",
        state_topic: str = "/franka_state_controller/franka_states",
        recovery_action: str = "/franka_control/error_recovery",
        verbose: bool = True,
        node_name: str = "prism_publisher",
    ):
        self.publish_rate_hz = publish_rate_hz
        self.frame_id = frame_id
        self.workspace = workspace or WorkspaceBox()
        self.rate_limit = rate_limit or RateLimit()
        self.verbose = verbose

        # ROS init (idempotent — safe to call multiple times)
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True, disable_signals=True)

        # ── Publisher (cartesian impedance equilibrium pose) ──
        self._pub = rospy.Publisher(equilibrium_topic, PoseStamped, queue_size=1)

        # ── State subscriber ──
        self._state: FrankaState | None = None
        self._state_lock = threading.Lock()
        self._state_sub = rospy.Subscriber(
            state_topic, FrankaState, self._state_cb, queue_size=1)

        # ── Error recovery action client ──
        self._recovery_client = actionlib.SimpleActionClient(
            recovery_action, ErrorRecoveryAction)
        if self.verbose:
            self._log("waiting up to 5 s for error_recovery action server...")
        self._recovery_available = self._recovery_client.wait_for_server(
            rospy.Duration(5.0))
        if not self._recovery_available and self.verbose:
            rospy.logwarn(
                "[prism_publisher] error_recovery action server not reachable; "
                "reflex recovery will be DISABLED. Reflex will require manual "
                "reset on the lab control PC.")

        # ── Pose state ──
        self._pose_lock = threading.Lock()
        self._target = PoseStamped()
        self._target.header.frame_id = self.frame_id
        self._initialized = False
        self._init_target_from_current()

        # ── Reflex state ──
        self._reflex_pending = threading.Event()
        self._stop_event = threading.Event()

        # ── Threads ──
        self._pub_thread = threading.Thread(
            target=self._publish_loop, name="prism-publish", daemon=True)
        self._pub_thread.start()
        self._reflex_thread = threading.Thread(
            target=self._reflex_loop, name="prism-reflex", daemon=True)
        self._reflex_thread.start()

        if self.verbose:
            self._log(f"ready. publish_rate={publish_rate_hz:.0f}Hz, "
                      f"recovery_available={self._recovery_available}")

    # ────────────────────────── public API ──────────────────────────

    def set_target_pose(
        self,
        xyz: np.ndarray | tuple,
        quat_xyzw: np.ndarray | tuple,
    ) -> dict:
        """Update the target EE pose. Returns a dict with what got applied.

        - xyz: 3-vec in panda_link0 frame, meters (clipped to workspace box)
        - quat_xyzw: 4-vec, will be unit-normalized; rate-limited via SLERP

        Returns: {'applied_xyz', 'applied_quat', 'translation_clipped',
                  'rotation_clipped', 'workspace_clipped'}
        """
        xyz_in = np.asarray(xyz, dtype=np.float64).reshape(3)
        q_in = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        q_in = q_in / max(np.linalg.norm(q_in), 1e-12)

        # ── workspace clip ──
        ws = self.workspace
        x_clipped = float(np.clip(xyz_in[0], ws.x_min, ws.x_max))
        y_clipped = float(np.clip(xyz_in[1], ws.y_min, ws.y_max))
        z_clipped = float(np.clip(xyz_in[2], ws.z_min, ws.z_max))
        ws_clipped = (x_clipped != xyz_in[0]
                      or y_clipped != xyz_in[1]
                      or z_clipped != xyz_in[2])

        with self._pose_lock:
            cur_xyz = np.array([
                self._target.pose.position.x,
                self._target.pose.position.y,
                self._target.pose.position.z,
            ])
            cur_q = np.array([
                self._target.pose.orientation.x,
                self._target.pose.orientation.y,
                self._target.pose.orientation.z,
                self._target.pose.orientation.w,
            ])

            # ── translation rate limit ──
            target_xyz = np.array([x_clipped, y_clipped, z_clipped])
            d_xyz = target_xyz - cur_xyz
            d_norm = float(np.linalg.norm(d_xyz))
            translation_clipped = False
            if d_norm > self.rate_limit.max_translation_per_update:
                scale = self.rate_limit.max_translation_per_update / d_norm
                target_xyz = cur_xyz + d_xyz * scale
                translation_clipped = True

            # ── rotation rate limit (SLERP) ──
            angle = _quat_angle(cur_q, q_in)
            rotation_clipped = False
            if angle > self.rate_limit.max_rotation_per_update:
                t = self.rate_limit.max_rotation_per_update / angle
                target_q = _quat_slerp(cur_q, q_in, t)
                rotation_clipped = True
            else:
                target_q = q_in

            # commit
            self._target.pose.position.x = float(target_xyz[0])
            self._target.pose.position.y = float(target_xyz[1])
            self._target.pose.position.z = float(target_xyz[2])
            self._target.pose.orientation.x = float(target_q[0])
            self._target.pose.orientation.y = float(target_q[1])
            self._target.pose.orientation.z = float(target_q[2])
            self._target.pose.orientation.w = float(target_q[3])

        return {
            "applied_xyz": target_xyz.copy(),
            "applied_quat": target_q.copy(),
            "translation_clipped": translation_clipped,
            "rotation_clipped": rotation_clipped,
            "workspace_clipped": ws_clipped,
        }

    def get_current_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current target (xyz, quat_xyzw). Both numpy arrays."""
        with self._pose_lock:
            xyz = np.array([
                self._target.pose.position.x,
                self._target.pose.position.y,
                self._target.pose.position.z,
            ])
            q = np.array([
                self._target.pose.orientation.x,
                self._target.pose.orientation.y,
                self._target.pose.orientation.z,
                self._target.pose.orientation.w,
            ])
        return xyz, q

    def get_robot_state(self) -> FrankaState | None:
        """Return latest FrankaState, or None if no message received yet."""
        with self._state_lock:
            return self._state

    def get_robot_ee(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (xyz, quat_xyzw) of the robot's current EE pose, or None."""
        s = self.get_robot_state()
        if s is None:
            return None
        return _o_t_ee_to_xyz_quat(s.O_T_EE)

    def reset_target_to_current(self) -> bool:
        """Force target = current EE pose (e.g. after error recovery)."""
        ee = self.get_robot_ee()
        if ee is None:
            self._log("reset_target_to_current: no state yet, skipping", level="warn")
            return False
        xyz, q = ee
        with self._pose_lock:
            self._target.pose.position.x = float(xyz[0])
            self._target.pose.position.y = float(xyz[1])
            self._target.pose.position.z = float(xyz[2])
            self._target.pose.orientation.x = float(q[0])
            self._target.pose.orientation.y = float(q[1])
            self._target.pose.orientation.z = float(q[2])
            self._target.pose.orientation.w = float(q[3])
        return True

    def shutdown(self) -> None:
        """Stop publish thread and reflex thread cleanly."""
        self._stop_event.set()
        self._pub_thread.join(timeout=2.0)
        self._reflex_thread.join(timeout=2.0)
        if self.verbose:
            self._log("shutdown complete")

    # ────────────────────────── internals ──────────────────────────

    def _state_cb(self, msg: FrankaState) -> None:
        with self._state_lock:
            self._state = msg
        if msg.robot_mode == ROBOT_MODE_REFLEX and not self._reflex_pending.is_set():
            self._log(f"reflex detected (mode={msg.robot_mode}); requesting recovery",
                      level="warn")
            self._reflex_pending.set()

    def _reflex_loop(self) -> None:
        while not self._stop_event.is_set():
            triggered = self._reflex_pending.wait(timeout=0.5)
            if not triggered:
                continue
            if not self._recovery_available:
                rospy.logwarn(
                    "[prism_publisher] reflex pending but recovery client unavailable")
                self._reflex_pending.clear()
                continue

            self._log("sending ErrorRecoveryAction goal...")
            self._recovery_client.send_goal(ErrorRecoveryGoal())
            done = self._recovery_client.wait_for_result(rospy.Duration(5.0))
            if done:
                # Reset target FIRST (so controller doesn't snap back to old target)
                ok = self.reset_target_to_current()
                self._log(f"recovery done (target_reset={ok})")
            else:
                self._log("recovery did not complete in 5 s", level="warn")
            self._reflex_pending.clear()

    def _publish_loop(self) -> None:
        rate = rospy.Rate(self.publish_rate_hz)
        seq = 0
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            with self._pose_lock:
                self._target.header.stamp = rospy.Time.now()
                self._target.header.seq = seq
                msg = self._target
            self._pub.publish(msg)
            seq += 1
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _init_target_from_current(self) -> None:
        """Block briefly trying to discover the robot's current EE pose."""
        # 1) try the equilibrium topic — if any node already published one,
        #    that's the most current 'controller-aware' target and joining it
        #    is jerk-free.
        try:
            from rospy.exceptions import ROSException
        except ImportError:
            ROSException = Exception

        try:
            msg = rospy.wait_for_message(
                "/cartesian_impedance_example_controller/equilibrium_pose",
                PoseStamped,
                timeout=2.0,
            )
            with self._pose_lock:
                self._target = msg
                self._target.header.frame_id = self.frame_id
            self._initialized = True
            p = msg.pose.position
            self._log(f"init from /equilibrium_pose: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
            return
        except ROSException:
            pass
        except Exception as e:
            self._log(f"init via /equilibrium_pose failed: {e}", level="warn")

        # 2) try franka_states — read O_T_EE directly
        try:
            msg = rospy.wait_for_message(
                "/franka_state_controller/franka_states",
                FrankaState,
                timeout=2.0,
            )
            xyz, q = _o_t_ee_to_xyz_quat(msg.O_T_EE)
            with self._pose_lock:
                self._target.pose.position.x = float(xyz[0])
                self._target.pose.position.y = float(xyz[1])
                self._target.pose.position.z = float(xyz[2])
                self._target.pose.orientation.x = float(q[0])
                self._target.pose.orientation.y = float(q[1])
                self._target.pose.orientation.z = float(q[2])
                self._target.pose.orientation.w = float(q[3])
            self._initialized = True
            self._log(f"init from /franka_states: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
            return
        except ROSException:
            pass
        except Exception as e:
            self._log(f"init via /franka_states failed: {e}", level="warn")

        # 3) safe default
        self._log("could not init from any source; using safe default "
                  "(0.30, 0.0, 0.50, identity quat)", level="warn")
        with self._pose_lock:
            self._target.pose.position.x = 0.30
            self._target.pose.position.y = 0.00
            self._target.pose.position.z = 0.50
            self._target.pose.orientation.x = 0.0
            self._target.pose.orientation.y = 0.0
            self._target.pose.orientation.z = 0.0
            self._target.pose.orientation.w = 1.0
        self._initialized = True

    def _log(self, msg: str, level: str = "info") -> None:
        if not self.verbose:
            return
        full = f"[prism_publisher] {msg}"
        if level == "warn":
            rospy.logwarn(full)
        elif level == "err":
            rospy.logerr(full)
        else:
            rospy.loginfo(full)
