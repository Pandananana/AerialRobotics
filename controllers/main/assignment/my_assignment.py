import csv
import dataclasses
import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate


# ─── Data Classes ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class DroneState:
    """Typed wrapper around the raw sensor_data dict."""
    pos: np.ndarray  # [x, y, z] global
    vel: np.ndarray  # [v_x, v_y, v_z] global
    acc: np.ndarray  # [ax, ay, az] global (gravity-compensated)
    yaw: float
    roll: float
    pitch: float

    @classmethod
    def from_sensor_data(cls, sd):
        return cls(
            pos=np.array([sd['x_global'], sd['y_global'], sd['z_global']]),
            vel=np.array([sd['v_x'], sd['v_y'], sd['v_z']]),
            acc=np.array([sd['ax_global'], sd['ay_global'], sd['az_global']]),
            yaw=sd['yaw'],
            roll=sd['roll'],
            pitch=sd['pitch'],
        )


def rotation_matrix(roll, pitch, yaw):
    """Build rotation matrix from Euler angles (ZYX convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [  -sp,           cp*sr,           cp*cr   ]
    ])


def compute_gate_normal(corners):
    """Compute unit normal to the gate plane from 4 ordered corners. Returns None if degenerate."""
    edge1 = np.array(corners[1]) - np.array(corners[0])
    edge2 = np.array(corners[3]) - np.array(corners[0])
    normal = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-6:
        return None
    return normal / norm_len


def clamp_control_command(control_command, drone, max_speed=0.8, max_yaw_rate=0.4):
    """Clamp position displacement and yaw change to limit drone speed and rotation."""
    x_t, y_t, z_t, yaw_t = control_command
    displacement = np.array([x_t, y_t, z_t]) - drone.pos
    dist = np.linalg.norm(displacement)
    if dist > max_speed:
        displacement = displacement / dist * max_speed
    target = drone.pos + displacement

    dyaw = (yaw_t - drone.yaw + np.pi) % (2 * np.pi) - np.pi
    dyaw = np.clip(dyaw, -max_yaw_rate, max_yaw_rate)

    return [target[0], target[1], target[2], drone.yaw + dyaw]


class GateDetector:
    """Detects pink gates in camera images and projects corners to world coordinates."""

    def __init__(self, fov=1.5):
        self.f = 150 / np.tan(fov / 2)
        self.lower_pink = np.array([140, 42, 0])
        self.upper_pink = np.array([156, 255, 255])

    def detect(self, camera_data, drone):
        """Full pipeline: image → pixel corners → world corners.
        Returns (corners_world, center_world) or (None, None)."""
        corners_px = self._detect_pixels(camera_data)
        if corners_px is None:
            return None, None

        corners_world, center_world = self._to_world(corners_px, drone)
        if corners_world is None:
            return None, None

        if not self._dimensions_plausible(corners_world):
            return None, None

        return corners_world, center_world

    # TODO: We need to handle multiple gates in the same frame, and selecting only the closest gate to the drone
    def _detect_pixels(self, camera_data):
        """Detect a pink gate quadrilateral in the camera image. Returns 4 pixel corners or None."""
        bgr = camera_data[:, :, :3] if camera_data.shape[2] == 4 else camera_data
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        gate_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(gate_contour) < 50:
            return None

        epsilon = 0.02 * cv2.arcLength(gate_contour, True)
        approx = cv2.approxPolyDP(gate_contour, epsilon, True)
        if len(approx) != 4:
            return None

        corners = self._order_corners(approx)

        h, w = camera_data.shape[:2]
        margin = 15
        for c in corners:
            if c[0] < margin or c[0] > w - margin or c[1] < margin or c[1] > h - margin:
                return None

        return corners

    def _to_world(self, corners_px, drone):
        """Project pixel corners to world coordinates using drone pose and depth estimation."""
        R = rotation_matrix(drone.roll, drone.pitch, drone.yaw)
        upper_left, upper_right, lower_right, lower_left = corners_px
        Z_left, Z_right = self._estimate_depth(upper_left, lower_left, upper_right, lower_right)

        corners_world = []
        for idx, corner_px in enumerate(corners_px):
            Z = Z_left if idx in [0, 3] else Z_right
            world_pt = self._pixel_to_world(corner_px, R, drone.pos, Z)
            corners_world.append(world_pt)

        center_world = np.mean(corners_world, axis=0)
        return corners_world, center_world

    def _pixel_to_world(self, point, R, drone_pos, Z):
        cx, cy = 150, 150
        u, v = point
        X_cam = (u - cx) * Z / self.f
        Y_cam = (v - cy) * Z / self.f
        body_point = np.array([Z, -X_cam, -Y_cam])
        return R @ body_point + drone_pos

    def _estimate_depth(self, upper_left, lower_left, upper_right, lower_right):
        D_real = 0.4
        d_left = np.linalg.norm(np.array(lower_left) - np.array(upper_left))
        d_right = np.linalg.norm(np.array(lower_right) - np.array(upper_right))
        return self.f * D_real / d_left, self.f * D_real / d_right

    @staticmethod
    def _dimensions_plausible(corners, width_range=(0.2, 0.7), height_range=(0.2, 0.7)):
        tl, tr, br, bl = [np.array(c) for c in corners]
        avg_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        avg_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        return width_range[0] <= avg_width <= width_range[1] and height_range[0] <= avg_height <= height_range[1]

    @staticmethod
    def _order_corners(pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        pts = pts.reshape(4, 2).astype(float)
        rect = np.zeros((4, 2), dtype=int)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

class GateKalmanFilter:
    """Kalman filter for 4 gate corners (12D state). Gate is static, noise is from detection."""

    def __init__(self, corners_world, initial_uncertainty=0.5, process_noise=0.001, measurement_noise=0.1):
        self.dim = 12
        self.x = np.concatenate(corners_world).astype(float)
        self.P = np.eye(self.dim) * initial_uncertainty
        self.Q = np.eye(self.dim) * process_noise
        self.R = np.eye(self.dim) * measurement_noise

    def update(self, corners_world):
        """Predict (static model) then update with new measurement."""
        self.P = self.P + self.Q
        z = np.concatenate(corners_world).astype(float)
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim) - K) @ self.P

    def get_corners(self):
        return [self.x[i*3:(i+1)*3].copy() for i in range(4)]

    def get_center(self):
        return np.mean(self.get_corners(), axis=0)


class GateTracker:
    """Manages Kalman-filtered gate estimation and stores completed measurements."""

    def __init__(self):
        self._filter = None
        self.measurements = []  # list of {"center", "normal", "corners"}
        self.current_gate_index = 0
        self.approach_normal = None
        self.corners = None
        self.center = None

    @property
    def has_estimate(self):
        return self.corners is not None and self.center is not None

    @property
    def all_gates_measured(self):
        return self.current_gate_index >= 5

    def process_detection(self, corners_world):
        """Feed a new detection into the Kalman filter and update estimates."""
        if self._filter is None:
            self._filter = GateKalmanFilter(corners_world)
        else:
            self._filter.update(corners_world)
        self.corners = self._filter.get_corners()
        self.center = self._filter.get_center()

    def keep_estimate(self):
        """On missed detection, retain the filtered estimate if available."""
        if self._filter is not None:
            self.corners = self._filter.get_corners()
            self.center = self._filter.get_center()
        else:
            self.corners = None
            self.center = None

    def reset_filter(self):
        """Clear the Kalman filter (e.g., before the measuring phase)."""
        self._filter = None

    def record_measurement(self):
        """Snapshot current filtered gate into the measurements list."""
        self.measurements.append({
            "center": np.array(self.center).copy(),
            "normal": self.approach_normal.copy(),
            "corners": [np.array(c).copy() for c in self.corners],
        })

    def advance_gate(self):
        """Move to next gate and reset all tracking state."""
        self.current_gate_index += 1
        self._filter = None
        self.corners = None
        self.center = None
        self.approach_normal = None

    def oriented_normal(self):
        """Compute normal from filtered corners, oriented toward the approach side."""
        if self.corners is None:
            return None
        normal = compute_gate_normal(self.corners)
        if normal is None:
            return None
        if self.approach_normal is not None and np.dot(normal, self.approach_normal) < 0:
            normal = -normal
        return normal



class PolyTrajectory:
    """Minimum-jerk 5th-order polynomial trajectory through 3D waypoints, time-parameterized.

    Segment times are allocated proportional to segment length. A final time t_f is
    tuned so the sampled trajectory respects per-axis velocity and acceleration limits.
    Position/velocity/acceleration/jerk/snap are C4-continuous at internal waypoints;
    initial velocity/acceleration match the caller-supplied BCs (typically the drone's
    current state), and final velocity/acceleration default to zero.
    """

    VEL_LIM_XY = 3.0
    VEL_LIM_Z = 0.75
    ACC_LIM_XY = 6
    ACC_LIM_Z = 5.0
    DISC_STEPS = 20

    def __init__(self, waypoints, v_init, a_init, v_final=None, a_final=None):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.m = len(self.waypoints)
        self.v_init = np.asarray(v_init, dtype=float)
        self.a_init = np.asarray(a_init, dtype=float)
        self.v_final = np.zeros(3) if v_final is None else np.asarray(v_final, dtype=float)
        self.a_final = np.zeros(3) if a_final is None else np.asarray(a_final, dtype=float)

        diffs = np.diff(self.waypoints, axis=0)
        self._seg_lengths = np.linalg.norm(diffs, axis=1)
        self.total_length = float(self._seg_lengths.sum())

        # Conservative starting guess — will be scaled up by _tune to respect limits.
        t_f_guess = self.total_length / (0.5 * self.VEL_LIM_XY)
        self.total_time = self._tune(t_f_guess)
        self._solve(self.total_time)

    @staticmethod
    def _poly_matrix(t):
        # Row k = k-th derivative of [t^5, t^4, t^3, t^2, t, 1] · coeffs.
        return np.array([
            [t**5,    t**4,    t**3,    t**2, t, 1],
            [5*t**4,  4*t**3,  3*t**2,  2*t,  1, 0],
            [20*t**3, 12*t**2, 6*t,     2,    0, 0],
            [60*t**2, 24*t,    6,       0,    0, 0],
            [120*t,   24,      0,       0,    0, 0],
        ])

    def _seg_times(self, t_f):
        # Allocate time proportional to segment length.
        return t_f * self._seg_lengths / self._seg_lengths.sum()

    def _solve(self, t_f):
        seg_times = self._seg_times(t_f)
        self.times = np.concatenate([[0.0], np.cumsum(seg_times)])
        m = self.m
        n = 6 * (m - 1)
        self.coeffs = np.zeros((n, 3))
        A_0 = self._poly_matrix(0.0)

        for dim in range(3):
            A = np.zeros((n, n))
            b = np.zeros(n)
            pos = self.waypoints[:, dim]

            if m == 2:
                # Single segment: all 6 BCs are explicit.
                A_f = self._poly_matrix(seg_times[0])
                A[0, :6] = A_0[0]; b[0] = pos[0]
                A[1, :6] = A_f[0]; b[1] = pos[1]
                A[2, :6] = A_0[1]; b[2] = self.v_init[dim]
                A[3, :6] = A_f[1]; b[3] = self.v_final[dim]
                A[4, :6] = A_0[2]; b[4] = self.a_init[dim]
                A[5, :6] = A_f[2]; b[5] = self.a_final[dim]
            else:
                row = 0
                for i in range(m - 1):
                    A_f = self._poly_matrix(seg_times[i])
                    if i == 0:
                        A[row, :6] = A_0[0]; b[row] = pos[0]; row += 1
                        A[row, :6] = A_f[0]; b[row] = pos[1]; row += 1
                        A[row, :6] = A_0[1]; b[row] = self.v_init[dim]; row += 1
                        A[row, :6] = A_0[2]; b[row] = self.a_init[dim]; row += 1
                        # Continuity of vel/acc/jerk/snap at end of segment 0.
                        A[row:row+4, :6] = A_f[1:]
                        A[row:row+4, 6:12] = -A_0[1:]
                        row += 4
                    elif i < m - 2:
                        A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                        A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                        A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                        row += 4
                    else:
                        A[row, i*6:(i+1)*6] = A_0[0]; b[row] = pos[i]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[0]; b[row] = pos[i+1]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[1]; b[row] = self.v_final[dim]; row += 1
                        A[row, i*6:(i+1)*6] = A_f[2]; b[row] = self.a_final[dim]; row += 1

            self.coeffs[:, dim] = np.linalg.solve(A, b)

    def _tune(self, t_f, max_iters=10, safety=1.05, tol=0.01):
        """Scale t_f so the sampled trajectory just respects velocity/acceleration limits.

        Scaling time by k reparameterizes max|v| by 1/k and max|a| by 1/k²; iterate to
        converge (BCs on initial v/a break exact invariance).
        """
        for _ in range(max_iters):
            self._solve(t_f)
            v_xy, v_z, a_xy, a_z = self._sample_limits()
            k = max(
                v_xy / self.VEL_LIM_XY,
                v_z / self.VEL_LIM_Z,
                np.sqrt(a_xy / self.ACC_LIM_XY),
                np.sqrt(a_z / self.ACC_LIM_Z),
            )
            new_t_f = t_f * k * safety
            if abs(new_t_f - t_f) / t_f < tol:
                return t_f
            t_f = new_t_f
        return t_f

    def _sample_limits(self):
        ts = np.linspace(0.0, self.times[-1], self.DISC_STEPS * self.m)
        v_xy_max = v_z_max = a_xy_max = a_z_max = 0.0
        for t in ts:
            seg, t_local = self._seg_index(t)
            M = self._poly_matrix(t_local)
            c = self.coeffs[seg*6:(seg+1)*6, :]
            v = M[1] @ c
            a = M[2] @ c
            v_xy_max = max(v_xy_max, float(np.hypot(v[0], v[1])))
            v_z_max = max(v_z_max, float(abs(v[2])))
            a_xy_max = max(a_xy_max, float(np.hypot(a[0], a[1])))
            a_z_max = max(a_z_max, float(abs(a[2])))
        return v_xy_max, v_z_max, a_xy_max, a_z_max

    def _seg_index(self, t):
        t_clamped = float(np.clip(t, 0.0, self.times[-1]))
        seg = int(min(max(np.searchsorted(self.times, t_clamped) - 1, 0), self.m - 2))
        return seg, t_clamped - self.times[seg]

    def position_at(self, t):
        seg, t_local = self._seg_index(t)
        return self._poly_matrix(t_local)[0] @ self.coeffs[seg*6:(seg+1)*6, :]

    def velocity_at(self, t):
        seg, t_local = self._seg_index(t)
        return self._poly_matrix(t_local)[1] @ self.coeffs[seg*6:(seg+1)*6, :]

    def yaw_at(self, t):
        """Heading from horizontal velocity direction."""
        v = self.velocity_at(t)
        if np.hypot(v[0], v[1]) < 1e-6:
            return 0.0
        return float(np.arctan2(v[1], v[0]))


class State(ABC):
    """Base class for flight states. Returns (control_command, next_state_or_None)."""

    @abstractmethod
    def execute(self, drone, tracker, dt):
        ...


class TakeoffState(State):
    def execute(self, drone, tracker, dt):
        if drone.pos[2] < 1.8:
            return [drone.pos[0], drone.pos[1], 2.0, drone.yaw], None
        return [drone.pos[0], drone.pos[1], drone.pos[2], drone.yaw], SearchingState()

# TODO: We need another state that moves to an average height and also moves further outside of the map.
class SearchingState(State):
    def __init__(self):
        self.target_yaw = None

    def execute(self, drone, tracker, dt):
        if tracker.has_estimate:
            normal = tracker.oriented_normal()
            if normal is None:
                logger.debug("Gate normal degenerate, moving outward and re-yawing")
                tracker.corners = None
                tracker.center = None
                return self._move_outward_and_yaw(drone), None

            center = np.array(tracker.center)
            candidate1 = center + normal
            candidate2 = center - normal

            if np.linalg.norm(candidate1 - drone.pos) < np.linalg.norm(candidate2 - drone.pos):
                target_pos = candidate1
                tracker.approach_normal = normal
            else:
                target_pos = candidate2
                tracker.approach_normal = -normal

            direction = center - drone.pos
            target_yaw = np.arctan2(direction[1], direction[0])
            return [drone.pos[0], drone.pos[1], drone.pos[2], target_yaw], ApproachingState(target_pos, center)
        else:
            self.target_yaw = drone.yaw + 0.3

        yaw = self.target_yaw if self.target_yaw is not None else drone.yaw
        return [drone.pos[0], drone.pos[1], drone.pos[2], yaw], None

    @staticmethod
    def _move_outward_and_yaw(drone, distance=0.3, yaw_step=0.3):
        """Move outward from world center (5,5) and yaw left to get a better gate view."""
        center = np.array([5.0, 5.0])
        pos2d = drone.pos[:2]
        radial = pos2d - center
        radial_norm = radial / (np.linalg.norm(radial) + 1e-6)
        outward = pos2d + radial_norm * distance
        return [outward[0], outward[1], drone.pos[2], drone.yaw + yaw_step]


class ApproachingState(State):
    def __init__(self, target_pos, gate_center):
        self.target_pos = target_pos
        self.gate_center = gate_center
        self.approach_distance = 0.5

    def execute(self, drone, tracker, dt):
        if tracker.has_estimate:
            normal = compute_gate_normal(tracker.corners)
            if normal is not None:
                self.gate_center = np.array(tracker.center)
                # Pick the normal direction closest to the drone
                candidate_a = self.gate_center + normal * self.approach_distance
                candidate_b = self.gate_center - normal * self.approach_distance
                if np.linalg.norm(candidate_a - drone.pos) <= np.linalg.norm(candidate_b - drone.pos):
                    chosen_normal = normal
                else:
                    chosen_normal = -normal
                tracker.approach_normal = chosen_normal
                self.target_pos = self.gate_center + chosen_normal * self.approach_distance

        direction = self.gate_center - drone.pos
        target_yaw = np.arctan2(direction[1], direction[0])

        dist = np.linalg.norm(self.target_pos - drone.pos)
        yaw_error = abs(target_yaw - drone.yaw)
        cmd = [self.target_pos[0], self.target_pos[1], self.target_pos[2], target_yaw]

        if dist < 0.05 and yaw_error < 0.05:
            tracker.reset_filter()
            logger.debug("Approach point reached; resetting gate filter for measurement")
            return cmd, MeasuringState(self.target_pos.copy(), target_yaw)

        return cmd, None


class MeasuringState(State):
    def __init__(self, target_pos, target_yaw):
        self.target_pos = target_pos
        self.target_yaw = target_yaw
        self.wait_timer = 0.0

    def execute(self, drone, tracker, dt):
        self.wait_timer += dt

        if tracker.has_estimate:
            normal = tracker.oriented_normal()
            if normal is not None:
                tracker.approach_normal = normal

        cmd = [self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_yaw]

        if self.wait_timer >= 1.0 and tracker.corners is not None:
            tracker.record_measurement()
            center = np.array(tracker.center)

            direction = center - drone.pos
            direction_norm = direction / np.linalg.norm(direction)
            pass_target = center + direction_norm * 0.2
            pass_yaw = np.arctan2(direction_norm[1], direction_norm[0])
            return [pass_target[0], pass_target[1], pass_target[2], pass_yaw], PassingThroughState(pass_target, pass_yaw)

        return cmd, None


class PassingThroughState(State):
    def __init__(self, target_pos, target_yaw):
        self.target_pos = target_pos
        self.target_yaw = target_yaw

    def execute(self, drone, tracker, dt):
        cmd = [self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_yaw]
        dist = np.linalg.norm(self.target_pos - drone.pos)

        if dist < 0.05:
            tracker.advance_gate()
            if tracker.all_gates_measured:
                logger.info("All 5 gates measured. Computing racing trajectory.")
                for i, m in enumerate(tracker.measurements):
                    logger.debug(f"Gate {i}: center={m['center']}, normal={m['normal']}")
                trajectory = build_racing_trajectory(drone, tracker.measurements)
                return cmd, RacingState(trajectory)
            else:
                return [self.target_pos[0], self.target_pos[1], self.target_pos[2], drone.yaw], RepositioningState(drone)

        return cmd, None

class RepositioningState(State):
    """Move the drone further from the gate circle center and up to 2m altitude before searching."""
    CIRCLE_CENTER = np.array([4.0, 4.0])
    TARGET_Z = 1.75
    OUTWARD_DISTANCE = 2  # how much further from center to move
    ARENA_MIN = 0  # 0.5m margin from arena boundary (0)
    ARENA_MAX = 8  # 0.5m margin from arena boundary (8)

    def __init__(self, drone):
        self.target_pos = None
        self.target_yaw = drone.yaw - np.pi/4 # Rotate a little bit to the right when repositioning.

    def execute(self, drone, tracker, dt):
        if self.target_pos is None:
            pos2d = drone.pos[:2]
            radial = pos2d - self.CIRCLE_CENTER
            radial_norm = radial / (np.linalg.norm(radial) + 1e-6)
            outward2d = pos2d + radial_norm * self.OUTWARD_DISTANCE
            outward2d = np.clip(outward2d, self.ARENA_MIN, self.ARENA_MAX)
            self.target_pos = np.array([outward2d[0], outward2d[1], self.TARGET_Z])

        dist = np.linalg.norm(self.target_pos - drone.pos)
        cmd = [self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_yaw]

        if dist < 0.1:
            return cmd, SearchingState()

        return cmd, None


def build_racing_trajectory(drone, measurements, num_laps=2):
    """Build a PolyTrajectory from the drone's current state through all gates."""
    waypoints = [drone.pos.copy()]
    for _ in range(num_laps):
        for m in measurements:
            waypoints.append(m['center'].copy())
    # Final: return to gate 0 area to stop timer
    m0 = measurements[0]
    waypoints.append(m0['center'].copy())

    trajectory = PolyTrajectory(
        waypoints,
        v_init=drone.vel.copy(),
        a_init=drone.acc.copy(),
    )
    logger.info(
        f"Racing trajectory: {trajectory.total_length:.2f}m over "
        f"{trajectory.total_time:.2f}s through {len(waypoints)} waypoints "
        f"(avg speed {trajectory.total_length / trajectory.total_time:.2f} m/s)"
    )
    return trajectory


class RacingState(State):
    """Fly through all gates along a precomputed time-parameterized polynomial trajectory."""

    SPEED_INTERVAL_S = 2.0

    # Velocity feedforward via position-offset trick.
    # Position PID: v_cmd = P_pos * (x_set - x). Setting x_set = x_desired + v_ff / P_pos
    # yields v_cmd = P_pos * (x_desired - x) + v_ff, i.e. position feedback + velocity FF,
    # without needing to plumb a separate FF channel through main.py / setpoint_to_pwm.
    # P gains must match ex1_pid_control.py (exp_num=4 branch).
    P_POS_XY = 1.5
    P_POS_Z = 5.0

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.t_elapsed = 0.0
        self.interval_elapsed = 0.0
        self.interval_max_speed = 0.0
        self.interval_max_pos_error = 0.0
        self.race_max_pos_error = 0.0
        self.race_pos_error_sum = 0.0
        self.race_pos_error_samples = 0
        self.interval_index = 0

    def _apply_ff(self, pos, vel):
        return np.array([
            pos[0] + vel[0] / self.P_POS_XY,
            pos[1] + vel[1] / self.P_POS_XY,
            pos[2] + vel[2] / self.P_POS_Z,
        ])

    def execute(self, drone, tracker, dt):
        # Tracking error: compare drone position to where the trajectory says it should be.
        # High error means the PID is lagging the reference — slow the trajectory or retune.
        ref_t = min(self.t_elapsed, self.trajectory.total_time)
        ref_pos = self.trajectory.position_at(ref_t)
        pos_error = float(np.linalg.norm(ref_pos - drone.pos))
        speed_mag = float(np.linalg.norm(drone.vel))

        self.interval_max_speed = max(self.interval_max_speed, speed_mag)
        self.interval_max_pos_error = max(self.interval_max_pos_error, pos_error)
        self.race_max_pos_error = max(self.race_max_pos_error, pos_error)
        self.race_pos_error_sum += pos_error
        self.race_pos_error_samples += 1
        self.interval_elapsed += dt
        self._flush_completed_speed_intervals(speed_mag, pos_error)

        self.t_elapsed += dt

        if self.t_elapsed >= self.trajectory.total_time:
            self._log_final_partial_speed_interval()
            mean_err = (
                self.race_pos_error_sum / self.race_pos_error_samples
                if self.race_pos_error_samples else 0.0
            )
            logger.info(
                f"Racing complete: predicted={self.trajectory.total_time:.2f}s, "
                f"actual={self.t_elapsed:.2f}s "
                f"(delta={self.t_elapsed - self.trajectory.total_time:+.3f}s); "
                f"pos_error mean={mean_err:.3f}m, max={self.race_max_pos_error:.3f}m"
            )
            target = self.trajectory.position_at(self.trajectory.total_time)
            target_yaw = self.trajectory.yaw_at(self.trajectory.total_time)
            return [target[0], target[1], target[2], target_yaw], DoneState(target, target_yaw)

        target = self.trajectory.position_at(self.t_elapsed)
        vel_ff = self.trajectory.velocity_at(self.t_elapsed)
        target_ff = self._apply_ff(target, vel_ff)
        target_yaw = self.trajectory.yaw_at(self.t_elapsed)
        return [target_ff[0], target_ff[1], target_ff[2], target_yaw], None

    def _flush_completed_speed_intervals(self, current_speed, current_error):
        while self.interval_elapsed >= self.SPEED_INTERVAL_S:
            self.interval_index += 1
            logger.info(
                f"Race interval {self.interval_index} ({self.SPEED_INTERVAL_S:.1f}s): "
                f"max_speed={self.interval_max_speed:.2f}m/s, "
                f"max_pos_error={self.interval_max_pos_error:.3f}m"
            )
            self.interval_elapsed -= self.SPEED_INTERVAL_S
            if self.interval_elapsed > 0.0:
                self.interval_max_speed = current_speed
                self.interval_max_pos_error = current_error
            else:
                self.interval_max_speed = 0.0
                self.interval_max_pos_error = 0.0

    def _log_final_partial_speed_interval(self):
        if self.interval_elapsed <= 1e-6:
            return

        self.interval_index += 1
        logger.info(
            f"Race interval {self.interval_index} ({self.interval_elapsed:.2f}s, partial): "
            f"max_speed={self.interval_max_speed:.2f}m/s, "
            f"max_pos_error={self.interval_max_pos_error:.3f}m"
        )
        self.interval_elapsed = 0.0
        self.interval_max_speed = 0.0
        self.interval_max_pos_error = 0.0


class DoneState(State):
    def __init__(self, target_pos, target_yaw):
        self.target_pos = target_pos
        self.target_yaw = target_yaw

    def execute(self, drone, tracker, dt):
        return [self.target_pos[0], self.target_pos[1], self.target_pos[2], self.target_yaw], None

class MyAssignment:
    def __init__(self):
        self.detector = GateDetector(fov=1.5)
        self.tracker = GateTracker()
        self.state = TakeoffState()
        self.elapsed = 0.0
        self.last_transition_t = 0.0
        self.measurement_phase_start_t = None

    @property
    def predicted_corners_world(self):
        """Exposed for visualization in main.py."""
        return self.tracker.corners

    def compute_command(self, sensor_data, camera_data, dt):
        self.elapsed += dt
        drone = DroneState.from_sensor_data(sensor_data)

        # Detection runs only during states that need it (not during takeoff, passing through, or done)
        if not isinstance(self.state, (TakeoffState, PassingThroughState, RepositioningState, RacingState, DoneState)):
            corners_world, _ = self.detector.detect(camera_data, drone)
            if corners_world is not None:
                self.tracker.process_detection(corners_world)
            else:
                self.tracker.keep_estimate()

        prev_state = self.state
        cmd, next_state = self.state.execute(drone, self.tracker, dt)
        if next_state is not None:
            self.state = next_state

        # Clamping runs for all states except takeoff and racing
        if not isinstance(prev_state, (TakeoffState, RacingState)):
            cmd = clamp_control_command(cmd, drone)

        return cmd

    def take_photo(self, sensor_data, camera_data):
        image_filename = "data/gate1.png"
        cv2.imwrite(image_filename, camera_data)
        csv_file = "data/gate1.csv"
        position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
        orientation = [sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']]
        row = position + orientation
        with open(csv_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x_global', 'y_global', 'z_global', 'roll', 'pitch', 'yaw'])
            writer.writerow(row)


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
