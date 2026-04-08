import csv
import dataclasses
from abc import ABC, abstractmethod

import cv2
import numpy as np

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
    yaw: float
    roll: float
    pitch: float

    @classmethod
    def from_sensor_data(cls, sd):
        return cls(
            pos=np.array([sd['x_global'], sd['y_global'], sd['z_global']]),
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


def clamp_control_command(control_command, drone, max_speed=1, max_yaw_rate=0.4):
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



class State(ABC):
    """Base class for flight states. Returns (control_command, next_state_or_None)."""

    @abstractmethod
    def execute(self, drone, tracker, dt):
        ...


class TakeoffState(State):
    def execute(self, drone, tracker, dt):
        if drone.pos[2] < 1:
            return [drone.pos[0], drone.pos[1], 1.5, drone.yaw], None
        return [drone.pos[0], drone.pos[1], drone.pos[2], drone.yaw], SearchingState()

# TODO: We need another state that moves to an average height and also moves further outside of the map.
class SearchingState(State):
    def __init__(self):
        self.target_yaw = None

    def execute(self, drone, tracker, dt):
        if tracker.has_estimate:
            normal = tracker.oriented_normal()
            if normal is None:
                print("No gate detected, moving outward")
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
            print(f"[GATE {tracker.current_gate_index}] Detected. Flying to 1m in front.")
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
            print("Resetting gate filter")
            print(f"[GATE {tracker.current_gate_index}] Arrived. Measuring for 1s...")
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
            i = tracker.current_gate_index
            print(f"[GATE {i}] === Measured at {center} ===")
            for j, label in enumerate(["TL", "TR", "BR", "BL"]):
                print(f"[GATE {i}]   Corner {label}: {tracker.measurements[-1]['corners'][j]}")

            direction = center - drone.pos
            direction_norm = direction / np.linalg.norm(direction)
            pass_target = center + direction_norm * 0.2
            pass_yaw = np.arctan2(direction_norm[1], direction_norm[0])
            print(f"[GATE {i}] Passing through to {pass_target}")
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
                print("[GATE] All 5 gates measured! Measurements:")
                for i, m in enumerate(tracker.measurements):
                    print(f"  Gate {i}: center={m['center']}, normal={m['normal']}")
                return cmd, DoneState(self.target_pos, self.target_yaw)
            else:
                print(f"[GATE] Searching for gate {tracker.current_gate_index}...")
                return [self.target_pos[0], self.target_pos[1], self.target_pos[2], drone.yaw], SearchingState()

        return cmd, None


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

    @property
    def predicted_corners_world(self):
        """Exposed for visualization in main.py."""
        return self.tracker.corners

    def compute_command(self, sensor_data, camera_data, dt):
        drone = DroneState.from_sensor_data(sensor_data)

        # Detection runs only during states that need it (not during takeoff, passing through, or done)
        if not isinstance(self.state, (TakeoffState, PassingThroughState, DoneState)):
            corners_world, _ = self.detector.detect(camera_data, drone)
            if corners_world is not None:
                self.tracker.process_detection(corners_world)
            else:
                self.tracker.keep_estimate()

        prev_state = self.state
        cmd, next_state = self.state.execute(drone, self.tracker, dt)
        if next_state is not None:
            self.state = next_state

        # Clamping runs for all states except takeoff
        if not isinstance(prev_state, TakeoffState):
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
