import csv

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


class MyAssignment:
    def __init__(self):
        # ---- INITIALISE YOUR VARIABLES HERE ----
        self.photo_counter = 0
        self.fov = 1.5
        self.f = 150 / np.tan(self.fov / 2)

        # HSV thresholds for pink gate detection
        self.lower_pink = np.array([140, 42, 0])
        self.upper_pink = np.array([156, 255, 255])

        # Predicted gate corners in world coordinates (set each frame, read by main.py for visualization)
        self.predicted_corners_world = None
        self.gate_center_world = None

        # State machine for gate measurement
        self.state = "TAKEOFF"  # TAKEOFF -> SEARCHING -> APPROACHING -> MEASURING -> PASSING_THROUGH -> (loop)
        self.target_position = None
        self.wait_timer = 0.0
        self.target_yaw = None

        # Gate tracking
        self.current_gate_index = 0
        self.gate_measurements = []  # list of {"center", "normal", "corners"}
        self.approach_normal = None  # normal vector from gate toward approach side

    def compute_command(self, sensor_data, camera_data, dt):

        # Take off
        if self.state == "TAKEOFF":
            if sensor_data['z_global'] < 1:
                control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.5, sensor_data['yaw']]
            else:
                self.state = "SEARCHING"
                control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]
            
            return control_command

        # Always detect gate and transform corners
        self.update_gate_detection(camera_data, sensor_data)

        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

        if self.state == "SEARCHING":
            if self.gate_center_world is not None and self.predicted_corners_world is not None:
                normal = self.compute_gate_normal()
                if normal is None:
                    print("No gate detected, moving outward")
                    self.predicted_corners_world = None
                    self.gate_center_world = None
                    return self.move_outward_and_yaw(sensor_data)

                center = np.array(self.gate_center_world)

                # Two candidate positions 1m in front of gate — pick closest to drone
                candidate1 = center + normal
                candidate2 = center - normal

                if np.linalg.norm(candidate1 - drone_pos) < np.linalg.norm(candidate2 - drone_pos):
                    self.target_position = candidate1
                    self.approach_normal = normal  # points from center toward approach side
                else:
                    self.target_position = candidate2
                    self.approach_normal = -normal

                # Compute yaw to face the gate from the target position
                direction = center - self.target_position
                self.target_yaw = np.arctan2(direction[1], direction[0])
                self.state = "APPROACHING"
                print(f"[GATE {self.current_gate_index}] Detected. Flying to 1m in front.")

            else:
                # Rotate left to scan for gates
                self.target_yaw = sensor_data['yaw'] + 0.3

            control_command = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], self.target_yaw or sensor_data['yaw']]

        elif self.state == "APPROACHING":
            dist = np.linalg.norm(self.target_position - drone_pos)
            if dist < 0.05:
                self.state = "MEASURING"
                self.wait_timer = 0.0
                print(f"[GATE {self.current_gate_index}] Arrived. Measuring for 1s...")
            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        elif self.state == "MEASURING":
            self.wait_timer += dt
            # Keep updating measurement while hovering
            if self.gate_center_world is not None and self.predicted_corners_world is not None:
                normal = self.compute_gate_normal()
                if normal is not None:
                    # Keep normal pointing toward approach side
                    if np.dot(normal, self.approach_normal) < 0:
                        normal = -normal
                    self.approach_normal = normal

            if self.wait_timer >= 1.0 and self.predicted_corners_world is not None:
                # Store measurement
                center = np.array(self.gate_center_world).copy()
                corners = [np.array(c).copy() for c in self.predicted_corners_world]
                self.gate_measurements.append({
                    "center": center,
                    "normal": self.approach_normal.copy(),
                    "corners": corners,
                })
                print(f"[GATE {self.current_gate_index}] === Measured at {center} ===")
                for i, label in enumerate(["TL", "TR", "BR", "BL"]):
                    print(f"[GATE {self.current_gate_index}]   Corner {label}: {corners[i]}")

                # Compute pass-through target: 20cm past gate center along drone→center line
                direction = center - drone_pos
                direction_norm = direction / np.linalg.norm(direction)
                self.target_position = center + direction_norm * 0.2
                self.target_yaw = np.arctan2(direction_norm[1], direction_norm[0])
                self.state = "PASSING_THROUGH"
                print(f"[GATE {self.current_gate_index}] Passing through to {self.target_position}")

            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        elif self.state == "PASSING_THROUGH":
            dist = np.linalg.norm(self.target_position - drone_pos)
            if dist < 0.05:
                self.current_gate_index += 1
                if self.current_gate_index >= 5:
                    self.state = "DONE"
                    print("[GATE] All 5 gates measured! Measurements:")
                    for i, m in enumerate(self.gate_measurements):
                        print(f"  Gate {i}: center={m['center']}, normal={m['normal']}")
                else:
                    self.state = "SEARCHING"
                    self.target_yaw = sensor_data['yaw']
                    self.gate_center_world = None
                    self.predicted_corners_world = None
                    print(f"[GATE] Searching for gate {self.current_gate_index}...")
            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        else:  # DONE
            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        return self.clamp_control_command(control_command, sensor_data)

    def update_gate_detection(self, camera_data, sensor_data):
        """Detect gate in camera image and update world coordinates."""
        corners = self.detect_gate(camera_data)
        if corners is None:
            self.predicted_corners_world = None
            self.gate_center_world = None
            return
        self.transform_gate_corners_to_world(corners, sensor_data)

    def compute_gate_normal(self):
        """Compute gate normal from predicted corners. Returns None if degenerate (edge-on gate)."""
        c = self.predicted_corners_world
        edge1 = np.array(c[1]) - np.array(c[0])
        edge2 = np.array(c[3]) - np.array(c[0])
        normal = np.cross(edge1, edge2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            return None
        return normal / norm_len

    def move_outward_and_yaw(self, sensor_data, distance=0.3, yaw_step=0.3):
        """Move outward from world center (5,5) and yaw left to get a better gate view."""
        center = np.array([5.0, 5.0])
        pos2d = np.array([sensor_data['x_global'], sensor_data['y_global']])
        radial = pos2d - center
        radial_norm = radial / (np.linalg.norm(radial) + 1e-6)
        outward_target = pos2d + radial_norm * distance
        return [outward_target[0], outward_target[1], sensor_data['z_global'], sensor_data['yaw'] + yaw_step]

    # TODO: We need to handle multiple gates in the same frame, and selecting only the closest gate to the drone
    def detect_gate(self, camera_data):
        """Detect a pink gate in the camera image and return its 4 corners + center pixel."""
        # Convert BGRA to BGR if needed
        if camera_data.shape[2] == 4:
            bgr = camera_data[:, :, :3]
        else:
            bgr = camera_data

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        gate_contour = max(contours, key=cv2.contourArea)

        # Filter out noise — require minimum area
        if cv2.contourArea(gate_contour) < 50:
            return None

        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(gate_contour, True)
        approx = cv2.approxPolyDP(gate_contour, epsilon, True)

        if len(approx) != 4:
            return None

        corners = self.order_corners(approx)

        # Reject if any corner is too close to the frame edge
        h, w = camera_data.shape[:2]
        margin = 15
        for c in corners:
            if c[0] < margin or c[0] > w - margin or c[1] < margin or c[1] > h - margin:
                return None

        # corners ordered as: top-left, top-right, bottom-right, bottom-left
        return corners

    def transform_gate_corners_to_world(self, corners, sensor_data):
        """Transform gate corners to world coordinates."""
        # Exit early if no corners are detected
        if corners is None:
            return

        # Build rotation matrix from drone attitude
        R = self.rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

        # Estimate depth from known gate height (corners: TL, TR, BR, BL)
        upper_left, upper_right, lower_right, lower_left = corners
        Z_left, Z_right = self.estimate_gate_depth(upper_left, lower_left, upper_right, lower_right)

        # Project each corner to world coordinates
        self.predicted_corners_world = []
        for idx, corner_px in enumerate(corners):
            Z = Z_left if idx in [0, 3] else Z_right
            world_pt = self.point_to_world_coordinate(corner_px, R, np.append(drone_pos, Z))
            self.predicted_corners_world.append(world_pt)

        # Calculate the center of the gate
        self.gate_center_world = np.mean(self.predicted_corners_world, axis=0)

    def clamp_control_command(self, control_command, sensor_data, max_speed=0.5, max_yaw_rate=0.15):
        """Clamp position displacement and yaw change to limit drone speed and rotation."""
        x_t, y_t, z_t, yaw_t = control_command
        x, y, z = sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']
        yaw = sensor_data['yaw']

        # Clamp position: limit displacement magnitude
        displacement = np.array([x_t - x, y_t - y, z_t - z])
        dist = np.linalg.norm(displacement)
        if dist > max_speed:
            displacement = displacement / dist * max_speed
        target = np.array([x, y, z]) + displacement

        # Clamp yaw: wrap difference to [-pi, pi] then limit magnitude
        dyaw = (yaw_t - yaw + np.pi) % (2 * np.pi) - np.pi
        dyaw = np.clip(dyaw, -max_yaw_rate, max_yaw_rate)

        return [target[0], target[1], target[2], yaw + dyaw]

    @staticmethod
    def order_corners(pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        pts = pts.reshape(4, 2).astype(float)
        rect = np.zeros((4, 2), dtype=int)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right
        d = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(d)]   # top-right
        rect[3] = pts[np.argmax(d)]   # bottom-left
        return rect

    def point_to_world_coordinate(self, point, rotation, transform):
        cx, cy = 150, 150

        u, v = point
        Z = transform[3]  # depth passed as 4th element

        # Pixel -> camera frame (X-right, Y-down, Z-forward)
        X_cam = (u - cx) * Z / self.f
        Y_cam = (v - cy) * Z / self.f
        Z_cam = Z

        # Camera frame -> body frame (X-forward, Y-left, Z-up)
        body_point = np.array([Z_cam, -X_cam, -Y_cam])

        # Body frame -> world frame
        world_point = rotation @ body_point + transform[:3]
        return world_point

    def estimate_gate_depth(self, upper_left, lower_left, upper_right, lower_right):
        D_real = 0.4  # known vertical distance in meters

        # Pixel distance for each side
        d_left = np.linalg.norm(np.array(lower_left) - np.array(upper_left))
        d_right = np.linalg.norm(np.array(lower_right) - np.array(upper_right))

        Z_left = self.f * D_real / d_left
        Z_right = self.f * D_real / d_right

        return Z_left, Z_right

    @staticmethod
    def rotation_matrix(roll, pitch, yaw):
        """Build rotation matrix from Euler angles (ZYX convention)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [  -sp,           cp*sr,           cp*cr  ]
        ])
        return R

    def take_photo(self, sensor_data, camera_data):
        # Save the camera image
        image_filename = "data/gate1.png"
        cv2.imwrite(image_filename, camera_data)

        # Prepare data to be saved
        csv_file = "data/gate1.csv"
        position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
        orientation = [sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']]
        row = position + orientation

        # Overwrite CSV each time (header + current row)
        with open(csv_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['x_global', 'y_global', 'z_global', 'roll', 'pitch', 'yaw'])
            writer.writerow(row)


# Module-level singleton so main.py can call assignment.get_command() unchanged
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
