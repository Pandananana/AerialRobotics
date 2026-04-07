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
        self.state = "SEARCHING"  # SEARCHING -> APPROACHING -> WAITING -> DONE
        self.target_position = None  # 1m in front of gate
        self.wait_timer = 0.0
        self.gate_measurement = None  # final gate location to print
        self.target_yaw = None  # yaw to face the gate

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command

        # ---- YOUR CODE HERE ----

        # Always detect gate and transform corners
        detection = self.detect_gate(camera_data)
        self.transform_gate_corners_to_world(detection, sensor_data)

        drone_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

        if self.state == "SEARCHING":
            if self.gate_center_world is not None and self.predicted_corners_world is not None:
                # Compute gate normal from corners (TL, TR, BR, BL)
                c = self.predicted_corners_world
                edge1 = np.array(c[1]) - np.array(c[0])  # TL -> TR
                edge2 = np.array(c[3]) - np.array(c[0])  # TL -> BL
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)

                center = np.array(self.gate_center_world)

                # Two candidate positions 1m in front of gate
                candidate1 = center + normal
                candidate2 = center - normal

                # Pick the closest one to the drone
                if np.linalg.norm(candidate1 - drone_pos) < np.linalg.norm(candidate2 - drone_pos):
                    self.target_position = candidate1
                else:
                    self.target_position = candidate2

                self.gate_measurement = center.copy()
                # Compute yaw to face the gate from the target position
                direction = center - self.target_position
                self.target_yaw = np.arctan2(direction[1], direction[0])
                self.state = "APPROACHING"
                print(f"[GATE] Detected gate at {self.gate_measurement}. Flying to {self.target_position}")

            # Keep hovering while searching
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]

        elif self.state == "APPROACHING":
            dist = np.linalg.norm(self.target_position - drone_pos)
            if dist < 0.05:
                self.state = "WAITING"
                self.wait_timer = 0.0
                print("[GATE] Arrived in front of gate. Waiting 1s for stable measurement...")
            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        elif self.state == "WAITING":
            self.wait_timer += dt
            # Keep updating gate measurement while waiting
            if self.gate_center_world is not None:
                self.gate_measurement = np.array(self.gate_center_world).copy()

            if self.wait_timer >= 1.0:
                self.state = "DONE"
                print(f"[GATE] === Gate world location: {self.gate_measurement} ===")
                if self.predicted_corners_world is not None:
                    for i, label in enumerate(["TL", "TR", "BR", "BL"]):
                        print(f"[GATE]   Corner {label}: {self.predicted_corners_world[i]}")

            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        else:  # DONE
            control_command = [self.target_position[0], self.target_position[1], self.target_position[2], self.target_yaw]

        return control_command

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
