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

        pass

    def compute_command(self, sensor_data, camera_data, dt):

        # NOTE: Displaying the camera image with cv2.imshow() will throw an error because GUI operations should be performed in the main thread.
        # If you want to display the camera image you can call it in main.py.

        # Take off example
        if sensor_data['z_global'] < 0.49:
            control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]
            return control_command

        # ---- YOUR CODE HERE ----

        # Detect gate in camera image
        detection = self.detect_gate(camera_data)
        if detection is not None:
            corners, center_px = detection

            # Estimate depth from known gate height
            upper_left, upper_right, lower_right, lower_left = corners
            Z_left, Z_right = self.estimate_gate_depth(upper_left, lower_left, upper_right, lower_right)
            Z_avg = (Z_left + Z_right) / 2

            # Build rotation matrix from drone attitude
            R = self.rotation_matrix(sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw'])
            transform = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])

            # Project gate center to world coordinates
            gate_world = self.point_to_world_coordinate(center_px, R, np.append(transform, Z_avg))

        control_command = [sensor_data['x_global'], sensor_data['y_global'], 1.0, sensor_data['yaw']]

        return control_command # Ordered as array with: [pos_x_cmd, pos_y_cmd, pos_z_cmd, yaw_cmd] in meters and radians

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
        center_px = corners.mean(axis=0).astype(int)

        # corners ordered as: top-left, top-right, bottom-right, bottom-left
        return corners, center_px

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

        # Pixel -> camera frame
        X_cam = (u - cx) * Z / self.f
        Y_cam = (v - cy) * Z / self.f
        Z_cam = Z

        # Camera frame -> world frame
        world_point = rotation @ np.array([X_cam, Y_cam, Z_cam]) + transform[:3]
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
