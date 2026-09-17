#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot with independent YOLO streaming display
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
Keyboard control is identical to 5_so100_keyboard_ee_control.py

YOLO stream displays object detection but does NOT control the robot
Video stream and robot control are completely independent
"""

import time
import logging
import traceback
import math
import cv2
import numpy as np
import threading
from ultralytics import YOLOE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Joint calibration coefficients - manually edit
# Format: [joint_name, zero_position_offset(degrees), scale_factor]
# JOINT_CALIBRATION = [
#     ['shoulder_pan', 6.0, 1.0],      # Joint1: zero position offset, scale factor
#     ['shoulder_lift', 2.0, 0.97],     # Joint2: zero position offset, scale factor
#     ['elbow_flex', 0.0, 1.05],        # Joint3: zero position offset, scale factor
#     ['wrist_flex', 0.0, 0.94],        # Joint4: zero position offset, scale factor
#     ['wrist_roll', 0.0, 0.5],        # Joint5: zero position offset, scale factor
#     ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scale factor
# ]
JOINT_CALIBRATION = [
    ['shoulder_pan', 0.0, 1.0],      # Joint1: zero position offset, scale factor
    ['shoulder_lift', 0.0, 1.0],     # Joint2: zero position offset, scale factor
    ['elbow_flex', 0.0, 1.0],        # Joint3: zero position offset, scale factor
    ['wrist_flex', 0.0, 1.0],        # Joint4: zero position offset, scale factor
    ['wrist_roll', 0.0, 1.0],        # Joint5: zero position offset, scale factor
    ['gripper', 0.0, 1.0],           # Joint6: zero position offset, scale factor
]

# 2D visual-servo tuning. Test these signs at low speed with the camera on the end effector.
# This is image servoing, not 3D position control: horizontal pixels steer shoulder_pan,
# while vertical pixels move one axis of the existing planar end-effector controller.
VISUAL_SERVO_CONFIG = {
    "control_hz": 20.0,  # Deliberately slower than P control so target updates remain smooth.
    "deadband_px": 10.0,  # Ignore small detector/tracker jitter near the image center.
    # Desired target center relative to the image center, in pixels: +x is right and +y is down.
    # For example, (80, -40) keeps the tracked object 80 px right and 40 px above image center.
    "target_offset_x_px": 0.0,
    "target_offset_y_px": -50.0,
    "smoothing_alpha": 0.2,  # EMA weight: higher reacts faster but transmits more visual noise.
    "target_timeout_s": 0.40,  # Stop moving if the locked track has not appeared recently.
    "pan_degrees_per_pixel": 0.05,  # Horizontal-pixel to shoulder_pan sign/gain; tune on hardware.
    "max_pan_step_deg": 1.0,  # Hard rate limit per visual-servo update.
    "image_y_meters_per_pixel": -0.01,  # Vertical-pixel to planar EE sign/gain; tune on hardware.
    "max_ee_step_m": 0.002,  # Hard limit on the EE target displacement per update.
    "image_y_ee_axis": "y",  # Use "x" instead only after verifying the physical response.
}


class VisualServoState:
    """Thread-safe bridge: video writes measurements, robot control reads them."""

    def __init__(self, target_label, camera_on_end_effector):
        self.target_label = target_label
        self.camera_on_end_effector = camera_on_end_effector
        self.enabled = False
        self.locked_id = None
        self.reacquire_requested = False
        self.smoothed_error = None
        self.last_seen_time = 0.0
        self._lock = threading.Lock()

    def toggle_enabled(self):
        with self._lock:
            # With a fixed external camera, moving the arm cannot recenter the object image.
            # Require an eye-in-hand camera before allowing autonomous motion.
            if not self.camera_on_end_effector:
                return False, "Visual servo requires a camera rigidly mounted on the end effector."

            self.enabled = not self.enabled
            if self.enabled:
                # A new activation never reuses an old ID or a stale filtered error.
                self.locked_id = None
                self.reacquire_requested = True
                self.smoothed_error = None
                return True, f"Visual servo enabled; acquiring a '{self.target_label}' track..."

            return False, "Visual servo paused."

    def request_reacquire(self):
        with self._lock:
            # The next video frame with a matching detection will select a fresh ID.
            self.locked_id = None
            self.reacquire_requested = True
            self.smoothed_error = None

    def update_from_candidates(self, candidates, frame_width, frame_height):
        """Lock a requested target and update its filtered desired-point error."""
        desired_x = frame_width / 2.0 + VISUAL_SERVO_CONFIG["target_offset_x_px"]
        desired_y = frame_height / 2.0 + VISUAL_SERVO_CONFIG["target_offset_y_px"]
        now = time.monotonic()
        lock_message = None

        with self._lock:
            candidate = None
            if self.reacquire_requested and candidates:
                # At acquisition time, select the requested class nearest the desired reticle.
                # Subsequent frames follow only this ByteTrack ID, avoiding class-level jumps.
                candidate = min(
                    candidates,
                    key=lambda item: (item["center"][0] - desired_x) ** 2 + (item["center"][1] - desired_y) ** 2,
                )
                self.locked_id = candidate["track_id"]
                self.reacquire_requested = False
                self.smoothed_error = None
                lock_message = f"Locked '{self.target_label}' tracking ID {self.locked_id}."

            if self.locked_id is not None:
                candidate = next((item for item in candidates if item["track_id"] == self.locked_id), None)

            if candidate is not None:
                raw_error = (
                    candidate["center"][0] - desired_x,
                    candidate["center"][1] - desired_y,
                )
                if self.smoothed_error is None:
                    self.smoothed_error = raw_error
                else:
                    # Exponential moving average damps detector and tracker frame-to-frame jitter.
                    alpha = VISUAL_SERVO_CONFIG["smoothing_alpha"]
                    self.smoothed_error = (
                        alpha * raw_error[0] + (1.0 - alpha) * self.smoothed_error[0],
                        alpha * raw_error[1] + (1.0 - alpha) * self.smoothed_error[1],
                    )
                self.last_seen_time = now

        return candidate, lock_message

    def get_control_error(self):
        """Return a fresh filtered error only while autonomous visual servo is enabled."""
        with self._lock:
            is_fresh = time.monotonic() - self.last_seen_time <= VISUAL_SERVO_CONFIG["target_timeout_s"]
            # Return None for a lost target so the robot holds its last target rather than searching.
            if not self.enabled or self.locked_id is None or self.smoothed_error is None or not is_fresh:
                return None
            return self.smoothed_error

    def get_display_status(self):
        with self._lock:
            if self.locked_id is None:
                status = "acquiring" if self.reacquire_requested else "not locked"
            elif time.monotonic() - self.last_seen_time > VISUAL_SERVO_CONFIG["target_timeout_s"]:
                status = f"ID {self.locked_id} lost"
            else:
                status = f"ID {self.locked_id} locked"
            mode = "ON" if self.enabled else "OFF"
            return f"Servo {mode} | {self.target_label} | {status}"


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def apply_deadband(error, deadband):
    """Remove a centered region from the error while preserving motion outside it."""
    if abs(error) <= deadband:
        return 0.0
    return error - math.copysign(deadband, error)


def extract_tracking_candidates(result, target_label):
    """Extract tracked instances of the requested YOLO class from one result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0 or boxes.id is None:
        return []

    # boxes.id is populated by model.track(); an ordinary model(...) inference has no IDs.
    boxes_xyxy = boxes.xyxy.cpu().tolist()
    class_ids = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist()
    candidates = []

    for box, class_id, confidence, track_id in zip(boxes_xyxy, class_ids, confidences, track_ids, strict=True):
        class_name = str(result.names[class_id])
        if class_name.casefold() != target_label.casefold():
            continue
        left, top, right, bottom = box
        candidates.append(
            {
                "track_id": int(track_id),
                "confidence": confidence,
                "center": ((left + right) / 2.0, (top + bottom) / 2.0),
            }
        )

    return candidates


def draw_visual_servo_overlay(frame, visual_servo, tracked_candidate):
    """Show the desired tracking point and selected target without changing detections."""
    height, width = frame.shape[:2]
    desired_point = (
        round(width / 2.0 + VISUAL_SERVO_CONFIG["target_offset_x_px"]),
        round(height / 2.0 + VISUAL_SERVO_CONFIG["target_offset_y_px"]),
    )
    cv2.drawMarker(frame, desired_point, (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

    if tracked_candidate is not None:
        # Yellow circle = locked object's center; arrow = residual error from desired point.
        target_center = tuple(round(value) for value in tracked_candidate["center"])
        cv2.circle(frame, target_center, 8, (0, 255, 255), 2)
        cv2.arrowedLine(frame, desired_point, target_center, (0, 255, 255), 2, tipLength=0.15)

    cv2.putText(
        frame,
        visual_servo.get_display_status(),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def apply_visual_servo(robot_targets, current_x, current_y, visual_servo):
    """Convert a filtered image error into conservative existing EE and pan targets."""
    error = visual_servo.get_control_error()
    if error is None:
        return current_x, current_y

    # Work only with the part of the error outside the centered deadband.
    horizontal_error = apply_deadband(error[0], VISUAL_SERVO_CONFIG["deadband_px"])
    vertical_error = apply_deadband(error[1], VISUAL_SERVO_CONFIG["deadband_px"])

    # Horizontal image error changes base pan. Reverse pan_degrees_per_pixel if hardware moves away.
    pan_step = clamp(
        horizontal_error * VISUAL_SERVO_CONFIG["pan_degrees_per_pixel"],
        -VISUAL_SERVO_CONFIG["max_pan_step_deg"],
        VISUAL_SERVO_CONFIG["max_pan_step_deg"],
    )
    robot_targets["shoulder_pan"] += pan_step

    # Vertical image error moves the selected planar EE axis through the existing IK implementation.
    # It intentionally does not command depth: a monocular image has no metric distance measurement.
    ee_step = clamp(
        vertical_error * VISUAL_SERVO_CONFIG["image_y_meters_per_pixel"],
        -VISUAL_SERVO_CONFIG["max_ee_step_m"],
        VISUAL_SERVO_CONFIG["max_ee_step_m"],
    )
    if VISUAL_SERVO_CONFIG["image_y_ee_axis"] == "x":
        current_x += ee_step
    else:
        current_y += ee_step

    # Keep all shoulder/elbow motion on the script's existing x/y -> inverse_kinematics path.
    shoulder_lift, elbow_flex = inverse_kinematics(current_x, current_y)
    robot_targets["shoulder_lift"] = shoulder_lift
    robot_targets["elbow_flex"] = elbow_flex
    return current_x, current_y


def apply_joint_calibration(joint_name, raw_position):
    """
    Apply joint calibration coefficients

    Args:
        joint_name: Joint name
        raw_position: Raw position value

    Returns:
        calibrated_position: Calibrated position value
    """
    for joint_cal in JOINT_CALIBRATION:
        if joint_cal[0] == joint_name:
            offset = joint_cal[1]  # Zero position offset
            scale = joint_cal[2]   # Scale factor
            calibrated_position = (raw_position - offset) * scale
            return calibrated_position
    return raw_position  # If no calibration coefficient found, return raw value


def inverse_kinematics(x, y, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

    Parameters:
        x: End effector x coordinate
        y: End effector y coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

    # Calculate distance from origin to target point
    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2  # Maximum reachable distance

    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        y *= scale_factor
        r = r_max

    # If target point is less than minimum workspace (|l1-l2|), scale it
    r_min = abs(l1 - l2)
    if r < r_min and r > 0:
        scale_factor = r_min / r
        x *= scale_factor
        y *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    # Convert theta1 and theta2 to joint2 and joint3 angles
    joint2 = theta1 + theta1_offset
    joint3 = theta2 + theta2_offset

    # Ensure angles are within URDF limits
    joint2 = max(-0.1, min(3.45, joint2))
    joint3 = max(-0.2, min(math.pi, joint3))

    # Convert from radians to degrees
    joint2_deg = math.degrees(joint2)
    joint3_deg = math.degrees(joint3)

    joint2_deg = 90-joint2_deg
    joint3_deg = joint3_deg-90

    return joint2_deg, joint3_deg


def move_to_zero_position(robot, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: Robot instance
        duration: Time required to move to zero position (seconds)
        kp: Proportional gain
    """
    print("Using P control to slowly move robot to zero position...")

    # Get current robot state
    current_obs = robot.get_observation()

    # Extract current joint positions
    current_positions = {}
    for key, value in current_obs.items():
        if key.endswith('.pos'):
            motor_name = key.removesuffix('.pos')
            current_positions[motor_name] = value

    # Zero position target
    zero_positions = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }

    # Calculate control steps
    control_freq = 50  # 60Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(f"Will move to zero position in {duration} seconds using P control, control frequency: {control_freq}Hz, proportional gain: {kp}")

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                # Apply calibration coefficients
                calibrated_value = apply_joint_calibration(motor_name, value)
                current_positions[motor_name] = calibrated_value

        # P control calculation
        robot_action = {}
        for joint_name, target_pos in zero_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)

        # Show progress
        if step % (control_freq // 2) == 0:  # Show progress every 0.5 seconds
            progress = (step / total_steps) * 100
            print(f"Moving to zero position progress: {progress:.1f}%")

        time.sleep(step_time)

    print("Robot has moved to zero position")


def return_to_start_position(robot, start_positions, kp=0.5, control_freq=50):
    """
    Use P control to return to start position

    Args:
        robot: Robot instance
        start_positions: Start joint positions dictionary
        kp: Proportional gain
        control_freq: Control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                current_positions[motor_name] = value  # Don't apply calibration coefficients

        # P control calculation
        robot_action = {}
        total_error = 0
        for joint_name, target_pos in start_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]
                error = target_pos - current_pos
                total_error += abs(error)

                # P control: output = Kp * error
                control_output = kp * error

                # Convert control output to position command
                new_position = current_pos + control_output
                robot_action[f"{joint_name}.pos"] = new_position

        # Send action to robot
        if robot_action:
            robot.send_action(robot_action)

        # Check if start position is reached
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


# Video thread: owns capture/inference/display and publishes only image-space measurements.
# The robot-control thread remains the sole writer of motor targets.
def video_stream_loop(model, cap, stop_event, visual_servo):
    """
    Run low-latency YOLO display independently from robot control.

    The reported display FPS includes camera capture, model inference, annotation,
    and OpenCV window handling.
    """
    print("Starting YOLO video stream...")
    fps_window_start = time.perf_counter()
    fps_frame_count = 0

    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available")
                continue

            # persist=True retains ByteTrack state across frames, making boxes.id stable while visible.
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.25, verbose=False)
            result = results[0]
            candidates = extract_tracking_candidates(result, visual_servo.target_label)
            tracked_candidate, lock_message = visual_servo.update_from_candidates(
                candidates, frame.shape[1], frame.shape[0]
            )
            if lock_message:
                print(lock_message)

            if result.boxes is None or len(result.boxes) == 0:
                # No objects detected - show original frame
                annotated_frame = frame
            else:
                # Show detection results
                annotated_frame = result.plot()

            # Overlay makes the selected ID and residual centering error visible before enabling motion.
            draw_visual_servo_overlay(annotated_frame, visual_servo, tracked_candidate)

            # Show detection results in a window
            cv2.imshow("YOLO Live Detection", annotated_frame)

            # 'q' controls shoulder_pan, so use an otherwise unused key here.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("v"):
                stop_event.set()
                break

            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_window_start
            if elapsed >= 1.0:
                print(f"Display FPS: {fps_frame_count / elapsed:.1f}")
                fps_frame_count = 0
                fps_window_start = time.perf_counter()

        except Exception as e:
            print(f"Video stream error: {e}")
            break

    print("Video stream ended")
    cv2.destroyAllWindows()


def connect_robot(robot, attempts=3, retry_delay_s=1.0):
    """Connect the motor bus, retrying transient serial write timeouts."""
    for attempt in range(1, attempts + 1):
        try:
            robot.connect()
            return
        except ConnectionError:
            if robot.bus.is_connected:
                robot.bus.disconnect(disable_torque=False)

            if attempt == attempts:
                raise

            print(f"Robot connection attempt {attempt}/{attempts} failed; retrying in {retry_delay_s:.1f}s...")
            time.sleep(retry_delay_s)


def p_control_loop(
    robot, keyboard, target_positions, start_positions, current_x, current_y, visual_servo, kp=0.5, control_freq=50
):
    """
    P control loop - identical to 5_so100_keyboard_ee_control.py

    Args:
        robot: robot instance
        keyboard: keyboard instance
        target_positions: target joint position dictionary
        start_positions: start joint position dictionary
        current_x: current x coordinate
        current_y: current y coordinate
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    control_period = 1.0 / control_freq

    # Initialize pitch control variables
    pitch = 0.0  # Initial pitch adjustment
    pitch_step = 1  # Pitch adjustment step size
    # Edge detection makes C and L one-shot commands, even if a key is held down.
    previous_pressed_keys = set()
    last_visual_servo_update = 0.0
    # Manual motion temporarily has priority over autonomous visual-servo target updates.
    manual_motion_keys = {"q", "a", "w", "s", "e", "d", "r", "f", "t", "g", "y", "h"}

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()

            pressed_keys = set(keyboard_action) if keyboard_action else set()
            newly_pressed_keys = pressed_keys - previous_pressed_keys
            previous_pressed_keys = pressed_keys

            for key in newly_pressed_keys:
                if key == "c":
                    _, message = visual_servo.toggle_enabled()
                    print(message)
                elif key == "l":
                    visual_servo.request_reacquire()
                    print(f"Reacquiring a '{visual_servo.target_label}' track...")

            if keyboard_action:
                # Process keyboard input, update target positions
                for key, value in keyboard_action.items():
                    if key == "x":
                        # Exit program, first return to start position
                        print("Exit command detected, returning to start position...")
                        return_to_start_position(robot, start_positions, 0.2, control_freq)
                        return

                    # Joint control mapping
                    joint_controls = {
                        "q": ("shoulder_pan", -1),  # Joint 1 decrease
                        "a": ("shoulder_pan", 1),  # Joint 1 increase
                        "t": ("wrist_roll", -1),  # Joint 5 decrease
                        "g": ("wrist_roll", 1),  # Joint 5 increase
                        "y": ("gripper", -1),  # Joint 6 decrease
                        "h": ("gripper", 1),  # Joint 6 increase
                    }

                    # x,y coordinate control
                    xy_controls = {
                        "w": ("x", -0.004),  # x decrease
                        "s": ("x", 0.004),  # x increase
                        "e": ("y", -0.004),  # y decrease
                        "d": ("y", 0.004),  # y increase
                    }

                    # Pitch control
                    if key == "r":
                        pitch += pitch_step
                        print(f"Increase pitch adjustment: {pitch:.3f}")
                    elif key == "f":
                        pitch -= pitch_step
                        print(f"Decrease pitch adjustment: {pitch:.3f}")

                    if key in joint_controls:
                        joint_name, delta = joint_controls[key]
                        if joint_name in target_positions:
                            current_target = target_positions[joint_name]
                            new_target = int(current_target + delta)
                            target_positions[joint_name] = new_target
                            print(f"Update target position {joint_name}: {current_target} -> {new_target}")

                    elif key in xy_controls:
                        coord, delta = xy_controls[key]
                        if coord == "x":
                            current_x += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions["shoulder_lift"] = joint2_target
                            target_positions["elbow_flex"] = joint3_target
                            print(
                                f"Update x coordinate: {current_x:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )
                        elif coord == "y":
                            current_y += delta
                            # Calculate target angles for joint2 and joint3
                            joint2_target, joint3_target = inverse_kinematics(current_x, current_y)
                            target_positions["shoulder_lift"] = joint2_target
                            target_positions["elbow_flex"] = joint3_target
                            print(
                                f"Update y coordinate: {current_y:.4f}, joint2={joint2_target:.3f}, joint3={joint3_target:.3f}"
                            )

            manual_motion_active = bool(pressed_keys & manual_motion_keys)
            now = time.monotonic()
            # The P loop remains at 50 Hz. Visual servo contributes a new high-level target at 8 Hz.
            # Holding a manual movement key freezes this autonomous contribution until it is released.
            if (
                not manual_motion_active
                and now - last_visual_servo_update >= 1.0 / VISUAL_SERVO_CONFIG["control_hz"]
            ):
                current_x, current_y = apply_visual_servo(target_positions, current_x, current_y, visual_servo)
                last_visual_servo_update = now

            # Apply pitch adjustment to wrist_flex
            # Calculate wrist_flex target position based on shoulder_lift and elbow_flex
            if "shoulder_lift" in target_positions and "elbow_flex" in target_positions:
                target_positions["wrist_flex"] = (
                    -target_positions["shoulder_lift"] - target_positions["elbow_flex"] + pitch
                )
                # Show current pitch value (display every 100 steps to avoid screen flooding)
                if hasattr(p_control_loop, "step_counter"):
                    p_control_loop.step_counter += 1
                else:
                    p_control_loop.step_counter = 0

                if p_control_loop.step_counter % 100 == 0:
                    print(
                        f"Current pitch adjustment: {pitch:.3f}, wrist_flex target: {target_positions['wrist_flex']:.3f}"
                    )

            # Get current robot state
            current_obs = robot.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    # Apply calibration coefficients
                    calibrated_value = apply_joint_calibration(motor_name, value)
                    current_positions[motor_name] = calibrated_value

            # P control calculation
            robot_action = {}
            for joint_name, target_pos in target_positions.items():
                if joint_name in current_positions:
                    current_pos = current_positions[joint_name]
                    error = target_pos - current_pos

                    # P control: output = Kp * error
                    control_output = kp * error

                    # Convert control output to position command
                    new_position = current_pos + control_output
                    robot_action[f"{joint_name}.pos"] = new_position

            # Send action to robot
            if robot_action:
                robot.send_action(robot_action)

            time.sleep(control_period)

        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break


def main():
    """Main function"""
    print("LeRobot Keyboard Control + Independent YOLO Display")
    print("="*60)

    try:
        # Import necessary modules
        # from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

        from lerobot.robots.so_follower.so_follower import SO100Follower
        from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
        # from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
        from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

        # Get port
        port = input("Please enter SO100 robot USB port (e.g.: /dev/ttyACM0): ").strip()

        # If Enter is pressed directly, use default port
        if not port:
            port = "/dev/ttyACM0"
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")

        # Configure robot
        # robot_config = SO100FollowerConfig(port=port)
        robot_config = SO100FollowerConfig(
            port=port,
            id="xlerobot_left",     # xlerobot_left (ttyACM1), xlerobot_right (ttyACM0)
            use_degrees=True,
        )
        robot = SO100Follower(robot_config)

        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)

        # Connect devices
        connect_robot(robot)
        keyboard.connect()

        print("Devices connected successfully!")

        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ['y', 'yes']:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ['n', 'no']:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")

        # Read starting joint angles
        print("Reading starting joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith('.pos'):
                motor_name = key.removesuffix('.pos')
                start_positions[motor_name] = int(value)  # Don't apply calibration coefficients

        print("Starting joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")

        # Move to zero position
        move_to_zero_position(robot, duration=3.0)

        # Initialize target positions as current positions (integers)
        target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }

        # Initialize x,y coordinate control
        x0, y0 = 0.1629, 0.1131
        current_x, current_y = x0, y0
        print(f"Initialize end effector position: x={current_x:.4f}, y={current_y:.4f}")

        # Initialize YOLO and camera
        model = YOLOE("yoloe-26l-seg.pt")

        # Get detection targets from user input
        print("\n" + "="*60)
        print("YOLO Detection Target Setup")
        print("="*60)
        target_input = input("Enter objects to detect (separate multiple objects with commas, e.g., bottle,cup,mouse): ").strip()

        # If Enter is pressed directly, use default targets
        if not target_input:
            target_objects = ["bottle"]
            print(f"Using default targets: {target_objects}")
        else:
            # Parse multiple objects separated by commas
            target_objects = [obj.strip() for obj in target_input.split(',') if obj.strip()]
            print(f"Detection targets: {target_objects}")

        # Set text prompt to detect the specified objects
        model.set_classes(target_objects, model.get_text_pe(target_objects))

        target_lookup = {target.casefold(): target for target in target_objects}
        while True:
            servo_target_input = input(
                f"Select 2D visual-servo target from {target_objects} (default: {target_objects[0]}): "
            ).strip()
            if not servo_target_input:
                servo_target = target_objects[0]
                break
            servo_target = target_lookup.get(servo_target_input.casefold())
            if servo_target is not None:
                break
            print("Choose one of the configured detection targets.")

        # The answer gates C: centered-image servo is meaningful only for a camera that moves with the arm.
        camera_on_end_effector = input(
            "Is the camera rigidly mounted on the end effector? (y/n, required for centered-target servo): "
        ).strip().lower() in {"y", "yes"}
        visual_servo = VisualServoState(servo_target, camera_on_end_effector)

        # List available cameras and prompt user
        def list_cameras(max_index=5):
            available = []
            for idx in range(max_index):
                cap_test = cv2.VideoCapture(idx)
                if cap_test.isOpened():
                    available.append(idx)
                    cap_test.release()
            return available

        cameras = list_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print(f"Available cameras: {cameras}")
        selected = int(input(f"Select camera index from {cameras}: "))
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            print("Camera not found!")
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Control instructions:")
        print("Keyboard control (independent of video stream):")
        print("- Q/A: Joint 1 (shoulder_pan) decrease/increase")
        print("- W/S: Control end effector x coordinate (joint2+3)")
        print("- E/D: Control end effector y coordinate (joint2+3)")
        print("- R/F: Pitch adjustment increase/decrease (affects wrist_flex)")
        print("- T/G: Joint 5 (wrist_roll) decrease/increase")
        print("- Y/H: Joint 6 (gripper) close/open")
        print("- C: Toggle 2D visual servo (requires end-effector-mounted camera)")
        print("- L: Reacquire and lock the selected tracking target")
        print("- X: Exit program (return to start position first)")
        print("- ESC: Exit program")
        print("")
        print("Video stream:")
        print("- Independent YOLO detection display (no robot control)")
        print("- Display FPS is printed once per second")
        print(f"- 2D servo target: {servo_target}; starts paused and locks a ByteTrack ID when enabled")
        print("- V (in YOLO window): Exit video stream")
        print("="*60)
        print("Note: Video stream and keyboard control are completely independent")

        # Start video stream in a separate thread
        video_stop_event = threading.Event()
        video_thread = threading.Thread(
            target=video_stream_loop,
            args=(model, cap, video_stop_event, visual_servo),
        )
        video_thread.start()

        # Start keyboard control loop (main thread)
        p_control_loop(
            robot,
            keyboard,
            target_positions,
            start_positions,
            current_x,
            current_y,
            visual_servo,
            kp=0.5,
            control_freq=50,
        )

        # Stop the display in the same thread that owns the OpenCV window.
        video_stop_event.set()
        video_thread.join(timeout=2.0)

        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        cap.release()
        print("Program ended")

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Is the robot correctly connected")
        print("2. Is the USB port correct")
        print("3. Do you have sufficient permissions to access USB device")
        print("4. Is the robot correctly configured")


if __name__ == "__main__":
    main()
