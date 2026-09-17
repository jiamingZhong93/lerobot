#!/usr/bin/env python3
"""
Simplified keyboard control for SO100/SO101 robot
Fixed action format conversion issues
Uses P control, keyboard only changes target joint angles
"""

import logging
import math
import time
import traceback
from dataclasses import dataclass
from threading import Event
from joyconrobotics import JoyconRobotics


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# These are normalized LeRobot command targets. SOFollower already applies the
# motor calibration associated with its robot id when reading and writing.
ZERO_JOINT_TARGETS = (
    ("shoulder_pan", 0.0),
    ("shoulder_lift", 0.0),
    ("elbow_flex", 0.0),
    ("wrist_flex", 0.0),
    ("wrist_roll", 0.0),
    ("gripper", 0.0),
)


@dataclass(frozen=True)
class JoyconBindings:
    """Joy-Con API and event names for one physical controller side."""

    vertical_stick_getter: str
    horizontal_stick_getter: str
    z_up_getter: str
    z_down_getter: str
    x_forward_getter: str
    x_backward_getter: str
    zero_position_getter: str
    recalibrate_event: str
    return_to_start_event: str
    gripper_toggle_event: str
    z_up_label: str
    z_down_label: str
    zero_position_label: str
    return_to_start_label: str
    gripper_toggle_label: str


@dataclass(frozen=True)
class JoyconControlCalibration:
    """Raw stick and button settings for one Joy-Con."""

    bindings: JoyconBindings
    vertical_center: int = 2300
    horizontal_center: int = 2000
    stick_deadzone: int = 300
    stick_range: int = 1000
    position_step_m: float = 0.0008


@dataclass(frozen=True)
class CartesianControlCalibration:
    """IK workspace reference and scaling from Joy-Con motion to robot targets."""

    # These are Cartesian coordinates for the two-link IK model, not joint
    # offsets. The geometric origin is inside the arm's minimum workspace.
    ik_reference_x_m: float = 0.1629
    ik_reference_z_m: float = 0.1131
    shoulder_pan_degrees_per_m: float = 300.0
    pitch_degrees_per_rad: float = -60.0
    roll_degrees_per_rad: float = 50.0
    # Keep this at zero to preserve the original direct roll/pitch response.
    # Raise it only if measured IMU noise moves the wrist while untouched.
    orientation_deadzone_rad: float = 0.0


@dataclass(frozen=True)
class GripperCalibration:
    open_target_percent: float = 60.0
    closed_target_percent: float = 0.0
    initial_joycon_state: float = 0.0


@dataclass(frozen=True)
class ConnectionSettings:
    """Recovery settings for transient motor-bus communication failures."""

    max_attempts: int = 3
    retry_delay_s: float = 1.0


@dataclass(frozen=True)
class MotionSettings:
    """P-control parameters shared by startup and Joy-Con zeroing."""

    zero_position_duration_s: float = 3.0
    zero_position_kp: float = 0.5


@dataclass(frozen=True)
class ArmConfiguration:
    """All side-specific settings. Change only ARM_SIDE to switch arms."""

    robot_id: str
    default_port: str
    joycon_device: str
    joycon: JoyconControlCalibration
    zero_joint_targets: tuple[tuple[str, float], ...] = ZERO_JOINT_TARGETS
    dof_speed: tuple[float, ...] = (2, 2, 2, 1, 1, 1)
    direction_reverse: tuple[int, ...] = (1, 1, 1)

    def zero_targets(self):
        return dict(self.zero_joint_targets)


LEFT_JOYCON_BINDINGS = JoyconBindings(
    vertical_stick_getter="get_stick_left_vertical",
    horizontal_stick_getter="get_stick_left_horizontal",
    z_up_getter="get_button_l",
    z_down_getter="get_button_l_stick",
    x_forward_getter="get_button_up",
    x_backward_getter="get_button_down",
    zero_position_getter="get_button_capture",
    recalibrate_event="minus",
    return_to_start_event="left",
    gripper_toggle_event="zl",
    z_up_label="L",
    z_down_label="L stick",
    zero_position_label="Capture",
    return_to_start_label="D-pad left",
    gripper_toggle_label="ZL",
)

RIGHT_JOYCON_BINDINGS = JoyconBindings(
    vertical_stick_getter="get_stick_right_vertical",
    horizontal_stick_getter="get_stick_right_horizontal",
    z_up_getter="get_button_r",
    z_down_getter="get_button_r_stick",
    x_forward_getter="get_button_x",
    x_backward_getter="get_button_b",
    zero_position_getter="get_button_home",
    recalibrate_event="plus",
    return_to_start_event="y",
    gripper_toggle_event="zr",
    z_up_label="R",
    z_down_label="R stick",
    zero_position_label="Home",
    return_to_start_label="Y",
    gripper_toggle_label="ZR",
)

ARM_SIDE = "right"  # Set to "right" to control the right arm with the right Joy-Con.
ARM_CONFIGURATIONS = {
    "left": ArmConfiguration(
        robot_id="xlerobot_left",
        default_port="/dev/xlerobot_left",
        joycon_device="left",
        joycon=JoyconControlCalibration(bindings=LEFT_JOYCON_BINDINGS),
    ),
    "right": ArmConfiguration(
        robot_id="xlerobot_right",
        default_port="/dev/xlerobot_right",
        joycon_device="right",
        joycon=JoyconControlCalibration(
            bindings=RIGHT_JOYCON_BINDINGS,
            vertical_center=1900,
            horizontal_center=2100,
        ),
    ),
}
if ARM_SIDE not in ARM_CONFIGURATIONS:
    raise ValueError(f"ARM_SIDE must be one of {tuple(ARM_CONFIGURATIONS)}")
ACTIVE_ARM_CONFIGURATION = ARM_CONFIGURATIONS[ARM_SIDE]

CARTESIAN_CONTROL_CALIBRATION = CartesianControlCalibration()
GRIPPER_CALIBRATION = GripperCalibration()
CONNECTION_SETTINGS = ConnectionSettings()
MOTION_SETTINGS = MotionSettings()


class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, control_calibration, **kwargs):
        # The parent constructor starts a control thread, so initialize fields
        # used by common_update before calling it.
        self.control_calibration = control_calibration
        self._pose_recenter_requested = Event()
        self._zero_position_requested = Event()
        self._return_to_start_requested = Event()
        self._zero_position_pressed = False
        super().__init__(device, **kwargs)

    def _stick_delta(self, raw_value, center):
        offset = raw_value - center
        if abs(offset) <= self.control_calibration.stick_deadzone:
            return 0.0
        return self.control_calibration.position_step_m * offset / self.control_calibration.stick_range

    def _joycon_value(self, getter_name):
        return getattr(self.joycon, getter_name)()

    def consume_pose_recenter_request(self):
        if not self._pose_recenter_requested.is_set():
            return False
        self._pose_recenter_requested.clear()
        return True

    def consume_zero_position_request(self):
        if not self._zero_position_requested.is_set():
            return False
        self._zero_position_requested.clear()
        return True

    def consume_return_to_start_request(self):
        if not self._return_to_start_requested.is_set():
            return False
        self._return_to_start_requested.clear()
        return True

    def common_update(self):
        bindings = self.control_calibration.bindings

        # Vertical stick controls X only; horizontal stick controls Y only.
        joycon_stick_v = self._joycon_value(bindings.vertical_stick_getter)
        self.position[0] += self._stick_delta(joycon_stick_v, self.control_calibration.vertical_center) * self.dof_speed[0] * self.direction_reverse[0]

        joycon_stick_h = self._joycon_value(bindings.horizontal_stick_getter)
        self.position[1] += self._stick_delta(joycon_stick_h, self.control_calibration.horizontal_center) * self.dof_speed[1] * self.direction_reverse[1]

        # Z is controlled only by buttons.
        joycon_button_up = self._joycon_value(bindings.z_up_getter)
        if joycon_button_up == 1:
            self.position[2] += self.control_calibration.position_step_m * self.dof_speed[2] * self.direction_reverse[2]

        joycon_button_down = self._joycon_value(bindings.z_down_getter)
        if joycon_button_down == 1:
            self.position[2] -= self.control_calibration.position_step_m * self.dof_speed[2] * self.direction_reverse[2]

        # Additional X-axis controls.
        joycon_button_xup = self._joycon_value(bindings.x_forward_getter)
        joycon_button_xback = self._joycon_value(bindings.x_backward_getter)
        if joycon_button_xup == 1:
            self.position[0] += self.control_calibration.position_step_m * self.dof_speed[0] * self.direction_reverse[0]
        elif joycon_button_xback == 1:
            self.position[0] -= self.control_calibration.position_step_m * self.dof_speed[0] * self.direction_reverse[0]

        # Capture/Home returns the robot to its configured zero position.
        joycon_button_home = self._joycon_value(bindings.zero_position_getter)
        if joycon_button_home == 1:
            self.position = self.offset_position_m.copy()
            if not self._zero_position_pressed:
                self._zero_position_requested.set()
        self._zero_position_pressed = joycon_button_home == 1

        # Gripper and control events.
        for event_type, status in self.button.events():
            if event_type == bindings.recalibrate_event and status == 1:
                self.reset_joycon()
                self.position = self.offset_position_m.copy()
                self._pose_recenter_requested.set()
            elif event_type == bindings.return_to_start_event and status == 1:
                self._return_to_start_requested.set()
            elif event_type == bindings.gripper_toggle_event and status == 1:
                if self.gripper_state == self.gripper_open:
                    self.gripper_state = self.gripper_close
                else:
                    self.gripper_state = self.gripper_open

        return self.position, self.gripper_state, 0


class JoyconTargetTracker:
    """Tracks incremental Cartesian targets while preserving a neutral pose."""

    def __init__(self, calibration, start_x_m=None, start_z_m=None):
        self.calibration = calibration
        self.x_m = calibration.ik_reference_x_m if start_x_m is None else start_x_m
        self.z_m = calibration.ik_reference_z_m if start_z_m is None else start_z_m
        self.shoulder_pan_degrees = 0.0
        self.pitch_degrees = 0.0
        self.roll_degrees = 0.0
        self._last_position = None
        self._orientation_reference = None
        self._pitch_reference_degrees = 0.0
        self._roll_reference_degrees = 0.0

    def recenter(self, pose):
        x, y, z, roll, pitch, yaw = pose
        self._last_position = (x, y, z)
        self._orientation_reference = (roll, pitch)
        self._pitch_reference_degrees = self.pitch_degrees
        self._roll_reference_degrees = self.roll_degrees

    def reset_to_zero(self, pose):
        """Restore the Cartesian and wrist targets to the configured zero pose."""
        self.x_m = self.calibration.ik_reference_x_m
        self.z_m = self.calibration.ik_reference_z_m
        self.shoulder_pan_degrees = 0.0
        self.pitch_degrees = 0.0
        self.roll_degrees = 0.0
        self.recenter(pose)

    def update(self, pose):
        x, y, z, roll, pitch, yaw = pose
        if self._last_position is None or self._orientation_reference is None:
            self.recenter(pose)
            return self.targets()

        previous_x, previous_y, previous_z = self._last_position
        self.x_m += x - previous_x
        self.z_m += z - previous_z
        self.shoulder_pan_degrees += (y - previous_y) * self.calibration.shoulder_pan_degrees_per_m
        self._last_position = (x, y, z)

        reference_roll, reference_pitch = self._orientation_reference
        roll_offset = roll - reference_roll
        pitch_offset = pitch - reference_pitch
        if abs(roll_offset) <= self.calibration.orientation_deadzone_rad:
            roll_offset = 0.0
        if abs(pitch_offset) <= self.calibration.orientation_deadzone_rad:
            pitch_offset = 0.0
        self.roll_degrees = self._roll_reference_degrees + roll_offset * self.calibration.roll_degrees_per_rad
        self.pitch_degrees = self._pitch_reference_degrees + pitch_offset * self.calibration.pitch_degrees_per_rad

        return self.targets()

    def targets(self):
        return (
            self.x_m,
            self.z_m,
            self.shoulder_pan_degrees,
            self.pitch_degrees,
            self.roll_degrees,
        )



def inverse_kinematics(x, z, l1=0.1159, l2=0.1350):
    """
    Calculate inverse kinematics for a 2-link robotic arm, considering joint offsets

    Parameters:
        x: End effector x coordinate
        z: End effector z coordinate
        l1: Upper arm length (default 0.1159 m)
        l2: Lower arm length (default 0.1350 m)

    Returns:
        joint2, joint3: Joint angles in radians as defined in the URDF file
    """
    # Calculate joint2 and joint3 offsets in theta1 and theta2
    theta1_offset = math.atan2(0.028, 0.11257)  # theta1 offset when joint2=0
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset  # theta2 offset when joint3=0

    # Calculate distance from origin to target point
    r = math.hypot(x, z)
    r_max = l1 + l2  # Maximum reachable distance

    # If target point is beyond maximum workspace, scale it to the boundary
    if r > r_max:
        scale_factor = r_max / r
        x *= scale_factor
        z *= scale_factor
        r = r_max

    # Project inner-workspace targets to the minimum reachable radius. At the
    # exact origin there is no direction, so use positive X deterministically.
    r_min = abs(l1 - l2)
    if r < r_min:
        if r == 0:
            x, z = r_min, 0.0
        else:
            scale_factor = r_min / r
            x *= scale_factor
            z *= scale_factor
        r = r_min

    # Use law of cosines to calculate theta2
    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))

    # Calculate theta2 (elbow angle)
    theta2 = math.pi - math.acos(cos_theta2)

    # Calculate theta1 (shoulder angle)
    beta = math.atan2(z, x)
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

    joint2_deg = 90 - joint2_deg
    joint3_deg = joint3_deg - 90

    return joint2_deg, joint3_deg


def move_to_zero_position(robot, zero_positions, duration=3.0, kp=0.5):
    """
    Use P control to slowly move robot to zero position

    Args:
        robot: robot instance
        zero_positions: normalized joint targets for this arm's control home
        duration: time to move to zero position (seconds)
        kp: proportional gain
    """
    print("Using P control to slowly move robot to zero position...")

    zero_positions = zero_positions.copy()

    # Calculate control steps
    control_freq = 50  # 50Hz control frequency
    total_steps = int(duration * control_freq)
    step_time = 1.0 / control_freq

    print(
        f"Will use P control to move to zero position in {duration} seconds, control frequency: {control_freq}Hz, proportional gain: {kp}"
    )

    for step in range(total_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                current_positions[motor_name] = value

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
        robot: robot instance
        start_positions: start joint position dictionary
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    print("Returning to start position...")

    control_period = 1.0 / control_freq
    max_steps = int(5.0 * control_freq)  # Maximum 5 seconds

    for step in range(max_steps):
        # Get current robot state
        current_obs = robot.get_observation()
        current_positions = {}
        for key, value in current_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
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

        # Check if reached start position
        if total_error < 2.0:  # If total error is less than 2 degrees, consider reached
            print("Returned to start position")
            break

        time.sleep(control_period)

    print("Return to start position completed")


def p_control_loop(
    robot, keyboard, target_positions, start_positions, current_x, current_z, joycon_controller, kp=0.5, control_freq=50
):
    """
    P control loop

    Args:
        robot: robot instance
        keyboard: keyboard instance
        target_positions: target joint position dictionary
        start_positions: start joint position dictionary
        current_x: current x coordinate
        current_z: current z coordinate
        joycon_controller: joycon robotics instance
        kp: proportional gain
        control_freq: control frequency (Hz)
    """
    control_period = 1.0 / control_freq
    joint_zero_targets = target_positions.copy()
    target_tracker = JoyconTargetTracker(CARTESIAN_CONTROL_CALIBRATION, current_x, current_z)

    # The IK reference is an internal Cartesian anchor. Its joint solution is
    # subtracted so a neutral Joy-Con holds this arm's configured zero targets.
    zero_shoulder_lift, zero_elbow_flex = inverse_kinematics(current_x, current_z)

    print(f"Starting P control loop, control frequency: {control_freq}Hz, proportional gain: {kp}")

    while True:
        try:
            # Get keyboard input
            keyboard_action = keyboard.get_action()
            if keyboard_action and "x" in keyboard_action:
                print("Return command detected, returning to start position...")
                return_to_start_position(robot, start_positions, 0.2, control_freq)
                return

            pose, gripper, _ = joycon_controller.get_control()
            if joycon_controller.consume_return_to_start_request():
                print("Joy-Con return command detected, returning to start position...")
                return_to_start_position(robot, start_positions, 0.2, control_freq)
                return
            if joycon_controller.consume_zero_position_request():
                print("Joy-Con zero-position command detected, moving to zero position...")
                joycon_controller.gripper_state = joycon_controller.gripper_close
                move_to_zero_position(
                    robot,
                    joint_zero_targets,
                    duration=MOTION_SETTINGS.zero_position_duration_s,
                    kp=MOTION_SETTINGS.zero_position_kp,
                )
                target_positions.clear()
                target_positions.update(joint_zero_targets)
                zero_pose, _, _ = joycon_controller.get_control()
                target_tracker.reset_to_zero(zero_pose)
                print("Robot returned to zero position")
                continue
            if joycon_controller.consume_pose_recenter_request():
                target_tracker.recenter(pose)
                print("Joy-Con control reference recentered")

            current_x, current_z, shoulder_pan, pitch, roll = target_tracker.update(pose)

            # Convert Cartesian changes relative to the IK anchor into joint
            # changes relative to the configured control-home targets.
            joint2_target, joint3_target = inverse_kinematics(current_x, current_z)
            shoulder_lift_delta = joint2_target - zero_shoulder_lift
            elbow_flex_delta = joint3_target - zero_elbow_flex
            target_positions["shoulder_pan"] = joint_zero_targets["shoulder_pan"] + shoulder_pan
            target_positions["shoulder_lift"] = joint_zero_targets["shoulder_lift"] + shoulder_lift_delta
            target_positions["elbow_flex"] = joint_zero_targets["elbow_flex"] + elbow_flex_delta
            target_positions["wrist_flex"] = joint_zero_targets["wrist_flex"] - shoulder_lift_delta - elbow_flex_delta + pitch
            target_positions["wrist_roll"] = joint_zero_targets["wrist_roll"] + roll

            if gripper == joycon_controller.gripper_open:
                target_positions["gripper"] = GRIPPER_CALIBRATION.open_target_percent
            else:
                target_positions["gripper"] = GRIPPER_CALIBRATION.closed_target_percent
            
            # Get current robot state
            current_obs = robot.get_observation()

            # Extract current joint positions
            current_positions = {}
            for key, value in current_obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    current_positions[motor_name] = value

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


def connect_robot(robot, settings=CONNECTION_SETTINGS):
    """Retry connection after an intermittent motor-bus status timeout."""
    for attempt in range(1, settings.max_attempts + 1):
        try:
            robot.connect()
            return
        except ConnectionError:
            if robot.bus.is_connected:
                # Close only the serial port before retrying. Do not send more
                # torque commands while recovering from a write timeout.
                robot.bus.disconnect(disable_torque=False)

            if attempt == settings.max_attempts:
                raise

            print(
                f"Robot connection attempt {attempt}/{settings.max_attempts} failed; "
                f"retrying in {settings.retry_delay_s:.1f}s..."
            )
            time.sleep(settings.retry_delay_s)


def main():
    """Main function"""
    print("LeRobot Simplified Keyboard Control Example (P Control)")
    print("=" * 50)

    try:
        # Import necessary modules
        # from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig

        from lerobot.robots.so_follower.so_follower import SO100Follower
        from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig
        # from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig

        from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
        from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

        # Get port
        port = input(
            f"Please enter the USB port for the {ARM_SIDE} SO100 robot "
            f"(press Enter for {ACTIVE_ARM_CONFIGURATION.default_port}): "
        ).strip()

        # If directly press Enter, use default port
        if not port:
            port = ACTIVE_ARM_CONFIGURATION.default_port
            print(f"Using default port: {port}")
        else:
            print(f"Connecting to port: {port}")

        # Configure robot
        robot_config = SO100FollowerConfig(
            port=port,
            id=ACTIVE_ARM_CONFIGURATION.robot_id,
            use_degrees=True,
        )
        robot = SO100Follower(robot_config)

        # Configure keyboard
        keyboard_config = KeyboardTeleopConfig()
        keyboard = KeyboardTeleop(keyboard_config)

        # Connect devices
        connect_robot(robot)
        keyboard.connect()
        joycon_controller = FixedAxesJoyconRobotics(
            ACTIVE_ARM_CONFIGURATION.joycon_device,
            control_calibration=ACTIVE_ARM_CONFIGURATION.joycon,
            dof_speed=list(ACTIVE_ARM_CONFIGURATION.dof_speed),
            direction_reverse=list(ACTIVE_ARM_CONFIGURATION.direction_reverse),
            gripper_state=GRIPPER_CALIBRATION.initial_joycon_state,
        )

        bindings = ACTIVE_ARM_CONFIGURATION.joycon.bindings
        print(f"Joy-Con fixed-axis controls for the {ARM_SIDE} arm:")
        print("Vertical stick: X axis; horizontal stick: shoulder_pan")
        print(f"{bindings.z_up_label}: Z up; {bindings.z_down_label}: Z down")
        print(f"{bindings.zero_position_label}: return the robot to zero position")
        print(f"{bindings.return_to_start_label}: return to starting position")
        print(f"{bindings.gripper_toggle_label}: toggle gripper; keyboard X: return to starting position")
        print("Press Ctrl+C to stop")
        print()


        print("Device connection successful!")

        # Ask whether to recalibrate
        while True:
            calibrate_choice = input("Do you want to recalibrate the robot? (y/n): ").strip().lower()
            if calibrate_choice in ["y", "yes"]:
                print("Starting recalibration...")
                robot.calibrate()
                print("Calibration completed!")
                break
            elif calibrate_choice in ["n", "no"]:
                print("Using previous calibration file")
                break
            else:
                print("Please enter y or n")

        # Read initial joint angles
        print("Reading initial joint angles...")
        start_obs = robot.get_observation()
        start_positions = {}
        for key, value in start_obs.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                start_positions[motor_name] = value

        print("Initial joint angles:")
        for joint_name, position in start_positions.items():
            print(f"  {joint_name}: {position}°")

        # Move to zero position
        zero_joint_targets = ACTIVE_ARM_CONFIGURATION.zero_targets()
        move_to_zero_position(
            robot,
            zero_joint_targets,
            duration=MOTION_SETTINGS.zero_position_duration_s,
            kp=MOTION_SETTINGS.zero_position_kp,
        )

        target_positions = zero_joint_targets.copy()

        # Initialize the internal IK reference; it is not a joint offset.
        current_x = CARTESIAN_CONTROL_CALIBRATION.ik_reference_x_m
        current_z = CARTESIAN_CONTROL_CALIBRATION.ik_reference_z_m
        print(f"Initialize IK reference: x={current_x:.4f}, z={current_z:.4f}")

        print("Control notes:")
        print("- Neutral Joy-Con holds the configured zero joint targets")
        print(f"- {bindings.zero_position_label} returns the robot to zero position")
        print(f"- {bindings.return_to_start_label} or keyboard X returns to the starting position")
        print("=" * 50)
        print("Note: Robot will continuously move to target positions")

        # Start P control loop
        p_control_loop(
            robot, keyboard, target_positions, start_positions, current_x, current_z, joycon_controller, kp=0.5, control_freq=50
        )

        # Disconnect
        robot.disconnect()
        keyboard.disconnect()
        print("Program ended")

    except Exception as e:
        print(f"Program execution failed: {e}")
        traceback.print_exc()
        print("Please check:")
        print("1. Whether the robot is properly connected")
        print("2. Whether the USB port is correct")
        print("3. Whether you have sufficient permissions to access USB devices")
        print("4. Whether the robot is properly configured")


if __name__ == "__main__":
    main()
