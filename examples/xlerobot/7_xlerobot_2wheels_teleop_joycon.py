# Run the host:
'''
PYTHONPATH=src python -m lerobot.robots.xlerobot_2wheels.xlerobot_2wheels_host --robot.id=my_xlerobot_2wheels
'''

# Run teleoperation from this repository:
'''
PYTHONPATH=src:/home/jiaming/xlerobot/joycon-robotics python examples/xlerobot/7_xlerobot_2wheels_teleop_joycon.py
'''

# Base: X forward, B backward, Y rotate left, A rotate right.
# Capture/Home zero the left/right arm. Press both within 250 ms to return
# both arms and the head to their startup pose.

import math
import time
from dataclasses import dataclass
from threading import Event

from lerobot.robots.xlerobot_2wheels import XLerobot2WheelsConfig, XLerobot2Wheels
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from joyconrobotics import JoyconRobotics


CONTROL_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ
ROBOT_NAME = "my_xlerobot_2wheels_lab"
CONNECT_ATTEMPTS = 3
CONNECT_RETRY_DELAY_S = 1.0
RETURN_CONTROL_HZ = 50
RETURN_KP = 0.2
RETURN_TIMEOUT_S = 5.0
RETURN_POSITION_TOLERANCE = 2.0
ZERO_POSITION_DURATION_S = 3.0
ZERO_POSITION_KP = 0.5
ZERO_CHORD_WINDOW_S = 0.25
LEFT_WHEEL_DIRECTION = -1
RIGHT_WHEEL_DIRECTION = 1

# Keep these semantic names aligned with 4_xlerobot_2wheels_teleop_keyboard.py.
BASE_KEYMAP = {
    "forward": "i",
    "backward": "k",
    "rotate_left": "j",
    "rotate_right": "l",
    "speed_up": "n",
    "speed_down": "m",
    "quit": "b",
}


@dataclass(frozen=True)
class JoyconBindings:
    vertical_stick_getter: str
    horizontal_stick_getter: str
    z_up_getter: str
    z_down_getter: str
    zero_position_getter: str
    calibration_event: str
    gripper_toggle_event: str
    zero_position_label: str


@dataclass(frozen=True)
class JoyconControlCalibration:
    bindings: JoyconBindings
    vertical_center: int
    horizontal_center: int
    stick_deadzone: int = 300
    stick_range: int = 1000
    position_step_m: float = 0.0008


@dataclass(frozen=True)
class CartesianControlCalibration:
    # These are Cartesian IK coordinates, not motor-position offsets.
    ik_reference_x_m: float = 0.1629
    ik_reference_z_m: float = 0.1131
    shoulder_pan_degrees_per_m: float = 300.0
    pitch_degrees_per_rad: float = -60.0
    roll_degrees_per_rad: float = 50.0
    orientation_deadzone_rad: float = 0.0


LEFT_JOYCON_BINDINGS = JoyconBindings(
    vertical_stick_getter="get_stick_left_vertical",
    horizontal_stick_getter="get_stick_left_horizontal",
    z_up_getter="get_button_l",
    z_down_getter="get_button_l_stick",
    zero_position_getter="get_button_capture",
    calibration_event="minus",
    gripper_toggle_event="zl",
    zero_position_label="Capture",
)
RIGHT_JOYCON_BINDINGS = JoyconBindings(
    vertical_stick_getter="get_stick_right_vertical",
    horizontal_stick_getter="get_stick_right_horizontal",
    z_up_getter="get_button_r",
    z_down_getter="get_button_r_stick",
    zero_position_getter="get_button_home",
    calibration_event="plus",
    gripper_toggle_event="zr",
    zero_position_label="Home",
)
JOYCON_CALIBRATIONS = {
    "left": JoyconControlCalibration(LEFT_JOYCON_BINDINGS, vertical_center=2300, horizontal_center=2000),
    "right": JoyconControlCalibration(RIGHT_JOYCON_BINDINGS, vertical_center=1900, horizontal_center=2100),
}
CARTESIAN_CALIBRATION = CartesianControlCalibration()

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class FixedAxesJoyconRobotics(JoyconRobotics):
    def __init__(self, device, control_calibration, **kwargs):
        # The parent constructor starts the update thread.
        self.control_calibration = control_calibration
        self._zero_position_requested = Event()
        self._zero_position_pressed = False
        super().__init__(device, **kwargs)
    def _joycon_value(self, getter_name):
        return getattr(self.joycon, getter_name)()

    def _stick_delta(self, raw_value, center):
        offset = raw_value - center
        if abs(offset) <= self.control_calibration.stick_deadzone:
            return 0.0
        return self.control_calibration.position_step_m * offset / self.control_calibration.stick_range

    def consume_zero_position_request(self):
        if not self._zero_position_requested.is_set():
            return False
        self._zero_position_requested.clear()
        return True

    def common_update(self):
        calibration = self.control_calibration
        bindings = calibration.bindings

        vertical_stick = self._joycon_value(bindings.vertical_stick_getter)
        self.position[0] += self._stick_delta(vertical_stick, calibration.vertical_center) * self.dof_speed[0] * self.direction_reverse[0]

        horizontal_stick = self._joycon_value(bindings.horizontal_stick_getter)
        self.position[1] += self._stick_delta(horizontal_stick, calibration.horizontal_center) * self.dof_speed[1] * self.direction_reverse[1]

        if self._joycon_value(bindings.z_up_getter) == 1:
            self.position[2] += calibration.position_step_m * self.dof_speed[2] * self.direction_reverse[2]
        if self._joycon_value(bindings.z_down_getter) == 1:
            self.position[2] -= calibration.position_step_m * self.dof_speed[2] * self.direction_reverse[2]

        zero_position_pressed = self._joycon_value(bindings.zero_position_getter) == 1
        if zero_position_pressed:
            self.position = self.offset_position_m.copy()
            if not self._zero_position_pressed:
                self._zero_position_requested.set()
        self._zero_position_pressed = zero_position_pressed

        for event_type, status in self.button.events():
            if event_type == bindings.calibration_event and status == 1:
                self.reset_joycon()
                self.position = self.offset_position_m.copy()
            elif event_type == bindings.gripper_toggle_event and status == 1:
                if self.gripper_state == self.gripper_open:
                    self.gripper_state = self.gripper_close
                else:
                    self.gripper_state = self.gripper_open

        return self.position, self.gripper_state, 0


class JoyconTargetTracker:
    """Track Cartesian and orientation targets relative to an IK home point."""

    def __init__(self, calibration):
        self.calibration = calibration
        self.x_m = calibration.ik_reference_x_m
        self.z_m = calibration.ik_reference_z_m
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

    def reset_to_home(self, pose):
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
        return self.x_m, self.z_m, self.shoulder_pan_degrees, self.pitch_degrees, self.roll_degrees


def inverse_kinematics(x, z, l1=0.1159, l2=0.1350):
    """Return SO101 shoulder-lift and elbow targets in degrees within the reachable workspace."""
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
    radius = math.hypot(x, z)
    radius_min = abs(l1 - l2)
    radius_max = l1 + l2

    if radius > radius_max:
        scale = radius_max / radius
        x, z, radius = x * scale, z * scale, radius_max
    elif radius < radius_min:
        if radius == 0:
            x, z = radius_min, 0.0
        else:
            scale = radius_min / radius
            x, z = x * scale, z * scale
        radius = radius_min

    cos_theta2 = -(radius**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(max(-1.0, min(1.0, cos_theta2)))
    theta1 = math.atan2(z, x) + math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))
    return 90.0 - math.degrees(joint2), math.degrees(joint3) - 90.0


class SimpleTeleopArm:
    def __init__(self, joint_map, initial_obs, prefix, calibration):
        self.joint_map = joint_map
        self.prefix = prefix
        self.start_positions = {
            joint: initial_obs[f"{prefix}_arm_{joint}.pos"] for joint in joint_map
        }
        self.zero_positions = dict.fromkeys(joint_map, 0.0)
        self.control_home = self.zero_positions.copy()
        self.target_positions = self.zero_positions.copy()
        self.target_tracker = JoyconTargetTracker(calibration)
        self.zero_shoulder_lift, self.zero_elbow_flex = inverse_kinematics(
            calibration.ik_reference_x_m, calibration.ik_reference_z_m
        )

    def set_control_home(self, targets, pose):
        self.control_home = targets.copy()
        self.target_positions = targets.copy()
        self.target_tracker.reset_to_home(pose)

    def reset_to_zero(self, pose):
        self.set_control_home(self.zero_positions, pose)

    def return_to_start_target(self, pose):
        self.set_control_home(self.start_positions, pose)

    def update_from_joycon(self, pose, gripper_state, joycon):
        x_m, z_m, shoulder_pan, pitch, roll = self.target_tracker.update(pose)
        shoulder_lift, elbow_flex = inverse_kinematics(x_m, z_m)
        shoulder_lift_delta = shoulder_lift - self.zero_shoulder_lift
        elbow_flex_delta = elbow_flex - self.zero_elbow_flex
        self.target_positions["shoulder_pan"] = self.control_home["shoulder_pan"] + shoulder_pan
        self.target_positions["shoulder_lift"] = self.control_home["shoulder_lift"] + shoulder_lift_delta
        self.target_positions["elbow_flex"] = self.control_home["elbow_flex"] + elbow_flex_delta
        self.target_positions["wrist_flex"] = (
            self.control_home["wrist_flex"] - shoulder_lift_delta - elbow_flex_delta + pitch
        )
        self.target_positions["wrist_roll"] = self.control_home["wrist_roll"] + roll
        self.target_positions["gripper"] = 60.0 if gripper_state == joycon.gripper_open else 0.0

    def p_control_action(self, observation, kp=0.5):
        current = {
            joint: observation[f"{self.prefix}_arm_{joint}.pos"] for joint in self.joint_map
        }
        return {
            f"{self.joint_map[joint]}.pos": current[joint] + kp * (self.target_positions[joint] - current[joint])
            for joint in self.target_positions
        }

    def target_error(self, observation):
        return sum(
            abs(self.target_positions[joint] - observation[f"{self.prefix}_arm_{joint}.pos"])
            for joint in self.target_positions
        )

class SimpleHeadControl:
    def __init__(self, initial_obs):
        self.target_positions = {
            "head_motor_1": initial_obs.get("head_motor_1.pos", 0.0),
            "head_motor_2": initial_obs.get("head_motor_2.pos", 0.0),
        }
        self.start_positions = self.target_positions.copy()
        self.zero_pos = {"head_motor_1": 0.0, "head_motor_2": 0.0}
        self.degree_step = 1.0

    def move_to_zero_position(self):
        self.target_positions = self.zero_pos.copy()

    def return_to_start_target(self):
        self.target_positions = self.start_positions.copy()

    def handle_joycon_input(self, joycon):
        if joycon.joycon.get_button_up() == 1:
            self.target_positions["head_motor_2"] += self.degree_step
        if joycon.joycon.get_button_down() == 1:
            self.target_positions["head_motor_2"] -= self.degree_step
        if joycon.joycon.get_button_left() == 1:
            self.target_positions["head_motor_1"] += self.degree_step
        if joycon.joycon.get_button_right() == 1:
            self.target_positions["head_motor_1"] -= self.degree_step

    def p_control_action(self, observation, kp=0.5):
        current = {motor: observation[f"{motor}.pos"] for motor in HEAD_MOTOR_MAP}
        return {
            f"{motor}.pos": current[motor] + kp * (self.target_positions[motor] - current[motor])
            for motor in self.target_positions
        }

    def target_error(self, observation):
        return sum(
            abs(self.target_positions[motor] - observation[f"{motor}.pos"])
            for motor in HEAD_MOTOR_MAP
        )

def get_base_actions(joycon):
    """Map right Joy-Con face buttons onto the semantic base actions from ``4_``."""
    actions = set()
    if joycon.joycon.get_button_x() == 1:
        actions.add("forward")
    if joycon.joycon.get_button_b() == 1:
        actions.add("backward")
    if joycon.joycon.get_button_y() == 1:
        actions.add("rotate_left")
    if joycon.joycon.get_button_a() == 1:
        actions.add("rotate_right")
    return actions

# Base speed control parameters - adjustable slopes
BASE_ACCELERATION_RATE = 10.0  # acceleration slope (speed/second)
BASE_DECELERATION_RATE = 10.0  # deceleration slope (speed/second)
BASE_MAX_SPEED = 6.0           # maximum speed multiplier
MIN_VELOCITY_THRESHOLD = 0.02 # minimum velocity to send to motors during deceleration

class SmoothBaseController:
    """Acceleration/deceleration controller for the differential-drive base."""

    def __init__(self):
        self.current_speed = 0.0
        self.current_direction = {"x.vel": 0.0, "theta.vel": 0.0}
        self.last_time = time.monotonic()

    def update(self, requested_actions, robot):
        now = time.monotonic()
        dt = min(max(now - self.last_time, 0.001), 0.05)
        self.last_time = now
        speed_setting = robot.speed_levels[robot.speed_index]
        requested_direction = {
            "x.vel": speed_setting["linear"] * ("forward" in requested_actions)
            - speed_setting["linear"] * ("backward" in requested_actions),
            "theta.vel": speed_setting["angular"] * ("rotate_left" in requested_actions)
            - speed_setting["angular"] * ("rotate_right" in requested_actions),
        }
        requested_motion = any(requested_direction.values())

        if requested_motion and requested_direction != self.current_direction:
            self.current_speed = max(0.0, self.current_speed - BASE_DECELERATION_RATE * dt)
            if self.current_speed == 0.0:
                self.current_direction = requested_direction
        elif requested_motion:
            self.current_speed = min(BASE_MAX_SPEED, self.current_speed + BASE_ACCELERATION_RATE * dt)
        else:
            self.current_speed = max(0.0, self.current_speed - BASE_DECELERATION_RATE * dt)

        action = {
            key: value * self.current_speed for key, value in self.current_direction.items()
        }
        for key, value in self.current_direction.items():
            if value and self.current_speed > 0.01 and abs(action[key]) < MIN_VELOCITY_THRESHOLD:
                action[key] = MIN_VELOCITY_THRESHOLD if value > 0 else -MIN_VELOCITY_THRESHOLD
        return action

def build_full_body_action(observation, left_arm, right_arm, head_control, base_action, kp=0.5):
    return {
        **left_arm.p_control_action(observation, kp=kp),
        **right_arm.p_control_action(observation, kp=kp),
        **head_control.p_control_action(observation, kp=kp),
        **base_action,
    }


def move_to_current_targets(robot, left_arm, right_arm, head_control, duration_s, kp=0.5):
    """Move smoothly while continuously holding the entire body and stopping the base."""
    deadline = time.monotonic() + duration_s
    control_period = 1.0 / RETURN_CONTROL_HZ
    while time.monotonic() < deadline:
        observation = robot.get_observation()
        action = build_full_body_action(
            observation,
            left_arm,
            right_arm,
            head_control,
            {"x.vel": 0.0, "theta.vel": 0.0},
            kp=kp,
        )
        robot.send_action(action)
        total_error = (
            left_arm.target_error(observation)
            + right_arm.target_error(observation)
            + head_control.target_error(observation)
        )
        if total_error <= RETURN_POSITION_TOLERANCE:
            return
        precise_sleep(control_period)


def move_arms_to_zero_position(robot, left_arm, right_arm, head_control, left_pose, right_pose):
    print("[MAIN] Moving both arms to the configured zero position...")
    left_arm.reset_to_zero(left_pose)
    right_arm.reset_to_zero(right_pose)
    move_to_current_targets(
        robot,
        left_arm,
        right_arm,
        head_control,
        duration_s=ZERO_POSITION_DURATION_S,
        kp=ZERO_POSITION_KP,
    )


def return_to_start_position(robot, left_arm, right_arm, head_control, left_pose, right_pose):
    print("[MAIN] Returning arms and head to the startup position...")
    left_arm.return_to_start_target(left_pose)
    right_arm.return_to_start_target(right_pose)
    head_control.return_to_start_target()
    move_to_current_targets(
        robot,
        left_arm,
        right_arm,
        head_control,
        duration_s=RETURN_TIMEOUT_S,
        kp=RETURN_KP,
    )


def connect_robot(robot):
    """Retry an intermittent motor-bus status timeout without recalibrating."""
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            robot.connect()
            return
        except ConnectionError:
            for bus in (robot.bus1, robot.bus2):
                if bus.is_connected:
                    bus.disconnect(disable_torque=False)
            if attempt == CONNECT_ATTEMPTS:
                raise
            print(
                f"[MAIN] Robot connection attempt {attempt}/{CONNECT_ATTEMPTS} failed; "
                f"retrying in {CONNECT_RETRY_DELAY_S:.1f}s..."
            )
            time.sleep(CONNECT_RETRY_DELAY_S)

def main():
    robot_config = XLerobot2WheelsConfig(
        id=ROBOT_NAME,
        use_degrees=True,
        teleop_keys=BASE_KEYMAP,
        left_wheel_direction=LEFT_WHEEL_DIRECTION,
        right_wheel_direction=RIGHT_WHEEL_DIRECTION,
    )
    robot = XLerobot2Wheels(robot_config)
    joycon_left = None
    joycon_right = None

    try:
        connect_robot(robot)
        init_rerun(session_name="xlerobot_2wheels_teleop")
        joycon_left = FixedAxesJoyconRobotics(
            "left",
            control_calibration=JOYCON_CALIBRATIONS["left"],
            dof_speed=[2, 2, 2, 1, 1, 1],
            gripper_state=0.0,
        )
        joycon_right = FixedAxesJoyconRobotics(
            "right",
            control_calibration=JOYCON_CALIBRATIONS["right"],
            dof_speed=[2, 2, 2, 1, 1, 1],
            gripper_state=0.0,
        )

        observation = robot.get_observation()
        left_arm = SimpleTeleopArm(
            LEFT_JOINT_MAP, observation, "left", CARTESIAN_CALIBRATION
        )
        right_arm = SimpleTeleopArm(
            RIGHT_JOINT_MAP, observation, "right", CARTESIAN_CALIBRATION
        )
        head_control = SimpleHeadControl(observation)
        base_controller = SmoothBaseController()

        left_pose, _, _ = joycon_left.get_control()
        right_pose, _, _ = joycon_right.get_control()
        move_arms_to_zero_position(robot, left_arm, right_arm, head_control, left_pose, right_pose)

        print("\n" + "=" * 72)
        print("XLeRobot 2Wheels Joy-Con Full-Body Control")
        print("Base: X forward, B backward, Y rotate left, A rotate right")
        print("Left arm: stick, L/L-stick, ZL gripper, Capture zero")
        print("Right arm: stick, R/R-stick, ZR gripper, Home zero")
        print("Left D-pad controls the head; Minus/Plus calibrate the matching Joy-Con")
        print("Capture + Home together: return arms and head to startup pose")
        print("=" * 72 + "\n")

        pending_zero_arms = set()
        zero_request_deadline = 0.0
        while True:
            loop_start = time.monotonic()
            pose_right, gripper_right, _ = joycon_right.get_control()
            pose_left, gripper_left, _ = joycon_left.get_control()
            left_zero_requested = joycon_left.consume_zero_position_request()
            right_zero_requested = joycon_right.consume_zero_position_request()

            if left_zero_requested:
                pending_zero_arms.add("left")
            if right_zero_requested:
                pending_zero_arms.add("right")
            if pending_zero_arms and zero_request_deadline == 0.0:
                zero_request_deadline = time.monotonic() + ZERO_CHORD_WINDOW_S

            if pending_zero_arms == {"left", "right"}:
                return_to_start_position(
                    robot, left_arm, right_arm, head_control, pose_left, pose_right
                )
                pending_zero_arms.clear()
                zero_request_deadline = 0.0
                base_controller = SmoothBaseController()
                continue

            if pending_zero_arms and time.monotonic() >= zero_request_deadline:
                if "left" in pending_zero_arms:
                    joycon_left.gripper_state = joycon_left.gripper_close
                    left_arm.reset_to_zero(pose_left)
                if "right" in pending_zero_arms:
                    joycon_right.gripper_state = joycon_right.gripper_close
                    right_arm.reset_to_zero(pose_right)
                move_to_current_targets(
                    robot, left_arm, right_arm, head_control, ZERO_POSITION_DURATION_S, ZERO_POSITION_KP
                )
                pending_zero_arms.clear()
                zero_request_deadline = 0.0
                base_controller = SmoothBaseController()
                continue

            left_arm.update_from_joycon(pose_left, gripper_left, joycon_left)
            right_arm.update_from_joycon(pose_right, gripper_right, joycon_right)
            head_control.handle_joycon_input(joycon_left)
            base_action = base_controller.update(get_base_actions(joycon_right), robot)
            observation = robot.get_observation()
            action = build_full_body_action(
                observation, left_arm, right_arm, head_control, base_action
            )
            sent_action = robot.send_action(action)
            # log_rerun_data(observation, sent_action)
            precise_sleep(max(0.0, CONTROL_PERIOD_S - (time.monotonic() - loop_start)))
    finally:
        if joycon_left is not None:
            joycon_left.disconnect()
        if joycon_right is not None:
            joycon_right.disconnect()
        if robot.is_connected:
            robot.disconnect()
        print("Teleoperation ended.")

if __name__ == "__main__":
    main()
