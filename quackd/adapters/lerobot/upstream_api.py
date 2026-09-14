"""The only file in quackd allowed to spell a LeRobot name (ADR-0022).

Every constant is tagged VERIFIED (read from upstream source, link given) or UNVERIFIED
(an assumption of ours, with what quackd does about it). `docs/adapters/lerobot.md` is the
human-readable version; `tests/test_upstream_api.py` proves UNVERIFIED names are only
reachable from the experimental `real` backend.

Source of truth: https://github.com/huggingface/lerobot at commit
fbb811fca92504439792b97d216f0d00c2268382 (main, 2026-09-01). First read 2026-09-02; read
again on 2026-09-13, at the same commit, when the arm adapter was hardened, which is where
most of the rows below come from. PyPI had 0.6.1 on both days; the pinned tree calls itself
0.6.2. Nothing here has been run against an arm, and LeRobot is never imported outside the
`real` backend.
"""

from __future__ import annotations

from quackd.transport.upstream_api import UpstreamRef

REPO = "https://github.com/huggingface/lerobot"
PIN = "fbb811fca92504439792b97d216f0d00c2268382"
READ_ON = "2026-09-13"
PYPI_VERSION_READ = "0.6.1"


def src(path: str, line: int | None = None) -> str:
    return f"{REPO}/blob/{PIN}/{path}" + (f"#L{line}" if line else "")


_ROBOT = "src/lerobot/robots/robot.py"
_UTILS = "src/lerobot/robots/utils.py"
_SO = "src/lerobot/robots/so_follower/so_follower.py"
_SO_CFG = "src/lerobot/robots/so_follower/config_so_follower.py"
_TYPES = "src/lerobot/lerobot_types.py"
_BUS = "src/lerobot/motors/motors_bus.py"
_FEETECH = "src/lerobot/motors/feetech/feetech.py"
_TABLES = "src/lerobot/motors/feetech/tables.py"
_CONSTANTS = "src/lerobot/utils/constants.py"
_POLICY = "src/lerobot/policies/pretrained.py"
_FACTORY = "src/lerobot/policies/factory.py"
_CAMERA = "src/lerobot/cameras/camera.py"
_OPENCV = "src/lerobot/cameras/opencv/camera_opencv.py"
_OPENCV_CFG = "src/lerobot/cameras/opencv/configuration_opencv.py"
_CAM_CFG = "src/lerobot/cameras/configs.py"
_CAMERAS_INIT = "src/lerobot/cameras/__init__.py"

# ── package ─────────────────────────────────────────────────────────────────────────────

PACKAGE = UpstreamRef("lerobot", "VERIFIED", src("pyproject.toml", 27), "PyPI and import name")
PYTHON = UpstreamRef(
    ">=3.12",
    "VERIFIED",
    src("pyproject.toml", 32),
    "requires-python; quackd's floor is 3.11, so the extra carries a python_version marker",
)
VERSION_AT_PIN = UpstreamRef("0.6.2", "VERIFIED", src("pyproject.toml", 28), "PyPI had 0.6.1")
FEETECH_EXTRA = UpstreamRef(
    "lerobot[feetech]",
    "VERIFIED",
    src("pyproject.toml", 178),
    "feetech-servo-sdk (imported as scservo_sdk) and pyserial live in this extra and not in "
    "lerobot's base dependencies, so a plain lerobot imports cleanly and then cannot open the "
    "arm's port. quackd[lerobot] asks for lerobot[feetech] and doctor has a row for the SDK",
)

# ── the Robot interface ─────────────────────────────────────────────────────────────────

ROBOT_BASE = UpstreamRef(
    "lerobot.robots.Robot", "VERIFIED", src(_ROBOT, 30), "the abstract base every robot implements"
)
ROBOT_CONNECT = UpstreamRef(
    "Robot.connect(calibrate=True)",
    "VERIFIED",
    src(_ROBOT, 125),
    "quackd passes calibrate=False: calibration is interactive (see ROBOT_CALIBRATE). The SO "
    "follower's connect() runs bus.connect() and configure() and writes no calibration into "
    "the motors at all, so the file on disk and the arm must already agree",
)
ROBOT_DISCONNECT = UpstreamRef("Robot.disconnect()", "VERIFIED", src(_ROBOT, 209))
ROBOT_GET_OBSERVATION = UpstreamRef(
    "Robot.get_observation() -> dict",
    "VERIFIED",
    src(_ROBOT, 182),
    "a flat dict: '<motor>.pos' floats plus one array per camera, keyed by camera name",
)
ROBOT_SEND_ACTION = UpstreamRef(
    "Robot.send_action(action: dict) -> dict",
    "VERIFIED",
    src(_ROBOT, 194),
    "'<motor>.pos' -> goal; returns what was actually sent, possibly clipped",
)
ROBOT_OBSERVATION_FEATURES = UpstreamRef(
    "Robot.observation_features",
    "VERIFIED",
    src(_ROBOT, 90),
    "key -> float, or a (h, w, c) shape tuple for a camera; usable before connect()",
)
ROBOT_ACTION_FEATURES = UpstreamRef("Robot.action_features", "VERIFIED", src(_ROBOT, 104))
ROBOT_IS_CONNECTED = UpstreamRef("Robot.is_connected", "VERIFIED", src(_ROBOT, 117))
ROBOT_IS_CALIBRATED = UpstreamRef("Robot.is_calibrated", "VERIFIED", src(_ROBOT, 137))
ROBOT_CALIBRATE = UpstreamRef(
    "Robot.calibrate() is interactive",
    "VERIFIED",
    src(_SO, 118),
    "the SO follower's calibrate() calls input() (also line 131); quackd never triggers it "
    "and refuses to drive an uncalibrated arm",
)
ROBOT_CONFIGURE = UpstreamRef("Robot.configure()", "VERIFIED", src(_ROBOT, 174))
ROBOT_CONTEXT = UpstreamRef(
    "Robot.__enter__/__exit__", "VERIFIED", src(_ROBOT, 61), "connect on enter, disconnect on exit"
)
TYPES = UpstreamRef(
    "RobotAction = dict[str, Any]; RobotObservation = dict[str, Any]",
    "VERIFIED",
    src(_TYPES, 40),
)
ROBOT_CALIBRATION_ATTR = UpstreamRef(
    "Robot.calibration",
    "VERIFIED",
    src(_ROBOT, 54),
    "motor name -> MotorCalibration, loaded from the file in __init__ when one exists. It is "
    "where quackd reads each joint's travel, so it is populated before connect() and empty "
    "when this arm has never been calibrated on this machine",
)
ROBOT_CALIBRATION_FPATH = UpstreamRef(
    "Robot.calibration_fpath",
    "VERIFIED",
    src(_ROBOT, 53),
    "calibration_dir / '<id>.json'; quackd reports the path, so a wrong id is visible",
)
CALIBRATION_DIR = UpstreamRef(
    "HF_LEROBOT_CALIBRATION/robots/so_follower/",
    "VERIFIED",
    src(_CONSTANTS, 86),
    "the default calibration directory: $HF_LEROBOT_CALIBRATION, else $HF_LEROBOT_HOME/"
    "calibration, else $HF_HOME/lerobot/calibration, then 'robots' and the robot class's own "
    "name. Two arms sharing an id share a file, and nothing in it names a serial number",
)

# ── the SO-101 follower (the arm the adapter targets by default) ────────────────────────

MAKE_ROBOT = UpstreamRef(
    "lerobot.robots.make_robot_from_config(config)", "VERIFIED", src(_UTILS, 27)
)
ROBOT_TYPE_SO101 = UpstreamRef(
    "so101_follower",
    "VERIFIED",
    src(_SO_CFG, 56),
    "the registered config type; make_robot_from_config dispatches on it (utils.py line 41)",
)
SO_FOLLOWER = UpstreamRef(
    "lerobot.robots.so_follower.SO101Follower",
    "VERIFIED",
    src(_SO, 242),
    "an alias of SOFollower (SO100Follower too); exported by the package __init__",
)
SO_NAME = UpstreamRef(
    "SOFollower.name is so_follower",
    "VERIFIED",
    src(_SO, 44),
    "the class name, and therefore the calibration subdirectory: an SO-100 and an SO-101 "
    "share one, because at this commit they are the same class",
)
SO_CONFIG = UpstreamRef(
    "SO101FollowerConfig(port, disable_torque_on_disconnect=True, max_relative_target=None, "
    "cameras={}, use_degrees=True, position_p_coefficient=16, position_i_coefficient=0, "
    "position_d_coefficient=32, num_read_retries=2)",
    "VERIFIED",
    src(_SO_CFG, 25),
    "an alias of SOFollowerRobotConfig; id and calibration_dir come from RobotConfig. quackd "
    "passes every safety-shaped field explicitly rather than inheriting a default it has not "
    "read, and sets max_relative_target, which upstream leaves at None",
)
SO_MOTORS = UpstreamRef(
    "shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper",
    "VERIFIED",
    src(_SO, 54),
    "six Feetech sts3215 motors, ids 1..6",
)
SO_OBSERVATION_KEYS = UpstreamRef(
    "'<motor>.pos'", "VERIFIED", src(_SO, 184), "joint positions; the same keys are the action"
)
SO_OBSERVATION_IS_POSITION_ONLY = UpstreamRef(
    "get_observation() reads Present_Position and nothing else",
    "VERIFIED",
    src(_SO, 183),
    "one sync_read of the positions, with num_read_retries extra attempts, plus a frame per "
    "camera. No torque state, no current, no temperature and no error flag, so a joint that "
    "has tripped its own protection looks exactly like one that has arrived",
)
SO_CAMERA_KEYS = UpstreamRef(
    "camera name -> array",
    "VERIFIED",
    src(_SO, 192),
    "get_observation() adds cam.read_latest() under each configured camera's name",
)
SO_ACTION_CLAMP = UpstreamRef(
    "max_relative_target caps each step",
    "VERIFIED",
    src(_SO, 223),
    "ensure_safe_goal_position (utils.py line 93) clips a goal to present +/- the cap and "
    "logs a warning when it does; None, which is upstream's default, means no cap. Setting "
    "it costs one extra sync_read of the present position per send_action",
)
SO_ACTION_CLAMP_IS_FLOAT = UpstreamRef(
    "max_relative_target must be a float or a dict per motor",
    "VERIFIED",
    src(_UTILS, 98),
    "ensure_safe_goal_position tests isinstance(float), then isinstance(dict), and raises "
    "TypeError on anything else, so an int cap raises rather than capping. quackd casts",
)
SO_SEND_ACTION_RETURN = UpstreamRef(
    "send_action() returns the goal actually sent",
    "VERIFIED",
    src(_SO, 230),
    "the clipped goal in '<motor>.pos' keys, which is not the measured position. quackd "
    "keeps it rather than assuming the goal it asked for was the one written",
)
SO_DEGREES = UpstreamRef(
    "use_degrees=True -> body joints in degrees",
    "VERIFIED",
    src(_SO, 50),
    "MotorNormMode.DEGREES; False means a -100..100 range",
)
SO_GRIPPER_RANGE = UpstreamRef(
    "gripper is 0..100", "VERIFIED", src(_SO, 59), "MotorNormMode.RANGE_0_100 whatever use_degrees"
)
SO_DISCONNECT_TORQUE = UpstreamRef(
    "disconnect() disables torque by default",
    "VERIFIED",
    src(_SO, 234),
    "disable_torque_on_disconnect defaults to True (config line 31): LeRobot lets the arm go "
    "limp when the session ends, and quackd keeps that default and says so. It fires on "
    "every clean exit, a doctor probe included, and not at all when the process is killed",
)
SO_GRIPPER_TORQUE_LIMIT = UpstreamRef(
    "Max_Torque_Limit 500 on the gripper",
    "VERIFIED",
    src(_SO, 169),
    "configure() caps the gripper at 50 % torque, 50 % current (Protection_Current 250) and "
    "25 % torque once overloaded (Overload_Torque 25): the native safety authority",
)
SO_BODY_HAS_NO_TORQUE_CAP = UpstreamRef(
    "the five body joints get no torque or current cap",
    "VERIFIED",
    src(_SO, 168),
    "the three caps above are inside a check for the gripper. Every other joint keeps "
    "whatever its firmware defaults to, so the manifest's torque_limit authority covers the "
    "gripper only and an elbow against an obstacle is the servo's own protection and nothing "
    "else",
)
SO_CONFIGURE_MOTORS = UpstreamRef(
    "configure_motors() writes Return_Delay_Time 0 and Acceleration 254",
    "VERIFIED",
    src(_FEETECH, 209),
    "plus Maximum_Acceleration 254 on protocol 0 and an sts3215 Phase fix. configure() calls "
    "it inside torque_disabled(), so connecting always drops torque briefly",
)
SO_IS_CONNECTED = UpstreamRef(
    "SOFollower.is_connected is the serial port plus the cameras",
    "VERIFIED",
    src(_SO, 88),
    "bus.is_connected and every camera's; BUS_IS_CONNECTED is what the bus half means",
)
SO_BUS = UpstreamRef(
    "SOFollower.bus is a FeetechMotorsBus",
    "VERIFIED",
    src(_SO, 51),
    "the attribute quackd reads registers through when the Robot interface has no answer",
)
NO_CLIENT_DEADMAN = UpstreamRef(
    "no deadman: nothing stops the arm when the client goes quiet",
    "VERIFIED",
    src(_SO, 205),
    "read end to end at the pin: SOFollower has no thread, timer, timeout or watchdog, and "
    "send_action writes Goal_Position and returns. A position-controlled arm holds its last "
    "goal under torque until the next write or disconnect(). quackd's stop re-sends the "
    "present position as the goal (hold) and never disables torque",
)

# ── the Feetech bus, below the Robot interface, where the registers are ─────────────────

BUS_DISABLE_TORQUE = UpstreamRef(
    "MotorsBus.disable_torque()", "VERIFIED", src(_BUS, 118), "NEVER called by quackd (limp)"
)
BUS_ENABLE_TORQUE = UpstreamRef("MotorsBus.enable_torque()", "VERIFIED", src(_BUS, 113))
BUS_DISCONNECT = UpstreamRef("MotorsBus.disconnect(disable_torque=True)", "VERIFIED", src(_BUS, 82))
BUS_IS_CONNECTED = UpstreamRef(
    "MotorsBus.is_connected is port_handler.is_open",
    "VERIFIED",
    src(_BUS, 511),
    "a serial port's open flag, not a reply from a motor. Unplug the arm and it stays True "
    "until a read fails, which is why quackd's heartbeat reads the arm rather than the flag",
)
BUS_IS_CALIBRATED = UpstreamRef(
    "FeetechMotorsBus.is_calibrated reads the motors back",
    "VERIFIED",
    src(_FEETECH, 228),
    "it reads Min_Position_Limit, Max_Position_Limit and Homing_Offset off every motor and "
    "compares them with the cached file. A missing file, a stale file, and the file of a "
    "different arm all come back False, which is the check quackd refuses on",
)
BUS_WRITE_CALIBRATION = UpstreamRef(
    "write_calibration() is reached only through calibrate()",
    "VERIFIED",
    src(_FEETECH, 268),
    "it writes the limits and the homing offset into the motors; connect(calibrate=False) "
    "never calls it, so quackd cannot move an arm's zero even by accident",
)
BUS_SYNC_READ = UpstreamRef(
    "MotorsBus.sync_read(data_name, motors=None, normalize=True, num_retry=0)",
    "VERIFIED",
    src(_BUS, 1128),
    "one framed transaction for every motor named; quackd uses it for the registers "
    "get_observation() does not read",
)
BUS_NORMALIZED_DATA = UpstreamRef(
    "NORMALIZED_DATA is Goal_Position and Present_Position",
    "VERIFIED",
    src(_FEETECH, 47),
    "the only two names sync_read normalises (motors_bus.py line 1167), so every other "
    "register comes back raw whatever normalize says. quackd passes normalize=False anyway, "
    "because a raw register is what it means to read",
)
MOTOR_CALIBRATION = UpstreamRef(
    "MotorCalibration(id, drive_mode, homing_offset, range_min, range_max)",
    "VERIFIED",
    src(_BUS, 176),
    "range_min and range_max are raw encoder ticks recorded by calibration, not degrees",
)
DEGREES_FORMULA = UpstreamRef(
    "degrees = (raw - mid) * 360 / 4095",
    "VERIFIED",
    src(_BUS, 874),
    "mid is (range_min + range_max) / 2 and 4095 is the resolution less one, so a joint's "
    "travel in degrees is (range_max - range_min) * 360 / 4095, centred on zero. That is how "
    "quackd turns a calibration file into the range it will accept a goal inside",
)
DEGREES_NO_CLAMP = UpstreamRef(
    "a degrees goal is not clamped to the calibrated range",
    "VERIFIED",
    src(_BUS, 904),
    "_unnormalize bounds the RANGE_0_100 and RANGE_M100_100 modes and does not bound "
    "DEGREES: the tick it computes is written to Goal_Position as-is. So the gripper is "
    "clamped by LeRobot and the five body joints are not, and what the firmware does with a "
    "tick outside Min_Position_Limit is Feetech's. quackd refuses the goal instead",
)
STS3215_RESOLUTION = UpstreamRef(
    "sts3215 resolution 4096",
    "VERIFIED",
    src(_TABLES, 190),
    "12 bits over a full turn, so one tick is about 0.088 degrees",
)
STS3215_REGISTERS = UpstreamRef(
    "Torque_Enable (40, 1) and Present_Temperature (63, 1)",
    "VERIFIED",
    src(_TABLES, 76),
    "address and length in the STS/SMS control table, temperature at line 87. Present_Load, "
    "Present_Current, Present_Voltage, Status and Max_Temperature_Limit are in the same "
    "table and quackd reads none of them yet. Nothing upstream reads these two either: the "
    "arm's own protection can drop torque and get_observation() will not mention it",
)

# ── cameras ─────────────────────────────────────────────────────────────────────────────

CAMERA_ASYNC_READ = UpstreamRef(
    "Camera.async_read(timeout_ms)", "VERIFIED", src(_CAMERA, 122), "the most recent new frame"
)
CAMERA_READ = UpstreamRef("Camera.read()", "VERIFIED", src(_CAMERA, 111))
CAMERA_RGB_CONVERSION = UpstreamRef(
    "OpenCVCamera converts BGR to RGB when color_mode is RGB",
    "VERIFIED",
    src(_OPENCV, 446),
    "so a camera array's channel order is a config choice, not a constant",
)
CAMERA_COLOR_MODE_DEFAULT = UpstreamRef(
    "OpenCVCameraConfig.color_mode defaults to ColorMode.RGB",
    "VERIFIED",
    src(_OPENCV_CFG, 62),
    "which settles the channel order quackd used to assume: a stock OpenCV camera hands "
    "over RGB, and quackd passes it explicitly anyway",
)
OPENCV_CAMERA = UpstreamRef(
    "lerobot.cameras.opencv.OpenCVCamera(config)",
    "VERIFIED",
    src(_OPENCV, 52),
    "the class quackd builds and owns itself, beside the follower rather than inside it",
)
OPENCV_CAMERA_CONFIG = UpstreamRef(
    "OpenCVCameraConfig(index_or_path, fps=None, width=None, height=None, "
    "color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION, warmup_s=1, fourcc=None, "
    "backend=Cv2Backends.ANY)",
    "VERIFIED",
    src(_OPENCV_CFG, 25),
    "an index or a device path, and a fourcc that must be four characters",
)
CAMERA_EXPORTS = UpstreamRef(
    "lerobot.cameras exports Camera, CameraConfig, ColorMode, Cv2Backends, Cv2Rotation",
    "VERIFIED",
    src(_CAMERAS_INIT, 16),
    "OpenCVCameraConfig is deliberately NOT among them (a note at line 19 says so): it "
    "comes from lerobot.cameras.opencv, which is why quackd imports from both",
)
CV2_BACKENDS = UpstreamRef(
    "Cv2Backends: ANY, V4L2, DSHOW, AVFOUNDATION, MSMF",
    "VERIFIED",
    src(_CAM_CFG, 45),
    "the backend is a config field, so the Windows fix people circulate as a source patch "
    "is a query key here: ?backend=msmf. ANY is the default and lets OpenCV choose",
)
CAMERA_CONNECT = UpstreamRef(
    "Camera.connect(warmup=True)",
    "VERIFIED",
    src(_CAMERA, 79),
    "opens the device and reads frames for warmup_s before returning, so a camera that "
    "opens and never delivers fails here rather than at the first observe",
)
OPENCV_OPEN_FAILS = UpstreamRef(
    "connect() raises ConnectionError on an index that will not open",
    "VERIFIED",
    src(_OPENCV, 171),
    "its own words name `lerobot-find-cameras opencv`, so quackd passes them through",
)
OPENCV_MODE_IS_A_DEMAND = UpstreamRef(
    "a requested fps or size that the camera refuses raises RuntimeError",
    "VERIFIED",
    src(_OPENCV, 260),
    "width and height at line 297; fourcc only warns (line 262). So pinning a mode is a "
    "refusal on a camera that cannot do it, which is why quackd asks for none by default",
)
OPENCV_MODE_DEFAULTS_TO_THE_CAMERA = UpstreamRef(
    "an unset fps, width or height keeps the camera's own mode",
    "VERIFIED",
    src(_OPENCV, 228),
    "the guards read the device's defaults back instead of setting anything (fps at line "
    "237), which is what quackd relies on when it is pointed at an unknown webcam",
)
CAMERA_READ_LATEST = UpstreamRef(
    "Camera.read_latest(max_age_ms=500)",
    "VERIFIED",
    src(_CAMERA, 137),
    "the newest buffered frame, non-blocking; raises TimeoutError when it is older than "
    "max_age_ms (opencv line 610) and RuntimeError before the first frame or if the read "
    "thread has died. quackd catches all of it and reports a camera error instead",
)
CAMERA_DISCONNECT = UpstreamRef("Camera.disconnect()", "VERIFIED", src(_CAMERA, 161))
SO_CAMERAS_ARE_THE_FOLLOWERS = UpstreamRef(
    "a follower's cameras are part of its connected state",
    "VERIFIED",
    src(_SO, 88),
    "is_connected is the bus AND every camera, and send_action and disconnect() are "
    "decorated on it, so a webcam that drops would make every move and every hold raise. "
    "That is why quackd builds its camera beside the follower and passes cameras={}",
)
FIND_CAMERAS = UpstreamRef(
    "lerobot-find-cameras opencv",
    "VERIFIED",
    src("pyproject.toml", 352),
    "lerobot.scripts.lerobot_find_cameras:main; prints an Id per camera and saves one "
    "outputs/captured_images/opencv_<id>.png (line 289), which is how an owner learns "
    "which index is which. Off Linux it scans indices 0 to 59",
)

# ── policies (pick is a LeRobot policy, never a quackd control law) ─────────────────────

POLICY_BASE = UpstreamRef(
    "lerobot.policies.pretrained.PreTrainedPolicy", "VERIFIED", src(_POLICY, 61)
)
POLICY_FROM_PRETRAINED = UpstreamRef(
    "PreTrainedPolicy.from_pretrained(path, *, config=None, local_files_only=False, "
    "revision=None, strict=False)",
    "VERIFIED",
    src(_POLICY, 147),
    "a local directory or a Hub repo id; sets eval mode",
)
POLICY_SELECT_ACTION = UpstreamRef(
    "PreTrainedPolicy.select_action(batch: dict[str, Tensor]) -> Tensor",
    "VERIFIED",
    src(_POLICY, 292),
    "one action per call, the policy handles its own action-chunk cache",
)
POLICY_RESET = UpstreamRef("PreTrainedPolicy.reset()", "VERIFIED", src(_POLICY, 223))
GET_POLICY_CLASS = UpstreamRef(
    "lerobot.policies.factory.get_policy_class(name)", "VERIFIED", src(_FACTORY, 80)
)
MAKE_PRE_POST_PROCESSORS = UpstreamRef(
    "lerobot.policies.factory.make_pre_post_processors(policy_cfg, pretrained_path)",
    "VERIFIED",
    src(_FACTORY, 151),
    "a raw observation goes through the pre-processor and the action tensor through the "
    "post-processor before it is a RobotAction",
)
MAKE_POLICY = UpstreamRef(
    "lerobot.policies.factory.make_policy(cfg)", "VERIFIED", src(_FACTORY, 260)
)

# ── UNVERIFIED: our assumptions, and what quackd does about each ────────────────────────

POLICY_PIPELINE = UpstreamRef(
    "POLICY_PIPELINE",
    "UNVERIFIED",
    src(_FACTORY, 151),
    "wiring a PreTrainedPolicy end to end (pre-processor, select_action, post-processor, "
    "device) has never been run by us. The real backend takes an injected policy with "
    "act(observation, task=...) -> action; load_policy() builds one from the verified names "
    "and is untested. A policy's actions go through the same step cap and the same range "
    "refusal as a verb's, which is quackd's rule and not upstream's",
)
GRIPPER_OPEN_VALUE = UpstreamRef(
    "GRIPPER_OPEN_VALUE",
    "UNVERIFIED",
    src(_SO, 59),
    "which end of the gripper's 0..100 range is open. 0 is the range_min tick of that "
    "motor's calibration and 100 the range_max, so which one is the open jaw is how the arm "
    "was assembled and calibrated. quackd assumes 100 is open, and the checklist asks for it "
    "to be confirmed by hand before anything is believed about holding",
)
HOLDING_INFERRED = UpstreamRef(
    "HOLDING_INFERRED",
    "UNVERIFIED",
    src(_SO, 183),
    "nothing reports grip force, so holding is inferred: the gripper was told to close, its "
    "reading has settled, and it settled short of shut. The band is quackd's own guess, an "
    "empty hand that binds reads as holding, and a thin enough object may not",
)
TEMPERATURE_C = UpstreamRef(
    "TEMPERATURE_C",
    "UNVERIFIED",
    src(_TABLES, 87),
    "Present_Temperature is one byte, and Feetech's documentation calls it degrees Celsius, "
    "which nothing in LeRobot reads or converts. quackd reports it raw and refuses to move a "
    "joint at or above 60, which is the datasheet's operating maximum and below the servo's "
    "own 70 cut-off, so the number and the threshold are both ours to be wrong about",
)
CAMERA_INDEX_MOVES = UpstreamRef(
    "CAMERA_INDEX_MOVES",
    "UNVERIFIED",
    src(_OPENCV, 308),
    "an OpenCV index is a scan position, not an identity: it can change when a camera is "
    "replugged or the machine reboots, and a laptop's own webcam usually holds 0. quackd "
    "records the index it opened and the frame size it got, and cannot tell you it is the "
    "camera you meant. `lerobot-find-cameras opencv` saves a frame per index for that",
)
WINDOWS_CAMERA_BACKEND = UpstreamRef(
    "WINDOWS_CAMERA_BACKEND",
    "UNVERIFIED",
    src(_CAM_CFG, 45),
    "which OpenCV backend a given Windows machine needs for a given webcam is not knowable "
    "in advance: the common report is a camera that lists and then will not open under the "
    "default. quackd leaves upstream's ANY alone and gives the owner ?backend=msmf rather "
    "than guessing per platform",
)
JOINT_RANGES = UpstreamRef(
    "JOINT_RANGES",
    "UNVERIFIED",
    src(_SO, 50),
    "the reachable range of each joint is whatever calibration recorded, and no vendor "
    "publishes what it ought to be. quackd computes each joint's travel from the calibration "
    "file and refuses a goal outside it rather than writing a tick LeRobot will not clamp "
    "(DEGREES_NO_CLAMP); whether that travel is the real mechanical limit is unverified",
)
SERIAL_PORT = UpstreamRef(
    "SERIAL_PORT",
    "UNVERIFIED",
    src(_SO_CFG, 29),
    "the arm's serial port (/dev/ttyACM0, COM5) comes from --address. quackd checks its "
    "shape and nothing more: which port is the arm, and whether a CH340 or CP210x driver is "
    "installed, is between the owner and their machine",
)
THREAD_SAFETY = UpstreamRef(
    "THREAD_SAFETY",
    "UNVERIFIED",
    src(_ROBOT),
    "Robot is synchronous and not documented as thread-safe, over a half-duplex serial bus "
    "where two talkers is a corrupt packet. quackd serialises every call under one lock in a "
    "worker thread with a deadline, and when a call blows its deadline it refuses every "
    "later call rather than starting a second thread on the same bus",
)


def all_refs() -> list[UpstreamRef]:
    return [v for v in globals().values() if isinstance(v, UpstreamRef)]


def refs_by_status(status: str) -> list[UpstreamRef]:
    return [r for r in all_refs() if r.status == status]
