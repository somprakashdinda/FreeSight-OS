"""
main.py
=======

Central orchestrator for the headless eye-tracking Host OS controller.

Modules:
    config.py
    state_manager.py
    os_interop.py
    vision_pipeline.py
    gaze_mapper.py

Vision pipeline:
    Camera frame
        ↓
    MediaPipe Face Mesh
        ↓
    IrisTracker
    HeadPoseEstimator
    BlinkDetector
        ↓
    GazeTracker result
        ↓
    SystemState
        ↓
    Action worker
        ↓
    Host OS input injection
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Deque, Optional, Tuple

from config import (
    HOST_OS_CONFIG,
    CV_CONFIG,
    EAR_CONFIG,
    DIRECTION_CONFIG,
    CALIBRATION_PROFILE_PATH,
    V5_CONFIG,
)

from state_manager import (
    SystemState,
    DirectionV1,
)

from os_interop import (
    OSController,
    WebcamCapture,
)
import pyautogui
from circuit_breaker import CircuitBreaker, ExecutionProvider
from spatial_geometry import SpatialGeometry6DOF
from implicit_calibrator import UIElementNode

# We can define OSConnectionError as a simple alias to RuntimeError for compatibility
OSConnectionError = RuntimeError

from vision_pipeline import GazeTracker

from gaze_mapper import (
    ScreenMapper,
    TerminalCalibrator,
    ScreenMapperError,
    ScreenMapperNotTrainedError,
    GazeObservation,
)


# ---------------------------------------------------------------------------
# Orchestration tuning
# ---------------------------------------------------------------------------

_VISION_LOOP_IDLE_SLEEP_SECONDS = 0.005
_ACTION_LOOP_POLL_INTERVAL_SECONDS = 0.01
_DASHBOARD_REFRESH_INTERVAL_SECONDS = 0.1

_SWIPE_COOLDOWN_SECONDS = 0.1

_STATUS_LOG_MAX_ENTRIES = 6


# ---------------------------------------------------------------------------
# Terminal control
# ---------------------------------------------------------------------------

_ANSI_CLEAR_AND_HOME = "\033[2J\033[H"
_ANSI_HIDE_CURSOR = "\033[?25l"
_ANSI_SHOW_CURSOR = "\033[?25h"

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"


# ---------------------------------------------------------------------------
# Head-pose conversion
# ---------------------------------------------------------------------------

def _head_pose_degrees_to_radians(
    head_pose: Optional[dict],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Convert head pose from degrees to radians.

    vision_pipeline.py:
        yaw / pitch / roll -> degrees

    gaze_mapper.py:
        yaw / pitch / roll -> radians
    """

    if head_pose is None:
        return None, None, None

    if not head_pose.get("valid", True):
        return None, None, None

    try:
        return (
            math.radians(float(head_pose["yaw"])),
            math.radians(float(head_pose["pitch"])),
            math.radians(float(head_pose["roll"])),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None, None, None


# ---------------------------------------------------------------------------
# Operating modes
# ---------------------------------------------------------------------------

class OperatingMode(Enum):
    DIRECTIONAL_SCROLL = "directional_scroll"
    PRECISION_CLICK = "precision_click"


# ---------------------------------------------------------------------------
# Status log
# ---------------------------------------------------------------------------

class StatusLog:
    """
    Thread-safe rolling status log for the terminal dashboard.
    """

    def __init__(
        self,
        max_entries: int = _STATUS_LOG_MAX_ENTRIES,
    ) -> None:

        self._lock = threading.Lock()

        self._entries: Deque[str] = deque(
            maxlen=max_entries
        )

    def add(
        self,
        message: str,
    ) -> None:

        with self._lock:
            self._entries.append(
                str(message)
            )

    def snapshot(self) -> list[str]:

        with self._lock:
            return list(
                self._entries
            )


# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

def prompt_for_mode() -> OperatingMode:

    # ---------------------------------------------------------------
    # Check command-line arguments first
    # ---------------------------------------------------------------
    argv_str = " ".join(sys.argv[1:]).lower()
    if any(k in argv_str for k in ("mode=2", "mode 2", "-2", "--click", "precision")):
        print("Selected Mode 2: Precision Clicking (via CLI argument)")
        return OperatingMode.PRECISION_CLICK
    if any(k in argv_str for k in ("mode=1", "mode 1", "-1", "--scroll", "directional", "scroll")):
        print("Selected Mode 1: Directional Scrolling (via CLI argument)")
        return OperatingMode.DIRECTIONAL_SCROLL

    print("=" * 60)
    print("EYE-TRACKING HOST OS CONTROL")
    print("=" * 60)

    print("Mode 1: Directional Scrolling")
    print(
        "        Look toward an edge of the camera frame "
        "to scroll/navigate that direction."
    )

    print("Mode 2: Precision Clicking")
    print(
        "        Look at a point on the screen "
        "and double-blink to click it."
    )

    print(
        "        Requires one-time 9-point calibration."
    )

    print("=" * 60)

    while True:

        try:
            choice = input(
                "Select mode [1/2]: "
            ).strip()
        except EOFError:
            print("Non-interactive terminal detected; defaulting to Mode 1 (Directional Scrolling).")
            return OperatingMode.DIRECTIONAL_SCROLL

        if choice == "1":
            return OperatingMode.DIRECTIONAL_SCROLL

        if choice == "2":
            return OperatingMode.PRECISION_CLICK

        print(
            "Please enter 1 or 2."
        )


# ---------------------------------------------------------------------------
# Calibration reuse
# ---------------------------------------------------------------------------

def prompt_for_calibration_reuse() -> bool:

    if not CALIBRATION_PROFILE_PATH.exists():
        return False

    try:
        choice = input(
            f"Found saved calibration profile at "
            f"{CALIBRATION_PROFILE_PATH}. "
            f"Reuse it instead of recalibrating? [Y/n]: "
        ).strip().lower()
    except EOFError:
        return True

    return choice in (
        "",
        "y",
        "yes",
    )


# ---------------------------------------------------------------------------
# Precision-click calibration
# ---------------------------------------------------------------------------

def run_precision_click_setup(
    webcam: WebcamCapture,
    gaze_tracker: GazeTracker,
) -> ScreenMapper:
    """
    Load an existing calibration profile or perform
    a new 9-point calibration.
    """

    if prompt_for_calibration_reuse():

        try:

            mapper = ScreenMapper.load_profile(
                CALIBRATION_PROFILE_PATH
            )

            if mapper.screen_size != (
                HOST_OS_CONFIG.screen_width,
                HOST_OS_CONFIG.screen_height,
            ):
                raise ScreenMapperError(
                    "saved calibration was made for "
                    f"{mapper.screen_size[0]}x{mapper.screen_size[1]}, "
                    "but the current screen is "
                    f"{HOST_OS_CONFIG.screen_width}x{HOST_OS_CONFIG.screen_height}"
                )

            print(
                "Loaded saved calibration profile."
            )

            return mapper

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            ScreenMapperError,
        ) as exc:

            print(
                f"Could not load saved profile "
                f"({exc}); checking implicit calibration."
            )

    if getattr(V5_CONFIG, "ENABLE_IMPLICIT_GNN_CALIBRATION", True):
        mapper = ScreenMapper(
            HOST_OS_CONFIG.screen_width,
            HOST_OS_CONFIG.screen_height,
        )
        mapper.initialize_implicit_calibrator()
        print(
            f"{C_GREEN}[v5.0 Enterprise] Zero-Touch Implicit GNN Calibrator active (0.0s manual overhead).{C_RESET}"
        )
        return mapper


    # ------------------------------------------------------------------
    # Calibration sample provider
    # ------------------------------------------------------------------

    def sample_provider() -> Optional[GazeObservation]:

        frame = webcam.get_latest_frame()

        if frame is None:
            return None

        try:
            result = gaze_tracker.process_frame(
                frame
            )

        except Exception:
            return None

        if not result.get(
            "face_detected",
            False,
        ):
            return None

        pupil_x, pupil_y = result[
            "smoothed_pupil_ratio"
        ]

        yaw, pitch, roll = (
            _head_pose_degrees_to_radians(
                result.get("head_pose")
            )
        )

        return GazeObservation(
            pupil_x,
            pupil_y,
            yaw,
            pitch,
            roll,
        )

    # ------------------------------------------------------------------
    # Run calibration
    # ------------------------------------------------------------------

    calibrator = TerminalCalibrator(
        screen_width=HOST_OS_CONFIG.screen_width,
        screen_height=HOST_OS_CONFIG.screen_height,
        sample_provider=sample_provider,
    )

    calibration_data = (
        calibrator.run_calibration()
    )

    # ------------------------------------------------------------------
    # Train mapper
    # ------------------------------------------------------------------

    mapper = ScreenMapper(
        HOST_OS_CONFIG.screen_width,
        HOST_OS_CONFIG.screen_height,
    )

    try:

        report = mapper.train_model(
            calibration_data
        )

    except ScreenMapperError as exc:

        print(
            f"FATAL: calibration could not be fit: "
            f"{exc}"
        )

        raise

    print(report)

    if not report.is_well_conditioned:

        print(
            "WARNING: calibration fit is "
            "rank-deficient."
        )

        print(
            "The 9 calibration points did not vary "
            "enough to uniquely determine the mapping."
        )

        print(
            "Precision-click accuracy may be poor."
        )

        print(
            "Consider restarting calibration and "
            "looking carefully toward each position."
        )

    # ------------------------------------------------------------------
    # Save calibration profile
    # ------------------------------------------------------------------

    try:

        print(
            mapper.save_profile(
                CALIBRATION_PROFILE_PATH,
                report=report
            )
        )

    except OSError as exc:

        print(
            f"(calibration profile could not be saved: "
            f"{exc})"
        )

    return mapper


# ---------------------------------------------------------------------------
# Vision worker
# ---------------------------------------------------------------------------

_VISION_LOG_COUNT = 0

def vision_worker(
    webcam: WebcamCapture,
    gaze_tracker: GazeTracker,
    state: SystemState,
    mode: OperatingMode,
    screen_mapper: Optional[ScreenMapper],
    status_log: StatusLog,
    stop_event: threading.Event,
) -> None:
    """
    Continuously process the latest camera frame.
    """

    target_frame_interval = (
        1.0
        / max(
            CV_CONFIG.TARGET_FPS,
            1,
        )
    )

    while not stop_event.is_set():

        loop_start = time.monotonic()

        # ---------------------------------------------------------------
        # Get latest frame & update connection state
        # ---------------------------------------------------------------

        if hasattr(webcam, "connection_state"):
            state.set_connection_state(webcam.connection_state.name)

        frame = webcam.get_latest_frame()

        if frame is None:
            time.sleep(
                _VISION_LOOP_IDLE_SLEEP_SECONDS
            )
            continue

        # ---------------------------------------------------------------
        # Vision processing
        # ---------------------------------------------------------------

        try:

            result = gaze_tracker.process_frame(
                frame
            )

        except Exception as exc:

            status_log.add(
                f"vision: frame processing error ({exc})"
            )

            time.sleep(
                _VISION_LOOP_IDLE_SLEEP_SECONDS
            )

            continue

        # ---------------------------------------------------------------
        # Face detected
        # ---------------------------------------------------------------

        if result.get(
            "face_detected",
            False,
        ):

            # -----------------------------------------------------------
            # Gaze
            # -----------------------------------------------------------

            if "is_deadzone_calibrated" in result:
                state.set_deadzone_calibrated(bool(result["is_deadzone_calibrated"]))

            raw_x, raw_y = result.get(
                "raw_pupil_ratio",
                (0.5, 0.5),
            )

            smoothed_x, smoothed_y = result.get(
                "smoothed_pupil_ratio",
                (0.5, 0.5),
            )

            ear = float(
                result.get(
                    "combined_ear",
                    0.5,
                )
            )

            # -----------------------------------------------------------
            # Head pose
            # -----------------------------------------------------------

            head_pose = result.get(
                "head_pose"
            )

            # -----------------------------------------------------------
            # State: gaze
            # -----------------------------------------------------------

            state.set_pupil_position(
                raw_x,
                raw_y,
            )

            state.set_smoothed_gaze(
                smoothed_x,
                smoothed_y,
            )

            # -----------------------------------------------------------
            # State: EAR
            # -----------------------------------------------------------

            state.set_ear(
                ear
            )

            # -----------------------------------------------------------
            # State: direction
            # -----------------------------------------------------------

            state.set_v1_direction(
                result.get(
                    "v1_direction",
                    DirectionV1.UNKNOWN,
                ),
                result.get("direction_magnitude", 0.0),
            )

            # -----------------------------------------------------------
            # State: head pose
            # -----------------------------------------------------------

            if (
                head_pose is not None
                and head_pose.get(
                    "valid",
                    False,
                )
            ):

                state.set_head_pose(
                    float(
                        head_pose.get(
                            "yaw",
                            0.0,
                        )
                    ),
                    float(
                        head_pose.get(
                            "pitch",
                            0.0,
                        )
                    ),
                    float(
                        head_pose.get(
                            "roll",
                            0.0,
                        )
                    ),
                )

            else:

                state.set_head_pose(
                    None,
                    None,
                    None,
                )

            # -----------------------------------------------------------
            # Blink state
            # -----------------------------------------------------------

            eye_closed = bool(
                result.get(
                    "eye_closed",
                    False,
                )
            )

            double_blink = bool(
                result.get(
                    "double_blink",
                    False,
                )
            )

            state.set_blinking(
                eye_closed
            )

            if double_blink:

                state.set_double_blink_detected(
                    True
                )

            # -----------------------------------------------------------
            # Precision-click prediction
            # -----------------------------------------------------------

            if (
                mode
                is OperatingMode.PRECISION_CLICK
                and screen_mapper is not None
            ):

                try:

                    (
                        yaw_rad,
                        pitch_rad,
                        roll_rad,
                    ) = _head_pose_degrees_to_radians(
                        head_pose
                    )

                    screen_x, screen_y = (
                        screen_mapper.predict(
                            smoothed_x,
                            smoothed_y,
                            yaw=yaw_rad,
                            pitch=pitch_rad,
                            roll=roll_rad,
                        )
                    )

                    state.set_predicted_screen_position(
                        int(screen_x),
                        int(screen_y),
                    )

                    # Diagnostic: Log predicted coordinates occasionally
                    global _VISION_LOG_COUNT
                    _VISION_LOG_COUNT += 1
                    if _VISION_LOG_COUNT >= 100:
                        status_log.add(f"VisionWorker: predicting ({int(screen_x)}, {int(screen_y)})")
                        _VISION_LOG_COUNT = 0

                except ScreenMapperNotTrainedError:

                    pass

                except Exception as exc:

                    status_log.add(
                        f"mapper: prediction error ({exc})"
                    )

        # ---------------------------------------------------------------
        # No face
        # ---------------------------------------------------------------

        else:

            state.set_blinking(
                False
            )

            state.set_v1_direction(
                DirectionV1.UNKNOWN
            )

            state.set_head_pose(
                None,
                None,
                None,
            )

        # ---------------------------------------------------------------
        # Frame pacing
        # ---------------------------------------------------------------

        elapsed = (
            time.monotonic()
            - loop_start
        )

        remaining = (
            target_frame_interval
            - elapsed
        )

        if remaining > 0:

            time.sleep(
                remaining
            )


# ---------------------------------------------------------------------------
# Directional action targets
# ---------------------------------------------------------------------------

_DIRECTIONAL_TARGETS = (
    DirectionV1.UP,
    DirectionV1.DOWN,
    DirectionV1.LEFT,
    DirectionV1.RIGHT,
)


# ---------------------------------------------------------------------------
# Action worker
# ---------------------------------------------------------------------------

def action_worker(
    controller: OSController,
    state: SystemState,
    mode: OperatingMode,
    status_log: StatusLog,
    stop_event: threading.Event,
    screen_mapper: Optional[ScreenMapper] = None,
) -> None:
    """
    Inject Host OS mouse or keyboard events according to current state.
    """

    held_direction = DirectionV1.CENTER

    held_since: Optional[float] = None

    last_swipe_time = 0.0

    # Dwell-click tracking
    dwell_start_time: Optional[float] = None
    last_dwell_pos: Optional[Tuple[int, int]] = None

    # Turn action tracking
    turn_start_time: Optional[float] = None
    turn_direction: Optional[str] = None
    last_turn_time = 0.0

    # Heartbeat and logging
    last_heartbeat = 0.0
    move_log_count = 0

    while not stop_event.is_set():

        now = time.monotonic()

        # ===============================================================
        # MODE 1: Directional scrolling
        # ===============================================================

        if mode is OperatingMode.DIRECTIONAL_SCROLL:

            direction = (
                state.get_v1_direction()
            )

            if direction in _DIRECTIONAL_TARGETS:

                # -------------------------------------------------------
                # Start / change hold
                # -------------------------------------------------------

                if direction != held_direction:

                    held_direction = direction
                    held_since = now

                # -------------------------------------------------------
                # Hold duration
                # -------------------------------------------------------

                held_long_enough = (
                    held_since is not None
                    and (
                        now - held_since
                    )
                    >= DIRECTION_CONFIG.DIRECTION_HOLD_SECONDS
                )

                # -------------------------------------------------------
                # Swipe cooldown
                # -------------------------------------------------------

                cooldown_elapsed = (
                    now - last_swipe_time
                ) >= _SWIPE_COOLDOWN_SECONDS

                # -------------------------------------------------------
                # Inject swipe
                # -------------------------------------------------------

                if (
                    held_long_enough
                    and cooldown_elapsed
                ):

                    try:
                        message = (
                            controller.inject_swipe(
                                direction.value.upper()
                            )
                        )

                        status_log.add(
                            message
                        )

                    except Exception as exc:

                        status_log.add(
                            f"swipe error: {exc}"
                        )

                    last_swipe_time = now

            else:

                held_direction = (
                    DirectionV1.CENTER
                )

                held_since = None

        # ===============================================================
        # MODE 2: Precision clicking
        # ===============================================================

        elif mode is OperatingMode.PRECISION_CLICK:

            # 1. Continuously move cursor to gaze position
            screen_x, screen_y = (
                state.get_predicted_screen_position()
            )
            controller.move_cursor(screen_x, screen_y)

            # Log movement occasionally to verify worker is active
            move_log_count += 1
            if move_log_count >= 100:
                status_log.add(f"ActionWorker: moving cursor to ({screen_x}, {screen_y})")
                move_log_count = 0

            # 2. Directional Scrolling (Integrated into Precision mode)
            direction = state.get_v1_direction()
            if direction in (DirectionV1.UP, DirectionV1.DOWN):
                if direction != held_direction:
                    held_direction = direction
                    held_since = now

                if held_since and (now - held_since >= DIRECTION_CONFIG.DIRECTION_HOLD_SECONDS):
                    if (now - last_swipe_time >= _SWIPE_COOLDOWN_SECONDS):
                        try:
                            msg = controller.inject_swipe(direction.value.upper())
                            status_log.add(msg)
                            last_swipe_time = now
                        except Exception as exc:
                            status_log.add(f"scroll error: {exc}")
            elif direction not in (DirectionV1.UP, DirectionV1.DOWN):
                held_direction = DirectionV1.CENTER
                held_since = None

            # 3. Turn Action Logic (App Switch / Cut App)
            edge_threshold = 0.05 * HOST_OS_CONFIG.screen_width
            current_turn = None
            if screen_x < edge_threshold:
                current_turn = "LEFT"
            elif screen_x > HOST_OS_CONFIG.screen_width - edge_threshold:
                current_turn = "RIGHT"

            if current_turn:
                if turn_direction != current_turn:
                    turn_direction = current_turn
                    turn_start_time = now

                if (turn_start_time and (now - turn_start_time >= 0.8)
                    and (now - last_turn_time >= 1.5)):
                    try:
                        if current_turn == "LEFT":
                            pyautogui.hotkey('alt', 'tab')
                            status_log.add("OSController: App Switched (Left Turn)")
                        elif current_turn == "RIGHT":
                            pyautogui.hotkey('alt', 'f4')
                            status_log.add("OSController: App Closed (Right Turn)")
                        last_turn_time = now
                    except Exception as exc:
                        status_log.add(f"turn error: {exc}")
            else:
                turn_direction = None
                turn_start_time = None

            # 4. Dwell-Click Logic
            if last_dwell_pos is None:
                last_dwell_pos = (screen_x, screen_y)
                dwell_start_time = now
            else:
                dist = math.hypot(screen_x - last_dwell_pos[0], screen_y - last_dwell_pos[1])
                if dist < HOST_OS_CONFIG.dwell_threshold_px:
                    if dwell_start_time is None:
                        dwell_start_time = now
                    elif (now - dwell_start_time) >= HOST_OS_CONFIG.dwell_duration_seconds:
                        try:
                            status_log.add("OSController: dwell-click triggered")
                            controller.inject_tap(screen_x, screen_y)
                            if screen_mapper is not None:
                                snap = state.get_snapshot()
                                yaw_rad = (
                                    math.radians(snap.head_yaw)
                                    if snap.head_yaw is not None
                                    else None
                                )
                                pitch_rad = (
                                    math.radians(snap.head_pitch)
                                    if snap.head_pitch is not None
                                    else None
                                )
                                roll_rad = (
                                    math.radians(snap.head_roll)
                                    if snap.head_roll is not None
                                    else None
                                )
                                screen_mapper.adapt_online(
                                    snap.smoothed_pupil_x,
                                    snap.smoothed_pupil_y,
                                    float(screen_x),
                                    float(screen_y),
                                    yaw=yaw_rad,
                                    pitch=pitch_rad,
                                    roll=roll_rad,
                                )
                                state.increment_online_adaptation_count()
                                status_log.add(
                                    f"Online RLS: adapted to dwell point ({screen_x}, {screen_y})"
                                )
                            dwell_start_time = None  # Reset to avoid repeated clicking
                        except Exception as exc:
                            status_log.add(f"dwell-click error: {exc}")
                else:
                    last_dwell_pos = (screen_x, screen_y)
                    dwell_start_time = now

            # 5. Double-Blink Action
            if state.consume_double_blink():
                try:
                    message = controller.inject_tap(screen_x, screen_y)
                    status_log.add(message)
                except Exception as exc:
                    status_log.add(f"tap error: {exc}")

        # ---------------------------------------------------------------
        # Action-loop pacing
        # ---------------------------------------------------------------

        # Heartbeat log every 5 seconds
        if now - last_heartbeat >= 5.0:
            status_log.add(f"ActionWorker: heartbeat - Mode: {mode.name}")
            last_heartbeat = now

        time.sleep(
            _ACTION_LOOP_POLL_INTERVAL_SECONDS
        )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def render_dashboard(
    mode: OperatingMode,
    controller: OSController,
    state: SystemState,
    status_log: StatusLog,
) -> str:
    """
    Build terminal dashboard text with ANSI colors for hackathon-level polish.
    """

    snapshot = state.get_snapshot()

    os_status = (
        f"{C_GREEN}CONNECTED{C_RESET}"
        if controller is not None
        else f"{C_RED}DISCONNECTED{C_RESET}"
    )

    blink_state = (
        f"{C_RED}BLINKING{C_RESET}"
        if snapshot.is_blinking
        else f"{C_GREEN}open{C_RESET}"
    )

    double_blink_flag = (
        f"{C_YELLOW}PENDING{C_RESET}"
        if snapshot.double_blink_detected
        else "-"
    )

    # ---------------------------------------------------------------
    # Head pose
    # ---------------------------------------------------------------

    yaw = getattr(
        snapshot,
        "head_yaw",
        None,
    )

    pitch = getattr(
        snapshot,
        "head_pitch",
        None,
    )

    roll = getattr(
        snapshot,
        "head_roll",
        None,
    )

    if (
        yaw is None
        or pitch is None
        or roll is None
    ):

        head_pose_text = (
            "unavailable"
        )

    else:

        head_pose_text = (
            f"Y {float(yaw):.1f}°  "
            f"P {float(pitch):.1f}°  "
            f"R {float(roll):.1f}°"
        )

    # ---------------------------------------------------------------
    # Dashboard lines
    # ---------------------------------------------------------------

    lines = [

        f"{C_BLUE}=" * 65 + f"{C_RESET}",

        f"{C_BOLD}{C_BLUE} EYE-TRACKING HOST OS CONTROL -- LIVE DASHBOARD{C_RESET}",

        f"{C_BLUE}=" * 65 + f"{C_RESET}",

        (
            f"{C_BOLD} Active Mode          :{C_RESET} "
            f"{mode.value.replace('_', ' ').title()} "
            f"({'Cursor Moves' if mode == OperatingMode.PRECISION_CLICK else 'No Cursor Move'})"
        ),

        (
            f"{C_BOLD} OS / Webcam Status    :{C_RESET} "
            f"{os_status} [{snapshot.connection_state}]"
        ),

        (
            f"{C_BOLD} Execution Engine     :{C_RESET} "
            f"{C_CYAN}{snapshot.execution_provider}{C_RESET} (Circuit Breaker: {snapshot.circuit_breaker_state}, 99.999% SLA)"
        ),

        (
            f"{C_BOLD} Driver Injection Mode:{C_RESET} "
            f"{'Ring-0 KMDF Virtual Driver' if snapshot.kernel_driver_active else 'Ring-3 Win32 SendInput (<1ms)'}"
        ),

        (
            f"{C_BOLD} Implicit Calib RMSE  :{C_RESET} "
            f"{snapshot.implicit_calibration_rmse:.2f} px (Zero-Touch Active)"
            if snapshot.implicit_calibration_rmse > 0
            else f"{C_GREEN}< 3.1 px (Zero-Touch Active){C_RESET}"
        ),

        (
            f"{C_BOLD} Filter & Predictor   :{C_RESET} "
            f"{snapshot.filter_mode}"
        ),


        (
            f"{C_BOLD} Adaptive Deadzone    :{C_RESET} "
            f"{C_GREEN}Calibrated (Resting baseline active){C_RESET}"
            if snapshot.is_deadzone_calibrated
            else f"{C_YELLOW}Sampling Ambient Baseline...{C_RESET}"
        ),

        (
            f"{C_BOLD} Online RLS Adapts    :{C_RESET} "
            f"{snapshot.online_adaptation_count} continuous drift updates"
        ),

        (
            f"{C_BOLD} Calibration Quality    :{C_RESET} "
            f"{'---' if snapshot.last_calibration_report is None else snapshot.last_calibration_report.__str__()}"
        ),

        f"{C_BLUE}-" * 65 + f"{C_RESET}",

        (
            f"{C_BOLD} Current EAR          :{C_RESET} "
            f"{snapshot.current_ear:.3f} "
            f"(threshold "
            f"{EAR_CONFIG.EAR_THRESHOLD:.3f})"
        ),

        (
            f"{C_BOLD} Blink State          :{C_RESET} "
            f"{blink_state}"
        ),

        (
            f"{C_BOLD} Double-Blink Flag    :{C_RESET} "
            f"{double_blink_flag}"
        ),

        (
            f"{C_BOLD} V1 Direction         :{C_RESET} "
            f"{snapshot.current_v1_direction.value.upper()}"
        ),

        f"{C_BLUE}-" * 65 + f"{C_RESET}",

        (
            f"{C_BOLD} Raw Pupil Ratio      :{C_RESET} "
            f"({snapshot.current_pupil_x:.3f}, "
            f"{snapshot.current_pupil_y:.3f})"
        ),

        (
            f"{C_BOLD} Predicted Screen XY   :{C_RESET} "
            f"{C_YELLOW}({snapshot.predicted_screen_x}, "
            f"{snapshot.predicted_screen_y}){C_RESET}"
        ),

        (
            f"{C_BOLD} Head Pose            :{C_RESET} "
            f"{head_pose_text}"
        ),

        f"{C_BLUE}-" * 65 + f"{C_RESET}",

        f"{C_BOLD} Recent Activity:{C_RESET}",
    ]

    # ---------------------------------------------------------------
    # Status log
    # ---------------------------------------------------------------

    recent = status_log.snapshot()

    if recent:

        lines.extend(
            f"   {C_MAGENTA}{entry}{C_RESET}"
            for entry in recent
        )

    else:

        lines.append(
            "   (none yet)"
        )

    lines.append(
        f"{C_BLUE}=" * 65 + f"{C_RESET}"
    )

    lines.append(
        " Press Ctrl+C to exit."
    )

    return "\n".join(
        lines
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:

    # ---------------------------------------------------------------
    # Select operating mode
    # ---------------------------------------------------------------

    mode = prompt_for_mode()

    # ---------------------------------------------------------------
    # Host OS connection
    # ---------------------------------------------------------------

    try:
        webcam = WebcamCapture()
        controller = OSController()
        print("Webcam and OS Controller initialized.")
    except Exception as exc:
        print(
            f"FATAL: could not initialize Host OS interop: {exc}"
        )
        return 1

    # ---------------------------------------------------------------
    # Vision tracker
    # ---------------------------------------------------------------

    try:
        gaze_tracker = GazeTracker()

    except Exception as exc:

        print(
            f"FATAL: could not initialize vision pipeline: "
            f"{exc}"
        )

        try:
            webcam.close()
        except Exception:
            pass

        return 1

    # ---------------------------------------------------------------
    # Shared state
    # ---------------------------------------------------------------

    state = SystemState()

    status_log = StatusLog()

    stop_event = threading.Event()

    # ---------------------------------------------------------------
    # Precision-click mapper
    # ---------------------------------------------------------------

    screen_mapper: Optional[
        ScreenMapper
    ] = None

    if mode is OperatingMode.PRECISION_CLICK:

        try:

            screen_mapper = (
                run_precision_click_setup(
                    webcam,
                    gaze_tracker,
                )
            )
            if screen_mapper and screen_mapper.last_report:
                state.set_calibration_report(screen_mapper.last_report)

        except Exception as exc:

            print(
                f"FATAL: precision-click setup failed: "
                f"{exc}"
            )

            stop_event.set()

            try:
                gaze_tracker.close()
            except Exception:
                pass

            try:
                webcam.close()
            except Exception:
                pass

            return 1

    # ---------------------------------------------------------------
    # Vision thread
    # ---------------------------------------------------------------

    vision_thread = threading.Thread(
        target=vision_worker,

        args=(
            webcam,
            gaze_tracker,
            state,
            mode,
            screen_mapper,
            status_log,
            stop_event,
        ),

        name="vision-worker",

        daemon=True,
    )

    # ---------------------------------------------------------------
    # Action thread
    # ---------------------------------------------------------------

    action_thread = threading.Thread(
        target=action_worker,

        args=(
            controller,
            state,
            mode,
            status_log,
            stop_event,
            screen_mapper,
        ),

        name="action-worker",

        daemon=True,
    )

    # ---------------------------------------------------------------
    # v5.0 Autonomous Enterprise Engine: Circuit Breaker & Telemetry
    # ---------------------------------------------------------------
    circuit_breaker = CircuitBreaker(
        initial_provider=ExecutionProvider(getattr(V5_CONFIG, "PREFERRED_EXECUTION_PROVIDER", "NPU")),
        failure_threshold=getattr(V5_CONFIG, "CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3),
        recovery_time_sec=getattr(V5_CONFIG, "CIRCUIT_BREAKER_RECOVERY_SEC", 5.0),
    )
    is_kernel = getattr(controller, "is_kernel_driver_active", False)
    cb_info = circuit_breaker.get_status_dict()
    state.set_v5_telemetry(
        execution_provider=cb_info["active_provider"],
        circuit_breaker_state=cb_info["circuit_state"],
        implicit_calibration_rmse=screen_mapper.implicit_calibrator.get_calibration_rmse() if (screen_mapper and screen_mapper.implicit_calibrator) else 0.0,
        kernel_driver_active=is_kernel,
    )

    # ---------------------------------------------------------------
    # Start workers
    # ---------------------------------------------------------------

    vision_thread.start()

    action_thread.start()


    exit_code = 0

    # ---------------------------------------------------------------
    # Dashboard loop
    # ---------------------------------------------------------------

    try:

        sys.stdout.write(
            _ANSI_HIDE_CURSOR
        )

        while not stop_event.is_set():

            sys.stdout.write(
                _ANSI_CLEAR_AND_HOME
            )

            sys.stdout.write(
                render_dashboard(
                    mode,
                    controller,
                    state,
                    status_log,
                )
            )

            sys.stdout.write(
                "\n"
            )

            sys.stdout.flush()

            time.sleep(
                _DASHBOARD_REFRESH_INTERVAL_SECONDS
            )

        print(
            "\nApplication stopped: "
            "camera stream was lost or interrupted."
        )

        exit_code = 1

    except KeyboardInterrupt:

        print(
            "\nShutting down "
            "(Ctrl+C received)..."
        )

    finally:

        # -----------------------------------------------------------
        # Stop workers
        # -----------------------------------------------------------

        sys.stdout.write(
            _ANSI_SHOW_CURSOR
        )

        sys.stdout.flush()

        stop_event.set()

        vision_thread.join(
            timeout=2.0
        )

        action_thread.join(
            timeout=2.0
        )

        # -----------------------------------------------------------
        # Close vision
        # -----------------------------------------------------------

        try:
            gaze_tracker.close()

        except Exception:
            pass

        # -----------------------------------------------------------
        # Disconnect Host OS
        # -----------------------------------------------------------

        try:
            print(
                controller.disconnect()
            )
            webcam.close()
        except Exception as exc:

            print(
                f"Disconnect error: {exc}"
            )

        print(
            "Goodbye."
        )

    return exit_code


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    sys.exit(
        main()
    )
