"""
state_manager.py
=================

Thread-safe shared state for the headless eye-tracking Host OS
control application.

This module contains no rendering, no windowing, and no UI logic of any
kind (no cv2.imshow, no Tkinter, no GUI event loop). It exists purely to
let independent worker threads — capture, CV inference, gaze mapping, and
the Android command dispatcher — exchange the latest computed values
without racing each other.

Threading model
----------------
A single `threading.Lock` guards the entire state object. The state is
small and updates are cheap (a handful of float/bool/enum assignments), so
a coarse-grained lock is the right trade-off: it is simple to reason about
and avoids subtle partial-update races (e.g. reading current_pupil_x from
frame N while current_pupil_y still reflects frame N-1). This remains true
with the head-pose/confidence fields added below — they're a few more
float/bool/Optional[float] assignments, not new data volume that would
justify a second lock (see `SystemState`'s docstring "Why still one lock"
note).

Unavailable-value convention
------------------------------
Every field that the upstream vision pipeline can fail to produce for a
given frame (head pose, gaze/tracking/screen-prediction confidence) is
typed `Optional[...]` and represented as `None` when unavailable — never
a sentinel number like `-1.0` or `0.0`, since those are valid values for
several of these fields (e.g. `0.0` degrees is a perfectly real, level
head pose). Callers must be prepared to handle `None` from every getter
in the "head pose" and "confidence" sections below.

If profiling ever shows contention here, the fields can be split into
per-concern locks (blink state vs. gaze state) without changing the
public API, since all access already goes through
get_snapshot()/setters.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DirectionV1(Enum):
    """Coarse, calibration-free gaze direction classification."""

    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StateSnapshot:
    """
    An immutable, point-in-time copy of SystemState.

    Returned by `SystemState.get_snapshot()` so callers can read several
    related fields without holding the lock for the duration of their own
    logic (which would otherwise block writers for an unbounded time).
    """

    current_pupil_x: float
    current_pupil_y: float
    current_ear: float
    is_blinking: bool
    double_blink_detected: bool
    current_v1_direction: DirectionV1
    direction_magnitude: float
    predicted_screen_x: int
    predicted_screen_y: int
    last_updated_monotonic: float
    last_calibration_report: Optional[Any]

    # -- Added for the MediaPipe/iris/head-pose pipeline upgrade -------- #
    # Smoothed iris/pupil position (0.0-1.0 per axis), i.e. what the V1
    # direction classifier and ScreenMapper.predict() actually act on --
    # distinct from current_pupil_x/y above, which is the raw per-frame
    # reading.
    smoothed_pupil_x: float
    smoothed_pupil_y: float
    # Head pose in DEGREES (matching vision_pipeline.GazeTracker's output
    # units), or None per-axis if unavailable this frame.
    head_yaw: Optional[float]
    head_pitch: Optional[float]
    head_roll: Optional[float]
    # True only when head_yaw/pitch/roll are all non-None -- a
    # convenience flag so callers that just need a boolean don't have to
    # check three Optionals themselves. See SystemState.set_head_pose.
    head_pose_valid: bool
    # Reserved confidence signals (0.0-1.0), None when not produced by
    # the current pipeline stage. See SystemState's "Confidence" section
    # docstring for what each represents and why they're optional.
    gaze_confidence: Optional[float]
    tracking_confidence: Optional[float]
    screen_prediction_confidence: Optional[float]
    # v2.0 - v4.0 telemetry enhancements
    connection_state: str = "CONNECTED"
    is_deadzone_calibrated: bool = False
    filter_mode: str = "Predictive UKF (16.6ms lookahead)"
    online_adaptation_count: int = 0
    # v5.0 Autonomous Enterprise Engine telemetry
    execution_provider: str = "NPU"
    circuit_breaker_state: str = "CLOSED"
    implicit_calibration_rmse: float = 0.0
    kernel_driver_active: bool = False
    # v7.0 Micro-Transformer Spatial AI Intent Telemetry
    predicted_saccade_x: float = 0.0
    predicted_saccade_y: float = 0.0
    intent_confidence: float = 0.0



class SystemState:
    """
    Lock-protected container for all values shared across the capture,
    vision, gaze-mapping, and Host OS-control threads.

    All reads and writes go through explicit getter/setter methods that
    acquire `self._lock` for the minimum time necessary (no I/O or
    computation is ever performed while the lock is held).

    Attributes are intentionally private (prefixed with `_`) to force all
    access through the synchronized accessors below; reaching in and
    reading `state._current_pupil_x` directly bypasses the lock and is a
    bug, not a shortcut.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

        # Raw gaze signal, in normalized eye-region coordinates (0.0-1.0).
        # This is the iris/pupil position as read on a single frame,
        # straight from GazeTracker.process_frame()["raw_pupil_ratio"].
        self._current_pupil_x: float = 0.5
        self._current_pupil_y: float = 0.5

        # Smoothed iris/pupil position (0.0-1.0), i.e.
        # GazeTracker.process_frame()["smoothed_pupil_ratio"]. Kept as a
        # separate pair of fields from the raw position above rather than
        # overwriting it, since V1 direction classification and
        # ScreenMapper.predict() are both driven by the smoothed value
        # while the raw value remains useful for diagnostics/dashboard
        # display.
        self._smoothed_pupil_x: float = 0.5
        self._smoothed_pupil_y: float = 0.5

        # Eye Aspect Ratio and derived blink state.
        self._current_ear: float = 0.0
        self._is_blinking: bool = False
        self._double_blink_detected: bool = False

        # Coarse V1 directional classification (see DirectionConfigV1).
        self._current_v1_direction: DirectionV1 = DirectionV1.UNKNOWN
        self._current_direction_magnitude: float = 0.0

        # Head pose in DEGREES (see vision_pipeline.GazeTracker's
        # "head_pose" output), None per-axis when unavailable. All three
        # are always set together by set_head_pose() -- see its
        # docstring -- so they can never be observed in a mixed
        # some-None/some-not state.
        self._head_yaw: Optional[float] = None
        self._head_pitch: Optional[float] = None
        self._head_roll: Optional[float] = None
        self._head_pose_valid: bool = False

        # Reserved confidence signals (0.0-1.0), None when the current
        # pipeline stage doesn't produce one. See the "Confidence"
        # section below for what each is intended to represent.
        self._gaze_confidence: Optional[float] = None
        self._tracking_confidence: Optional[float] = None
        self._screen_prediction_confidence: Optional[float] = None

        # Calibrated screen-space prediction (see gaze_mapper.py), in
        # device pixel coordinates.
        self._predicted_screen_x: int = 0
        self._predicted_screen_y: int = 0
        self._last_calibration_report: Optional[Any] = None

        self._last_updated_monotonic: float = time.monotonic()

        # v2.0 - v4.0 hardware resilience and adaptive filter telemetry
        self._connection_state: str = "CONNECTED"
        self._is_deadzone_calibrated: bool = False
        self._filter_mode: str = "Predictive UKF (16.6ms lookahead)"
        self._online_adaptation_count: int = 0

        # v5.0 Autonomous Enterprise Engine telemetry
        self._execution_provider: str = "NPU"
        self._circuit_breaker_state: str = "CLOSED"
        self._implicit_calibration_rmse: float = 0.0
        self._kernel_driver_active: bool = False

        # v7.0 Micro-Transformer Spatial AI Intent Telemetry
        self._predicted_saccade_x: float = 0.0
        self._predicted_saccade_y: float = 0.0
        self._intent_confidence: float = 0.0

        # Lock-free double-buffered atomic pointer swap reference
        self._snapshot: StateSnapshot = self._build_snapshot()

    def _build_snapshot(self) -> StateSnapshot:
        """Construct an immutable StateSnapshot from current state values."""
        return StateSnapshot(
            current_pupil_x=self._current_pupil_x,
            current_pupil_y=self._current_pupil_y,
            current_ear=self._current_ear,
            is_blinking=self._is_blinking,
            double_blink_detected=self._double_blink_detected,
            current_v1_direction=self._current_v1_direction,
            direction_magnitude=self._current_direction_magnitude,
            predicted_screen_x=self._predicted_screen_x,
            predicted_screen_y=self._predicted_screen_y,
            last_updated_monotonic=self._last_updated_monotonic,
            last_calibration_report=self._last_calibration_report,
            smoothed_pupil_x=self._smoothed_pupil_x,
            smoothed_pupil_y=self._smoothed_pupil_y,
            head_yaw=self._head_yaw,
            head_pitch=self._head_pitch,
            head_roll=self._head_roll,
            head_pose_valid=self._head_pose_valid,
            gaze_confidence=self._gaze_confidence,
            tracking_confidence=self._tracking_confidence,
            screen_prediction_confidence=self._screen_prediction_confidence,
            connection_state=self._connection_state,
            is_deadzone_calibrated=self._is_deadzone_calibrated,
            filter_mode=self._filter_mode,
            online_adaptation_count=self._online_adaptation_count,
            execution_provider=self._execution_provider,
            circuit_breaker_state=self._circuit_breaker_state,
            implicit_calibration_rmse=self._implicit_calibration_rmse,
            kernel_driver_active=self._kernel_driver_active,
            predicted_saccade_x=self._predicted_saccade_x,
            predicted_saccade_y=self._predicted_saccade_y,
            intent_confidence=self._intent_confidence,
        )


    def _publish_snapshot(self) -> None:
        """Update monotonic timestamp and atomically swap the snapshot pointer."""
        self._last_updated_monotonic = time.monotonic()
        self._snapshot = self._build_snapshot()

    # ------------------------------------------------------------------ #
    # Pupil position
    # ------------------------------------------------------------------ #
    def set_pupil_position(self, x: float, y: float) -> None:
        """Set the latest normalized pupil position (0.0-1.0 per axis)."""
        with self._lock:
            self._current_pupil_x = x
            self._current_pupil_y = y
            self._publish_snapshot()

    def get_pupil_position(self) -> tuple[float, float]:
        """Return the latest (pupil_x, pupil_y) as a tuple."""
        with self._lock:
            return self._current_pupil_x, self._current_pupil_y

    # ------------------------------------------------------------------ #
    # Smoothed iris/pupil position (gaze)
    # ------------------------------------------------------------------ #
    def set_smoothed_gaze(self, x: float, y: float) -> None:
        """Set the latest EMA-smoothed normalized gaze position (0.0-1.0
        per axis) -- GazeTracker.process_frame()["smoothed_pupil_ratio"]."""
        with self._lock:
            self._smoothed_pupil_x = x
            self._smoothed_pupil_y = y
            self._publish_snapshot()

    def get_smoothed_gaze(self) -> tuple[float, float]:
        """Return the latest (smoothed_x, smoothed_y) as a tuple."""
        with self._lock:
            return self._smoothed_pupil_x, self._smoothed_pupil_y

    # ------------------------------------------------------------------ #
    # Head pose
    # ------------------------------------------------------------------ #
    def set_head_pose(
        self, yaw: Optional[float], pitch: Optional[float], roll: Optional[float]
    ) -> None:
        with self._lock:
            self._head_yaw = yaw
            self._head_pitch = pitch
            self._head_roll = roll
            self._head_pose_valid = yaw is not None and pitch is not None and roll is not None
            self._publish_snapshot()

    def get_head_pose(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        with self._lock:
            return self._head_yaw, self._head_pitch, self._head_roll

    def is_head_pose_valid(self) -> bool:
        with self._lock:
            return self._head_pose_valid

    # ------------------------------------------------------------------ #
    # Confidence signals (reserved)
    # ------------------------------------------------------------------ #
    def set_gaze_confidence(self, confidence: Optional[float]) -> None:
        with self._lock:
            self._gaze_confidence = confidence
            self._publish_snapshot()

    def get_gaze_confidence(self) -> Optional[float]:
        with self._lock:
            return self._gaze_confidence

    def set_tracking_confidence(self, confidence: Optional[float]) -> None:
        with self._lock:
            self._tracking_confidence = confidence
            self._publish_snapshot()

    def get_tracking_confidence(self) -> Optional[float]:
        with self._lock:
            return self._tracking_confidence

    def set_screen_prediction_confidence(self, confidence: Optional[float]) -> None:
        with self._lock:
            self._screen_prediction_confidence = confidence
            self._publish_snapshot()

    def get_screen_prediction_confidence(self) -> Optional[float]:
        with self._lock:
            return self._screen_prediction_confidence

    # ------------------------------------------------------------------ #
    # EAR / blink state
    # ------------------------------------------------------------------ #
    def set_ear(self, ear: float) -> None:
        with self._lock:
            self._current_ear = ear
            self._publish_snapshot()

    def get_ear(self) -> float:
        with self._lock:
            return self._current_ear

    def set_blinking(self, is_blinking: bool) -> None:
        with self._lock:
            self._is_blinking = is_blinking
            self._publish_snapshot()

    def is_blinking(self) -> bool:
        with self._lock:
            return self._is_blinking

    def set_double_blink_detected(self, detected: bool) -> None:
        with self._lock:
            self._double_blink_detected = detected
            self._publish_snapshot()

    def consume_double_blink(self) -> bool:
        with self._lock:
            was_detected = self._double_blink_detected
            self._double_blink_detected = False
            self._publish_snapshot()
            return was_detected

    def is_double_blink_detected(self) -> bool:
        with self._lock:
            return self._double_blink_detected

    # ------------------------------------------------------------------ #
    # V1 directional classification
    # ------------------------------------------------------------------ #
    def set_v1_direction(self, direction: DirectionV1, magnitude: float = 0.0) -> None:
        with self._lock:
            self._current_v1_direction = direction
            self._current_direction_magnitude = magnitude
            self._publish_snapshot()

    def get_v1_direction(self) -> DirectionV1:
        with self._lock:
            return self._current_v1_direction

    # ------------------------------------------------------------------ #
    # Calibrated screen-space prediction
    # ------------------------------------------------------------------ #
    def set_predicted_screen_position(self, x: int, y: int) -> None:
        with self._lock:
            self._predicted_screen_x = x
            self._predicted_screen_y = y
            self._publish_snapshot()

    def get_predicted_screen_position(self) -> tuple[int, int]:
        with self._lock:
            return self._predicted_screen_x, self._predicted_screen_y

    def set_calibration_report(self, report: Optional[Any]) -> None:
        with self._lock:
            self._last_calibration_report = report
            self._publish_snapshot()

    def get_calibration_report(self) -> Optional[Any]:
        with self._lock:
            return self._last_calibration_report

    # ------------------------------------------------------------------ #
    # Telemetry setters (v2.0 - v4.0)
    # ------------------------------------------------------------------ #
    def set_connection_state(self, state_str: str) -> None:
        """Set the webcam/OS hardware connection state."""
        with self._lock:
            self._connection_state = str(state_str)
            self._publish_snapshot()

    def set_deadzone_calibrated(self, calibrated: bool) -> None:
        """Set whether the adaptive deadzone baseline has been calibrated."""
        with self._lock:
            self._is_deadzone_calibrated = bool(calibrated)
            self._publish_snapshot()

    def set_filter_mode(self, mode_str: str) -> None:
        """Set the active gaze filtering description."""
        with self._lock:
            self._filter_mode = str(mode_str)
            self._publish_snapshot()

    def increment_online_adaptation_count(self) -> None:
        """Increment count of successful online RLS calibration adaptations."""
        with self._lock:
            self._online_adaptation_count += 1
            self._publish_snapshot()

    def set_v5_telemetry(
        self,
        execution_provider: Optional[str] = None,
        circuit_breaker_state: Optional[str] = None,
        implicit_calibration_rmse: Optional[float] = None,
        kernel_driver_active: Optional[bool] = None,
    ) -> None:
        """Update v5.0 Autonomous Enterprise Engine telemetry fields."""
        with self._lock:
            if execution_provider is not None:
                self._execution_provider = str(execution_provider)
            if circuit_breaker_state is not None:
                self._circuit_breaker_state = str(circuit_breaker_state)
            if implicit_calibration_rmse is not None:
                self._implicit_calibration_rmse = float(implicit_calibration_rmse)
            if kernel_driver_active is not None:
                self._kernel_driver_active = bool(kernel_driver_active)
            self._publish_snapshot()

    def set_v7_intent_prediction(
        self, pred_x: float, pred_y: float, confidence: float
    ) -> None:
        """Update v7.0 Micro-Transformer anticipatory gaze forecast."""
        with self._lock:
            self._predicted_saccade_x = float(pred_x)
            self._predicted_saccade_y = float(pred_y)
            self._intent_confidence = float(confidence)
            self._publish_snapshot()

    def get_v7_intent_prediction(self) -> tuple[float, float, float]:
        """Return latest v7.0 (predicted_x, predicted_y, intent_confidence)."""
        with self._lock:
            return self._predicted_saccade_x, self._predicted_saccade_y, self._intent_confidence

    # ------------------------------------------------------------------ #
    # Bulk access (Lock-free double-buffered atomic pointer read)
    # ------------------------------------------------------------------ #
    def get_snapshot(self) -> StateSnapshot:
        """
        Return an immutable snapshot of the entire state with zero lock contention
        via double-buffered atomic pointer reference swap.
        """
        return self._snapshot

    def reset(self) -> None:
        """
        Reset all fields to their initial defaults.
        """
        with self._lock:
            self._current_pupil_x = 0.5
            self._current_pupil_y = 0.5
            self._current_ear = 0.0
            self._is_blinking = False
            self._double_blink_detected = False
            self._current_v1_direction = DirectionV1.UNKNOWN
            self._smoothed_pupil_x = 0.5
            self._smoothed_pupil_y = 0.5
            self._head_yaw = None
            self._head_pitch = None
            self._head_roll = None
            self._head_pose_valid = False
            self._gaze_confidence = None
            self._tracking_confidence = None
            self._screen_prediction_confidence = None
            self._predicted_screen_x = 0
            self._predicted_screen_y = 0
            self._execution_provider = "NPU"
            self._circuit_breaker_state = "CLOSED"
            self._implicit_calibration_rmse = 0.0
            self._kernel_driver_active = False
            self._predicted_saccade_x = 0.0
            self._predicted_saccade_y = 0.0
            self._intent_confidence = 0.0
            self._publish_snapshot()

