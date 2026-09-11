"""
vision_pipeline.py
==================

Main headless computer-vision pipeline.

Pipeline:

    BGR frame
        |
        v
    MediaPipe Face Mesh
        |
        +----> IrisTracker
        |
        +----> HeadPoseEstimator
        |
        +----> BlinkDetector
        |
        v
    Gaze / direction / blink / head-pose result

No GUI.
No cv2.imshow().
No camera capture.
No Android communication.

This module receives BGR frames from android_interop.py and returns
plain Python dictionaries containing gaze, iris, EAR, blink, direction,
and head-pose information.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import (
    CVConfig,
    EARConfig,
    DirectionConfigV1,
    CV_CONFIG,
    EAR_CONFIG,
    DIRECTION_CONFIG,
)

from state_manager import DirectionV1
from vision.iris_tracker import IrisTracker
from vision.head_pose import HeadPoseEstimator
from vision.blink_detector import BlinkDetector
from utils import EMASmoother


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EyeLandmarks6 = Sequence[Tuple[float, float]]


# ---------------------------------------------------------------------------
# MediaPipe landmark constants
# ---------------------------------------------------------------------------

# These are kept here only for face-box generation / safety checks.
_EXPECTED_LANDMARK_COUNT_BASE = 468
_EXPECTED_LANDMARK_COUNT_WITH_IRIS = 478


# ---------------------------------------------------------------------------
# Dynamic Lighting & Anatomy Deadzone Auto-Tuning (v2.0+)
# ---------------------------------------------------------------------------

class AdaptiveDeadzoneManager:
    """
    Dynamic Lighting & Anatomy Deadzone Auto-Tuning.

    Measures baseline pupil center offsets and eye geometry during a
    3-second baseline calibration phase on startup, dynamically setting
    inner/outer deadzone bounds for Mode 1 directional scrolling.
    """

    def __init__(
        self,
        baseline_seconds: float = 3.0,
        target_fps: int = 30,
        half_width_x: float = 0.10,
        half_height_y: float = 0.10,
        pose_compensation_factor: float = 0.002,
    ) -> None:
        self.baseline_frames = max(10, int(baseline_seconds * target_fps))
        self.half_width_x = float(half_width_x)
        self.half_height_y = float(half_height_y)
        self.pose_compensation_factor = float(pose_compensation_factor)

        self._samples_x: Deque[float] = deque(maxlen=self.baseline_frames)
        self._samples_y: Deque[float] = deque(maxlen=self.baseline_frames)
        self._baseline_center_x: float = 0.50
        self._baseline_center_y: float = 0.50
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def baseline_center(self) -> Tuple[float, float]:
        return self._baseline_center_x, self._baseline_center_y

    def add_sample(self, pupil_x: float, pupil_y: float) -> bool:
        """Add a stable gaze observation to baseline estimation."""
        if not (0.0 <= pupil_x <= 1.0 and 0.0 <= pupil_y <= 1.0):
            return False
        self._samples_x.append(float(pupil_x))
        self._samples_y.append(float(pupil_y))

        if not self._is_calibrated and len(self._samples_x) >= self.baseline_frames:
            self._baseline_center_x = float(np.median(self._samples_x))
            self._baseline_center_y = float(np.median(self._samples_y))
            self._is_calibrated = True

        return self._is_calibrated

    def classify(
        self,
        ratio_x: float,
        ratio_y: float,
        head_pose: Optional[Dict[str, Any]] = None,
        fallback_cfg: Optional[DirectionConfigV1] = None,
    ) -> Tuple[DirectionV1, float]:
        """Classify gaze direction using adaptive baseline bounds."""
        # Update baseline if not yet calibrated
        if not self._is_calibrated:
            self.add_sample(ratio_x, ratio_y)

        # Dynamic deadzone boundaries around resting baseline
        if self._is_calibrated:
            cx, cy = self._baseline_center_x, self._baseline_center_y
            left_bound = max(0.1, cx - self.half_width_x)
            right_bound = min(0.9, cx + self.half_width_x)
            up_bound = max(0.1, cy - self.half_height_y)
            down_bound = min(0.9, cy + self.half_height_y)
        else:
            # Warmup fallback to configured static thresholds
            left_bound = getattr(fallback_cfg, "X_RATIO_LEFT", 0.40)
            right_bound = getattr(fallback_cfg, "X_RATIO_RIGHT", 0.60)
            up_bound = getattr(fallback_cfg, "Y_RATIO_UP", 0.40)
            down_bound = getattr(fallback_cfg, "Y_RATIO_DOWN", 0.60)

        # Head Pose Pitch Compensation
        adjusted_y = ratio_y
        if head_pose and head_pose.get("valid", False):
            pitch = float(head_pose.get("pitch", 0.0))
            comp_factor = (
                getattr(fallback_cfg, "POSE_COMPENSATION_FACTOR", self.pose_compensation_factor)
                if fallback_cfg
                else self.pose_compensation_factor
            )
            adjusted_y -= pitch * comp_factor

        # Horizontal classification
        if ratio_x < left_bound:
            mag = (left_bound - ratio_x) / max(left_bound, 1e-6)
            return DirectionV1.LEFT, min(mag, 1.0)
        if ratio_x > right_bound:
            mag = (ratio_x - right_bound) / max(1.0 - right_bound, 1e-6)
            return DirectionV1.RIGHT, min(mag, 1.0)

        # Vertical classification
        if adjusted_y < up_bound:
            mag = (up_bound - adjusted_y) / max(up_bound, 1e-6)
            return DirectionV1.UP, min(mag, 1.0)
        if adjusted_y > down_bound:
            mag = (adjusted_y - down_bound) / max(1.0 - down_bound, 1e-6)
            return DirectionV1.DOWN, min(mag, 1.0)

        return DirectionV1.CENTER, 0.0


# ---------------------------------------------------------------------------
# Main gaze tracker
# ---------------------------------------------------------------------------

class GazeTracker:
    """
    Stateful per-session gaze / blink / head-pose tracker.

    One instance should be reused for the entire video stream.
    """

    def __init__(
        self,
        cv_config: CVConfig = CV_CONFIG,
        ear_config: EARConfig = EAR_CONFIG,
        direction_config: DirectionConfigV1 = DIRECTION_CONFIG,
        smoothing_buffer_size: int = 5,
    ) -> None:

        self._cv_config = cv_config
        self._ear_config = ear_config
        self._direction_config = direction_config

        # ---------------------------------------------------------------
        # Adaptive Deadzone Auto-Tuning (v2.0+)
        # ---------------------------------------------------------------
        self._deadzone_manager = AdaptiveDeadzoneManager(
            baseline_seconds=getattr(direction_config, "ADAPTIVE_BASELINE_SECONDS", 3.0),
            target_fps=getattr(cv_config, "TARGET_FPS", 30),
            half_width_x=getattr(direction_config, "ADAPTIVE_HALF_WIDTH_X", 0.10),
            half_height_y=getattr(direction_config, "ADAPTIVE_HALF_HEIGHT_Y", 0.10),
            pose_compensation_factor=getattr(
                direction_config, "POSE_COMPENSATION_FACTOR", 0.002
            ),
        )

        # ---------------------------------------------------------------
        # MediaPipe Face Mesh
        # ---------------------------------------------------------------

        self._face_mesh = self._load_face_mesh(
            cv_config
        )

        # ---------------------------------------------------------------
        # Iris tracker
        # ---------------------------------------------------------------

        self._iris_tracker = IrisTracker()

        # ---------------------------------------------------------------
        # Head-pose estimator
        # ---------------------------------------------------------------

        self._head_pose_estimator = HeadPoseEstimator()

        # ---------------------------------------------------------------
        # Blink detector
        # ---------------------------------------------------------------

        self._blink_detector = BlinkDetector(
            ear_threshold=ear_config.EAR_THRESHOLD,
            min_closed_frames=(
                ear_config.CONSECUTIVE_FRAMES_FOR_BLINK
            ),
            double_blink_max_delay=(
                ear_config.DOUBLE_BLINK_MAX_DELAY
            ),
            refractory_period=(
                ear_config.BLINK_REFRACTORY_PERIOD
            ),
        )

        # ---------------------------------------------------------------
        # Gaze smoothing
        # ---------------------------------------------------------------

        smoothing_alpha = getattr(
            cv_config,
            "LANDMARK_SMOOTHING_ALPHA",
            0.6,
        )

        self._pupil_x_smoother = EMASmoother(
            alpha=smoothing_alpha,
            buffer_size=smoothing_buffer_size,
        )

        self._pupil_y_smoother = EMASmoother(
            alpha=smoothing_alpha,
            buffer_size=smoothing_buffer_size,
        )

        self._ear_smoother = EMASmoother(
            alpha=smoothing_alpha,
            buffer_size=smoothing_buffer_size,
        )

    # ------------------------------------------------------------------
    # MediaPipe setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_face_mesh(
        cv_config: CVConfig,
    ):
        """
        Create MediaPipe Face Mesh in video mode.

        refine_landmarks=True is REQUIRED for iris landmarks 468-477.
        """

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "GazeTracker requires MediaPipe. "
                "Install it with: pip install mediapipe"
            ) from exc

        max_faces = getattr(
            cv_config,
            "MAX_NUM_FACES",
            getattr(
                cv_config,
                "FACE_MESH_MAX_NUM_FACES",
                1,
            ),
        )

        min_detection_confidence = getattr(
            cv_config,
            "MEDIAPIPE_MIN_DETECTION_CONFIDENCE",
            0.5,
        )

        min_tracking_confidence = getattr(
            cv_config,
            "MEDIAPIPE_MIN_TRACKING_CONFIDENCE",
            0.5,
        )

        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=(
                min_detection_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
        )

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process one BGR frame.

        Returns a dictionary containing:

            face_detected
            face_box

            left_region
            right_region

            raw_pupil_ratio
            smoothed_pupil_ratio

            combined_ear

            eye_closed
            blink_completed
            double_blink

            head_pose

            v1_direction
        """

        try:
            return self._process_frame_inner(frame)

        except Exception:
            # A single bad frame must never kill the video loop.
            self._reset_tracking_state()

            return self._no_face_result()

    # ------------------------------------------------------------------
    # Internal frame processing
    # ------------------------------------------------------------------

    def _process_frame_inner(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:

        # ---------------------------------------------------------------
        # Validate frame
        # ---------------------------------------------------------------

        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
        ):
            return self._no_face_result()

        if frame.ndim != 3:
            return self._no_face_result()

        if frame.shape[2] != 3:
            return self._no_face_result()

        # ---------------------------------------------------------------
        # Detect MediaPipe landmarks
        # ---------------------------------------------------------------

        face_landmarks = self._detect_face_landmarks(
            frame
        )

        if face_landmarks is None:
            self._reset_tracking_state()
            return self._no_face_result()

        # ---------------------------------------------------------------
        # Convert MediaPipe landmarks to pixel coordinates
        # ---------------------------------------------------------------

        landmarks_px = self._landmarks_to_pixels(
            face_landmarks,
            frame.shape,
        )

        if landmarks_px is None:
            self._reset_tracking_state()
            return self._no_face_result()

        # ---------------------------------------------------------------
        # Face box
        # ---------------------------------------------------------------

        face_box = self._compute_face_box(
            landmarks_px,
            frame.shape,
        )

        # ---------------------------------------------------------------
        # Iris tracking
        # ---------------------------------------------------------------

        iris_result = self._run_iris_tracker(
            face_landmarks,
            landmarks_px,
        )

        left_region = self._extract_eye_result(
            iris_result,
            "left",
        )

        right_region = self._extract_eye_result(
            iris_result,
            "right",
        )

        # ---------------------------------------------------------------
        # Gaze ratios
        # ---------------------------------------------------------------

        valid_ratios = []

        if left_region["pupil_found"]:
            valid_ratios.append(
                left_region["pupil_ratio"]
            )

        if right_region["pupil_found"]:
            rx, ry = right_region["pupil_ratio"]
            # Right eye's outer->inner axis runs opposite to the left
            # eye's (see iris_tracker.py comment at its OUTER/INNER
            # landmark constants). Invert X so both eyes agree that
            # "ratio increasing" = gaze moving toward the subject's
            # right, before they get averaged together.
            valid_ratios.append((1.0 - rx, ry))

        raw_x, raw_y = self._average_ratio(
            valid_ratios
        )

        smoothed_x = (
            self._pupil_x_smoother.update(raw_x)
        )

        smoothed_y = (
            self._pupil_y_smoother.update(raw_y)
        )

        # ---------------------------------------------------------------
        # EAR
        # ---------------------------------------------------------------

        valid_ears = []

        left_ear = left_region.get(
            "ear_estimate"
        )

        right_ear = right_region.get(
            "ear_estimate"
        )

        if left_ear is not None:
            valid_ears.append(
                float(left_ear)
            )

        if right_ear is not None:
            valid_ears.append(
                float(right_ear)
            )

        if valid_ears:
            raw_ear = float(
                np.mean(valid_ears)
            )
        else:
            raw_ear = self._ear_smoother.value

        smoothed_ear = (
            self._ear_smoother.update(raw_ear)
        )

        # ---------------------------------------------------------------
        # Blink detector
        # ---------------------------------------------------------------

        blink_result = self._blink_detector.update(
            smoothed_ear
        )

        # ---------------------------------------------------------------
        # Head pose
        # ---------------------------------------------------------------

        head_pose_result = (
            self._run_head_pose_estimator(
                face_landmarks,
                landmarks_px,
            )
        )

        # ---------------------------------------------------------------
        # Direction
        # ---------------------------------------------------------------

        direction, magnitude = self.classify_v1_direction(
            smoothed_x,
            smoothed_y,
            head_pose={
                "yaw": head_pose_result.yaw,
                "pitch": head_pose_result.pitch,
                "roll": head_pose_result.roll,
                "valid": head_pose_result.valid,
            },
        )

        # ---------------------------------------------------------------
        # Final result
        # ---------------------------------------------------------------

        return {
            "face_detected": True,

            "face_box": face_box,

            "left_region": left_region,
            "right_region": right_region,

            "raw_pupil_ratio": (
                raw_x,
                raw_y,
            ),

            "smoothed_pupil_ratio": (
                smoothed_x,
                smoothed_y,
            ),

            "combined_ear": smoothed_ear,

            "eye_closed": bool(
                blink_result.eye_closed
            ),

            "blink_completed": bool(
                blink_result.blink_completed
            ),

            "double_blink": bool(
                blink_result.double_blink
            ),

            "head_pose": {
                "yaw": float(
                    head_pose_result.yaw
                ),
                "pitch": float(
                    head_pose_result.pitch
                ),
                "roll": float(
                    head_pose_result.roll
                ),
                "valid": bool(
                    head_pose_result.valid
                ),
                "confidence": float(
                    head_pose_result.confidence
                ),
            },

            "v1_direction": direction,
            "direction_magnitude": magnitude,
            "is_deadzone_calibrated": self._deadzone_manager.is_calibrated,
            "baseline_center": self._deadzone_manager.baseline_center,
        }

    # ------------------------------------------------------------------
    # MediaPipe detection
    # ------------------------------------------------------------------

    def _detect_face_landmarks(
        self,
        frame: np.ndarray,
    ):
        """
        Run MediaPipe Face Mesh.

        Returns the actual MediaPipe landmark list, or None.
        """

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame_rgb.flags.writeable = False

        try:
            results = self._face_mesh.process(
                frame_rgb
            )
        finally:
            frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return None

        # MAX_NUM_FACES=1 means we use the first face.
        return results.multi_face_landmarks[0].landmark

    # ------------------------------------------------------------------
    # Landmark conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_pixels(
        landmarks,
        frame_shape: Tuple[int, ...],
    ) -> Optional[np.ndarray]:
        """
        Convert MediaPipe normalized landmarks into
        pixel coordinates.
        """

        if landmarks is None:
            return None

        if len(landmarks) < _EXPECTED_LANDMARK_COUNT_BASE:
            return None

        height = frame_shape[0]
        width = frame_shape[1]

        points = []

        for landmark in landmarks:

            x = float(landmark.x) * width
            y = float(landmark.y) * height

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                return None

            points.append(
                (x, y)
            )

        return np.asarray(
            points,
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Iris tracker adapter
    # ------------------------------------------------------------------

    def _run_iris_tracker(
        self,
        face_landmarks,
        landmarks_px: np.ndarray,
    ):
        """
        Run the modular IrisTracker.

        The tracker created for this project works with the
        MediaPipe landmark collection.

        A small compatibility fallback is included for versions
        where the tracker expects pixel coordinates.
        """

        try:
            return self._iris_tracker.process(
                face_landmarks
            )

        except Exception:
            return self._iris_tracker.process(
                landmarks_px
            )

    # ------------------------------------------------------------------
    # Iris result conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_eye_result(
        iris_result,
        side: str,
    ) -> Dict[str, Any]:
        """
        Convert IrisTrackingResult / EyeGeometry into the
        dictionary format used by the rest of the project.

        Handles both dataclass-style and dictionary-style
        tracker results.
        """

        eye = None

        # ---------------------------------------------------------------
        # Dataclass-style result
        # ---------------------------------------------------------------

        if hasattr(iris_result, side):
            eye = getattr(
                iris_result,
                side,
            )

        # ---------------------------------------------------------------
        # Dictionary-style result
        # ---------------------------------------------------------------

        elif isinstance(
            iris_result,
            dict,
        ):
            eye = iris_result.get(side)

            if eye is None:
                eye = iris_result.get(
                    f"{side}_region"
                )

        # ---------------------------------------------------------------
        # Nothing available
        # ---------------------------------------------------------------

        if eye is None:
            return {
                "box": (
                    0,
                    0,
                    0,
                    0,
                ),
                "pupil_found": False,
                "pupil_ratio": (
                    0.5,
                    0.5,
                ),
                "ear_estimate": None,
            }

        # ---------------------------------------------------------------
        # EyeGeometry dataclass
        # ---------------------------------------------------------------

        pupil_ratio = getattr(
            eye,
            "iris_ratio",
            None,
        )

        if pupil_ratio is None:
            pupil_ratio = getattr(
                eye,
                "normalized_iris",
                None,
            )

        if pupil_ratio is None:
            pupil_ratio = getattr(
                eye,
                "pupil_ratio",
                None,
            )

        # EyeGeometry's real fields are normalized_iris_x /
        # normalized_iris_y (two separate floats, not a combined
        # attribute) -- none of the names checked above ever match it,
        # which is why pupil_ratio was previously always falling through
        # to the (0.5, 0.5) default at the bottom of this block.
        if pupil_ratio is None:
            normalized_x = getattr(eye, "normalized_iris_x", None)
            normalized_y = getattr(eye, "normalized_iris_y", None)
            if normalized_x is not None and normalized_y is not None:
                pupil_ratio = (normalized_x, normalized_y)

        if pupil_ratio is None and isinstance(
            eye,
            dict,
        ):
            pupil_ratio = eye.get(
                "pupil_ratio"
            )

            if pupil_ratio is None:
                normalized_x = eye.get("normalized_iris_x")
                normalized_y = eye.get("normalized_iris_y")
                if normalized_x is not None and normalized_y is not None:
                    pupil_ratio = (normalized_x, normalized_y)

        if pupil_ratio is None:
            pupil_ratio = (
                0.5,
                0.5,
            )

        # ---------------------------------------------------------------
        # EAR
        # ---------------------------------------------------------------

        ear = getattr(
            eye,
            "ear",
            None,
        )

        if ear is None:
            ear = getattr(
                eye,
                "ear_estimate",
                None,
            )

        if ear is None and isinstance(
            eye,
            dict,
        ):
            ear = eye.get(
                "ear_estimate"
            )

        # ---------------------------------------------------------------
        # Validity
        # ---------------------------------------------------------------

        valid = getattr(
            eye,
            "valid",
            None,
        )

        if valid is None:
            valid = getattr(
                eye,
                "pupil_found",
                None,
            )

        if valid is None and isinstance(
            eye,
            dict,
        ):
            valid = eye.get(
                "pupil_found",
                False,
            )

        valid = bool(
            valid
        )

        # ---------------------------------------------------------------
        # Eye box
        # ---------------------------------------------------------------

        box = getattr(
            eye,
            "box",
            None,
        )

        if box is None and isinstance(
            eye,
            dict,
        ):
            box = eye.get(
                "box"
            )

        if box is None:
            box = (
                0,
                0,
                0,
                0,
            )

        return {
            "box": box,
            "pupil_found": valid,
            "pupil_ratio": (
                float(pupil_ratio[0]),
                float(pupil_ratio[1]),
            ),
            "ear_estimate": (
                float(ear)
                if ear is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Head pose adapter
    # ------------------------------------------------------------------

    def _run_head_pose_estimator(
        self,
        face_landmarks,
        landmarks_px: np.ndarray,
    ):
        """
        Run HeadPoseEstimator.

        Prefer the MediaPipe landmark collection. A pixel-coordinate
        fallback is included for compatibility.
        """

        # HeadPoseEstimator consumes the original normalized MediaPipe
        # landmarks and exposes ``estimate`` (not ``process``).
        return self._head_pose_estimator.estimate(face_landmarks)

    # ------------------------------------------------------------------
    # Average gaze ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _average_ratio(
        ratios: Sequence[
            Tuple[float, float]
        ],
    ) -> Tuple[float, float]:
        """
        Average valid eye gaze ratios.
        """

        if not ratios:
            return (
                0.5,
                0.5,
            )

        arr = np.asarray(
            ratios,
            dtype=np.float64,
        )

        if arr.ndim != 2 or arr.shape[1] != 2:
            return (
                0.5,
                0.5,
            )

        mean = np.mean(
            arr,
            axis=0,
        )

        return (
            float(
                np.clip(
                    mean[0],
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    mean[1],
                    0.0,
                    1.0,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Face box
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_face_box(
        landmarks_px: np.ndarray,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        """
        Calculate a face bounding box from all detected
        MediaPipe landmarks.
        """

        height = frame_shape[0]
        width = frame_shape[1]

        x_min = float(
            np.min(
                landmarks_px[:, 0]
            )
        )

        y_min = float(
            np.min(
                landmarks_px[:, 1]
            )
        )

        x_max = float(
            np.max(
                landmarks_px[:, 0]
            )
        )

        y_max = float(
            np.max(
                landmarks_px[:, 1]
            )
        )

        x1 = int(
            np.clip(
                x_min,
                0,
                width,
            )
        )

        y1 = int(
            np.clip(
                y_min,
                0,
                height,
            )
        )

        x2 = int(
            np.clip(
                x_max,
                0,
                width,
            )
        )

        y2 = int(
            np.clip(
                y_max,
                0,
                height,
            )
        )

        return (
            x1,
            y1,
            x2,
            y2,
        )

    @property
    def deadzone_manager(self) -> AdaptiveDeadzoneManager:
        """Dynamic Lighting & Anatomy Deadzone Auto-Tuner."""
        return self._deadzone_manager

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def classify_v1_direction(
        self,
        ratio_x: float,
        ratio_y: float,
        head_pose: Optional[Dict[str, Any]] = None,
    ) -> Tuple[DirectionV1, float]:
        """
        Convert normalized gaze position into V1 direction and a magnitude.
        Magnitude (0.0 to 1.0) represents how far the gaze is into the zone.
        Uses AdaptiveDeadzoneManager for dynamic baseline tuning.
        """
        if hasattr(self, "_deadzone_manager") and getattr(
            self._direction_config, "ENABLE_ADAPTIVE_DEADZONE", True
        ):
            return self._deadzone_manager.classify(
                ratio_x,
                ratio_y,
                head_pose=head_pose,
                fallback_cfg=self._direction_config,
            )

        cfg = self._direction_config

        # --- Head Pose Compensation ---
        adjusted_y = ratio_y
        if head_pose and head_pose.get("valid", False):
            pitch = float(head_pose.get("pitch", 0.0))
            compensation = pitch * getattr(cfg, "POSE_COMPENSATION_FACTOR", 0.003)
            adjusted_y -= compensation

        # Horizontal Check
        if ratio_x < cfg.X_RATIO_LEFT:
            mag = (cfg.X_RATIO_LEFT - ratio_x) / cfg.X_RATIO_LEFT
            return DirectionV1.LEFT, min(mag, 1.0)

        if ratio_x > cfg.X_RATIO_RIGHT:
            mag = (ratio_x - cfg.X_RATIO_RIGHT) / (1.0 - cfg.X_RATIO_RIGHT)
            return DirectionV1.RIGHT, min(mag, 1.0)

        # Vertical Check
        if adjusted_y < cfg.Y_RATIO_UP:
            mag = (cfg.Y_RATIO_UP - adjusted_y) / cfg.Y_RATIO_UP
            return DirectionV1.UP, min(mag, 1.0)

        if adjusted_y > cfg.Y_RATIO_DOWN:
            mag = (adjusted_y - cfg.Y_RATIO_DOWN) / (1.0 - cfg.Y_RATIO_DOWN)
            return DirectionV1.DOWN, min(mag, 1.0)

        return DirectionV1.CENTER, 0.0

    # ------------------------------------------------------------------
    # No-face result
    # ------------------------------------------------------------------

    def _no_face_result(
        self,
    ) -> Dict[str, Any]:
        """
        Safe result when no face is detected.
        """

        return {
            "face_detected": False,

            "smoothed_pupil_ratio": (
                self._pupil_x_smoother.value,
                self._pupil_y_smoother.value,
            ),

            "combined_ear": (
                self._ear_smoother.value
            ),

            "eye_closed": False,
            "blink_completed": False,
            "double_blink": False,

            "head_pose": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "valid": False,
                "confidence": 0.0,
            },

            "v1_direction": (
                DirectionV1.UNKNOWN
            ),
        }

    # ------------------------------------------------------------------
    # Reset tracking
    # ------------------------------------------------------------------

    def _reset_tracking_state(
        self,
    ) -> None:
        """
        Reset all temporal tracking state after tracking loss.
        """

        self._pupil_x_smoother.reset()
        self._pupil_y_smoother.reset()
        self._ear_smoother.reset()

        try:
            self._blink_detector.reset()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Release MediaPipe resources.
        """

        try:
            self._face_mesh.close()
        except Exception:
            pass

        try:
            close_method = getattr(
                self._iris_tracker,
                "close",
                None,
            )

            if callable(close_method):
                close_method()

        except Exception:
            pass

        try:
            close_method = getattr(
                self._head_pose_estimator,
                "close",
                None,
            )

            if callable(close_method):
                close_method()

        except Exception:
            pass