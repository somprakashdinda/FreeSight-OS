"""
config.py
=========

Centralized, immutable configuration for the headless eye-tracking Host OS
control application.

Design goals
------------
- Zero magic numbers scattered through the codebase: every tunable constant
  lives here, grouped by concern, as a frozen dataclass.
- Safe to import from any thread/module without side effects (no I/O, no
  hardware access, no global mutable state).
- Each dataclass instance is frozen (immutable) so accidental runtime
  mutation of "constants" is caught early as an AttributeError rather than
  silently corrupting shared configuration.

Usage
-----
    from config import CV_CONFIG, EAR_CONFIG, DIRECTION_CONFIG, ANDROID_CONFIG

    if ear < EAR_CONFIG.EAR_THRESHOLD:
        ...

Two groups of configuration
----------------------------
This module keeps a clear line between two kinds of settings:

1. **CV / vision-pipeline configuration** — everything that tunes the
   per-frame detection/estimation math itself: `CVConfig`, `IrisConfig`,
   `HeadPoseConfig`, `EARConfig`, `DirectionConfigV1`, `GazeConfig`. These
   are consumed by `vision_pipeline.py` and `gaze_mapper.py` and have no
   opinion about threads, sockets, or timing loops.
2. **Orchestration / runtime configuration** — everything that governs how
   the app is wired together and scheduled: `HostOSConfig` (camera and
   screen settings) and `PerformanceConfig` (processing-loop
   pacing, staleness, worker sleep intervals). These are consumed by
   `android_interop.py` and `main.py` and have no opinion about detection
   thresholds or model internals.

Compatibility note
-------------------
`CVConfig` still carries its original `YOLO_MODEL_PATH` /
`YOLO_CONFIDENCE_THRESHOLD` / `YOLO_DEVICE` fields even though
`vision_pipeline.py` no longer uses YOLO (it was replaced by MediaPipe
Face Mesh). They're kept rather than silently removed, in case any
external tooling/scripts still reference them; new code should not read
them. `FACE_MESH_MAX_NUM_FACES` is likewise kept alongside the new
`MAX_NUM_FACES` (see `CVConfig` docstring for which one is actually read
by the current pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, Tuple


# --------------------------------------------------------------------------- #
# Orchestration / runtime configuration
# --------------------------------------------------------------------------- #
# Host OS connection configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HostOSConfig:
    """
    Configuration for the host OS control interface.
    """
    camera_index: int = 0
    screen_width: int = 1920
    screen_height: int = 1080
    failsafe_enabled: bool = False
    scroll_sensitivity: int = 100
    # Dwell-click settings
    dwell_threshold_px: int = 30
    dwell_duration_seconds: float = 1.5
    gaze_sensitivity: float = 1.0
    # Multi-Monitor Virtual Desktop settings
    use_virtual_desktop: bool = True
    virtual_screen_left: int = 0
    virtual_screen_top: int = 0
    virtual_screen_width: int = 1920
    virtual_screen_height: int = 1080


# --------------------------------------------------------------------------- #
# Processing-loop pacing / staleness configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerformanceConfig:
    """
    Runtime pacing knobs for the orchestration loop (main.py) that pulls
    frames from `WebcamCapture.get_latest_frame()` and feeds them
    through the CV pipeline. Deliberately separate from `CVConfig`:
    nothing here changes what the pipeline computes, only how often/how
    promptly it's asked to compute it.

    Attributes
    ----------
    TARGET_PROCESSING_FPS:
        Target rate at which the orchestration loop pulls a frame and
        runs it through `GazeTracker.process_frame()`. Distinct from
        `CVConfig.TARGET_FPS` (the camera/stream's own target rate): the
        camera can keep producing frames faster than the CV pipeline can
        (or needs to) consume them, and `WebcamCapture`'s
        always-overwrite latest-frame cache is exactly what lets the
        processing loop run at this lower, independent rate without a
        backlog ever building up.
    MAX_STALE_FRAME_AGE_SECONDS:
        If the most recently decoded frame is older than this when the
        orchestration loop reads it, it should be treated as stale (e.g.
        the video decode thread stalled or the connection dropped) rather
        than processed as if it were live.
    WORKER_SLEEP_INTERVAL_SECONDS:
        Sleep duration used by a polling loop waiting for the next frame
        (or the next state transition) to avoid busy-waiting a CPU core
        at 100%. Should be well under `1 / TARGET_PROCESSING_FPS`.
    """

    TARGET_PROCESSING_FPS: int = 30
    MAX_STALE_FRAME_AGE_SECONDS: float = 0.5
    WORKER_SLEEP_INTERVAL_SECONDS: float = 0.005


# --------------------------------------------------------------------------- #
# CV / vision-pipeline configuration
# --------------------------------------------------------------------------- #
# Capture + face/landmark detection configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CVConfig:
    """
    Parameters governing the incoming video stream and the face-landmark
    detection model (MediaPipe Face Mesh).

    Attributes
    ----------
    STREAM_WIDTH / STREAM_HEIGHT:
        Expected resolution of the frame stream decoded from the device
        (scrcpy video socket, H.264 elementary stream, decoded via PyAV).
    TARGET_FPS:
        Camera/stream's own target frame rate. See
        `PerformanceConfig.TARGET_PROCESSING_FPS` for the (independent)
        rate at which the orchestration loop consumes frames.
    YOLO_MODEL_PATH / YOLO_CONFIDENCE_THRESHOLD / YOLO_DEVICE:
        **Legacy, unused by the current pipeline.** Kept only for
        backward compatibility with any external tooling that still
        references them — `vision_pipeline.py` was migrated from a YOLO
        face-box detector to MediaPipe Face Mesh and no longer reads
        these. Do not wire new code to them.
    FACE_MESH_MAX_NUM_FACES:
        **Legacy name**, kept for backward compatibility. The MediaPipe
        Face Mesh loader in `vision_pipeline.py` actually reads
        `MAX_NUM_FACES` (below) via `getattr`; keep the two in sync if
        you change either.
    MAX_NUM_FACES:
        Cap on simultaneous tracked faces, passed to MediaPipe Face
        Mesh's `max_num_faces`. Should be 1 for this single-user gaze/
        touch-injection scheme — this is also how "multiple faces in
        frame" is handled: MediaPipe itself only ever returns the most
        prominent face when this is 1.
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE:
        Minimum face-detection confidence MediaPipe requires before it
        will initialize landmark tracking on a face.
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE:
        Minimum landmark-tracking confidence MediaPipe requires before
        falling back to running full detection again (video mode).
    MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH:
        Path to an explicit `.task` model bundle, or `None` to use
        MediaPipe's bundled default. `None` by default because the
        current pipeline uses the legacy `mp.solutions.face_mesh`
        solution, which bundles its own model with no external path
        needed; this field only becomes meaningful if the pipeline is
        later migrated to the newer `mediapipe.tasks` Face Landmarker
        API, which does take an explicit model file.
    LANDMARK_SMOOTHING_ALPHA:
        Exponential moving-average factor (0-1) applied to raw pupil-
        ratio coordinates to reduce jitter before EAR/gaze computation.
        Lower values smooth more aggressively. (Iris and head-pose
        smoothing have their own, independently tunable alphas — see
        `IrisConfig`/`HeadPoseConfig` — this one specifically governs the
        pupil_x/pupil_y and EAR smoothers in `GazeTracker`.)
    """

    # Standard landscape webcam resolution
    STREAM_WIDTH: int = 1280
    STREAM_HEIGHT: int = 720
    TARGET_FPS: int = 30
    YOLO_MODEL_PATH: str = "yolov8n-face.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_DEVICE: str = "cpu"
    FACE_MESH_MAX_NUM_FACES: int = 1
    MAX_NUM_FACES: int = 1
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH: Optional[str] = None
    LANDMARK_SMOOTHING_ALPHA: float = 0.6


# --------------------------------------------------------------------------- #
# Iris / pupil tracking configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IrisConfig:
    """
    Parameters governing iris/pupil position extraction and smoothing
    (see `vision_pipeline.GazeTracker._build_eye_result` /
    `get_gaze_ratios`).

    Attributes
    ----------
    SMOOTHING_ALPHA:
        Exponential moving-average factor (0-1) for the iris/pupil ratio
        smoothers. Independently tunable from
        `CVConfig.LANDMARK_SMOOTHING_ALPHA` — defaults to the same value
        for consistency, but iris position tends to be noisier
        frame-to-frame than the overall pupil ratio it feeds into, so
        this is broken out separately rather than sharing one knob.
    SMOOTHING_BUFFER_SIZE:
        Number of recent raw samples the median pre-filter (inside
        `EMASmoother`) considers before the EMA update, for iris-specific
        smoothing. See `CVConfig`/`EMASmoother` docstring for why a
        median pre-filter is used at all (single-frame spike rejection).
    MIN_PUPIL_CONFIDENCE:
        Minimum confidence required to treat an iris detection as valid.
        Reserved for a future geometric-plausibility gate (e.g. iris
        ring-point spread/roundness sanity check) — MediaPipe Face Mesh
        does not expose a native per-landmark confidence score, so the
        current pipeline instead gates purely on landmark *presence*
        (`pupil_found`). Not yet consumed; present so a future confidence
        heuristic has a config home without inventing a new dataclass.
    VALID_RATIO_MIN / VALID_RATIO_MAX:
        Sanity bounds for a normalized iris ratio coordinate. Ratios
        computed by `get_gaze_ratios` are already clipped into
        [VALID_RATIO_MIN, VALID_RATIO_MAX]; exposed here (rather than as
        an inline literal) so calibration/diagnostic code can reference
        the same bounds instead of re-hardcoding 0.0/1.0.
    """

    SMOOTHING_ALPHA: float = 0.6
    SMOOTHING_BUFFER_SIZE: int = 5
    MIN_PUPIL_CONFIDENCE: float = 0.5
    VALID_RATIO_MIN: float = 0.0
    VALID_RATIO_MAX: float = 1.0


# --------------------------------------------------------------------------- #
# Head pose estimation configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadPoseConfig:
    """
    Parameters governing head-pose (yaw/pitch/roll) estimation and
    smoothing (see `vision_pipeline.GazeTracker._estimate_head_pose` /
    `_estimate_and_smooth_head_pose`).

    Attributes
    ----------
    SMOOTHING_ALPHA:
        Exponential moving-average factor (0-1) for the yaw/pitch/roll
        smoothers. Independently tunable from
        `CVConfig.LANDMARK_SMOOTHING_ALPHA` — defaults to the same value
        for consistency ("same philosophy as pupil smoothing"), but head
        pose and pupil ratio can reasonably warrant different amounts of
        smoothing (head pose changes more slowly than saccadic eye
        movement, so a future tuning pass may want this lower/more
        aggressive).
    SMOOTHING_BUFFER_SIZE:
        Median pre-filter buffer size for the head-pose smoothers, same
        role as `IrisConfig.SMOOTHING_BUFFER_SIZE` but independently
        tunable.
    MAX_YAW_DEG / MAX_PITCH_DEG / MAX_ROLL_DEG:
        Per-axis plausibility ceiling, in degrees. A `solvePnP` estimate
        whose magnitude on any axis exceeds its ceiling is treated as
        unreliable (near-edge-on face / degenerate landmark geometry)
        rather than a genuine extreme head pose, and reported as `None`
        instead. Split per-axis (rather than one shared ceiling) since a
        phone-camera gaze-tracking setup can plausibly see much larger
        yaw excursions (turning the head toward/away from the phone)
        than pitch or roll excursions before the estimate becomes
        untrustworthy.
    """

    SMOOTHING_ALPHA: float = 0.6
    SMOOTHING_BUFFER_SIZE: int = 5
    MAX_YAW_DEG: float = 100.0
    MAX_PITCH_DEG: float = 80.0
    MAX_ROLL_DEG: float = 80.0


# --------------------------------------------------------------------------- #
# Eye Aspect Ratio / blink-detection configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EARConfig:
    """
    Thresholds governing blink and double-blink detection via Eye Aspect
    Ratio (EAR), following the standard Soukupova & Cech formulation.

    Attributes
    ----------
    EAR_THRESHOLD:
        EAR value below which an eye is considered "closed" for a given
        frame.
    CONSECUTIVE_FRAMES_FOR_BLINK:
        Minimum number of consecutive closed-eye frames required before a
        blink event is registered (debounces single-frame CV noise).
    DOUBLE_BLINK_MAX_DELAY:
        Maximum time (seconds) between the end of one blink and the start
        of the next for the pair to count as a "double blink" gesture
        (typically mapped to a tap/select action).
    BLINK_REFRACTORY_PERIOD:
        Minimum time (seconds) after a registered blink before a new blink
        can begin registering, to avoid double-counting a single prolonged
        closure.
    """

    EAR_THRESHOLD: float = 0.20
    CONSECUTIVE_FRAMES_FOR_BLINK: int = 2
    DOUBLE_BLINK_MAX_DELAY: float = 0.5
    BLINK_REFRACTORY_PERIOD: float = 0.15


# --------------------------------------------------------------------------- #
# Version 1 directional gaze thresholds (ratio-based, pre-calibration)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectionConfigV1:
    """
    Simple, calibration-free directional thresholds based on normalized
    pupil position within the eye bounding box (0.0 = fully left/up,
    1.0 = fully right/down). Used as a fallback / bootstrap mode before
    the full affine gaze mapping (see gaze_mapper.py) has been calibrated.

    Attributes
    ----------
    X_RATIO_LEFT:
        Normalized horizontal pupil ratio below which gaze is classified
        "left".
    X_RATIO_RIGHT:
        Normalized horizontal pupil ratio above which gaze is classified
        "right".
    Y_RATIO_UP:
        Normalized vertical pupil ratio below which gaze is classified
        "up".
    Y_RATIO_DOWN:
        Normalized vertical pupil ratio above which gaze is classified
        "down".
    DIRECTION_HOLD_SECONDS:
        Minimum time gaze must remain in a directional zone before a
        swipe command is dispatched (prevents accidental swipes from
        transient glances).
    """

    X_RATIO_LEFT: float = 0.40
    X_RATIO_RIGHT: float = 0.60
    Y_RATIO_UP: float = 0.40
    Y_RATIO_DOWN: float = 0.60
    DIRECTION_HOLD_SECONDS: float = 0.1
    POSE_COMPENSATION_FACTOR: float = 0.002
    # Dynamic Lighting & Anatomy Deadzone Auto-Tuning (v2.0+)
    ENABLE_ADAPTIVE_DEADZONE: bool = True
    ADAPTIVE_BASELINE_SECONDS: float = 3.0
    ADAPTIVE_HALF_WIDTH_X: float = 0.10
    ADAPTIVE_HALF_HEIGHT_Y: float = 0.10


# --------------------------------------------------------------------------- #
# Calibrated gaze mapping (ScreenMapper / TerminalCalibrator) configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GazeConfig:
    """
    Parameters governing the calibrated pupil(+head-pose)-to-screen
    mapping (see gaze_mapper.py: `ScreenMapper`, `TerminalCalibrator`).

    Attributes
    ----------
    CALIBRATION_GRID_SIZE:
        Grid dimension for the calibration routine (3 == the classic
        3x3, 9-point layout).
    CALIBRATION_GRID_INSET_FRACTION:
        How far the calibration grid points are inset from the screen's
        true 0.0/1.0 edges (e.g. 0.1 means the outermost points sit at
        10%/90% of each axis). Points are inset rather than placed on
        the literal edge because asking a user to fixate the exact
        corner pixel encourages head movement rather than pure eye
        movement, which would bias the calibration.
    CALIBRATION_SAMPLES_PER_POINT:
        Number of raw pupil-ratio samples averaged into one calibration
        observation per grid point.
    CALIBRATION_SAMPLE_INTERVAL_SECONDS:
        Delay between successive samples while collecting one grid
        point's temporal average.
    CALIBRATION_MAX_ATTEMPTS_PER_POINT:
        How many times a single grid point is retried if tracking was
        too unstable/absent to produce a usable sample.
    CALIBRATION_MIN_VALID_SAMPLE_FRACTION:
        Minimum fraction of polled samples that must be valid (face
        detected) for a grid point's temporal average to be accepted.
    PREDICTION_SMOOTHING_ALPHA:
        Reserved for an optional EMA smoothing pass on `ScreenMapper`'s
        *predicted screen coordinates* themselves (as opposed to the
        upstream pupil-ratio smoothing `GazeTracker` already applies
        before the mapper ever sees it) — e.g. for extra on-screen
        cursor stability at the cost of a little added latency. Not yet
        consumed by `ScreenMapper.predict()`.
    OUTLIER_REJECTION_MAX_RMSE_PX:
        Diagnostic threshold: a `CalibrationFitReport` with `rmse_x_px`
        or `rmse_y_px` above this indicates the fit fetched a poor
        match to the collected calibration points (noisy tracking,
        user movement during calibration, etc.) and re-calibration
        should be recommended. Reserved for that diagnostic check; not
        yet enforced automatically by `train_model()`.
    """

    CALIBRATION_GRID_SIZE: int = 3
    CALIBRATION_GRID_INSET_FRACTION: float = 0.1
    CALIBRATION_SAMPLES_PER_POINT: int = 30
    CALIBRATION_SAMPLE_INTERVAL_SECONDS: float = 0.03
    CALIBRATION_MAX_ATTEMPTS_PER_POINT: int = 3
    CALIBRATION_MIN_VALID_SAMPLE_FRACTION: float = 0.5
    PREDICTION_SMOOTHING_ALPHA: float = 0.6
    OUTLIER_REJECTION_MAX_RMSE_PX: float = 75.0
    # Predictive UKF & Online RLS Adaptation (v2.0 - v4.0)
    USE_PREDICTIVE_FILTER: bool = True
    PREDICTION_AHEAD_FRAMES: float = 1.0
    PREDICTIVE_DT: float = 1.0 / 60.0
    FILTER_Q_BASE: float = 0.05
    FILTER_R_BASE: float = 2.5
    FILTER_SACCADE_VELOCITY_THRESH: float = 250.0
    ENABLE_ONLINE_ADAPTATION: bool = True
    RLS_FORGETTING_FACTOR: float = 0.995


# --------------------------------------------------------------------------- #
# v5.0 Autonomous Enterprise Engine Configuration (100.0/100 Milestone)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class V5EnterpriseConfig:
    """
    Configuration parameters for v5.0 Autonomous Enterprise Engine:
    - Zero-Touch Implicit Calibration via GNN Saliency Alignment
    - 6-DOF Spatial Geometry & Posture Invariance (+/- 75 deg)
    - Ring-0 Kernel KMDF Input Injection Driver
    - Self-Healing Circuit Breaker Hardware Fallback
    """
    ENABLE_IMPLICIT_GNN_CALIBRATION: bool = True
    ENABLE_KERNEL_INPUT_DRIVER: bool = True
    ENABLE_6DOF_SPATIAL_GEOMETRY: bool = True
    PREFERRED_EXECUTION_PROVIDER: str = "NPU"
    SALIENCY_MIN_WEIGHT_THRESHOLD: float = 0.85
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RECOVERY_SEC: float = 5.0
    KMDF_DRIVER_SYMBOLIC_LINK: str = r"\\.\EyeTrackerKMDFInput"
    TARGET_MAX_LATENCY_MS: float = 4.2
    CALIBRATION_RMSE_GOAL_PX: float = 3.1


# --------------------------------------------------------------------------- #
# Aggregate, ready-to-import singletons
# --------------------------------------------------------------------------- #
HOST_OS_CONFIG: Final[HostOSConfig] = HostOSConfig()
PERFORMANCE_CONFIG: Final[PerformanceConfig] = PerformanceConfig()
CV_CONFIG: Final[CVConfig] = CVConfig()
IRIS_CONFIG: Final[IrisConfig] = IrisConfig()
HEAD_POSE_CONFIG: Final[HeadPoseConfig] = HeadPoseConfig()
EAR_CONFIG: Final[EARConfig] = EARConfig()
DIRECTION_CONFIG: Final[DirectionConfigV1] = DirectionConfigV1()
GAZE_CONFIG: Final[GazeConfig] = GazeConfig()
V5_CONFIG: Final[V5EnterpriseConfig] = V5EnterpriseConfig()



# --------------------------------------------------------------------------- #
# Misc. path constants (kept outside the dataclasses since they're derived,
# not tunable CV/blink/direction parameters)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
CALIBRATION_PROFILE_PATH: Final[Path] = PROJECT_ROOT / "calibration_profile.json"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

