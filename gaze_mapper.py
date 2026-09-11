"""Calibration and screen-coordinate mapping for the gaze controller."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Sequence
import numpy as np
from utils import EMASmoother

from config import CALIBRATION_PROFILE_PATH, GAZE_CONFIG, HOST_OS_CONFIG
from predictive_filter import PredictiveGazeUKF


class OnlineGazeAdapter:
    """
    Recursive Least Squares (RLS) continuous online calibration adapter.
    Refines screen-space gaze regression parameters during fixation/dwell events
    to prevent physical posture drift over extended sessions.
    """

    def __init__(
        self,
        num_features: int,
        lambda_factor: float = 0.995,
        initial_theta_x: Optional[np.ndarray] = None,
        initial_theta_y: Optional[np.ndarray] = None,
        initial_p_diag: float = 10.0,
    ) -> None:
        self.num_features = int(num_features)
        self.lambda_factor = float(lambda_factor)
        self.theta_x = (
            np.zeros(self.num_features, dtype=np.float64)
            if initial_theta_x is None
            else np.array(initial_theta_x, dtype=np.float64).copy()
        )
        self.theta_y = (
            np.zeros(self.num_features, dtype=np.float64)
            if initial_theta_y is None
            else np.array(initial_theta_y, dtype=np.float64).copy()
        )
        self.P_x = np.eye(self.num_features, dtype=np.float64) * initial_p_diag
        self.P_y = np.eye(self.num_features, dtype=np.float64) * initial_p_diag
        self.adaptations_count = 0

    def update(self, phi: np.ndarray, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        RLS update step with measurement target (target_x, target_y) given feature vector phi.
        Returns the prediction residuals (error_x, error_y).
        """
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        if phi.shape[0] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features}, got {phi.shape[0]}"
            )

        # Update X
        p_phi_x = self.P_x @ phi
        denom_x = self.lambda_factor + float(phi @ p_phi_x)
        K_x = p_phi_x / max(denom_x, 1e-12)
        pred_x = float(phi @ self.theta_x)
        err_x = float(target_x) - pred_x
        self.theta_x += K_x * err_x
        self.P_x = (self.P_x - np.outer(K_x, p_phi_x)) / self.lambda_factor

        # Update Y
        p_phi_y = self.P_y @ phi
        denom_y = self.lambda_factor + float(phi @ p_phi_y)
        K_y = p_phi_y / max(denom_y, 1e-12)
        pred_y = float(phi @ self.theta_y)
        err_y = float(target_y) - pred_y
        self.theta_y += K_y * err_y
        self.P_y = (self.P_y - np.outer(K_y, p_phi_y)) / self.lambda_factor

        self.adaptations_count += 1
        return err_x, err_y


class GazeObservation(NamedTuple):
    pupil_x: float
    pupil_y: float
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None


class ScreenMapperError(RuntimeError):
    """Raised for invalid calibration or prediction input."""


class ScreenMapperNotTrainedError(ScreenMapperError):
    """Raised before a mapper has calibration coefficients."""


@dataclass(frozen=True)
class CalibrationFitReport:
    num_samples: int
    rmse_x_px: float
    rmse_y_px: float
    design_matrix_rank: int
    num_features: int
    is_well_conditioned: bool
    used_pose_features: bool
    pose_fallback_reason: Optional[str] = None

    def __str__(self) -> str:
        mode = "pose-aware" if self.used_pose_features else "pupil-only"
        warning = "" if self.is_well_conditioned else " [rank-deficient]"
        return (f"ScreenMapper trained on {self.num_samples} samples ({mode}, "
                f"rank {self.design_matrix_rank}/{self.num_features}); "
                f"RMSE x={self.rmse_x_px:.1f}px, y={self.rmse_y_px:.1f}px{warning}")


_GRID = (("top-left", .10, .10), ("top-centre", .50, .10),
         ("top-right", .90, .10), ("middle-left", .10, .50),
         ("centre", .50, .50), ("middle-right", .90, .50),
         ("bottom-left", .10, .90), ("bottom-centre", .50, .90),
         ("bottom-right", .90, .90))


class ScreenMapper:
    """Regularized quadratic gaze-to-screen mapper with safe clamping."""
    def __init__(self, screen_width: int, screen_height: int) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        self._screen_width, self._screen_height = int(screen_width), int(screen_height)
        self._coeffs_x: Optional[np.ndarray] = None
        self._coeffs_y: Optional[np.ndarray] = None
        self._coeffs_x_lin: Optional[np.ndarray] = None
        self._coeffs_y_lin: Optional[np.ndarray] = None
        self._use_pose_features = False
        self._last_report: Optional[CalibrationFitReport] = None
        self._smooth_x = EMASmoother(alpha=GAZE_CONFIG.PREDICTION_SMOOTHING_ALPHA)
        self._smooth_y = EMASmoother(alpha=GAZE_CONFIG.PREDICTION_SMOOTHING_ALPHA)

        # Predictive UKF & Online RLS (v2.0 - v4.0)
        self._predictive_filter: Optional[PredictiveGazeUKF] = (
            PredictiveGazeUKF(
                dt=getattr(GAZE_CONFIG, "PREDICTIVE_DT", 1.0 / 60.0),
                prediction_ahead_frames=getattr(GAZE_CONFIG, "PREDICTION_AHEAD_FRAMES", 1.0),
                q_base=getattr(GAZE_CONFIG, "FILTER_Q_BASE", 0.05),
                r_base=getattr(GAZE_CONFIG, "FILTER_R_BASE", 2.5),
                saccade_velocity_thresh=getattr(
                    GAZE_CONFIG, "FILTER_SACCADE_VELOCITY_THRESH", 250.0
                ),
            )
            if getattr(GAZE_CONFIG, "USE_PREDICTIVE_FILTER", True)
            else None
        )
        self._online_adapter: Optional[OnlineGazeAdapter] = None
        self._implicit_calibrator = None

    def initialize_implicit_calibrator(self):
        """Initialize zero-touch implicit GNN calibration engine."""
        from implicit_calibrator import ImplicitGNNCalibrator
        self._implicit_calibrator = ImplicitGNNCalibrator(self._screen_width, self._screen_height)
        return self._implicit_calibrator

    @property
    def implicit_calibrator(self):
        return self._implicit_calibrator

    @property
    def is_trained(self) -> bool:
        return (self._coeffs_x is not None and self._coeffs_y is not None) or (self._implicit_calibrator is not None)


    @property
    def screen_size(self) -> tuple[int, int]:
        """Screen dimensions for which this mapper was calibrated."""
        return self._screen_width, self._screen_height

    @property
    def last_report(self) -> Optional[CalibrationFitReport]:
        """The fit report from the last training or load operation."""
        return self._last_report

    @staticmethod
    def _features(x: float, y: float, pose: bool, yaw: Optional[float], pitch: Optional[float], roll: Optional[float]) -> np.ndarray:
        values = [1.0, x, y, x * x, y * y, x * y]
        if pose:
            values.extend([
                0.0 if yaw is None else yaw,
                0.0 if pitch is None else pitch,
                0.0 if roll is None else roll,
            ])
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def _sample(sample: Sequence[object]) -> tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]:
        if len(sample) == 4:
            px, py, sx, sy = sample
            return float(px), float(py), float(sx), float(sy), None, None, None
        if len(sample) == 7:
            px, py, sx, sy, yaw, pitch, roll = sample
            return (float(px), float(py), float(sx), float(sy),
                    None if yaw is None else float(yaw),
                    None if pitch is None else float(pitch),
                    None if roll is None else float(roll))
        raise ScreenMapperError("each calibration sample must have 4 or 7 values")

    def train_model(self, calibration_data: Sequence[Sequence[object]]) -> CalibrationFitReport:
        samples = [self._sample(item) for item in calibration_data]
        if len(samples) < 6:
            raise ScreenMapperError("at least six stable calibration points are required")

        for sample in samples:
            if not all(math.isfinite(value) for value in sample[:4]):
                raise ScreenMapperError("calibration coordinates must be finite")
            if any(value is not None and not math.isfinite(value) for value in sample[4:]):
                raise ScreenMapperError("head-pose calibration values must be finite")

        pose_complete = all(all(value is not None for value in item[4:]) for item in samples)
        self._use_pose_features = pose_complete and len(samples) >= 9
        columns = 9 if self._use_pose_features else 6
        matrix = np.vstack([self._features(item[0], item[1], self._use_pose_features, *item[4:]) for item in samples])
        x_target = np.asarray([item[2] for item in samples])
        y_target = np.asarray([item[3] for item in samples])

        # --- Linear fallback coefficients ---
        matrix_lin = np.vstack([[1.0, item[0], item[1]] for item in samples])
        try:
            self._coeffs_x_lin = np.linalg.lstsq(matrix_lin, x_target, rcond=None)[0]
            self._coeffs_y_lin = np.linalg.lstsq(matrix_lin, y_target, rcond=None)[0]
        except Exception:
            self._coeffs_x_lin = None
            self._coeffs_y_lin = None

        # --- Quadratic fit ---
        ridge = np.eye(columns) * 1e-4
        ridge[0, 0] = 0.0
        normal = matrix.T @ matrix + ridge
        try:
            coeffs_x = np.linalg.solve(normal, matrix.T @ x_target)
            coeffs_y = np.linalg.solve(normal, matrix.T @ y_target)
        except np.linalg.LinAlgError as exc:
            self._coeffs_x = None
            self._coeffs_y = None
            raise ScreenMapperError("calibration matrix could not be solved") from exc

        if not np.isfinite(coeffs_x).all() or not np.isfinite(coeffs_y).all():
            self._coeffs_x = None
            self._coeffs_y = None
            raise ScreenMapperError("calibration produced non-finite coefficients")

        self._coeffs_x = coeffs_x
        self._coeffs_y = coeffs_y
        if getattr(GAZE_CONFIG, "ENABLE_ONLINE_ADAPTATION", True):
            self._online_adapter = OnlineGazeAdapter(
                columns,
                lambda_factor=getattr(GAZE_CONFIG, "RLS_FORGETTING_FACTOR", 0.995),
                initial_theta_x=coeffs_x,
                initial_theta_y=coeffs_y,
            )
        rank = int(np.linalg.matrix_rank(matrix))
        report = CalibrationFitReport(
            len(samples), float(np.sqrt(np.mean((matrix @ self._coeffs_x - x_target) ** 2))),
            float(np.sqrt(np.mean((matrix @ self._coeffs_y - y_target) ** 2))),
            rank, columns, rank >= columns, self._use_pose_features,
            None if self._use_pose_features or not pose_complete else "pose data incomplete or insufficient")
        self._last_report = report
        return report

    def predict(self, pupil_x: float, pupil_y: float, *, yaw: Optional[float] = None, pitch: Optional[float] = None, roll: Optional[float] = None) -> tuple[int, int]:
        if not self.is_trained:
            raise ScreenMapperNotTrainedError("calibration is not trained")
        if not all(math.isfinite(float(v)) for v in (pupil_x, pupil_y)):
            raise ScreenMapperError("gaze features must be finite")
        if any(
            value is not None and not math.isfinite(float(value))
            for value in (yaw, pitch, roll)
        ):
            raise ScreenMapperError("head-pose features must be finite")

        # Zero-touch implicit calibration path
        if self._coeffs_x is None and self._implicit_calibrator is not None:
            raw_px = float(np.clip(pupil_x, 0, 1) * self._screen_width)
            raw_py = float(np.clip(pupil_y, 0, 1) * self._screen_height)
            imp_x, imp_y = self._implicit_calibrator.process_passive_gaze_sample((raw_px, raw_py))
            if self._predictive_filter is not None:
                pred_x, pred_y = self._predictive_filter.update_and_predict(imp_x, imp_y)
                return int(np.clip(round(pred_x), 0, self._screen_width - 1)), int(np.clip(round(pred_y), 0, self._screen_height - 1))
            return int(np.clip(round(imp_x), 0, self._screen_width - 1)), int(np.clip(round(imp_y), 0, self._screen_height - 1))

        # ---------------------------------------------------------------
        # Coordinate Prediction
        # ---------------------------------------------------------------
        row = self._features(float(np.clip(pupil_x, 0, 1)), float(np.clip(pupil_y, 0, 1)), self._use_pose_features, yaw, pitch, roll)


        coeffs_x = self._online_adapter.theta_x if self._online_adapter is not None else self._coeffs_x
        coeffs_y = self._online_adapter.theta_y if self._online_adapter is not None else self._coeffs_y

        # Use linear fallback if quadratic fit is rank-deficient or missing
        if coeffs_x is None or (self._last_report and not self._last_report.is_well_conditioned):
            if self._coeffs_x_lin is not None:
                matrix_lin = np.asarray([1.0, float(np.clip(pupil_x, 0, 1)), float(np.clip(pupil_y, 0, 1))])
                raw_x = float(matrix_lin @ self._coeffs_x_lin)
                raw_y = float(matrix_lin @ self._coeffs_y_lin)
            else:
                raw_x, raw_y = self._screen_width / 2, self._screen_height / 2
        else:
            raw_x = float(row @ coeffs_x)
            raw_y = float(row @ coeffs_y)

        # Apply sensitivity multiplier relative to screen center
        center_x, center_y = self._screen_width / 2, self._screen_height / 2
        sensitivity = HOST_OS_CONFIG.gaze_sensitivity
        raw_x = center_x + (raw_x - center_x) * sensitivity
        raw_y = center_y + (raw_y - center_y) * sensitivity

        if self._predictive_filter is not None:
            pred_x, pred_y = self._predictive_filter.update_and_predict(raw_x, raw_y)
            smoothed_x = pred_x
            smoothed_y = pred_y
        else:
            smoothed_x = self._smooth_x.update(raw_x)
            smoothed_y = self._smooth_y.update(raw_y)

        x = int(np.clip(round(smoothed_x), 0, self._screen_width - 1))
        y = int(np.clip(round(smoothed_y), 0, self._screen_height - 1))
        return x, y

    def adapt_online(
        self,
        pupil_x: float,
        pupil_y: float,
        target_screen_x: float,
        target_screen_y: float,
        *,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Online adaptation step during dwell fixation. Refines model weights
        using Recursive Least Squares to eliminate posture drift.
        """
        if not self.is_trained or self._online_adapter is None:
            return None
        row = self._features(
            float(np.clip(pupil_x, 0, 1)),
            float(np.clip(pupil_y, 0, 1)),
            self._use_pose_features,
            yaw,
            pitch,
            roll,
        )
        err = self._online_adapter.update(row, target_screen_x, target_screen_y)
        self._coeffs_x = self._online_adapter.theta_x.copy()
        self._coeffs_y = self._online_adapter.theta_y.copy()
        return err

    @property
    def online_adaptation_count(self) -> int:
        return self._online_adapter.adaptations_count if self._online_adapter is not None else 0

    def save_profile(self, path: Path = CALIBRATION_PROFILE_PATH, report: Optional[CalibrationFitReport] = None) -> str:
        if not self.is_trained:
            raise ScreenMapperNotTrainedError("cannot save untrained calibration")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "screen_width": self._screen_width,
            "screen_height": self._screen_height,
            "use_pose_features": self._use_pose_features,
            "coeffs_x": self._coeffs_x.tolist(),
            "coeffs_y": self._coeffs_y.tolist()
        }
        if report:
            data["report"] = {
                "rmse_x": report.rmse_x_px,
                "rmse_y": report.rmse_y_px,
                "is_well_conditioned": report.is_well_conditioned,
                "rank": report.design_matrix_rank,
                "num_features": report.num_features,
                "used_pose_features": report.used_pose_features,
            }

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return f"Saved calibration profile: {path}"

    @classmethod
    def load_profile(cls, path: Path = CALIBRATION_PROFILE_PATH) -> "ScreenMapper":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        mapper = cls(int(payload["screen_width"]), int(payload["screen_height"]))
        mapper._use_pose_features = bool(payload["use_pose_features"])
        mapper._coeffs_x = np.asarray(payload["coeffs_x"], dtype=np.float64)
        mapper._coeffs_y = np.asarray(payload["coeffs_y"], dtype=np.float64)

        report_data = payload.get("report")
        if report_data:
            mapper._last_report = CalibrationFitReport(
                num_samples=report_data.get("num_samples", 0),
                rmse_x_px=report_data["rmse_x"],
                rmse_y_px=report_data["rmse_y"],
                design_matrix_rank=report_data["rank"],
                num_features=report_data["num_features"],
                is_well_conditioned=report_data["is_well_conditioned"],
                used_pose_features=report_data["used_pose_features"],
                pose_fallback_reason=report_data.get("pose_fallback_reason")
            )

        expected = 9 if mapper._use_pose_features else 6
        if mapper._coeffs_x.shape != (expected,) or mapper._coeffs_y.shape != (expected,):
            raise ScreenMapperError("invalid coefficient dimensions in calibration profile")
        if not np.isfinite(mapper._coeffs_x).all() or not np.isfinite(mapper._coeffs_y).all():
            raise ScreenMapperError("calibration profile contains non-finite coefficients")

        if getattr(GAZE_CONFIG, "ENABLE_ONLINE_ADAPTATION", True):
            mapper._online_adapter = OnlineGazeAdapter(
                expected,
                lambda_factor=getattr(GAZE_CONFIG, "RLS_FORGETTING_FACTOR", 0.995),
                initial_theta_x=mapper._coeffs_x,
                initial_theta_y=mapper._coeffs_y,
            )

        return mapper


class TerminalCalibrator:
    """Collect stable terminal-guided nine-point calibration samples."""
    def __init__(self, screen_width: int, screen_height: int, sample_provider: Callable[[], Optional[GazeObservation]], samples_per_point: int = GAZE_CONFIG.CALIBRATION_SAMPLES_PER_POINT, sample_interval_seconds: float = GAZE_CONFIG.CALIBRATION_SAMPLE_INTERVAL_SECONDS, max_attempts_per_point: int = GAZE_CONFIG.CALIBRATION_MAX_ATTEMPTS_PER_POINT, min_valid_sample_fraction: float = GAZE_CONFIG.CALIBRATION_MIN_VALID_SAMPLE_FRACTION) -> None:
        self.width, self.height = int(screen_width), int(screen_height)
        self.provider = sample_provider
        self.count, self.interval = max(1, int(samples_per_point)), max(0.0, float(sample_interval_seconds))
        self.attempts, self.minimum = max(1, int(max_attempts_per_point)), min(1.0, max(.1, float(min_valid_sample_fraction)))

    def run_calibration(self) -> list[tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]]:
        print("Nine-point calibration: look at the named phone-screen position, then press Enter.")
        results = []
        for label, u, v in _GRID:
            for attempt in range(self.attempts):
                input(f"Look at {label}; press Enter to sample ({attempt + 1}/{self.attempts}): ")
                readings = []
                for _ in range(self.count):
                    reading = self.provider()
                    if reading is not None:
                        readings.append(reading)
                    time.sleep(self.interval)
                if len(readings) >= math.ceil(self.count * self.minimum):
                    average = lambda field: (float(np.mean([float(getattr(r, field)) for r in readings if getattr(r, field) is not None])) if any(getattr(r, field) is not None for r in readings) else None)
                    results.append((float(np.mean([r.pupil_x for r in readings])), float(np.mean([r.pupil_y for r in readings])), u * (self.width - 1), v * (self.height - 1), average("yaw"), average("pitch"), average("roll")))
                    break
                print("Tracking was unstable; try again.")
        if len(results) < 6:
            raise ScreenMapperError("calibration failed: fewer than six stable points")
        return results
