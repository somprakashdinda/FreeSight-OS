"""
predictive_filter.py
====================

6-State Predictive Kinematic Kalman Filter for eye gaze tracking, latency compensation,
and micro-saccadic jitter suppression.

Derived from the v4.0 Ultra-Low Latency Architecture specification in suggestion-v2.md.

State vector:
    x = [x, y, vx, vy, ax, ay]^T
where:
    (x, y)       = Screen coordinates (pixels)
    (vx, vy)     = Gaze velocity (pixels / sec)
    (ax, ay)     = Gaze acceleration (pixels / sec^2)

Look-ahead prediction:
    pred_x = x + vx * lead_time + 0.5 * ax * lead_time^2
    pred_y = y + vy * lead_time + 0.5 * ay * lead_time^2
"""

from __future__ import annotations

import math
from typing import Tuple, Optional
import numpy as np


class PredictiveGazeUKF:
    """
    6-State Predictive Kalman Filter for Lag Compensation and Saccade Smoothing.
    Compensates for display refresh latency (16.6ms at 60Hz) while eliminating
    ocular micro-saccade jitter during fixations.
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        prediction_ahead_frames: float = 1.0,
        q_base: float = 0.05,
        r_base: float = 2.5,
        saccade_velocity_thresh: float = 250.0,
    ) -> None:
        self.dt = float(dt)
        self.lead_time = float(dt * prediction_ahead_frames)
        self.saccade_thresh = float(saccade_velocity_thresh)
        self._q_base = float(q_base)
        self._r_base = float(r_base)

        # State vector: [x, y, vx, vy, ax, ay]^T
        self.x = np.zeros((6, 1), dtype=np.float64)

        # State covariance matrix P
        self.P = np.eye(6, dtype=np.float64) * 10.0

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.F[0, 4] = 0.5 * (self.dt ** 2)
        self.F[1, 5] = 0.5 * (self.dt ** 2)
        self.F[2, 4] = self.dt
        self.F[3, 5] = self.dt

        # Measurement matrix H (measures x and y position)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process and measurement noise
        self.Q = np.eye(6, dtype=np.float64) * self._q_base
        self.Q[2, 2] = 2.0
        self.Q[3, 3] = 2.0
        self.Q[4, 4] = 10.0
        self.Q[5, 5] = 10.0
        self.R = np.eye(2, dtype=np.float64) * self._r_base

        self._initialized = False

    def reset(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Reset the filter state and covariance matrix."""
        self.x.fill(0.0)
        if x is not None and y is not None:
            self.x[0, 0] = float(x)
            self.x[1, 0] = float(y)
            self._initialized = True
        else:
            self._initialized = False
        self.P = np.eye(6, dtype=np.float64) * 10.0

    @property
    def velocity(self) -> Tuple[float, float]:
        """Current estimated gaze velocity in pixels/second."""
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def speed(self) -> float:
        """Current estimated gaze speed in pixels/second."""
        vx, vy = self.velocity
        return math.hypot(vx, vy)

    def update_and_predict(self, measured_x: float, measured_y: float) -> Tuple[float, float]:
        """
        Perform measurement update and compute ahead-of-time predicted position.
        
        Adaptive noise scaling:
        - When velocity exceeds saccade_thresh, R decreases and Q increases to track
          rapid gaze jumps with zero lag.
        - When velocity is low (fixation), R increases and Q decreases to filter out
          micro-saccade tremor.
        """
        if not self._initialized:
            self.reset(measured_x, measured_y)
            return float(measured_x), float(measured_y)

        z = np.array([[float(measured_x)], [float(measured_y)]], dtype=np.float64)

        # Adaptive Noise Covariance Gating
        current_speed = self.speed
        if current_speed > self.saccade_thresh:
            # Saccade active: prioritize new measurement, lower measurement noise
            saccade_ratio = min(current_speed / self.saccade_thresh, 5.0)
            R_eff = self.R / saccade_ratio
            Q_eff = self.Q * saccade_ratio
        else:
            # Fixation / smooth pursuit: enhance jitter suppression
            R_eff = self.R * 1.5
            Q_eff = self.Q

        # 1. Predict Step
        x_prior = self.F @ self.x
        P_prior = self.F @ self.P @ self.F.T + Q_eff

        # 2. Update Step
        y_residual = z - (self.H @ x_prior)
        S = self.H @ P_prior @ self.H.T + R_eff
        try:
            K = P_prior @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P_prior @ self.H.T @ np.linalg.pinv(S)

        self.x = x_prior + (K @ y_residual)
        self.P = (np.eye(6, dtype=np.float64) - (K @ self.H)) @ P_prior

        # 3. Look-Ahead Prediction (Compensates for display refresh latency)
        pred_x = (
            self.x[0, 0]
            + (self.x[2, 0] * self.lead_time)
            + (0.5 * self.x[4, 0] * (self.lead_time ** 2))
        )
        pred_y = (
            self.x[1, 0]
            + (self.x[3, 0] * self.lead_time)
            + (0.5 * self.x[5, 0] * (self.lead_time ** 2))
        )

        return float(pred_x), float(pred_y)
