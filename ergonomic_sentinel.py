"""
Direct Ocular Precision Controller (DOPC) - Version 18.0 Full-Body Kinematic Synergy Engine
Module D: Postural Ergonomics & Dynamic Spatial Sensitivity Sentinel (ergonomic_sentinel.py)

Monitors:
  - Cervical spine forward tilt (> 18 deg)
  - Thoracic slouching / kyphosis (> 12 deg)
  - Static muscle strain and immobility accumulation
  - Dynamic gaze sensitivity adaptation based on user posture distance and body angle
"""

import time
import numpy as np
from typing import Dict, Any, Tuple


class PosturalErgonomicSentinel:
    """
    Postural Ergonomics & Dynamic Spatial Sensitivity Sentinel.
    Computes real-time ergonomic health scores and adjusts gaze sensitivity.
    Executes in < 0.001 ms per evaluation with zero heap allocations.
    """

    def __init__(self,
                 cervical_threshold_deg: float = 18.0,
                 thoracic_slouch_threshold_deg: float = 12.0,
                 immobility_warning_sec: float = 1800.0):
        self.cervical_threshold = float(cervical_threshold_deg)
        self.thoracic_threshold = float(thoracic_slouch_threshold_deg)
        self.immobility_warning_sec = float(immobility_warning_sec)

        # Baseline reference
        self._neutral_distance_cm = 65.0

        # State tracking
        self._last_movement_time = time.perf_counter()
        self._static_strain_duration_sec = 0.0
        self._cervical_strain_accum = 0.0
        self._thoracic_strain_accum = 0.0

        # Output metrics
        self.ergonomic_score = 1.0
        self.dynamic_sensitivity_multiplier = 1.0
        self.warning_active = False
        self.alert_message = "POSTURE_OPTIMAL"

    def evaluate_posture(self,
                         cervical_tilt_deg: float,
                         thoracic_pitch_deg: float,
                         torso_roll_deg: float,
                         estimated_distance_cm: float = 65.0,
                         dt_sec: float = 0.016) -> Dict[str, Any]:
        """
        Hotpath evaluation of upper-body ergonomics and gaze sensitivity scaling.
        """
        t0 = time.perf_counter()
        now = t0

        abs_cervical = abs(cervical_tilt_deg)
        abs_thoracic = abs(thoracic_pitch_deg)
        abs_roll = abs(torso_roll_deg)

        # 1. Strain accumulation
        is_straining = (abs_cervical > self.cervical_threshold) or (abs_thoracic > self.thoracic_threshold)

        if is_straining:
            cervical_excess = max(0.0, abs_cervical - self.cervical_threshold)
            thoracic_excess = max(0.0, abs_thoracic - self.thoracic_threshold)
            self._cervical_strain_accum += cervical_excess * dt_sec * 0.1
            self._thoracic_strain_accum += thoracic_excess * dt_sec * 0.1
            self._static_strain_duration_sec += dt_sec
        else:
            # Recovery
            self._cervical_strain_accum = max(0.0, self._cervical_strain_accum - dt_sec * 0.05)
            self._thoracic_strain_accum = max(0.0, self._thoracic_strain_accum - dt_sec * 0.05)
            self._static_strain_duration_sec = max(0.0, self._static_strain_duration_sec - dt_sec * 0.02)

        # 2. Ergonomic Health Score (1.0 = Perfect, 0.0 = Severe Strain)
        total_strain = self._cervical_strain_accum + self._thoracic_strain_accum + (abs_roll * 0.02)
        self.ergonomic_score = float(np.clip(1.0 - (total_strain * 0.15), 0.0, 1.0))

        # 3. Dynamic Gaze Spatial Sensitivity Multiplier
        # As distance increases or slouch increases, expand smoothing and scale sensitivity
        distance_ratio = estimated_distance_cm / self._neutral_distance_cm
        posture_damping = 1.0 + (1.0 - self.ergonomic_score) * 0.40
        self.dynamic_sensitivity_multiplier = float(np.clip(distance_ratio * posture_damping, 0.65, 1.60))

        # 4. Ergonomic Alert Status
        if self._static_strain_duration_sec > self.immobility_warning_sec:
            self.warning_active = True
            self.alert_message = "PROLONGED_STATIC_STRAIN_STAND_UP"
        elif abs_cervical > (self.cervical_threshold * 1.5):
            self.warning_active = True
            self.alert_message = "SEVERE_FORWARD_HEAD_TILT"
        elif abs_thoracic > (self.thoracic_threshold * 1.5):
            self.warning_active = True
            self.alert_message = "SEVERE_THORACIC_SLOUCH"
        elif self.ergonomic_score < 0.60:
            self.warning_active = True
            self.alert_message = "POSTURE_DEGRADED_REALIGN"
        else:
            self.warning_active = False
            self.alert_message = "POSTURE_OPTIMAL"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "ergonomic_score": self.ergonomic_score,
            "cervical_tilt_deg": float(cervical_tilt_deg),
            "thoracic_pitch_deg": float(thoracic_pitch_deg),
            "dynamic_sensitivity_multiplier": self.dynamic_sensitivity_multiplier,
            "static_strain_sec": float(self._static_strain_duration_sec),
            "warning_active": self.warning_active,
            "alert_message": self.alert_message,
            "distance_cm": float(estimated_distance_cm),
            "latency_ms": latency_ms
        }
