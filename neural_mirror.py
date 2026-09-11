"""
neural_mirror.py
================

Module C: Peripheral Sub-Visual Neural Mirroring & Haptic Focus for FreeSight-OS (DOPC v13.0).
Projects high-frequency sub-visual luminance micro-pulses (85Hz, 4% depth) in the user's extreme
visual periphery to confirm dwell fixation intent without drawing foveal attention away from
the primary task.

Specification from suggestion-v11.md (Section 3, Module C).
"""

from __future__ import annotations

import math
import time
from typing import Dict, Any, Tuple


class PeripheralNeuralMirror:
    """Peripheral Sub-Visual Neural Mirroring Engine."""

    def __init__(self, pulse_frequency_hz: float = 85.0, modulation_depth: float = 0.04):
        """
        Initialize the Peripheral Neural Mirror.

        Args:
            pulse_frequency_hz: High-frequency sub-visual carrier modulation (default: 85.0 Hz).
            modulation_depth: Relative amplitude of peripheral luminance pulse (default: 0.04 = 4%).
        """
        self.pulse_frequency_hz = pulse_frequency_hz
        self.modulation_depth = modulation_depth
        self.start_time = time.perf_counter()
        self.pulse_count = 0
        self.last_modulation = 0.0

    def calculate_peripheral_luminance(
        self, dwell_progress: float, current_time: float | None = None
    ) -> Dict[str, Any]:
        """
        Calculates the instantaneous peripheral luminance halo for sub-visual confirmation.

        Args:
            dwell_progress: Current fixation dwell progress from 0.0 to 1.0.
            current_time: Optional high-resolution timestamp.

        Returns:
            Dict containing edge luminance values, carrier wave phase, and confirmation signal.
        """
        t = current_time if current_time is not None else time.perf_counter()
        elapsed = t - self.start_time

        # Carrier wave at 85 Hz (high frequency, imperceptible flicker)
        carrier_phase = 2.0 * math.pi * self.pulse_frequency_hz * elapsed
        carrier_val = math.sin(carrier_phase)

        # Amplitude modulation proportional to dwell progress
        # Dwell progress ramps the intensity subtly in the extreme periphery
        pulse_amplitude = self.modulation_depth * math.pow(dwell_progress, 1.5)
        luminance_delta = pulse_amplitude * (0.5 + 0.5 * carrier_val)

        self.last_modulation = luminance_delta
        if dwell_progress >= 0.95:
            self.pulse_count += 1

        return {
            "active": bool(dwell_progress > 0.1),
            "dwell_progress": round(dwell_progress, 3),
            "carrier_hz": self.pulse_frequency_hz,
            "luminance_delta": round(luminance_delta, 5),
            "edge_halo_alpha": round(float(pulse_amplitude * 2.5), 4),
            "intent_confirmed": bool(dwell_progress >= 1.0),
            "pulses_emitted": self.pulse_count,
        }
