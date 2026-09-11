"""
Blink detection state machine.

This module contains only blink-detection logic and has no OpenCV dependency.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlinkResult:
    """Result produced after processing one EAR sample."""

    eye_closed: bool
    blink_completed: bool
    double_blink: bool


class BlinkDetector:
    """
    Lightweight EAR-based blink detector.

    Detection behavior:
    - EAR below threshold means the eyes are closed.
    - The EAR must remain below the threshold for the configured
      number of consecutive frames before closure is accepted.
    - When the eyes reopen after an accepted closure, a blink is completed.
    - A second completed blink within the configured double-blink
      delay is reported as a double blink.
    - A refractory period prevents repeated detections from being
      generated too quickly.
    - Invalid/missing EAR values are ignored safely.
    """

    def __init__(
        self,
        ear_threshold: float = 0.20,
        min_closed_frames: int = 2,
        double_blink_max_delay: float = 0.50,
        refractory_period: float = 0.15,
    ) -> None:
        """
        Args:
            ear_threshold:
                EAR value below which the eye is considered closed.

            min_closed_frames:
                Number of consecutive closed frames required before
                an eye closure is accepted.

            double_blink_max_delay:
                Maximum time in seconds between two completed blinks
                for them to count as a double blink.

            refractory_period:
                Minimum time in seconds between accepted blink events.
        """
        if ear_threshold <= 0:
            raise ValueError("ear_threshold must be greater than 0")

        if min_closed_frames < 1:
            raise ValueError("min_closed_frames must be at least 1")

        if double_blink_max_delay < 0:
            raise ValueError("double_blink_max_delay cannot be negative")

        if refractory_period < 0:
            raise ValueError("refractory_period cannot be negative")

        self.ear_threshold = float(ear_threshold)
        self.min_closed_frames = int(min_closed_frames)
        self.double_blink_max_delay = float(double_blink_max_delay)
        self.refractory_period = float(refractory_period)

        # Number of consecutive frames currently below the EAR threshold.
        self._closed_frames = 0

        # True once a valid eye closure has been accepted.
        self._eye_is_closed = False

        # Time of the most recently completed blink.
        self._last_blink_time: Optional[float] = None

        # Time of the most recently accepted blink event.
        self._last_event_time: Optional[float] = None

    def reset(self) -> None:
        """Reset the detector to its initial state."""
        self._closed_frames = 0
        self._eye_is_closed = False
        self._last_blink_time = None
        self._last_event_time = None

    @staticmethod
    def _valid_ear(ear: Optional[float]) -> bool:
        """Return True when EAR is a finite, usable number."""
        if ear is None:
            return False

        try:
            value = float(ear)
        except (TypeError, ValueError):
            return False

        return math.isfinite(value) and value >= 0.0

    def update(self, ear: Optional[float]) -> BlinkResult:
        """
        Process one EAR sample.

        Args:
            ear:
                Current Eye Aspect Ratio. None, NaN, infinity, or
                otherwise invalid values are ignored.

        Returns:
            BlinkResult containing:
                - eye_closed: current accepted eye-closure state
                - blink_completed: True when a blink has just completed
                - double_blink: True when a double blink has just completed
        """
        now = time.monotonic()

        # Invalid/missing EAR values do not modify the state machine.
        if not self._valid_ear(ear):
            return BlinkResult(
                eye_closed=self._eye_is_closed,
                blink_completed=False,
                double_blink=False,
            )

        ear_value = float(ear)

        # ----------------------------------------------------------
        # Eye is currently below the closure threshold.
        # ----------------------------------------------------------
        if ear_value < self.ear_threshold:
            self._closed_frames += 1

            if (
                not self._eye_is_closed
                and self._closed_frames >= self.min_closed_frames
            ):
                self._eye_is_closed = True

            return BlinkResult(
                eye_closed=self._eye_is_closed,
                blink_completed=False,
                double_blink=False,
            )

        # ----------------------------------------------------------
        # Eye has reopened.
        # ----------------------------------------------------------
        blink_completed = False
        double_blink = False

        if self._eye_is_closed:
            self._eye_is_closed = False
            self._closed_frames = 0

            # Refractory protection.
            can_accept_blink = (
                self._last_event_time is None
                or now - self._last_event_time >= self.refractory_period
            )

            if can_accept_blink:
                blink_completed = True
                self._last_event_time = now

                # Check whether this blink follows the previous
                # completed blink quickly enough to form a double blink.
                if (
                    self._last_blink_time is not None
                    and now - self._last_blink_time
                    <= self.double_blink_max_delay
                ):
                    double_blink = True
                    self._last_blink_time = None
                else:
                    self._last_blink_time = now

        else:
            self._closed_frames = 0

        return BlinkResult(
            eye_closed=self._eye_is_closed,
            blink_completed=blink_completed,
            double_blink=double_blink,
        )