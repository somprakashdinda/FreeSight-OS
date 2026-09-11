"""
General utilities for the gaze controller.
"""

from __future__ import annotations
from collections import deque
from typing import Optional
import numpy as np

class EMASmoother:
    """
    Exponential moving average with a short median buffer.

    The median buffer rejects isolated frame spikes before the EMA
    receives the value.
    """

    def __init__(
        self,
        alpha: float,
        buffer_size: int = 5,
    ) -> None:

        if not 0.0 < alpha <= 1.0:
            raise ValueError(
                f"alpha must be in (0, 1], got {alpha}"
            )

        if buffer_size < 1:
            raise ValueError(
                f"buffer_size must be >= 1, got {buffer_size}"
            )

        self._alpha = float(alpha)
        self._buffer: Deque[float] = deque(
            maxlen=buffer_size
        )
        self._ema_value: Optional[float] = None

    def update(self, raw_value: float) -> float:
        """
        Add a raw value and return the smoothed value.
        """

        if not np.isfinite(raw_value):
            return self.value

        self._buffer.append(float(raw_value))

        median_value = float(
            np.median(
                np.asarray(
                    self._buffer,
                    dtype=np.float64,
                )
            )
        )

        if self._ema_value is None:
            self._ema_value = median_value
        else:
            self._ema_value = (
                self._alpha * median_value
                + (1.0 - self._alpha) * self._ema_value
            )

        return float(self._ema_value)

    @property
    def value(self) -> float:
        """
        Current smoothed value.

        Returns neutral 0.5 when no value has been received yet.
        """

        if self._ema_value is None:
            return 0.5

        return float(self._ema_value)

    def reset(self) -> None:
        """Clear the smoother history."""

        self._buffer.clear()
        self._ema_value = None
