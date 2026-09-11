"""
smooth_scroller.py
==================

Physics-Informed Sub-Pixel Smooth Scrolling Engine (DOPC v9.0 Master).
Replaces discrete step-scrolling with continuous logarithmic velocity scaling,
dynamic inertial friction damping, and sub-pixel accumulation.

Mathematical Model:
1. Micro-Deadzone Filter: Rejects involuntary ocular tremor / microsaccades (|Δy| < deadzone).
2. Logarithmic Velocity Scaling:
   V_raw(Δy) = sign(Δy) * α * ln(1 + β * (|Δy| - deadzone) / (1 - deadzone))
3. Inertial Friction Damping:
   V_t = μ * V_{t-1} + (1 - μ) * V_raw(Δy)  (where μ ∈ [0.85, 0.95])
4. Sub-Pixel Accumulation & Whole Wheel Tick Extraction:
   Accumulator_t = Accumulator_{t-1} + V_t * Δt
   WheelTicks = int(Accumulator_t)
   Accumulator_t -= WheelTicks
"""

from __future__ import annotations

import math
import time
from typing import Tuple


class SubPixelSmoothScroller:
    """
    Physics-informed smooth scrolling engine with logarithmic velocity curves,
    inertial fluid friction damping, and sub-pixel accumulation.
    """

    def __init__(
        self,
        friction: float = 0.90,
        gain: float = 45.0,
        deadzone: float = 0.05,
        beta: float = 3.0,
    ) -> None:
        """
        Initialize the smooth scroller.

        Args:
            friction: Fluid momentum friction coefficient μ ∈ [0.85, 0.95].
            gain: Scaling amplitude α for maximum wheel velocity.
            deadzone: Threshold below which ocular motion is filtered out.
            beta: Logarithmic curvature parameter β.
        """
        self.friction = max(0.0, min(0.999, friction))
        self.gain = gain
        self.deadzone = max(0.0, min(0.5, deadzone))
        self.beta = beta

        # 1D Vertical State
        self.velocity = 0.0
        self.subpixel_accumulator = 0.0

        # 2D Horizontal State
        self.velocity_x = 0.0
        self.subpixel_accumulator_x = 0.0

        self.last_update_time = time.time()

    def process_ocular_displacement(self, normalized_offset: float) -> int:
        """
        Calculates sub-pixel scroll increments and returns integer wheel tick steps for 1D vertical scrolling.

        Args:
            normalized_offset: Signed ocular offset Δy = R_y - R_center ∈ [-1.0, 1.0].

        Returns:
            Whole wheel ticks (positive for scroll down, negative for scroll up).
        """
        now = time.time()
        dt = max(now - self.last_update_time, 0.001)
        self.last_update_time = now

        # 1. Apply Micro-Deadzone Filter
        abs_offset = abs(normalized_offset)
        if abs_offset < self.deadzone:
            target_velocity = 0.0
        else:
            # 2. Logarithmic Velocity Scaling
            effective_offset = (abs_offset - self.deadzone) / max(0.001, 1.0 - self.deadzone)
            direction = 1.0 if normalized_offset > 0 else -1.0
            target_velocity = direction * self.gain * math.log(1.0 + self.beta * min(1.0, effective_offset))

        # 3. Inertial Friction Damping
        self.velocity = (self.friction * self.velocity) + ((1.0 - self.friction) * target_velocity)

        # 4. Sub-Pixel Accumulation
        self.subpixel_accumulator += self.velocity * dt

        # 5. Extract Whole Wheel Ticks
        ticks = int(self.subpixel_accumulator)
        self.subpixel_accumulator -= ticks

        return ticks

    def process_ocular_displacement_2d(
        self, normalized_offset_x: float, normalized_offset_y: float
    ) -> Tuple[int, int]:
        """
        2D Vector Sub-Pixel Scrolling for horizontal and vertical navigation.

        Returns:
            (ticks_x, ticks_y)
        """
        ticks_y = self.process_ocular_displacement(normalized_offset_y)

        # Process X axis using same physics model
        abs_offset_x = abs(normalized_offset_x)
        if abs_offset_x < self.deadzone:
            target_vx = 0.0
        else:
            effective_x = (abs_offset_x - self.deadzone) / max(0.001, 1.0 - self.deadzone)
            dir_x = 1.0 if normalized_offset_x > 0 else -1.0
            target_vx = dir_x * self.gain * math.log(1.0 + self.beta * min(1.0, effective_x))

        now = time.time()
        dt = max(now - self.last_update_time, 0.001)
        self.velocity_x = (self.friction * self.velocity_x) + ((1.0 - self.friction) * target_vx)
        self.subpixel_accumulator_x += self.velocity_x * dt

        ticks_x = int(self.subpixel_accumulator_x)
        self.subpixel_accumulator_x -= ticks_x

        return ticks_x, ticks_y

    def reset(self) -> None:
        """Reset velocity and accumulator to zero."""
        self.velocity = 0.0
        self.velocity_x = 0.0
        self.subpixel_accumulator = 0.0
        self.subpixel_accumulator_x = 0.0
        self.last_update_time = time.time()

    @property
    def current_velocity(self) -> float:
        """Current instantaneous vertical velocity in pixels/sec."""
        return self.velocity

    @property
    def current_velocity_x(self) -> float:
        """Current instantaneous horizontal velocity in pixels/sec."""
        return self.velocity_x
