"""
screen_geometry.py
==================

Multi-Display Virtual Screen Geometry & Per-Monitor V2 DPI Awareness.

Implements Module C and Section 5.1 from suggestion-v2.md:
- Direct Win32 SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
- Virtual screen bounds querying (SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN)
- Affine homogeneous coordinate mapping for multi-monitor desktop traversals.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import platform
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Win32 System Metrics Constants
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Per-Monitor V2 DPI Awareness Context: -4
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


def initialize_dpi_awareness() -> bool:
    """
    Set Per-Monitor V2 DPI awareness for the process to prevent coordinate
    drift and blurry scaling across mixed-DPI multi-monitor environments.
    """
    if not _IS_WINDOWS:
        return False
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # SetProcessDpiAwarenessContext(-4)
            result = user32.SetProcessDpiAwarenessContext(
                ctypes.c_void_p(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
            )
            return bool(result)
        elif hasattr(ctypes.windll.shcore, "SetProcessDpiAwareness"):
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            return True
    except Exception as exc:
        logger.debug("Could not set DPI awareness context: %s", exc)
    return False


# Automatically attempt initialization on module import
_DPI_AWARENESS_SET = initialize_dpi_awareness()


@dataclass(frozen=True)
class VirtualDesktopBounds:
    """Bounding box of the virtual desktop spanning all active monitors."""
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class ScreenGeometryManager:
    """
    Manages multi-monitor screen bounds and homogeneous affine coordinate mappings.
    """

    def __init__(self, fallback_width: int = 1920, fallback_height: int = 1080) -> None:
        self.fallback_w = int(fallback_width)
        self.fallback_h = int(fallback_height)
        self._user32 = ctypes.windll.user32 if _IS_WINDOWS else None
        self.bounds = self.query_virtual_desktop_bounds()

    def query_virtual_desktop_bounds(self) -> VirtualDesktopBounds:
        """Fetch current virtual desktop dimensions across all connected monitors."""
        if self._user32:
            try:
                vx = int(self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
                vy = int(self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
                vw = int(self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
                vh = int(self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
                if vw > 0 and vh > 0:
                    return VirtualDesktopBounds(vx, vy, vw, vh)
            except Exception as exc:
                logger.warning("Error querying Win32 virtual screen metrics: %s", exc)

        return VirtualDesktopBounds(0, 0, self.fallback_w, self.fallback_h)

    def refresh(self) -> VirtualDesktopBounds:
        """Re-query virtual desktop bounds (e.g., after monitor plug/unplug)."""
        self.bounds = self.query_virtual_desktop_bounds()
        return self.bounds

    def normalize_to_sendinput(self, screen_x: float, screen_y: float) -> Tuple[int, int]:
        """
        Convert pixel coordinates in virtual desktop space into normalized
        Win32 SendInput absolute coordinates in [0, 65535].
        """
        b = self.bounds
        clamped_x = max(b.left, min(screen_x, b.right - 1))
        clamped_y = max(b.top, min(screen_y, b.bottom - 1))

        norm_x = int(((clamped_x - b.left) * 65535) / max(b.width, 1))
        norm_y = int(((clamped_y - b.top) * 65535) / max(b.height, 1))
        return norm_x, norm_y

    def map_normalized_ratio_to_screen(
        self,
        ratio_x: float,
        ratio_y: float,
        target_monitor_bounds: Optional[VirtualDesktopBounds] = None,
    ) -> Tuple[int, int]:
        """Map [0.0, 1.0] normalized gaze ratios to screen pixel coordinates."""
        target = target_monitor_bounds or self.bounds
        x = int(target.left + ratio_x * (target.width - 1))
        y = int(target.top + ratio_y * (target.height - 1))
        return x, y
