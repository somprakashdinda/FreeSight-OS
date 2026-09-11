"""
test_os_interop.py
==================

Unit tests for Win32 SendInput coordinates normalization,
ScreenGeometryManager multi-monitor metrics, and camera connection state machine.
"""

import unittest
from os_interop import NativeWin32Input, ConnectionState
from screen_geometry import ScreenGeometryManager, VirtualDesktopBounds


class TestOSInterop(unittest.TestCase):

    def test_screen_geometry_normalization(self):
        """Verify coordinate normalization maps properly into [0, 65535]."""
        mgr = ScreenGeometryManager(fallback_width=1920, fallback_height=1080)
        # Override bounds with known virtual desktop
        object.__setattr__(mgr, "bounds", VirtualDesktopBounds(left=0, top=0, width=1920, height=1080))

        # Top-left corner
        nx, ny = mgr.normalize_to_sendinput(0, 0)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

        # Bottom-right corner
        nx, ny = mgr.normalize_to_sendinput(1919, 1079)
        self.assertAlmostEqual(nx, 65535, delta=100)
        self.assertAlmostEqual(ny, 65535, delta=100)

        # Center
        nx, ny = mgr.normalize_to_sendinput(960, 540)
        self.assertAlmostEqual(nx, 32768, delta=100)
        self.assertAlmostEqual(ny, 32768, delta=100)

    def test_multi_monitor_virtual_desktop_offset(self):
        """Verify multi-monitor setup where primary monitor has negative virtual left."""
        mgr = ScreenGeometryManager(fallback_width=3840, fallback_height=1080)
        # 2x 1080p monitors: left monitor is at x in [-1920, 0), right is at [0, 1920)
        object.__setattr__(mgr, "bounds", VirtualDesktopBounds(left=-1920, top=0, width=3840, height=1080))

        # Left monitor far edge (-1920, 0)
        nx, ny = mgr.normalize_to_sendinput(-1920, 0)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

        # Monitor boundary at x=0
        nx, ny = mgr.normalize_to_sendinput(0, 540)
        self.assertAlmostEqual(nx, 32768, delta=100)

    def test_native_win32_input_coordinate_bounds(self):
        """Verify NativeWin32Input converts coordinates safely within bounds."""
        controller = NativeWin32Input()
        nx, ny = controller._to_normalized_coords(controller.vx, controller.vy)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

        # Extreme values clamped to [0, 65535]
        nx_neg, ny_neg = controller._to_normalized_coords(-99999, -99999)
        self.assertEqual(nx_neg, 0)
        self.assertEqual(ny_neg, 0)

        nx_huge, ny_huge = controller._to_normalized_coords(99999, 99999)
        self.assertEqual(nx_huge, 65535)
        self.assertEqual(ny_huge, 65535)

    def test_connection_state_enum(self):
        """Verify camera ConnectionState enum values exist."""
        self.assertEqual(ConnectionState.CONNECTED.name, "CONNECTED")
        self.assertEqual(ConnectionState.RECONNECTING.name, "RECONNECTING")
        self.assertEqual(ConnectionState.FAILED.name, "FAILED")
        self.assertEqual(ConnectionState.DISCONNECTED.name, "DISCONNECTED")


if __name__ == "__main__":
    unittest.main()
