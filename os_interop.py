"""
os_interop.py
==============

Native Host OS interface and hardware resilience layer for the eye-tracking controller.

Enhancements (v1.1 - v3.0):
1. Native Windows SendInput dispatch via ctypes (<1ms latency, UAC bypass).
2. Virtual Desktop multi-monitor coordinate transformation.
3. Automated Camera Reconnection State Machine with exponential backoff.
4. Graceful PyAutoGUI fallback for cross-platform support.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from enum import Enum, auto
import platform
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from config import HOST_OS_CONFIG, CV_CONFIG
from screen_geometry import initialize_dpi_awareness
from smooth_scroller import SubPixelSmoothScroller
from lean_scroller import TorsoLeanScroller

# Ensure Per-Monitor V2 DPI awareness is initialized
initialize_dpi_awareness()

# Optional PyAutoGUI fallback
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.001
    _PYAUTOGUI_AVAILABLE = True
except ImportError:
    _PYAUTOGUI_AVAILABLE = False


# ===========================================================================
# Win32 SendInput Structures and Constants
# ===========================================================================

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
    MOUSEEVENTF_WHEEL = 0x0800
    MOUSEEVENTF_HWHEEL = 0x01000
    MOUSEEVENTF_VIRTUALDESK = 0x4000
    MOUSEEVENTF_ABSOLUTE = 0x8000

    KEYEVENTF_KEYUP = 0x0002
    VK_LEFT = 0x25
    VK_UP = 0x26
    VK_RIGHT = 0x27
    VK_DOWN = 0x28
    WHEEL_DELTA = 120

    SM_XVIRTUALSCREEN = 76
    SM_YVIRTUALSCREEN = 77
    SM_CXVIRTUALSCREEN = 78
    SM_CYVIRTUALSCREEN = 79

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("_union",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("_union", _INPUT_UNION),
        ]


class NativeWin32Input:
    """
    Direct Win32 SendInput controller for sub-millisecond dispatch and
    multi-monitor virtual screen coordinate normalization.
    """

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32 if _IS_WINDOWS else None
        self._refresh_virtual_desktop_bounds()

    def _refresh_virtual_desktop_bounds(self) -> None:
        if self._user32:
            self.vx = self._user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
            self.vy = self._user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
            self.vw = max(1, self._user32.GetSystemMetrics(SM_CXVIRTUALSCREEN))
            self.vh = max(1, self._user32.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        else:
            self.vx, self.vy = 0, 0
            self.vw, self.vh = HOST_OS_CONFIG.screen_width, HOST_OS_CONFIG.screen_height

    def _to_normalized_coords(self, x: int, y: int) -> Tuple[int, int]:
        """Normalize absolute coordinates to 0..65535 across virtual desktop."""
        norm_x = int(((x - self.vx) * 65535) / self.vw)
        norm_y = int(((y - self.vy) * 65535) / self.vh)
        return max(0, min(65535, norm_x)), max(0, min(65535, norm_y))

    def move_cursor(self, x: int, y: int) -> bool:
        if not self._user32:
            return False
        norm_x, norm_y = self._to_normalized_coords(x, y)
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(
            dx=norm_x,
            dy=norm_y,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
            time=0,
            dwExtraInfo=None,
        )
        return self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1

    def click(self, x: int, y: int) -> bool:
        if not self._user32:
            return False
        norm_x, norm_y = self._to_normalized_coords(x, y)
        inputs = (INPUT * 2)()

        # Mouse down
        inputs[0].type = INPUT_MOUSE
        inputs[0].mi = MOUSEINPUT(
            dx=norm_x,
            dy=norm_y,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_LEFTDOWN,
            time=0,
            dwExtraInfo=None,
        )

        # Mouse up
        inputs[1].type = INPUT_MOUSE
        inputs[1].mi = MOUSEINPUT(
            dx=norm_x,
            dy=norm_y,
            mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_LEFTUP,
            time=0,
            dwExtraInfo=None,
        )

        return self._user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) == 2

    def scroll(self, amount: int) -> bool:
        if not self._user32:
            return False
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=amount,
            dwFlags=MOUSEEVENTF_WHEEL,
            time=0,
            dwExtraInfo=None,
        )
        return self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1

    def scroll_horizontal(self, amount: int) -> bool:
        if not self._user32:
            return False
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=amount,
            dwFlags=MOUSEEVENTF_HWHEEL,
            time=0,
            dwExtraInfo=None,
        )
        return self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1

    def right_click(self, x: int, y: int) -> bool:
        if not self._user32:
            return False
        norm_x, norm_y = self._to_normalized_coords(x, y)
        inputs = (INPUT * 2)()
        inputs[0].type = INPUT_MOUSE
        inputs[0].mi = MOUSEINPUT(
            dx=norm_x, dy=norm_y, mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_RIGHTDOWN,
            time=0, dwExtraInfo=None,
        )
        inputs[1].type = INPUT_MOUSE
        inputs[1].mi = MOUSEINPUT(
            dx=norm_x, dy=norm_y, mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_RIGHTUP,
            time=0, dwExtraInfo=None,
        )
        return self._user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) == 2

    def middle_click(self, x: int, y: int) -> bool:
        if not self._user32:
            return False
        norm_x, norm_y = self._to_normalized_coords(x, y)
        inputs = (INPUT * 2)()
        inputs[0].type = INPUT_MOUSE
        inputs[0].mi = MOUSEINPUT(
            dx=norm_x, dy=norm_y, mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_MIDDLEDOWN,
            time=0, dwExtraInfo=None,
        )
        inputs[1].type = INPUT_MOUSE
        inputs[1].mi = MOUSEINPUT(
            dx=norm_x, dy=norm_y, mouseData=0,
            dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_MIDDLEUP,
            time=0, dwExtraInfo=None,
        )
        return self._user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) == 2

    def press_key(self, vk_code: int) -> bool:
        if not self._user32:
            return False
        inputs = (INPUT * 2)()

        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0].ki = KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)

        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1].ki = KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)

        return self._user32.SendInput(2, inputs, ctypes.sizeof(INPUT)) == 2

    def move_mouse_absolute(self, x: int, y: int) -> bool:
        """Alias for move_cursor to support kernel driver fallback."""
        return self.move_cursor(x, y)

    def send_mouse_event(self, dw_flags: int, mouse_data: int = 0) -> bool:
        """Dispatches an explicit mouse event with custom flags."""
        if not self._user32:
            return False
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=mouse_data,
            dwFlags=dw_flags,
            time=0,
            dwExtraInfo=None,
        )
        return self._user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) == 1



# ===========================================================================
# Camera Reconnection State Machine
# ===========================================================================

class ConnectionState(Enum):
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()
    DISCONNECTED = auto()


class WebcamCapture:
    """
    Captures frames from system webcam with automated fault-tolerant
    reconnection state machine and exponential backoff recovery.
    """

    def __init__(
        self,
        camera_index: int = HOST_OS_CONFIG.camera_index,
        max_stale_seconds: float = 0.5,
        reconnect_max_backoff: float = 2.0,
    ) -> None:
        self._camera_index = camera_index
        self._max_stale_seconds = max_stale_seconds
        self._reconnect_max_backoff = reconnect_max_backoff
        self._backoff = 0.1
        self._cap: Optional[cv2.VideoCapture] = None
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self._last_successful_read_time = 0.0
        self._start_stream()

    @property
    def state(self) -> ConnectionState:
        return self._state

    def _start_stream(self) -> bool:
        """Initialize or re-initialize the webcam capture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if _IS_WINDOWS:
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._camera_index)
        else:
            self._cap = cv2.VideoCapture(self._camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CV_CONFIG.STREAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CV_CONFIG.STREAM_HEIGHT)

        if self._cap.isOpened():
            self._state = ConnectionState.CONNECTED
            self._last_successful_read_time = time.monotonic()
            self._backoff = 0.1
            return True
        else:
            self._state = ConnectionState.RECONNECTING
            return False

    def _attempt_reconnect(self) -> None:
        """Execute one exponential backoff reconnection step."""
        self._state = ConnectionState.RECONNECTING
        time.sleep(self._backoff)
        success = self._start_stream()
        if not success:
            self._backoff = min(self._reconnect_max_backoff, self._backoff * 1.5)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Grab latest frame from webcam. Automatically triggers recovery if
        stream is stalled or dropped.
        """
        now = time.monotonic()

        if self._cap is None or not self._cap.isOpened():
            self._attempt_reconnect()
            return None

        ret, frame = self._cap.read()
        if ret and frame is not None and frame.size > 0:
            self._last_successful_read_time = now
            self._state = ConnectionState.CONNECTED
            return frame

        # Read failed or stalled
        if now - self._last_successful_read_time > self._max_stale_seconds:
            self._attempt_reconnect()

        return None

    def close(self) -> None:
        """Release the webcam hardware cleanly."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._state = ConnectionState.DISCONNECTED

    @property
    def is_connected(self) -> bool:
        """Check if webcam is currently open and healthy."""
        return self._state == ConnectionState.CONNECTED and self._cap is not None and self._cap.isOpened()


# ===========================================================================
# Host OS Controller (High-Performance Native Dispatcher)
# ===========================================================================

class OSController:
    """
    Executes native mouse and keyboard commands on the Host OS.
    Integrates Ring-0 KMDF virtual input injection with sub-millisecond
    Win32 SendInput and PyAutoGUI fallback.
    """

    def __init__(self) -> None:
        self._native: Optional[NativeWin32Input] = NativeWin32Input() if _IS_WINDOWS else None
        self._kernel_driver = None
        try:
            from kernel_input_driver import KernelModeInputDriver
            self._kernel_driver = KernelModeInputDriver()
        except Exception:
            self._kernel_driver = None

        # v9.0 Sub-Pixel Smooth Scrolling Engine
        self._smooth_scroller = SubPixelSmoothScroller(friction=0.90, gain=45.0, deadzone=0.05)
        # v14.0 Torso Lean Kinetic Scrolling & Panning Engine
        self._lean_scroller = TorsoLeanScroller(deadzone_deg=2.0, pitch_gain=18.0, roll_gain=16.0, friction=0.92)

        # Prevent screen timeout / sleep while FreeSight-OS is active
        keep_display_active(True)
        configure_system_power_screen_awake(True)

    @property
    def is_kernel_driver_active(self) -> bool:
        """Returns True if input is dispatched via Ring-0 KMDF driver."""
        return self._kernel_driver is not None and self._kernel_driver.is_kernel_mode_active()

    def inject_tap(self, x: int, y: int) -> str:
        """Simulate a mouse click at the given screen coordinates."""
        clamped_x = max(5, min(x, HOST_OS_CONFIG.screen_width - 5))
        clamped_y = max(5, min(y, HOST_OS_CONFIG.screen_height - 5))

        if self._kernel_driver and self._kernel_driver.is_kernel_mode_active():
            if self._kernel_driver.inject_mouse_click(clamped_x, clamped_y):
                return f"OSController: KMDF kernel click at ({clamped_x}, {clamped_y})"

        if self._native and self._native.click(clamped_x, clamped_y):
            return f"OSController: native click at ({clamped_x}, {clamped_y})"


        if _PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.click(clamped_x, clamped_y)
                return f"OSController: pyautogui click at ({clamped_x}, {clamped_y})"
            except Exception as exc:
                return f"OSController: click failed ({exc})"

        return "OSController: no input backend available"

    def inject_swipe(self, direction: str, magnitude: float = 1.0) -> str:
        """
        Simulate a directional action (wheel scroll or arrow key press).
        """
        dir_key = str(direction).strip().upper()
        amount = int(HOST_OS_CONFIG.scroll_sensitivity * magnitude)

        if self._native:
            if dir_key == "UP":
                wheel_units = int(WHEEL_DELTA * magnitude)
                if self._native.scroll(wheel_units):
                    return f"OSController: native scrolled UP (mag: {magnitude:.2f})"
            elif dir_key == "DOWN":
                wheel_units = -int(WHEEL_DELTA * magnitude)
                if self._native.scroll(wheel_units):
                    return f"OSController: native scrolled DOWN (mag: {magnitude:.2f})"
            elif dir_key == "LEFT":
                if self._native.press_key(VK_LEFT):
                    return "OSController: native pressed LEFT"
            elif dir_key == "RIGHT":
                if self._native.press_key(VK_RIGHT):
                    return "OSController: native pressed RIGHT"

        if _PYAUTOGUI_AVAILABLE:
            try:
                if dir_key == "UP":
                    pyautogui.scroll(amount)
                    return f"OSController: scrolled UP (mag: {magnitude:.2f})"
                elif dir_key == "DOWN":
                    pyautogui.scroll(-amount)
                    return f"OSController: scrolled DOWN (mag: {magnitude:.2f})"
                elif dir_key == "LEFT":
                    pyautogui.press('left')
                    return "OSController: pressed LEFT"
                elif dir_key == "RIGHT":
                    pyautogui.press('right')
                    return "OSController: pressed RIGHT"
            except Exception as exc:
                return f"OSController: swipe failed ({exc})"

        return f"OSController: direction {dir_key} dispatched"

    def scroll_smoothly(self, normalized_offset: float) -> int:
        """
        Calculates sub-pixel scroll velocity with logarithmic acceleration and friction damping,
        dispatching native Win32 wheel events when integer ticks accumulate.
        """
        ticks = self._smooth_scroller.process_ocular_displacement(normalized_offset)
        if ticks != 0 and self._native:
            self._native.scroll(ticks * WHEEL_DELTA)
        return ticks

    def scroll_lean(self, pitch_deg: float, roll_deg: float, dt: float = 0.0333) -> Tuple[int, int]:
        """
        Processes torso pitch and roll for 2D continuous sub-pixel scrolling and panning (v14.0).
        """
        res = self._lean_scroller.process_lean(pitch_deg, roll_deg, dt)
        ticks_y = res.get("ticks_y", 0)
        ticks_x = res.get("ticks_x", 0)
        if ticks_y != 0 and self._native:
            # Positive pitch = lean forward = scroll down
            self._native.scroll(-ticks_y * WHEEL_DELTA)
        if ticks_x != 0 and self._native:
            self._native.scroll_horizontal(ticks_x * WHEEL_DELTA)
        return ticks_y, ticks_x

    def inject_secondary_action(self, action_type: str, x: int, y: int) -> str:
        """
        Executes secondary gesture actions (Right Click, Middle Click, Drag).
        """
        clamped_x = max(5, min(x, HOST_OS_CONFIG.screen_width - 5))
        clamped_y = max(5, min(y, HOST_OS_CONFIG.screen_height - 5))

        if action_type == "RIGHT_CLICK":
            if self._native and self._native.right_click(clamped_x, clamped_y):
                return f"OSController: native right-click at ({clamped_x}, {clamped_y})"
            if _PYAUTOGUI_AVAILABLE:
                pyautogui.rightClick(clamped_x, clamped_y)
                return f"OSController: pyautogui right-click at ({clamped_x}, {clamped_y})"

        elif action_type == "MIDDLE_CLICK":
            if self._native and self._native.middle_click(clamped_x, clamped_y):
                return f"OSController: native middle-click at ({clamped_x}, {clamped_y})"
            if _PYAUTOGUI_AVAILABLE:
                pyautogui.middleClick(clamped_x, clamped_y)
                return f"OSController: pyautogui middle-click at ({clamped_x}, {clamped_y})"

        elif action_type == "DRAG_TOGGLE":
            if _PYAUTOGUI_AVAILABLE:
                pyautogui.mouseDown(clamped_x, clamped_y)
                return f"OSController: mouseDown drag at ({clamped_x}, {clamped_y})"

        elif action_type == "DRAG_RELEASE":
            if _PYAUTOGUI_AVAILABLE:
                pyautogui.mouseUp(clamped_x, clamped_y)
                return f"OSController: mouseUp drag at ({clamped_x}, {clamped_y})"

        return f"OSController: secondary action {action_type} dispatched"

    def move_cursor(self, x: int, y: int) -> None:
        """Move the OS mouse cursor with virtual-desktop multi-monitor awareness."""
        clamped_x = max(5, min(x, HOST_OS_CONFIG.screen_width - 5))
        clamped_y = max(5, min(y, HOST_OS_CONFIG.screen_height - 5))

        if self._kernel_driver and self._kernel_driver.is_kernel_mode_active():
            if self._kernel_driver.inject_absolute_mouse_event(clamped_x, clamped_y, 0):
                return

        if self._native and self._native.move_cursor(clamped_x, clamped_y):
            return


        if _PYAUTOGUI_AVAILABLE:
            try:
                pyautogui.moveTo(clamped_x, clamped_y, _pause=False)
            except Exception:
                pass

    def disconnect(self) -> str:
        """No-op for Host OS, provided for API compatibility."""
        return "OSController: disconnected"

    def connect(self) -> str:
        """No-op for Host OS, provided for API compatibility."""
        return "OSController: connected to Host OS"


# ===========================================================================
# Continuous Display & System Sleep Prevention
# ===========================================================================

def keep_display_active(enable: bool = True) -> bool:
    """
    Prevents the display from turning off and the system from entering idle sleep.
    Uses Win32 SetThreadExecutionState:
    ES_CONTINUOUS (0x80000000) | ES_SYSTEM_REQUIRED (0x00000001) | ES_DISPLAY_REQUIRED (0x00000002)
    """
    if not _IS_WINDOWS:
        return False
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        if enable:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        else:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        return True
    except Exception:
        return False


def configure_system_power_screen_awake(enable: bool = True) -> bool:
    """
    Configures Windows power scheme monitor and standby timeouts.
    If enable=True, sets display and sleep timeouts to 0 (Never turn off).
    """
    if not _IS_WINDOWS:
        return False
    import subprocess
    try:
        timeout_val = "0" if enable else "15"
        subprocess.run(["powercfg", "/change", "monitor-timeout-ac", timeout_val], capture_output=True, check=False)
        subprocess.run(["powercfg", "/change", "monitor-timeout-dc", timeout_val], capture_output=True, check=False)
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", timeout_val], capture_output=True, check=False)
        subprocess.run(["powercfg", "/change", "standby-timeout-dc", timeout_val], capture_output=True, check=False)
        return True
    except Exception:
        return False

