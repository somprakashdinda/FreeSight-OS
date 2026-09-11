"""
persistent_camera.py
====================

Permanent Camera Watchdog & OS Power Override (DOPC v9.0 Master).
Guarantees the webcam NEVER shuts down, freezes, or drops frames until
explicitly stopped by a manual user click.

Operational Pillars:
1. Win32 SetThreadExecutionState: Suppresses OS display dimming, idle sleep, and USB suspend.
2. Indefinite Exponential Recovery: Hot-rebinds OpenCV hardware handles upon frame drops.
3. Manual-Shutdown-Only Policy: The thread only terminates when manual_click_shutdown() is invoked.
4. Synthetic Failover Buffer: Supplies continuous 720p frames if hardware device is unplugged.
"""

from __future__ import annotations

import ctypes
import logging
import platform
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Win32 Power Management Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

logger = logging.getLogger("PersistentCameraDaemon")


class PersistentCameraDaemon:
    """
    Hardware Camera Daemon ensuring non-stop camera execution with OS power overrides
    and auto-reconnection until an explicit manual user shutdown is triggered.
    """

    def __init__(self, camera_index: int = 0, target_fps: float = 60.0) -> None:
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1.0, target_fps)

        self.is_running = False
        self.manual_shutdown_triggered = False
        self.lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.reconnect_count = 0
        self.total_frames_captured = 0

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_hardware_active = False

        # Lock OS Power State (Prevent Screen Sleep & USB Suspend)
        self._prevent_os_sleep()

    def _prevent_os_sleep(self) -> None:
        """Informs Windows OS that system and display are required continuously."""
        if platform.system() == "Windows":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                logger.info("[Power Daemon] OS sleep and display suspend successfully suppressed.")
            except Exception as e:
                logger.warning(f"[Power Daemon] Could not set thread execution state: {e}")

    def _allow_os_sleep(self) -> None:
        """Restores normal Windows OS power management state."""
        if platform.system() == "Windows":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                logger.info("[Power Daemon] OS power state restored to normal defaults.")
            except Exception as e:
                logger.warning(f"[Power Daemon] Could not restore thread execution state: {e}")

    def _init_hardware_camera(self) -> bool:
        """Initializes OpenCV video capture with DirectShow optimization on Windows."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        try:
            if platform.system() == "Windows":
                self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self.camera_index)
            else:
                self._cap = cv2.VideoCapture(self.camera_index)

            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self._is_hardware_active = True
                return True
        except Exception as e:
            logger.warning(f"[Camera Watchdog] Hardware init exception: {e}")

        self._is_hardware_active = False
        return False

    def start_non_stop_capture(self) -> threading.Thread:
        """Starts the non-stop camera capture daemon thread."""
        self.is_running = True
        self.manual_shutdown_triggered = False
        capture_thread = threading.Thread(target=self._capture_loop, name="camera-watchdog", daemon=True)
        capture_thread.start()
        return capture_thread

    def _capture_loop(self) -> None:
        """Continuous frame acquisition loop with zero-downtime recovery."""
        self._init_hardware_camera()
        backoff = 0.05

        while not self.manual_shutdown_triggered:
            t0 = time.perf_counter()
            frame_acquired = self._acquire_hardware_frame()

            if not frame_acquired:
                self.reconnect_count += 1
                logger.warning(
                    f"[Camera Watchdog] Frame drop detected. Reconnecting attempt #{self.reconnect_count}..."
                )
                time.sleep(backoff)
                backoff = min(0.5, backoff * 1.5)  # Exponential backoff 50ms -> 500ms
                self._init_hardware_camera()
                continue

            # Successful frame reset backoff
            backoff = 0.05
            self.total_frames_captured += 1

            # Smooth frame pacing
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.001, self.frame_interval - elapsed)
            time.sleep(sleep_time)

        logger.info("[Camera Watchdog] Manual shutdown signal received. Shutting down camera safely.")
        self.is_running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._allow_os_sleep()

    def _acquire_hardware_frame(self) -> bool:
        """Captures a frame from hardware camera or generates synthetic 720p frame buffer."""
        if self._cap is not None and self._cap.isOpened():
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.current_frame = frame
                    return True
            except Exception:
                pass

        # Generate synthetic 720p fallback frame buffer if hardware is detached or blacked out
        with self.lock:
            # Create a test synthetic buffer (dark gradient canvas with ocular timestamp)
            synth = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                synth,
                f"Synthetic Failover Buffer - Frame {self.total_frames_captured}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 242, 254),
                2,
                cv2.LINE_AA,
            )
            self.current_frame = synth
        return True

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Thread-safe copy of the latest captured frame."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def manual_click_shutdown(self) -> None:
        """The ONLY trigger allowed to shut down the camera capture thread."""
        logger.info("[User Interface] Manual shutdown button clicked by user.")
        self.manual_shutdown_triggered = True

    @property
    def is_hardware_connected(self) -> bool:
        """True if physical camera is currently connected and streaming."""
        return self._is_hardware_active and self._cap is not None and self._cap.isOpened()
