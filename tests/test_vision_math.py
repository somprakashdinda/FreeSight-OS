"""
test_vision_math.py
===================

Unit tests for vision geometry, EAR calculation, and AdaptiveDeadzoneManager.
"""

import unittest
import numpy as np
from vision.blink_detector import BlinkDetector
from vision_pipeline import AdaptiveDeadzoneManager
from state_manager import DirectionV1


class TestVisionMath(unittest.TestCase):

    def test_blink_detector_logic(self):
        """Test single blink, double blink, and refractory period."""
        detector = BlinkDetector(
            ear_threshold=0.20,
            min_closed_frames=2,
            double_blink_max_delay=0.5,
            refractory_period=0.1,
        )

        # Eye open
        res1 = detector.update(0.35)
        self.assertFalse(res1.eye_closed)
        self.assertFalse(res1.blink_completed)

        # Eye closed frame 1 (debouncing)
        res2 = detector.update(0.15)
        self.assertFalse(res2.eye_closed)
        self.assertFalse(res2.blink_completed)

        # Eye closed frame 2 (accepted closure)
        res3 = detector.update(0.15)
        self.assertTrue(res3.eye_closed)
        self.assertFalse(res3.blink_completed)

        # Eye reopens -> blink completed
        res4 = detector.update(0.35)
        self.assertFalse(res4.eye_closed)
        self.assertTrue(res4.blink_completed)

    def test_adaptive_deadzone_calibration(self):
        """Test baseline collection and dynamic threshold calculation."""
        manager = AdaptiveDeadzoneManager(
            baseline_seconds=1.0,
            target_fps=10,  # 10 frames needed
            half_width_x=0.10,
            half_height_y=0.10,
        )

        self.assertFalse(manager.is_calibrated)

        # Feed 10 samples around center (0.48, 0.52)
        for _ in range(10):
            manager.add_sample(0.48, 0.52)

        self.assertTrue(manager.is_calibrated)
        cx, cy = manager.baseline_center
        self.assertAlmostEqual(cx, 0.48, places=2)
        self.assertAlmostEqual(cy, 0.52, places=2)

    def test_adaptive_deadzone_classification(self):
        """Test directional classification using calibrated baseline."""
        manager = AdaptiveDeadzoneManager(
            baseline_seconds=0.5,
            target_fps=10,  # 5 frames
            half_width_x=0.10,
            half_height_y=0.10,
        )
        for _ in range(5):
            manager.add_sample(0.50, 0.50)

        # Looking center
        dir_res, mag = manager.classify(0.50, 0.50)
        self.assertEqual(dir_res, DirectionV1.CENTER)
        self.assertEqual(mag, 0.0)

        # Looking left (ratio_x < 0.40)
        dir_res, mag = manager.classify(0.20, 0.50)
        self.assertEqual(dir_res, DirectionV1.LEFT)
        self.assertGreater(mag, 0.0)

        # Looking right (ratio_x > 0.60)
        dir_res, mag = manager.classify(0.80, 0.50)
        self.assertEqual(dir_res, DirectionV1.RIGHT)
        self.assertGreater(mag, 0.0)

        # Looking up (ratio_y < 0.40)
        dir_res, mag = manager.classify(0.50, 0.20)
        self.assertEqual(dir_res, DirectionV1.UP)
        self.assertGreater(mag, 0.0)

        # Looking down (ratio_y > 0.60)
        dir_res, mag = manager.classify(0.50, 0.80)
        self.assertEqual(dir_res, DirectionV1.DOWN)
        self.assertGreater(mag, 0.0)

    def test_pitch_compensation(self):
        """Verify head pose pitch compensates vertical gaze ratio."""
        manager = AdaptiveDeadzoneManager(
            baseline_seconds=0.5,
            target_fps=10,
            half_width_x=0.10,
            half_height_y=0.10,
            pose_compensation_factor=0.005,
        )
        for _ in range(5):
            manager.add_sample(0.50, 0.50)

        # At ratio_y = 0.55, without pitch compensation it might be center.
        # With positive pitch (+20 deg), adjusted_y = 0.55 - 0.10 = 0.45 (remains in center).
        # With negative pitch (-25 deg), adjusted_y = 0.55 - (-0.125) = 0.675 -> DOWN!
        dir_res, mag = manager.classify(0.50, 0.55, head_pose={"valid": True, "pitch": -25.0})
        self.assertEqual(dir_res, DirectionV1.DOWN)


if __name__ == "__main__":
    unittest.main()
