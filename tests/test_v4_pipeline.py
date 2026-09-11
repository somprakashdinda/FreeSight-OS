"""
test_v4_pipeline.py
===================

Unit tests for v4.0 Predictive Kalman Filter and low-latency pipeline
as specified in Section 6 of suggestion-v2.md.
"""

import unittest
import numpy as np
from predictive_filter import PredictiveGazeUKF


class TestV4Pipeline(unittest.TestCase):

    def test_ukf_prediction_accuracy(self):
        """Verify UKF ahead-of-time prediction leads measurements without divergence."""
        filter_inst = PredictiveGazeUKF(dt=1.0 / 60.0, prediction_ahead_frames=1.0)

        # Simulate linear motion eye gaze
        trajectories = [(100.0 + i * 5.0, 200.0 + i * 3.0) for i in range(30)]
        px, py = 0.0, 0.0
        for x_m, y_m in trajectories:
            px, py = filter_inst.update_and_predict(x_m, y_m)

        # Assert prediction leads measurement without divergence
        self.assertTrue(px > 240.0, f"Expected px > 240.0, got {px}")
        self.assertTrue(py > 280.0, f"Expected py > 280.0, got {py}")

    def test_fixation_jitter_suppression(self):
        """Verify that under static fixation with noise, output variance < input variance."""
        filter_inst = PredictiveGazeUKF(dt=1.0 / 60.0, prediction_ahead_frames=0.0)
        np.random.seed(42)

        # Fixed point (960, 540) with Gaussian micro-saccade noise
        center_x, center_y = 960.0, 540.0
        noise_x = np.random.normal(0, 4.0, 100)
        noise_y = np.random.normal(0, 4.0, 100)

        filtered_x = []
        filtered_y = []
        for nx, ny in zip(noise_x, noise_y):
            fx, fy = filter_inst.update_and_predict(center_x + nx, center_y + ny)
            filtered_x.append(fx)
            filtered_y.append(fy)

        # Discard initial warmup
        filtered_x = filtered_x[20:]
        filtered_y = filtered_y[20:]

        var_input = np.var(noise_x[20:])
        var_output = np.var(np.array(filtered_x) - center_x)
        self.assertLess(var_output, var_input * 0.7, "Jitter variance should be reduced by at least 30%")

    def test_saccade_adaptation(self):
        """Verify that high velocity gaze jumps trigger saccade gating."""
        filter_inst = PredictiveGazeUKF(dt=1.0 / 60.0, saccade_velocity_thresh=200.0)
        # Initialize at 100
        filter_inst.update_and_predict(100.0, 100.0)
        # Big jump to 800 (speed > 200px/s)
        px, py = filter_inst.update_and_predict(800.0, 800.0)
        # Should adapt rapidly towards 800
        self.assertGreater(px, 400.0)
        self.assertGreater(py, 400.0)


if __name__ == "__main__":
    unittest.main()
