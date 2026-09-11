"""
test_gaze_mapper.py
===================

Unit tests for ScreenMapper, 9-point calibration regression, and OnlineGazeAdapter.
"""

import unittest
import numpy as np
from gaze_mapper import ScreenMapper, OnlineGazeAdapter, CalibrationFitReport


class TestGazeMapper(unittest.TestCase):

    def setUp(self):
        self.width = 1920
        self.height = 1080
        self.mapper = ScreenMapper(self.width, self.height)

    def test_nine_point_calibration_training(self):
        """Test training on a standard 3x3 synthetic grid with low RMSE."""
        grid_u = [0.1, 0.5, 0.9]
        grid_v = [0.1, 0.5, 0.9]
        samples = []
        for u in grid_u:
            for v in grid_v:
                # Target screen pixel coordinates
                sx = u * (self.width - 1)
                sy = v * (self.height - 1)
                samples.append((u, v, sx, sy))

        report = self.mapper.train_model(samples)
        self.assertTrue(self.mapper.is_trained)
        self.assertTrue(report.is_well_conditioned)
        self.assertLess(report.rmse_x_px, 5.0, "Ideal linear grid fit should have RMSE < 5px")
        self.assertLess(report.rmse_y_px, 5.0)

    def test_prediction_clamping(self):
        """Verify predicted coordinates are strictly clamped inside screen bounds."""
        samples = [
            (0.1, 0.1, 100, 100),
            (0.5, 0.1, 960, 100),
            (0.9, 0.1, 1800, 100),
            (0.1, 0.5, 100, 540),
            (0.5, 0.5, 960, 540),
            (0.9, 0.5, 1800, 540),
            (0.1, 0.9, 100, 1000),
            (0.5, 0.9, 960, 1000),
            (0.9, 0.9, 1800, 1000),
        ]
        self.mapper.train_model(samples)

        # Out-of-bounds inputs
        px, py = self.mapper.predict(-0.5, -0.5)
        self.assertGreaterEqual(px, 0)
        self.assertGreaterEqual(py, 0)

        px, py = self.mapper.predict(1.5, 1.5)
        self.assertLessEqual(px, self.width - 1)
        self.assertLessEqual(py, self.height - 1)

    def test_online_rls_adapter_convergence(self):
        """Test that OnlineGazeAdapter weights converge on target targets."""
        adapter = OnlineGazeAdapter(num_features=3, lambda_factor=0.99)
        # Ground truth: y = 2*f1 + 3*f2 + 10
        features = np.array([1.0, 0.5, 0.5])
        target_x = 1.0 * 10 + 0.5 * 200 + 0.5 * 100  # 160.0
        target_y = 1.0 * 20 + 0.5 * 300 + 0.5 * 200  # 270.0

        for _ in range(30):
            err_x, err_y = adapter.update(features, target_x, target_y)

        self.assertLess(abs(err_x), 1.0, "RLS error for X should converge to near zero")
        self.assertLess(abs(err_y), 1.0, "RLS error for Y should converge to near zero")
        self.assertEqual(adapter.adaptations_count, 30)

    def test_screen_mapper_online_adaptation(self):
        """Verify ScreenMapper.adapt_online updates weights and adaptation count."""
        samples = [
            (0.1, 0.1, 100, 100),
            (0.5, 0.1, 960, 100),
            (0.9, 0.1, 1800, 100),
            (0.1, 0.5, 100, 540),
            (0.5, 0.5, 960, 540),
            (0.9, 0.5, 1800, 540),
            (0.1, 0.9, 100, 1000),
            (0.5, 0.9, 960, 1000),
            (0.9, 0.9, 1800, 1000),
        ]
        self.mapper.train_model(samples)
        initial_count = self.mapper.online_adaptation_count

        # Adapt to dwell point (500, 500)
        self.mapper.adapt_online(0.3, 0.4, 500.0, 500.0)
        self.assertEqual(self.mapper.online_adaptation_count, initial_count + 1)


if __name__ == "__main__":
    unittest.main()
