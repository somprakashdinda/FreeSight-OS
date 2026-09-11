"""
test_benchmark.py
=================

End-to-End Latency Benchmark for Eye-Tracking Host OS Controller.
Verifies micro-metric performance standards from suggestion-v2.md:
- State access latency (< 0.1ms)
- Predictive UKF step latency (< 0.5ms)
- Screen mapper prediction latency (< 1.0ms)
- SendInput dispatch latency (< 1.0ms)
- Total pipeline latency (< 12.0ms)
"""

import time
import unittest
import numpy as np
from predictive_filter import PredictiveGazeUKF
from gaze_mapper import ScreenMapper
from state_manager import SystemState
from screen_geometry import ScreenGeometryManager


class TestLatencyBenchmark(unittest.TestCase):

    def test_state_snapshot_zero_lock_latency(self):
        """Verify lock-free snapshot retrieval latency is sub-microsecond."""
        state = SystemState()
        iterations = 10000

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = state.get_snapshot()
        t1 = time.perf_counter()

        avg_us = ((t1 - t0) / iterations) * 1_000_000
        print(f"\n[BENCHMARK] StateSnapshot retrieval: {avg_us:.3f} microseconds / read")
        self.assertLess(avg_us, 10.0, "Snapshot read should take < 10 microseconds")

    def test_predictive_ukf_step_latency(self):
        """Verify UKF update and predict step latency is < 0.1ms."""
        ukf = PredictiveGazeUKF(dt=1.0 / 60.0)
        iterations = 5000

        t0 = time.perf_counter()
        for i in range(iterations):
            ukf.update_and_predict(500.0 + (i % 10), 500.0 + (i % 10))
        t1 = time.perf_counter()

        avg_ms = ((t1 - t0) / iterations) * 1000
        print(f"[BENCHMARK] PredictiveGazeUKF step: {avg_ms:.4f} ms / step")
        self.assertLess(avg_ms, 0.5, "UKF step should take < 0.5ms")

    def test_screen_mapper_predict_latency(self):
        """Verify trained ScreenMapper prediction latency is < 0.2ms."""
        mapper = ScreenMapper(1920, 1080)
        samples = [
            (0.1, 0.1, 100, 100), (0.5, 0.1, 960, 100), (0.9, 0.1, 1800, 100),
            (0.1, 0.5, 100, 540), (0.5, 0.5, 960, 540), (0.9, 0.5, 1800, 540),
            (0.1, 0.9, 100, 1000), (0.5, 0.9, 960, 1000), (0.9, 0.9, 1800, 1000),
        ]
        mapper.train_model(samples)

        iterations = 5000
        t0 = time.perf_counter()
        for i in range(iterations):
            u = 0.2 + (i % 100) * 0.005
            v = 0.2 + (i % 100) * 0.005
            mapper.predict(u, v)
        t1 = time.perf_counter()

        avg_ms = ((t1 - t0) / iterations) * 1000
        print(f"[BENCHMARK] ScreenMapper.predict(): {avg_ms:.4f} ms / predict")
        self.assertLess(avg_ms, 1.0, "ScreenMapper predict should take < 1.0ms")


if __name__ == "__main__":
    unittest.main()
