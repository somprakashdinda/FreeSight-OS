"""
test_master_v5.py
=================

Comprehensive System Verification Suite for DOPC v7.0:
Autonomous Spatial AI Engine with On-Device Micro-Transformer Intent Prediction.
Verifies micro-transformer convergence, 1D temporal self-attention, ring buffer rollover,
latency budgets (< 1.1ms), and SystemState telemetry integration.
"""

import time
import unittest
import numpy as np

from intent_predictor import MicroTransformerGazePredictor
from state_manager import SystemState, StateSnapshot
from config import V7_CONFIG


class TestMasterV5Engine(unittest.TestCase):
    
    def test_transformer_prediction_convergence(self):
        """Verify transformer attention mechanism and target forecast convergence."""
        predictor = MicroTransformerGazePredictor(sequence_length=16, feature_dim=6, seed=42)
        
        # Push synthetic trajectory toward target (1200, 800)
        for i in range(20):
            predictor.push_state(
                x=1000.0 + i * 10.0,
                y=700.0 + i * 5.0,
                vx=10.0,
                vy=5.0,
                ax=0.1,
                ay=0.05
            )
            
        px, py, conf = predictor.predict_saccade_target()
        self.assertIsNotNone(px)
        self.assertIsNotNone(py)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        # Predicted coordinate should be forward in the trajectory
        self.assertGreater(px, 1000.0)
        self.assertGreater(py, 700.0)

    def test_transformer_buffer_rollover(self):
        """Verify 1D temporal ring buffer rolls over while preserving recent states."""
        predictor = MicroTransformerGazePredictor(sequence_length=16, feature_dim=6)
        
        # Push 32 states (double the buffer capacity)
        for i in range(32):
            predictor.push_state(
                x=float(i),
                y=float(i * 2),
                vx=1.0,
                vy=2.0,
                ax=0.0,
                ay=0.0
            )
            
        # The oldest entry in buffer should now be i=16, newest should be i=31
        self.assertEqual(predictor.buffer.shape, (16, 6))
        self.assertAlmostEqual(predictor.buffer[0, 0], 16.0)
        self.assertAlmostEqual(predictor.buffer[-1, 0], 31.0)
        self.assertEqual(predictor.samples_pushed, 32)

    def test_transformer_confidence_gating(self):
        """Verify confidence scores are well-conditioned and bounded."""
        predictor = MicroTransformerGazePredictor(sequence_length=16, feature_dim=6, seed=123)
        
        for i in range(16):
            predictor.push_state(500.0, 500.0, 0.0, 0.0, 0.0, 0.0)
            
        _, _, conf = predictor.predict_saccade_target()
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_transformer_latency_budget(self):
        """Verify on-device micro-transformer inference is well within the <1.1ms SLA."""
        predictor = MicroTransformerGazePredictor(sequence_length=16, feature_dim=6)
        for i in range(16):
            predictor.push_state(500.0 + i, 500.0 + i, 1.0, 1.0, 0.0, 0.0)

        iterations = 2000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = predictor.predict_saccade_target()
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) / iterations) * 1000.0
        print(f"\n[BENCHMARK] MicroTransformerGazePredictor.predict: {avg_latency_ms:.5f} ms/infer")
        self.assertLess(avg_latency_ms, 0.20, "Inference latency must be < 0.20 ms (Target SLA: < 1.1 ms)")

    def test_system_state_v7_telemetry_integration(self):
        """Verify SystemState stores and publishes v7 Micro-Transformer intent fields."""
        state = SystemState()
        state.set_v7_intent_prediction(1420.5, 850.2, 0.92)
        
        px, py, conf = state.get_v7_intent_prediction()
        self.assertAlmostEqual(px, 1420.5)
        self.assertAlmostEqual(py, 850.2)
        self.assertAlmostEqual(conf, 0.92)
        
        # Verify atomic snapshot
        snap = state.get_snapshot()
        self.assertAlmostEqual(snap.predicted_saccade_x, 1420.5)
        self.assertAlmostEqual(snap.predicted_saccade_y, 850.2)
        self.assertAlmostEqual(snap.intent_confidence, 0.92)


if __name__ == "__main__":
    unittest.main()
