"""
test_state_manager.py
=====================

Unit tests for SystemState: lock-free double-buffered atomic state snapshotting
and high-concurrency reader/writer synchronization.
"""

import threading
import time
import unittest
from state_manager import SystemState, DirectionV1


class TestStateManager(unittest.TestCase):

    def test_snapshot_immutability(self):
        """Verify StateSnapshot is frozen and cannot be mutated by callers."""
        state = SystemState()
        snap = state.get_snapshot()
        with self.assertRaises(Exception):
            snap.current_pupil_x = 0.9  # FrozenInstanceError

    def test_atomic_double_blink_consumption(self):
        """Verify consume_double_blink clears the flag atomically."""
        state = SystemState()
        state.set_double_blink_detected(True)
        self.assertTrue(state.is_double_blink_detected())

        # First consume should be True
        self.assertTrue(state.consume_double_blink())
        # Subsequent consume should be False
        self.assertFalse(state.consume_double_blink())
        self.assertFalse(state.is_double_blink_detected())

    def test_high_concurrency_contention(self):
        """
        Verify lock-free read consistency under high contention:
        1 writer thread updating at high frequency, 4 reader threads querying snapshots.
        """
        state = SystemState()
        stop_event = threading.Event()
        read_counts = [0, 0, 0, 0]
        errors = []

        def writer():
            step = 0
            while not stop_event.is_set():
                val = float(step % 100) / 100.0
                state.set_pupil_position(val, val)
                state.set_ear(0.25 + val * 0.1)
                state.set_v1_direction(DirectionV1.LEFT if val < 0.5 else DirectionV1.RIGHT)
                step += 1
                time.sleep(0.0005)

        def reader(idx):
            while not stop_event.is_set():
                try:
                    snap = state.get_snapshot()
                    # Point-in-time consistency check: pupil_x and pupil_y were written together
                    if snap.current_pupil_x != snap.current_pupil_y:
                        errors.append(f"Inconsistent snapshot observed: x={snap.current_pupil_x}, y={snap.current_pupil_y}")
                    read_counts[idx] += 1
                except Exception as exc:
                    errors.append(str(exc))
                time.sleep(0.0002)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader, args=(i,)) for i in range(4)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        # Run for 0.25 seconds under full contention
        time.sleep(0.25)
        stop_event.set()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Encountered concurrency errors: {errors}")
        total_reads = sum(read_counts)
        self.assertGreater(total_reads, 100, f"Expected >100 total reads, got {total_reads}")


if __name__ == "__main__":
    unittest.main()
