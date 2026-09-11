"""
benchmark_performance.py
========================

Automated End-to-End Performance Profiler and Benchmark Suite for FreeSight-OS (v6.0 Ultra).
Evaluates real-time subsystem micro-latencies, hardware utilization, live vision pipeline FPS,
and produces an industry-grade SLA compliance report.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import json
import os
import platform
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Import FreeSight-OS Core Modules
from state_manager import SystemState, DirectionV1
from predictive_filter import PredictiveGazeUKF
from gaze_mapper import ScreenMapper
from screen_geometry import ScreenGeometryManager
from spatial_geometry import SpatialGeometry6DOF, HeadPose6DOF
from implicit_calibrator import ImplicitGNNCalibrator, UIElementNode
from neuromorphic_dvs import NeuromorphicDVSHAL, DVSEventRecord
from bci_intent_fusion import HybridBCIIntentFusion
from intent_predictor import MicroTransformerGazePredictor
from smooth_scroller import SubPixelSmoothScroller
from persistent_camera import PersistentCameraDaemon
from work_limit_enforcer import WorkLimitEnforcer
from os_interop import OSController, WebcamCapture
from vision_pipeline import GazeTracker
from config import CV_CONFIG, V5_CONFIG, V6_CONFIG, V9_CONFIG


# ---------------------------------------------------------------------------
# Memory Measurement (Win32 API)
# ---------------------------------------------------------------------------

class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

def get_process_memory_mb() -> float:
    """Returns current process Working Set (RAM) in megabytes."""
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0400 | 0x0010, False, os.getpid())
        if handle:
            ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
            ctypes.windll.kernel32.CloseHandle(handle)
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

class FreeSightPerformanceProfiler:

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.sla_targets: Dict[str, Tuple[float, str]] = {
            "state_snapshot_us": (10.0, "µs (< 10 µs lock-free SLA)"),
            "ukf_step_ms": (0.50, "ms (< 0.50 ms predictive filter SLA)"),
            "screen_mapper_predict_ms": (0.50, "ms (< 0.50 ms RLS mapping SLA)"),
            "spatial_6dof_compensation_ms": (0.20, "ms (< 0.20 ms 6DOF parallax SLA)"),
            "implicit_calibration_step_ms": (0.50, "ms (< 0.50 ms autonomous calibrator SLA)"),
            "neuromorphic_dvs_packet_ms": (0.80, "ms (< 0.80 ms event HAL SLA)"),
            "bci_intent_fusion_ms": (1.00, "ms (< 1.00 ms intent gating SLA)"),
            "os_input_dispatch_ms": (1.00, "ms (< 1.00 ms SendInput SLA)"),
            "micro_transformer_predict_ms": (1.10, "ms (< 1.10 ms v7 intent forecast SLA)"),
            "subpixel_smooth_scroll_ms": (0.05, "ms (< 0.05 ms v9 physics scroller SLA)"),
            "work_limit_enforce_ms": (0.05, "ms (< 0.05 ms v9 resource guard SLA)"),
            "pure_inference_latency_ms": (25.0, "ms (< 25.0 ms algorithmic budget)"),
            "live_camera_stream_fps": (28.0, "FPS (>= 28.0 FPS real-world webcam pacing SLA)"),
        }

    def run_micro_benchmarks(self):
        print("\n" + "=" * 76)
        print("  STAGE 1: SUBSYSTEM MICRO-LATENCY & THROUGHPUT BENCHMARKS")
        print("=" * 76)

        # 1. State Snapshot Latency
        state = SystemState()
        N = 25000
        t0 = time.perf_counter()
        for _ in range(N):
            _ = state.get_snapshot()
        t1 = time.perf_counter()
        state_us = ((t1 - t0) / N) * 1_000_000
        self.results["state_snapshot_us"] = round(state_us, 4)
        print(f"  [+] SystemState.get_snapshot():       {state_us:8.4f} µs/op    (Target: < 10.0 µs)")

        # 2. Predictive UKF Step Latency
        ukf = PredictiveGazeUKF(dt=1.0 / 60.0)
        N = 10000
        t0 = time.perf_counter()
        for i in range(N):
            ukf.update_and_predict(500.0 + (i % 20), 500.0 + (i % 20))
        t1 = time.perf_counter()
        ukf_ms = ((t1 - t0) / N) * 1000.0
        self.results["ukf_step_ms"] = round(ukf_ms, 5)
        print(f"  [+] PredictiveGazeUKF.step():         {ukf_ms:8.5f} ms/step  (Target: < 0.50 ms)")

        # 3. ScreenMapper Predict Latency
        mapper = ScreenMapper(1920, 1080)
        samples = [
            (0.1, 0.1, 100, 100), (0.5, 0.1, 960, 100), (0.9, 0.1, 1800, 100),
            (0.1, 0.5, 100, 540), (0.5, 0.5, 960, 540), (0.9, 0.5, 1800, 540),
            (0.1, 0.9, 100, 1000), (0.5, 0.9, 960, 1000), (0.9, 0.9, 1800, 1000),
        ]
        mapper.train_model(samples)
        N = 10000
        t0 = time.perf_counter()
        for i in range(N):
            mapper.predict(0.3 + (i % 100) * 0.004, 0.3 + (i % 100) * 0.004)
        t1 = time.perf_counter()
        mapper_ms = ((t1 - t0) / N) * 1000.0
        self.results["screen_mapper_predict_ms"] = round(mapper_ms, 5)
        print(f"  [+] ScreenMapper.predict():           {mapper_ms:8.5f} ms/op    (Target: < 0.50 ms)")

        # 4. Spatial Geometry 6DOF Parallax Compensation
        geo = SpatialGeometry6DOF()
        head_pose = HeadPose6DOF(
            yaw=2.0, pitch=-3.0, roll=0.5,
            translation=np.array([0.0, 0.0, 0.65], dtype=np.float64),
            rotation_matrix=np.eye(3, dtype=np.float64),
            focal_distance_cm=65.0
        )
        raw_vec = np.array([0.1, -0.2], dtype=np.float64)
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            geo.compensate_posture(raw_vec, head_pose)
        t1 = time.perf_counter()
        geo_ms = ((t1 - t0) / N) * 1000.0
        self.results["spatial_6dof_compensation_ms"] = round(geo_ms, 5)
        print(f"  [+] SpatialGeometry6DOF.compensate(): {geo_ms:8.5f} ms/op    (Target: < 0.20 ms)")

        # 5. Autonomous Implicit Calibrator Step
        calibrator = ImplicitGNNCalibrator(1920, 1080)
        nodes = [
            UIElementNode("btn1", (100.0, 100.0, 300.0, 150.0), 0.9),
            UIElementNode("btn2", (500.0, 500.0, 700.0, 550.0), 0.8),
        ]
        calibrator.update_ui_saliency_map(nodes)
        N = 5000
        t0 = time.perf_counter()
        for i in range(N):
            calibrator.process_passive_gaze_sample((0.15 + (i % 50) * 0.001, 0.12 + (i % 50) * 0.001))
        t1 = time.perf_counter()
        impl_ms = ((t1 - t0) / N) * 1000.0
        self.results["implicit_calibration_step_ms"] = round(impl_ms, 5)
        print(f"  [+] ImplicitCalibrator.step():        {impl_ms:8.5f} ms/op    (Target: < 0.50 ms)")

        # 6. Neuromorphic DVS Packet Processing
        dvs = NeuromorphicDVSHAL(sensor_width=1280, sensor_height=720)
        raw_events = [DVSEventRecord(x=100 + (i % 50), y=100 + (i % 50), timestamp_us=1000 + i, polarity=1) for i in range(500)]
        N = 1000
        t0 = time.perf_counter()
        for _ in range(N):
            dvs.process_event_stream(raw_events)
        t1 = time.perf_counter()
        dvs_ms = ((t1 - t0) / N) * 1000.0
        self.results["neuromorphic_dvs_packet_ms"] = round(dvs_ms, 5)
        events_per_sec = (500 * N) / (t1 - t0)
        self.results["neuromorphic_events_per_sec"] = round(events_per_sec, 0)
        print(f"  [+] Neuromorphic DVS Packet (500ev):  {dvs_ms:8.5f} ms/pkt   (Target: < 0.80 ms, {events_per_sec:,.0f} ev/s)")

        # 7. Hybrid BCI Intent Fusion
        bci = HybridBCIIntentFusion(dwell_threshold_ms=200.0, fixation_velocity_threshold=15.0, neural_intent_threshold=0.75)
        N = 10000
        t0 = time.perf_counter()
        for i in range(N):
            bci.evaluate_click_intent(gaze_velocity=8.5, eeg_p300_signal=0.88, dt_ms=16.6)
        t1 = time.perf_counter()
        bci_ms = ((t1 - t0) / N) * 1000.0
        self.results["bci_intent_fusion_ms"] = round(bci_ms, 5)
        print(f"  [+] HybridBCIIntentFusion.evaluate(): {bci_ms:8.5f} ms/op    (Target: < 1.00 ms)")

        # 8. OSController Input Dispatch Latency
        controller = OSController()
        controller.connect()
        N = 2000
        t0 = time.perf_counter()
        for i in range(N):
            controller.move_cursor(960 + (i % 10), 540 + (i % 10))
        t1 = time.perf_counter()
        os_ms = ((t1 - t0) / N) * 1000.0
        self.results["os_input_dispatch_ms"] = round(os_ms, 5)
        print(f"  [+] OSController.move_cursor():       {os_ms:8.5f} ms/op    (Target: < 1.00 ms)")

        # 9. v7.0 On-Device Micro-Transformer Intent Prediction
        predictor = MicroTransformerGazePredictor(sequence_length=16, feature_dim=6)
        for i in range(16):
            predictor.push_state(500.0 + i, 500.0 + i, 1.0, 1.0, 0.0, 0.0)
        N = 5000
        t0 = time.perf_counter()
        for _ in range(N):
            _ = predictor.predict_saccade_target()
        t1 = time.perf_counter()
        trans_ms = ((t1 - t0) / N) * 1000.0
        self.results["micro_transformer_predict_ms"] = round(trans_ms, 5)
        print(f"  [+] MicroTransformer.predict():       {trans_ms:8.5f} ms/infer (Target: < 1.10 ms)")

        # 10. v9.0 Sub-Pixel Smooth Scroller Dispatch Latency
        scroller = SubPixelSmoothScroller(friction=0.90, gain=45.0, deadzone=0.05)
        N = 10000
        t0 = time.perf_counter()
        for i in range(N):
            _ = scroller.process_ocular_displacement(0.20 + (i % 10) * 0.01)
        t1 = time.perf_counter()
        scroll_ms = ((t1 - t0) / N) * 1000.0
        self.results["subpixel_smooth_scroll_ms"] = round(scroll_ms, 5)
        print(f"  [+] SubPixelSmoothScroller:           {scroll_ms:8.5f} ms/op    (Target: < 0.05 ms)")

        # 11. v9.0 Hard Work Limit & Resource Guard Latency
        enforcer = WorkLimitEnforcer(max_memory_mb=12.5, target_fps=60.0)
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            _ = enforcer.check_resource_limits()
        t1 = time.perf_counter()
        enf_ms = ((t1 - t0) / N) * 1000.0
        self.results["work_limit_enforce_ms"] = round(enf_ms, 5)
        print(f"  [+] WorkLimitEnforcer.check():        {enf_ms:8.5f} ms/check (Target: < 0.05 ms)")

    def run_live_vision_pipeline_benchmark(self, frame_count: int = 40):
        print("\n" + "=" * 76)
        print(f"  STAGE 2: LIVE VISION PIPELINE BENCHMARK ({frame_count} Real Camera Frames)")
        print("=" * 76)

        ram_before = get_process_memory_mb()
        webcam = WebcamCapture()

        tracker = GazeTracker()

        latencies_ms: List[float] = []
        infer_latencies_ms: List[float] = []

        print("  Starting camera acquisition & MediaPipe FaceMesh processing...")
        t_start_total = time.perf_counter()

        for f_idx in range(frame_count):
            t_f0 = time.perf_counter()
            frame = webcam.get_latest_frame()
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            # Process pure vision pipeline: FaceMesh, IrisTracker, HeadPose, Blink
            t_infer_0 = time.perf_counter()
            _ = tracker.process_frame(frame)
            t_infer_1 = time.perf_counter()
            t_f1 = time.perf_counter()

            infer_ms = (t_infer_1 - t_infer_0) * 1000.0
            elapsed_ms = (t_f1 - t_f0) * 1000.0
            latencies_ms.append(elapsed_ms)
            infer_latencies_ms.append(infer_ms)

            if (f_idx + 1) % 10 == 0 or (f_idx + 1) == frame_count:
                sys.stdout.write(f"\r  Frames processed: {f_idx + 1}/{frame_count} | Infer: {infer_ms:5.2f}ms | Total: {elapsed_ms:5.2f}ms | Camera FPS: {1000.0/max(1.0, elapsed_ms):5.1f}")
                sys.stdout.flush()

        t_end_total = time.perf_counter()
        print()

        tracker.close()
        webcam.close()
        ram_after = get_process_memory_mb()

        total_wall_sec = t_end_total - t_start_total
        actual_fps = len(latencies_ms) / max(0.001, total_wall_sec)
        mean_lat_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0
        mean_infer_ms = float(np.mean(infer_latencies_ms)) if infer_latencies_ms else 0.0
        p50_lat_ms = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95_lat_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
        p99_lat_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0.0
        std_lat_ms = float(np.std(latencies_ms)) if latencies_ms else 0.0
        infer_capacity_fps = 1000.0 / max(0.1, mean_infer_ms)

        self.results["live_camera_stream_fps"] = round(actual_fps, 2)
        self.results["pure_inference_latency_ms"] = round(mean_infer_ms, 2)
        self.results["pure_inference_capacity_fps"] = round(infer_capacity_fps, 1)
        self.results["mean_frame_latency_ms"] = round(mean_lat_ms, 2)
        self.results["p50_frame_latency_ms"] = round(p50_lat_ms, 2)
        self.results["p95_frame_latency_ms"] = round(p95_lat_ms, 2)
        self.results["p99_frame_latency_ms"] = round(p99_lat_ms, 2)
        self.results["frame_jitter_std_ms"] = round(std_lat_ms, 2)
        self.results["ram_usage_mb"] = round(ram_after, 1)
        self.results["ram_delta_mb"] = round(ram_after - ram_before, 2)

        print()
        print(f"  [+] Pure Algorithmic Latency:         {mean_infer_ms:8.2f} ms/frame (Capacity: {infer_capacity_fps:.1f} FPS)")
        print(f"  [+] Live Camera Hardware Frame Rate:  {actual_fps:8.2f} FPS     (Target: >= 28.0 FPS)")
        print(f"  [+] Mean End-to-End Cycle Latency:    {mean_lat_ms:8.2f} ms")
        print(f"  [+] Median (p50) Frame Latency:       {p50_lat_ms:8.2f} ms")
        print(f"  [+] 95th Percentile (p95) Latency:    {p95_lat_ms:8.2f} ms")
        print(f"  [+] 99th Percentile (p99) Latency:    {p99_lat_ms:8.2f} ms")
        print(f"  [+] Frame Jitter (Std Dev):           {std_lat_ms:8.2f} ms")
        print(f"  [+] Active RAM Working Set:           {ram_after:8.1f} MB     (Delta: {ram_after - ram_before:+.2f} MB)")

    def evaluate_performance_level(self) -> Dict[str, Any]:
        print("\n" + "=" * 76)
        print("  STAGE 3: PERFORMANCE LEVEL & SLA COMPLIANCE EVALUATION")
        print("=" * 76)

        passed_slas = 0
        total_slas = len(self.sla_targets)
        eval_table = []

        for metric, (threshold, desc) in self.sla_targets.items():
            val = self.results.get(metric, 0.0)
            if "fps" in metric:
                passed = val >= threshold
                status = "PASS [EXCEEDED]" if val >= threshold * 1.2 else ("PASS" if passed else "FAIL")
            else:
                passed = val <= threshold
                status = "PASS [ULTRA-FAST]" if val <= threshold * 0.25 else ("PASS" if passed else "FAIL")

            if passed:
                passed_slas += 1

            eval_table.append({
                "metric": metric,
                "measured": val,
                "threshold": threshold,
                "description": desc,
                "status": status,
            })
            print(f"  {metric:<30} : {val:8.4f} | SLA: {desc:<34} -> [{status}]")

        compliance_pct = (passed_slas / total_slas) * 100.0

        if compliance_pct == 100.0:
            if self.results.get("ukf_step_ms", 1.0) < 0.1 and self.results.get("state_snapshot_us", 10.0) < 1.0:
                tier = "ENTERPRISE ULTRA (GRADE: A++)"
                score = 100.0
            else:
                tier = "ENTERPRISE GRADE (GRADE: A+)"
                score = 99.5
        elif compliance_pct >= 85.0:
            tier = "PRODUCTION READY (GRADE: A)"
            score = 95.0
        else:
            tier = "STANDARD TIER (GRADE: B)"
            score = 85.0

        self.results["sla_compliance_pct"] = compliance_pct
        self.results["performance_tier"] = tier
        self.results["overall_performance_score"] = score
        self.results["evaluation_table"] = eval_table

        print("\n" + "=" * 76)
        print(f"  FINAL PERFORMANCE LEVEL:  {tier}")
        print(f"  OVERALL PERFORMANCE SCORE: {score:.1f} / 100.0  ({compliance_pct:.0f}% SLA Compliance)")
        print("=" * 76 + "\n")

        # Save report
        with open("benchmark_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("  [✓] Detailed benchmark report saved to: benchmark_report.json\n")

        return self.results


def main():
    profiler = FreeSightPerformanceProfiler()
    profiler.run_micro_benchmarks()
    profiler.run_live_vision_pipeline_benchmark(frame_count=40)
    profiler.evaluate_performance_level()


if __name__ == "__main__":
    main()
