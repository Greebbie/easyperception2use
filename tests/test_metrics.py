"""Tests for pipeline metrics and health monitoring."""

import time
import pytest
from metrics import PipelineMetrics


class TestPipelineMetrics:
    def test_initial_state(self):
        m = PipelineMetrics()
        health = m.get_health()
        assert health["state"] == "initializing"
        assert health["total_frames"] == 0
        assert health["fps"] == 0.0

    def test_set_state(self):
        m = PipelineMetrics()
        m.set_state("running")
        assert m.get_health()["state"] == "running"
        m.set_state("degraded")
        assert m.get_health()["state"] == "degraded"

    def test_degraded_modules(self):
        m = PipelineMetrics()
        m.add_degraded_module("depth")
        assert "depth" in m.get_health()["degraded_modules"]
        m.add_degraded_module("depth")  # duplicate
        assert m.get_health()["degraded_modules"].count("depth") == 1
        m.remove_degraded_module("depth")
        assert "depth" not in m.get_health()["degraded_modules"]

    def test_record_frame(self):
        m = PipelineMetrics()
        m.record_frame({"total": 50.0, "detect": 30.0})
        m.record_frame({"total": 60.0, "detect": 40.0})
        assert m.get_health()["total_frames"] == 2

    def test_latency_percentiles(self):
        m = PipelineMetrics()
        for i in range(100):
            m.record_frame({"total": float(i)})
        p50 = m.get_latency_p50("total")
        p95 = m.get_latency_p95("total")
        assert 45 <= p50 <= 55
        assert 90 <= p95 <= 99

    def test_fps_calculation(self):
        m = PipelineMetrics()
        # Simulate 10 frames at ~100fps
        for i in range(10):
            m.record_frame({"total": 10.0})
            time.sleep(0.01)
        fps = m.get_fps()
        assert fps > 0  # Should be measurable

    def test_pipeline_info(self):
        m = PipelineMetrics()
        m.set_state("running")
        m.add_degraded_module("depth")
        info = m.get_pipeline_info()
        assert info["state"] == "running"
        assert "depth" in info["degraded_modules"]
        assert info["uptime_sec"] >= 0

    def test_reset(self):
        m = PipelineMetrics()
        m.record_frame({"total": 50.0})
        m.reset()
        assert m.get_health()["total_frames"] == 0
