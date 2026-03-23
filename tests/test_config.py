"""Tests for default configuration validation."""

import pytest
from config import DEFAULT_CONFIG


class TestDefaultConfig:
    def test_all_required_keys_present(self):
        required_keys = [
            "source", "model_path", "min_confidence", "process_fps",
            "tracker", "track_history_len", "track_lost_timeout",
            "track_confirm_frames", "track_lost_frames",
            "kalman_process_noise", "kalman_measurement_noise",
            "output_strategy", "output_interval_sec", "output_method",
            "ego_motion_source", "ego_settle_sec", "ego_auto_detect",
            "risk_thresholds", "center_region_x", "center_region_y",
            "depth_enabled", "depth_model_size", "depth_device",
            "ws_enabled", "ws_port",
        ]
        for key in required_keys:
            assert key in DEFAULT_CONFIG, f"Missing config key: {key}"

    def test_value_types(self):
        assert isinstance(DEFAULT_CONFIG["min_confidence"], float)
        assert isinstance(DEFAULT_CONFIG["process_fps"], int)
        assert DEFAULT_CONFIG["filter_classes"] is None or isinstance(
            DEFAULT_CONFIG["filter_classes"], list
        )
        assert isinstance(DEFAULT_CONFIG["risk_thresholds"], dict)
        assert "high" in DEFAULT_CONFIG["risk_thresholds"]
        assert "medium" in DEFAULT_CONFIG["risk_thresholds"]

    def test_risk_thresholds_ordering(self):
        thresholds = DEFAULT_CONFIG["risk_thresholds"]
        assert thresholds["high"] > thresholds["medium"]

    def test_center_region_tuples(self):
        cx = DEFAULT_CONFIG["center_region_x"]
        cy = DEFAULT_CONFIG["center_region_y"]
        assert len(cx) == 2
        assert len(cy) == 2
        assert cx[0] < cx[1]
        assert cy[0] < cy[1]
        assert 0.0 <= cx[0] <= 1.0
        assert 0.0 <= cx[1] <= 1.0

    def test_depth_disabled_by_default(self):
        assert DEFAULT_CONFIG["depth_enabled"] is False

    def test_value_ranges(self):
        """Critical config values should be in sane ranges."""
        assert 0.0 <= DEFAULT_CONFIG["min_confidence"] <= 1.0
        assert DEFAULT_CONFIG["process_fps"] > 0
        assert DEFAULT_CONFIG["track_lost_timeout"] > 0
        assert DEFAULT_CONFIG["kalman_process_noise"] > 0
        assert DEFAULT_CONFIG["kalman_measurement_noise"] > 0
        assert DEFAULT_CONFIG["ego_settle_sec"] > 0
        assert DEFAULT_CONFIG["output_interval_sec"] > 0

    def test_output_strategy_valid(self):
        valid = {"every_frame", "interval", "on_change", "hybrid", "stable"}
        assert DEFAULT_CONFIG["output_strategy"] in valid
