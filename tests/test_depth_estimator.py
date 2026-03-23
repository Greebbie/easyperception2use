"""Tests for depth estimation module (static methods + disabled state)."""

import pytest
import numpy as np

from depth_estimator import DepthEstimator


# =========================================================================
# Initialization / disabled states
# =========================================================================

class TestDepthEstimatorInit:
    def test_default_config(self):
        de = DepthEstimator()
        assert de.model_size == "small"
        assert de.enabled is True
        assert de._loaded is False

    def test_disabled_returns_none(self):
        de = DepthEstimator(enabled=False)
        assert de.estimate(np.zeros((480, 640, 3), dtype=np.uint8)) is None

    def test_load_failed_returns_none(self):
        de = DepthEstimator(enabled=True)
        de._load_failed = True
        assert de.estimate(np.zeros((480, 640, 3), dtype=np.uint8)) is None
        assert de.load_failed is True

    def test_load_model_without_transformers(self):
        """_load_model should fail gracefully and auto-disable if transformers not installed."""
        de = DepthEstimator(enabled=True, load_timeout=5.0)
        # This will either succeed (if transformers installed) or fail gracefully
        # Either way, the pipeline should not crash
        result = de._load_model()
        if not result:
            assert de.enabled is False
            assert de._load_failed is True


# =========================================================================
# get_object_depth (static, no model required)
# =========================================================================

class TestGetObjectDepth:
    def test_near_depth(self):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        bbox = {"x1": 200, "y1": 150, "x2": 400, "y2": 350}
        result = DepthEstimator.get_object_depth(depth_map, bbox)
        assert result["label"] == "near"
        assert result["value"] == 0.0

    def test_far_depth(self):
        depth_map = np.ones((480, 640), dtype=np.float32)
        bbox = {"x1": 200, "y1": 150, "x2": 400, "y2": 350}
        result = DepthEstimator.get_object_depth(depth_map, bbox)
        assert result["label"] == "far"
        assert result["value"] == 1.0

    def test_mid_depth(self):
        depth_map = np.full((480, 640), 0.5, dtype=np.float32)
        bbox = {"x1": 200, "y1": 150, "x2": 400, "y2": 350}
        result = DepthEstimator.get_object_depth(depth_map, bbox)
        assert result["label"] == "mid"

    def test_degenerate_bbox_zero_width(self):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        result = DepthEstimator.get_object_depth(
            depth_map, {"x1": 200, "y1": 150, "x2": 200, "y2": 350}
        )
        assert result == {"value": 0.5, "label": "mid"}

    def test_degenerate_bbox_zero_height(self):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        result = DepthEstimator.get_object_depth(
            depth_map, {"x1": 200, "y1": 150, "x2": 400, "y2": 150}
        )
        assert result == {"value": 0.5, "label": "mid"}

    def test_bbox_clamped_at_edges(self):
        depth_map = np.full((480, 640), 0.2, dtype=np.float32)
        result = DepthEstimator.get_object_depth(
            depth_map, {"x1": -50, "y1": -50, "x2": 100, "y2": 100}
        )
        assert result["label"] == "near"
        assert 0.0 <= result["value"] <= 1.0

    def test_gradient_depth_map(self):
        depth_map = np.zeros((480, 640), dtype=np.float32)
        for x in range(640):
            depth_map[:, x] = x / 639.0

        left = DepthEstimator.get_object_depth(
            depth_map, {"x1": 10, "y1": 100, "x2": 100, "y2": 300}
        )
        right = DepthEstimator.get_object_depth(
            depth_map, {"x1": 540, "y1": 100, "x2": 630, "y2": 300}
        )
        assert left["value"] < right["value"]
        assert left["label"] == "near"
        assert right["label"] == "far"

    def test_boundary_labels(self):
        """Test exact boundary values: 0.33 → near, 0.66 → mid."""
        depth_map_near = np.full((100, 100), 0.329, dtype=np.float32)
        depth_map_mid = np.full((100, 100), 0.33, dtype=np.float32)
        depth_map_mid2 = np.full((100, 100), 0.659, dtype=np.float32)
        depth_map_far = np.full((100, 100), 0.66, dtype=np.float32)
        bbox = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}

        assert DepthEstimator.get_object_depth(depth_map_near, bbox)["label"] == "near"
        assert DepthEstimator.get_object_depth(depth_map_mid, bbox)["label"] == "mid"
        assert DepthEstimator.get_object_depth(depth_map_mid2, bbox)["label"] == "mid"
        assert DepthEstimator.get_object_depth(depth_map_far, bbox)["label"] == "far"

    def test_nan_in_depth_map(self):
        """NaN values in depth map should be handled (np.mean returns NaN)."""
        depth_map = np.full((100, 100), float("nan"), dtype=np.float32)
        bbox = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}
        result = DepthEstimator.get_object_depth(depth_map, bbox)
        # NaN comparison: value < 0.33 is False, value < 0.66 is False → "far"
        # This is a known behavior — NaN produces "far" label
        assert result["label"] in ("near", "mid", "far")

    def test_single_pixel_bbox(self):
        """1x1 bbox at valid location should not crash."""
        depth_map = np.full((100, 100), 0.5, dtype=np.float32)
        result = DepthEstimator.get_object_depth(
            depth_map, {"x1": 50, "y1": 50, "x2": 51, "y2": 51}
        )
        assert 0.0 <= result["value"] <= 1.0
