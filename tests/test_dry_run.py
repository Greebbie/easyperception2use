"""Tests for DryRunGenerator and FakeObject synthetic data."""

import random
import pytest
import numpy as np

from dry_run import FakeObject, DryRunGenerator


# =========================================================================
# FakeObject
# =========================================================================

class TestFakeObject:
    def test_initialization_in_bounds(self):
        for seed in range(10):
            random.seed(seed)
            obj = FakeObject(1, 1280, 720)
            assert 0.05 <= obj.x <= 0.95
            assert 0.05 <= obj.y <= 0.95
            assert obj.cls_name in ("person", "car", "dog", "chair", "bottle")
            assert 0.5 <= obj.confidence <= 0.98

    def test_update_moves_position(self):
        random.seed(42)
        obj = FakeObject(1, 1280, 720)
        obj.vx = 0.1
        obj.vy = 0.05
        old_x, old_y = obj.x, obj.y
        # Use small dt to avoid random direction change
        obj.update(0.01)
        assert abs(obj.x - (old_x + 0.1 * 0.01)) < 0.001
        assert abs(obj.y - (old_y + 0.05 * 0.01)) < 0.001

    def test_bounce_off_right_edge(self):
        random.seed(42)
        obj = FakeObject(1, 1280, 720)
        obj.x = 0.96
        obj.vx = 0.5
        obj.vy = 0.0
        obj.update(0.01)
        assert obj.x <= 0.95
        assert obj.vx < 0

    def test_bounce_off_left_edge(self):
        random.seed(42)
        obj = FakeObject(1, 1280, 720)
        obj.x = 0.04
        obj.vx = -0.5
        obj.vy = 0.0
        obj.update(0.01)
        assert obj.x >= 0.05
        assert obj.vx > 0

    def test_get_bbox_px_valid(self):
        for seed in range(5):
            random.seed(seed)
            obj = FakeObject(1, 1280, 720)
            bbox = obj.get_bbox_px()
            assert bbox["x1"] < bbox["x2"]
            assert bbox["y1"] < bbox["y2"]
            assert all(isinstance(v, int) for v in bbox.values())


# =========================================================================
# DryRunGenerator
# =========================================================================

class TestDryRunGenerator:
    def test_generate_frame_shape(self):
        gen = DryRunGenerator(1280, 720, num_objects=3)
        frame = gen.generate_frame()
        assert frame.shape == (720, 1280, 3)
        assert frame.dtype == np.uint8

    def test_generate_scene_json_schema(self):
        gen = DryRunGenerator(1280, 720, num_objects=3)
        scene = gen.generate_scene_json(1.0)
        # All top-level keys must match scene_builder.py output
        required = {"frame_id", "schema_version", "timestamp", "frame_size",
                     "actionable", "trust", "camera_motion",
                     "objects", "scene", "meta"}
        assert required.issubset(scene.keys()), f"Missing: {required - scene.keys()}"
        assert scene["schema_version"] == "3.2"
        assert scene["actionable"] is True
        # Trust model
        for key in ("detection", "position", "motion", "scene"):
            assert key in scene["trust"]
        # Camera motion
        for key in ("tx", "ty", "compensated", "confidence", "ego_state"):
            assert key in scene["camera_motion"]
        assert scene["camera_motion"]["ego_state"] == "stopped"
        # Scene summary
        for key in ("stable", "time_since_change_sec", "snapshot_quality"):
            assert key in scene["scene"], f"Missing scene key: {key}"

    def test_scene_json_object_schema(self):
        gen = DryRunGenerator(1280, 720, num_objects=2)
        scene = gen.generate_scene_json(1.0)
        for obj in scene["objects"]:
            for key in ("track_id", "class", "confidence", "track_age",
                        "position_confidence", "velocity_confidence",
                        "position", "bbox_px", "motion"):
                assert key in obj, f"Missing key: {key}"
            # Position sub-fields (must match SceneBuilder output)
            for key in ("rel_x", "rel_y", "smoothed_x", "smoothed_y",
                        "rel_size", "region"):
                assert key in obj["position"], f"Missing position key: {key}"
            # Motion sub-fields
            for key in ("direction", "speed", "vx", "vy", "moving", "reliable"):
                assert key in obj["motion"], f"Missing motion key: {key}"

    def test_scene_json_object_count(self):
        gen = DryRunGenerator(1280, 720, num_objects=4)
        scene = gen.generate_scene_json(1.0)
        assert len(scene["objects"]) == 4
        assert scene["scene"]["object_count"] == 4

    def test_frame_id_increments(self):
        gen = DryRunGenerator(1280, 720, num_objects=1)
        s1 = gen.generate_scene_json(1.0)
        s2 = gen.generate_scene_json(2.0)
        assert s2["frame_id"] == s1["frame_id"] + 1

    def test_zero_objects_no_crash(self):
        """DryRunGenerator with zero objects should not crash."""
        gen = DryRunGenerator(1280, 720, num_objects=0)
        scene = gen.generate_scene_json(1.0)
        assert scene["objects"] == []
        assert scene["scene"]["object_count"] == 0
        assert scene["scene"]["dominant_object"] is None

    def test_direction_stationary_when_slow(self):
        """Objects with speed <= 0.02 should have direction 'stationary'."""
        gen = DryRunGenerator(1280, 720, num_objects=1)
        gen.objects[0].vx = 0.001
        gen.objects[0].vy = 0.001
        scene = gen.generate_scene_json(1.0)
        obj = scene["objects"][0]
        assert obj["motion"]["direction"] == "stationary"
        assert obj["motion"]["moving"] is False

    def test_region_consistent_with_position(self):
        """Region string must be consistent with rel_x/rel_y."""
        random.seed(42)
        gen = DryRunGenerator(1280, 720, num_objects=5)
        scene = gen.generate_scene_json(1.0)
        for obj in scene["objects"]:
            x = obj["position"]["rel_x"]
            y = obj["position"]["rel_y"]
            region = obj["position"]["region"]
            row, col = region.split("_")

            if x < 0.33:
                assert col == "left"
            elif x < 0.67:
                assert col == "center"
            else:
                assert col == "right"

            if y < 0.4:
                assert row == "top"
            elif y < 0.7:
                assert row == "middle"
            else:
                assert row == "bottom"
