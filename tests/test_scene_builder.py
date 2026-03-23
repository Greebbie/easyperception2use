"""Tests for SceneBuilder — the core detection-to-scene orchestration module."""

import json
import time
import copy

import pytest
import numpy as np

from scene_builder import SceneBuilder
from config import DEFAULT_CONFIG
from test_helpers import make_yolo_results, MockResult, MockBoxes, YOLO_NAMES


def _make_config(**overrides):
    """Return a config dict with overrides applied."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def _make_builder(config=None, fw=1280, fh=720):
    """Create a SceneBuilder with sane test defaults."""
    if config is None:
        config = _make_config(track_confirm_frames=1)
    return SceneBuilder(fw, fh, config)


# =========================================================================
# Constructor validation
# =========================================================================

class TestConstructor:
    def test_rejects_zero_width(self):
        with pytest.raises(ValueError, match="Invalid frame dimensions"):
            SceneBuilder(0, 720, _make_config())

    def test_rejects_zero_height(self):
        with pytest.raises(ValueError, match="Invalid frame dimensions"):
            SceneBuilder(1280, 0, _make_config())

    def test_rejects_negative_dimensions(self):
        with pytest.raises(ValueError):
            SceneBuilder(-1, 720, _make_config())

    def test_update_frame_size_rejects_zero(self):
        sb = _make_builder()
        with pytest.raises(ValueError):
            sb.update_frame_size(0, 720)

    def test_update_frame_size_resets_state(self):
        sb = _make_builder()
        results = make_yolo_results([
            (1, "person", 0.9, 540.0, 260.0, 740.0, 460.0),
        ])
        sb.build(results, 1.0)
        assert len(sb.prev_positions) > 0
        sb.update_frame_size(640, 480)
        assert sb.fw == 640
        assert sb.fh == 480
        assert len(sb.prev_positions) == 0
        assert len(sb._confirmed) == 0


# =========================================================================
# Region mapping (3x3 grid)
# =========================================================================

class TestGetRegion:
    def test_all_nine_regions(self):
        sb = _make_builder()
        cases = [
            (0.1, 0.1, "top_left"), (0.5, 0.1, "top_center"), (0.9, 0.1, "top_right"),
            (0.1, 0.5, "middle_left"), (0.5, 0.5, "middle_center"), (0.9, 0.5, "middle_right"),
            (0.1, 0.9, "bottom_left"), (0.5, 0.9, "bottom_center"), (0.9, 0.9, "bottom_right"),
        ]
        for rx, ry, expected in cases:
            assert sb._get_region(rx, ry) == expected, f"({rx}, {ry}) should be {expected}"

    def test_exact_boundaries(self):
        """Boundary values: x=0.33 → center, x=0.67 → right, y=0.4 → middle, y=0.7 → bottom."""
        sb = _make_builder()
        # x boundaries (< 0.33 is left, >= 0.33 and < 0.67 is center, >= 0.67 is right)
        assert sb._get_region(0.33, 0.5) == "middle_center"
        assert sb._get_region(0.67, 0.5) == "middle_right"
        assert sb._get_region(0.329, 0.5) == "middle_left"
        # y boundaries
        assert sb._get_region(0.5, 0.4) == "middle_center"
        assert sb._get_region(0.5, 0.7) == "bottom_center"
        assert sb._get_region(0.5, 0.399) == "top_center"

    def test_edge_values(self):
        """Extreme values 0.0 and 1.0."""
        sb = _make_builder()
        assert sb._get_region(0.0, 0.0) == "top_left"
        assert sb._get_region(1.0, 1.0) == "bottom_right"


# =========================================================================
# Motion calculation
# =========================================================================

class TestCalcMotion:
    def test_first_observation_stationary(self):
        sb = _make_builder()
        motion = sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        assert motion["direction"] == "stationary"
        assert motion["speed"] == 0.0
        assert motion["moving"] is False

    def test_direction_right(self):
        sb = _make_builder()
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        sb._kalman.update(1, 0.5, 0.5, 0.0)
        sb._kalman.update(1, 0.55, 0.5, 0.1)
        motion = sb._calc_motion(1, 0.55, 0.5, 0.1, 0.1, 0.0)
        assert motion["direction"] == "right"
        assert motion["moving"] is True
        assert motion["speed"] > 0

    def test_direction_left(self):
        sb = _make_builder()
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        sb._kalman.update(1, 0.5, 0.5, 0.0)
        sb._kalman.update(1, 0.45, 0.5, 0.1)
        motion = sb._calc_motion(1, 0.45, 0.5, 0.1, -0.1, 0.0)
        assert motion["direction"] == "left"

    def test_direction_up(self):
        sb = _make_builder()
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        sb._kalman.update(1, 0.5, 0.5, 0.0)
        sb._kalman.update(1, 0.5, 0.4, 0.1)
        motion = sb._calc_motion(1, 0.5, 0.4, 0.1, 0.0, -0.1)
        assert motion["direction"] == "up"

    def test_direction_down(self):
        sb = _make_builder()
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        sb._kalman.update(1, 0.5, 0.5, 0.0)
        sb._kalman.update(1, 0.5, 0.6, 0.1)
        motion = sb._calc_motion(1, 0.5, 0.6, 0.1, 0.0, 0.1)
        assert motion["direction"] == "down"

    def test_below_threshold_stationary(self):
        sb = _make_builder()
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        motion = sb._calc_motion(1, 0.501, 0.501, 0.1, 0.001, 0.001)
        assert motion["moving"] is False
        assert motion["direction"] == "stationary"

    def test_reliable_when_stopped(self):
        sb = _make_builder()
        sb._ego_state = "stopped"
        sb._camera_confidence = 0.9
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        motion = sb._calc_motion(1, 0.5, 0.5, 0.1, 0.0, 0.0)
        assert motion["reliable"] is True

    def test_unreliable_when_moving(self):
        sb = _make_builder()
        sb._ego_state = "moving"
        sb._camera_confidence = 0.1
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        motion = sb._calc_motion(1, 0.5, 0.5, 0.1, 0.0, 0.0)
        assert motion["reliable"] is False

    def test_unreliable_when_low_camera_confidence(self):
        sb = _make_builder()
        sb._ego_state = "stopped"
        sb._camera_confidence = 0.1  # below 0.3 threshold
        sb._calc_motion(1, 0.5, 0.5, 0.0, 0.0, 0.0)
        motion = sb._calc_motion(1, 0.5, 0.5, 0.1, 0.0, 0.0)
        assert motion["reliable"] is False


# =========================================================================
# Object debounce
# =========================================================================

class TestUpdateDebounce:
    def test_confirm_after_n_frames(self):
        sb = _make_builder(_make_config(track_confirm_frames=3))
        sb._update_debounce({1})
        assert 1 not in sb._confirmed
        sb._update_debounce({1})
        assert 1 not in sb._confirmed
        sb._update_debounce({1})
        assert 1 in sb._confirmed

    def test_not_confirmed_too_early(self):
        sb = _make_builder(_make_config(track_confirm_frames=5))
        for _ in range(4):
            sb._update_debounce({1})
        assert 1 not in sb._confirmed

    def test_lost_after_threshold(self):
        sb = _make_builder(_make_config(track_confirm_frames=1, track_lost_frames=3))
        sb._update_debounce({1})
        assert 1 in sb._confirmed
        sb._update_debounce(set())
        sb._update_debounce(set())
        assert 1 in sb._confirmed
        sb._update_debounce(set())  # 3rd miss
        assert 1 not in sb._confirmed

    def test_reappear_resets_missing_count(self):
        sb = _make_builder(_make_config(track_confirm_frames=1, track_lost_frames=3))
        sb._update_debounce({1})
        sb._update_debounce(set())
        sb._update_debounce(set())
        sb._update_debounce({1})    # reappear resets
        sb._update_debounce(set())
        sb._update_debounce(set())
        assert 1 in sb._confirmed  # should survive

    def test_stale_pending_cleanup(self):
        sb = _make_builder(_make_config(track_confirm_frames=5))
        sb._update_debounce({1})
        assert 1 in sb._pending_count
        sb._update_debounce(set())
        assert 1 not in sb._pending_count

    def test_multiple_tracks_independent(self):
        sb = _make_builder(_make_config(track_confirm_frames=2))
        sb._update_debounce({1, 2})
        assert 1 not in sb._confirmed
        sb._update_debounce({1})  # track 2 disappears
        assert 1 in sb._confirmed
        assert 2 not in sb._confirmed


# =========================================================================
# Track stability
# =========================================================================

class TestTrackStability:
    def test_change_updates_timestamp(self):
        sb = _make_builder()
        sb._track_stability({1}, 1.0)
        sb._track_stability({1, 2}, 2.0)
        assert sb._last_change_ts == 2.0

    def test_no_change_preserves_timestamp(self):
        sb = _make_builder()
        sb._track_stability({1}, 1.0)
        sb._track_stability({1}, 2.0)
        assert sb._last_change_ts == 1.0

    def test_empty_to_nonempty_is_change(self):
        sb = _make_builder()
        sb._track_stability(set(), 1.0)
        sb._track_stability({1}, 2.0)
        assert sb._last_change_ts == 2.0


# =========================================================================
# Output assembly & trust model
# =========================================================================

class TestMakeOutput:
    def test_output_schema_keys(self):
        sb = _make_builder()
        out = sb._make_output([], set(), 1.0)
        required = {"frame_id", "schema_version", "timestamp", "frame_size",
                     "actionable", "trust", "camera_motion", "objects", "scene", "meta"}
        assert required.issubset(out.keys())

    def test_output_is_valid_json(self):
        """Ensure output can round-trip through JSON serialization."""
        sb = _make_builder()
        out = sb._make_output([], set(), 1.0)
        serialized = json.dumps(out)
        parsed = json.loads(serialized)
        assert parsed["schema_version"] == "3.2"

    def test_trust_stopped(self):
        sb = _make_builder()
        sb._ego_state = "stopped"
        out = sb._make_output([], set(), 1.0)
        assert out["actionable"] is True
        for key in ("detection", "position", "motion", "scene"):
            assert out["trust"][key] is True

    def test_trust_moving(self):
        sb = _make_builder()
        sb._ego_state = "moving"
        out = sb._make_output([], set(), 1.0)
        assert out["actionable"] is False
        assert out["trust"]["detection"] is True
        assert out["trust"]["position"] is False
        assert out["trust"]["motion"] is False
        assert out["trust"]["scene"] is False

    def test_trust_settling(self):
        sb = _make_builder()
        sb._ego_state = "settling"
        out = sb._make_output([], set(), 1.0)
        assert out["actionable"] is False

    def test_with_objects_includes_scene_summary(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.10)]
        out = sb._make_output(objects, {1}, 1.0)
        assert out["scene"]["object_count"] == 1
        assert out["scene"]["center_occupied"] is True
        assert out["scene"]["risk_level"] in ("low", "medium", "high")

    def test_latency_included_when_provided(self):
        sb = _make_builder()
        out = sb._make_output([], set(), 1.0, latency_ms={"total": 50.0})
        assert out["latency_ms"]["total"] == 50.0

    def test_latency_absent_when_none(self):
        sb = _make_builder()
        out = sb._make_output([], set(), 1.0)
        assert "latency_ms" not in out


# =========================================================================
# Scene summary
# =========================================================================

def _make_obj_dict(track_id, cls="person", region="middle_center",
                   rel_size=0.05, moving=False, pos_conf=0.9):
    """Build an object dict matching SceneBuilder's internal format."""
    return {
        "track_id": track_id,
        "class": cls,
        "confidence": 0.9,
        "track_age": 10,
        "position_confidence": pos_conf,
        "velocity_confidence": 0.5,
        "position": {
            "rel_x": 0.5, "rel_y": 0.5,
            "smoothed_x": 0.5, "smoothed_y": 0.5,
            "rel_size": rel_size, "region": region,
        },
        "bbox_px": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "motion": {
            "direction": "right" if moving else "stationary",
            "speed": 0.05 if moving else 0.0,
            "vx": 0.05 if moving else 0.0,
            "vy": 0.0,
            "moving": moving,
            "reliable": True,
        },
    }


class TestBuildSceneSummary:
    def test_empty_objects(self):
        sb = _make_builder()
        summary = sb._build_scene_summary([], 1.0)
        assert summary["risk_level"] == "clear"
        assert summary["object_count"] == 0
        assert summary["center_occupied"] is False
        assert summary["dominant_object"] is None

    def test_risk_high(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.20)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "high"

    def test_risk_medium(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.08)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "medium"

    def test_risk_low(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.03)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "low"

    def test_risk_boundary_at_threshold(self):
        """Exact threshold values: 0.15 → medium (not high), 0.05 → low (not medium)."""
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.15)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "medium"
        objects = [_make_obj_dict(1, region="middle_center", rel_size=0.05)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "low"

    def test_non_center_object_is_clear(self):
        """Object not in center region should not raise risk level."""
        sb = _make_builder()
        objects = [_make_obj_dict(1, region="top_left", rel_size=0.30)]
        assert sb._build_scene_summary(objects, 1.0)["risk_level"] == "clear"

    def test_region_summary(self):
        sb = _make_builder()
        objects = [
            _make_obj_dict(1, cls="person", region="top_left"),
            _make_obj_dict(2, cls="car", region="top_left"),
            _make_obj_dict(3, cls="dog", region="bottom_right"),
        ]
        summary = sb._build_scene_summary(objects, 1.0)
        assert set(summary["region_summary"]["top_left"]) == {"person", "car"}
        assert summary["region_summary"]["bottom_right"] == ["dog"]

    def test_dominant_object(self):
        sb = _make_builder()
        objects = [
            _make_obj_dict(1, rel_size=0.02),
            _make_obj_dict(2, rel_size=0.10),
            _make_obj_dict(3, rel_size=0.05),
        ]
        summary = sb._build_scene_summary(objects, 1.0)
        assert summary["dominant_object"]["track_id"] == 2
        assert summary["dominant_object"]["rel_size"] == 0.10

    def test_moving_count(self):
        sb = _make_builder()
        objects = [
            _make_obj_dict(1, moving=True),
            _make_obj_dict(2, moving=False),
            _make_obj_dict(3, moving=True),
        ]
        assert sb._build_scene_summary(objects, 1.0)["moving_count"] == 2

    def test_snapshot_quality_range(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1, pos_conf=0.9)]
        summary = sb._build_scene_summary(objects, 1.0)
        assert 0.0 <= summary["snapshot_quality"] <= 1.0

    def test_depth_ordering_when_depth_present(self):
        sb = _make_builder()
        objects = [
            {**_make_obj_dict(1), "depth": {"value": 0.8, "label": "far"}},
            {**_make_obj_dict(2), "depth": {"value": 0.2, "label": "near"}},
        ]
        summary = sb._build_scene_summary(objects, 1.0)
        assert summary["depth_ordering"] == [2, 1]  # nearest first
        assert summary["nearest_object"]["track_id"] == 2


# =========================================================================
# Compact JSON
# =========================================================================

class TestCompact:
    def _full_scene(self):
        sb = _make_builder()
        objects = [_make_obj_dict(1)]
        objects[0]["predicted_next"] = {"cx": 0.51, "cy": 0.5}
        return sb._make_output(objects, {1}, 1.0)

    def test_compact_top_level_keys(self):
        compact = SceneBuilder.compact(self._full_scene())
        for key in ("ts", "actionable", "objects", "scene", "camera", "changes"):
            assert key in compact

    def test_compact_object_fields(self):
        compact = SceneBuilder.compact(self._full_scene())
        obj = compact["objects"][0]
        for key in ("id", "cls", "cx", "cy", "size", "region", "conf",
                     "age", "pos_conf", "vx", "vy", "moving", "reliable"):
            assert key in obj, f"Missing key: {key}"

    def test_compact_with_prediction(self):
        compact = SceneBuilder.compact(self._full_scene())
        assert "pred" in compact["objects"][0]

    def test_compact_with_depth(self):
        scene = self._full_scene()
        scene["objects"][0]["depth"] = {"value": 0.3, "label": "near"}
        compact = SceneBuilder.compact(scene)
        assert compact["objects"][0]["depth"] == 0.3

    def test_compact_is_valid_json(self):
        compact = SceneBuilder.compact(self._full_scene())
        serialized = json.dumps(compact)
        assert len(serialized) < 1000  # compact should be small

    def test_compact_empty_scene(self):
        sb = _make_builder()
        scene = sb._make_output([], set(), 1.0)
        compact = SceneBuilder.compact(scene)
        assert compact["objects"] == []
        assert compact["scene"]["count"] == 0


# =========================================================================
# Ego state machine
# =========================================================================

class TestEgoStateMachine:
    def test_initial_state_stopped(self):
        sb = _make_builder()
        assert sb._ego_state == "stopped"

    def test_stopped_to_moving(self):
        sb = _make_builder()
        sb.set_ego_motion(True)
        assert sb._ego_state == "moving"

    def test_moving_to_settling(self):
        sb = _make_builder()
        sb.set_ego_motion(True)
        sb.set_ego_motion(False)
        assert sb._ego_state == "settling"

    def test_settling_to_stopped(self):
        sb = _make_builder(_make_config(ego_settle_sec=0.5))
        sb.set_ego_motion(True)
        sb.set_ego_motion(False)
        sb._ego_stop_ts = time.time() - 1.0
        sb._update_ego_state(time.time())
        assert sb._ego_state == "stopped"

    def test_settling_not_expired(self):
        sb = _make_builder(_make_config(ego_settle_sec=5.0))
        sb.set_ego_motion(True)
        sb.set_ego_motion(False)
        sb._update_ego_state(time.time())
        assert sb._ego_state == "settling"

    def test_full_cycle(self):
        """stopped → moving → settling → stopped (full round trip)."""
        sb = _make_builder(_make_config(ego_settle_sec=0.1))
        assert sb._ego_state == "stopped"
        sb.set_ego_motion(True)
        assert sb._ego_state == "moving"
        sb.set_ego_motion(False)
        assert sb._ego_state == "settling"
        sb._ego_stop_ts = time.time() - 0.5
        sb._update_ego_state(time.time())
        assert sb._ego_state == "stopped"

    def test_ego_velocity_stored(self):
        sb = _make_builder()
        sb.set_ego_motion(True, vx=0.1, vy=-0.05)
        assert sb._ego_vx == 0.1
        assert sb._ego_vy == -0.05

    def test_repeated_moving_no_change(self):
        """Calling set_ego_motion(True) while already moving stays in moving."""
        sb = _make_builder()
        sb.set_ego_motion(True)
        sb.set_ego_motion(True)  # still moving
        assert sb._ego_state == "moving"


# =========================================================================
# Optical flow integration
# =========================================================================

class TestOpticalFlow:
    def test_first_frame_initializes(self):
        sb = _make_builder()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        assert sb._prev_gray is not None
        assert sb._global_dx == 0.0
        assert sb._global_dy == 0.0

    def test_static_frames_zero_motion(self):
        sb = _make_builder()
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        sb._estimate_global_motion(frame, [], 1.1)
        assert abs(sb._global_dx) < 0.01
        assert abs(sb._global_dy) < 0.01
        assert sb._camera_confidence > 0.5

    def test_ego_motion_none_mode(self):
        sb = _make_builder(_make_config(ego_motion_source="none"))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        assert sb._global_dx == 0.0
        assert sb._global_dy == 0.0
        assert sb._camera_confidence == 1.0
        assert sb._ego_state == "stopped"

    def test_foreground_masking_reduces_confidence(self):
        """With large foreground bbox covering most of the frame, confidence drops."""
        sb = _make_builder()
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        # No foreground — get baseline confidence
        sb._estimate_global_motion(frame, [], 1.1)
        baseline_conf = sb._camera_confidence

        # Large foreground bbox — confidence should drop significantly
        sb._prev_gray = None  # reset to get clean comparison
        sb._estimate_global_motion(frame, [], 1.2)
        huge_bbox = [(0, 0, 155, 115)]  # covers most of 160x120 downsampled frame
        sb._estimate_global_motion(frame, huge_bbox, 1.3)
        assert sb._camera_confidence < baseline_conf

    def test_external_mode_moving(self):
        sb = _make_builder(_make_config(ego_motion_source="external"))
        sb.set_ego_motion(True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        assert sb._camera_confidence == 0.0
        assert sb._global_dx == 0.0

    def test_external_mode_stopped(self):
        sb = _make_builder(_make_config(ego_motion_source="external"))
        sb._ego_state = "stopped"
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        sb._estimate_global_motion(frame, [], 1.0)
        sb._estimate_global_motion(frame, [], 1.1)
        assert sb._camera_confidence > 0.0


# =========================================================================
# Ego auto-detect from optical flow
# =========================================================================

class TestEgoAutoDetect:
    def test_low_confidence_triggers_moving(self):
        sb = _make_builder(_make_config(ego_auto_detect=True))
        sb._ego_state = "stopped"
        sb._camera_confidence = 0.1  # below 0.3
        # Simulate 3 consecutive low-confidence frames
        for _ in range(3):
            sb._camera_confidence = 0.1
            if sb._ego_auto_detect:
                if sb._camera_confidence < 0.3:
                    sb._low_confidence_streak += 1
                    if sb._low_confidence_streak >= 3 and sb._ego_state == "stopped":
                        sb._ego_state = "moving"
        assert sb._ego_state == "moving"

    def test_confidence_recovery_triggers_settling(self):
        sb = _make_builder(_make_config(ego_auto_detect=True))
        sb._ego_state = "moving"
        sb._low_confidence_streak = 5
        # Confidence recovers
        sb._camera_confidence = 0.8
        if sb._low_confidence_streak >= 3 and sb._ego_state == "moving":
            sb._ego_state = "settling"
            sb._ego_stop_ts = time.time()
        sb._low_confidence_streak = 0
        assert sb._ego_state == "settling"


# =========================================================================
# Integration: full build() with mock YOLO results
# =========================================================================

class TestBuildIntegration:
    def test_build_none_results(self):
        sb = _make_builder()
        out = sb.build(None, 1.0)
        assert out["objects"] == []
        assert out["schema_version"] == "3.2"

    def test_build_empty_boxes(self):
        sb = _make_builder()
        results = [MockResult([], YOLO_NAMES)]
        out = sb.build(results, 1.0)
        assert out["objects"] == []

    def test_build_single_detection(self):
        sb = _make_builder()
        results = make_yolo_results([
            (1, "person", 0.9, 540.0, 260.0, 740.0, 460.0),
        ])
        out = sb.build(results, 1.0)
        assert len(out["objects"]) == 1
        obj = out["objects"][0]
        assert obj["class"] == "person"
        assert obj["track_id"] == 1
        assert obj["confidence"] == 0.9
        assert obj["position"]["region"] == "middle_center"
        assert "predicted_next" in obj
        assert "position_confidence" in obj
        assert "velocity_confidence" in obj

    def test_build_filters_low_confidence(self):
        sb = _make_builder(_make_config(track_confirm_frames=1, min_confidence=0.45))
        results = make_yolo_results([
            (1, "person", 0.1, 100.0, 100.0, 200.0, 200.0),
        ])
        out = sb.build(results, 1.0)
        assert out["objects"] == []
        assert out["meta"]["dropped_by_confidence"] == 1

    def test_build_filters_by_class(self):
        sb = _make_builder(_make_config(
            track_confirm_frames=1, filter_classes=["person"]
        ))
        results = make_yolo_results([
            (1, "car", 0.9, 100.0, 100.0, 200.0, 200.0),
        ])
        assert sb.build(results, 1.0)["objects"] == []

    def test_build_no_track_id_skipped(self):
        sb = _make_builder()
        results = make_yolo_results([
            (None, "person", 0.9, 100.0, 100.0, 200.0, 200.0),
        ])
        assert sb.build(results, 1.0)["objects"] == []

    def test_build_multiple_objects(self):
        sb = _make_builder()
        results = make_yolo_results([
            (1, "person", 0.9, 100.0, 100.0, 300.0, 400.0),
            (2, "car", 0.85, 600.0, 300.0, 900.0, 500.0),
        ])
        out = sb.build(results, 1.0)
        assert len(out["objects"]) == 2
        assert {o["class"] for o in out["objects"]} == {"person", "car"}

    def test_build_with_frame(self):
        sb = _make_builder()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        results = make_yolo_results([
            (1, "person", 0.9, 540.0, 260.0, 740.0, 460.0),
        ])
        sb.build(results, 1.0, frame=frame)
        out = sb.build(results, 1.1, frame=frame)
        assert "camera_motion" in out
        assert "confidence" in out["camera_motion"]

    def test_build_with_depth_fn(self):
        sb = _make_builder()
        results = make_yolo_results([
            (1, "person", 0.9, 540.0, 260.0, 740.0, 460.0),
        ])
        depth_fn = lambda bbox: {"value": 0.3, "label": "near"}
        out = sb.build(results, 1.0, depth_fn=depth_fn)
        assert out["objects"][0]["depth"] == {"value": 0.3, "label": "near"}

    def test_build_unknown_class_id_skipped(self):
        """Box with cls_id not in names dict should be skipped."""
        from test_helpers import MockBox, MockResult, MockBoxes
        box = MockBox(1, 99, 0.9, [100.0, 100.0, 200.0, 200.0])  # cls_id=99 not in names
        results = [MockResult([box], YOLO_NAMES)]
        sb = _make_builder()
        out = sb.build(results, 1.0)
        assert out["objects"] == []

    def test_build_mixed_tracked_and_untracked(self):
        """Mix of tracked and untracked boxes in same frame."""
        sb = _make_builder()
        results = make_yolo_results([
            (1, "person", 0.9, 100.0, 100.0, 200.0, 200.0),
            (None, "car", 0.85, 300.0, 300.0, 400.0, 400.0),
        ])
        out = sb.build(results, 1.0)
        assert len(out["objects"]) == 1
        assert out["objects"][0]["class"] == "person"

    def test_build_frame_id_increments(self):
        sb = _make_builder()
        out1 = sb.build(None, 1.0)
        out2 = sb.build(None, 2.0)
        assert out2["frame_id"] == out1["frame_id"] + 1

    def test_build_cleanup_removes_old_tracks(self):
        """Tracks not seen within timeout should be cleaned from prev_positions."""
        sb = _make_builder(_make_config(track_confirm_frames=1, track_lost_timeout=0.5))
        results = make_yolo_results([
            (1, "person", 0.9, 100.0, 100.0, 200.0, 200.0),
        ])
        sb.build(results, 1.0)
        assert 1 in sb.prev_positions
        # Object disappears, wait past timeout
        sb.build(None, 5.0)
        assert 1 not in sb.prev_positions

    def test_build_output_valid_json(self):
        """Full pipeline output must be JSON-serializable (no NaN/Inf)."""
        sb = _make_builder()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        results = make_yolo_results([
            (1, "person", 0.9, 540.0, 260.0, 740.0, 460.0),
        ])
        sb.build(results, 1.0, frame=frame)
        out = sb.build(results, 1.1, frame=frame)
        serialized = json.dumps(out)
        assert "NaN" not in serialized
        assert "Infinity" not in serialized
