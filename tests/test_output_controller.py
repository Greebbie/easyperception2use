"""Tests for output frequency and strategy control."""

import pytest
from output_controller import OutputController


def _make_scene(ts, obj_count=1, risk="low", classes=None, objects=None):
    if classes is None:
        classes = ["person"]
    if objects is None:
        objects = [{"track_id": i, "position": {"region": "middle_center"},
                     "motion": {"moving": False}}
                    for i in range(obj_count)]
    return {
        "timestamp": ts,
        "objects": objects,
        "scene": {
            "object_count": obj_count,
            "risk_level": risk,
            "classes_present": classes,
            "dominant_object": {"rel_size": 0.05} if obj_count > 0 else None,
            "center_occupied": False,
            "moving_count": 0,
            "region_summary": {},
        },
    }


class TestOutputController:
    def test_every_frame(self):
        ctrl = OutputController(strategy="every_frame")
        assert ctrl.should_output(_make_scene(0))
        assert ctrl.should_output(_make_scene(0.01))
        assert ctrl.should_output(_make_scene(0.02))

    def test_interval(self):
        ctrl = OutputController(strategy="interval", interval_sec=1.0)
        # First output at t=1.0 (since last_output_time starts at 0)
        assert not ctrl.should_output(_make_scene(0.5))
        assert ctrl.should_output(_make_scene(1.0))
        assert not ctrl.should_output(_make_scene(1.5))
        assert ctrl.should_output(_make_scene(2.0))

    def test_on_change_first_output(self):
        ctrl = OutputController(strategy="on_change")
        assert ctrl.should_output(_make_scene(0))

    def test_on_change_no_change(self):
        ctrl = OutputController(strategy="on_change")
        scene = _make_scene(0)
        ctrl.should_output(scene)
        assert not ctrl.should_output(_make_scene(1))

    def test_on_change_object_count(self):
        ctrl = OutputController(strategy="on_change")
        ctrl.should_output(_make_scene(0, obj_count=1))
        assert ctrl.should_output(_make_scene(1, obj_count=2))

    def test_on_change_risk_level(self):
        ctrl = OutputController(strategy="on_change")
        ctrl.should_output(_make_scene(0, risk="low"))
        assert ctrl.should_output(_make_scene(1, risk="high"))

    def test_on_change_new_class(self):
        ctrl = OutputController(strategy="on_change")
        ctrl.should_output(_make_scene(0, classes=["person"]))
        assert ctrl.should_output(_make_scene(1, classes=["person", "car"]))

    def test_on_change_region_cross(self):
        ctrl = OutputController(strategy="on_change")
        obj1 = [{"track_id": 1, "position": {"region": "top_left"},
                  "motion": {"moving": False}}]
        ctrl.should_output(_make_scene(0, objects=obj1))

        obj2 = [{"track_id": 1, "position": {"region": "middle_center"},
                  "motion": {"moving": False}}]
        assert ctrl.should_output(_make_scene(1, objects=obj2))

    def test_hybrid(self):
        ctrl = OutputController(strategy="hybrid", interval_sec=1.0)
        assert ctrl.should_output(_make_scene(0))
        # No change, not enough time
        assert not ctrl.should_output(_make_scene(0.5))
        # Interval hit
        assert ctrl.should_output(_make_scene(1.0))

    def test_hybrid_change_triggers_early(self):
        ctrl = OutputController(strategy="hybrid", interval_sec=10.0)
        ctrl.should_output(_make_scene(0, risk="low"))
        # Change triggers output even before interval
        assert ctrl.should_output(_make_scene(0.5, risk="high"))
