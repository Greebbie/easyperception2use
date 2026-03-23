"""Tests for scene change detection and diff generation."""

import pytest
from scene_differ import SceneDiffer


def _make_scene(objects=None, risk="clear", classes=None, center=False):
    """Helper to build a minimal scene JSON for testing."""
    if objects is None:
        objects = []
    if classes is None:
        classes = list(set(o["class"] for o in objects)) if objects else []
    return {
        "timestamp": 1000.0,
        "frame_size": {"w": 1280, "h": 720},
        "objects": objects,
        "scene": {
            "object_count": len(objects),
            "center_occupied": center,
            "dominant_object": None,
            "risk_level": risk,
            "classes_present": classes,
            "moving_count": 0,
            "region_summary": {},
        },
        "meta": {"active_tracks": 0, "total_tracks_in_memory": 0,
                 "dropped_by_confidence": 0},
    }


def _make_obj(track_id, cls="person", region="middle_center",
              rel_size=0.05, moving=False, direction="stationary"):
    return {
        "track_id": track_id,
        "class": cls,
        "confidence": 0.9,
        "position": {"rel_x": 0.5, "rel_y": 0.5, "rel_size": rel_size,
                      "region": region},
        "bbox_px": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
        "motion": {"direction": direction, "speed": 0.03 if moving else 0.0,
                    "vx": 0.0, "vy": 0.0, "moving": moving},
    }


class TestSceneDiffer:
    def test_first_scene_no_objects(self):
        differ = SceneDiffer()
        scene = _make_scene()
        changes = differ.diff(scene)
        assert changes == []

    def test_first_scene_with_objects(self):
        differ = SceneDiffer()
        scene = _make_scene(
            objects=[_make_obj(1)],
            classes=["person"],
        )
        changes = differ.diff(scene)
        assert len(changes) == 1
        assert "scene_start" in changes[0]

    def test_object_entered(self):
        differ = SceneDiffer()
        differ.diff(_make_scene())  # empty first scene

        scene2 = _make_scene(
            objects=[_make_obj(1, region="top_left")],
            classes=["person"],
        )
        changes = differ.diff(scene2)
        assert any("object_entered" in c for c in changes)
        assert any("#1" in c for c in changes)

    def test_object_left(self):
        differ = SceneDiffer()
        scene1 = _make_scene(
            objects=[_make_obj(1)],
            classes=["person"],
        )
        differ.diff(scene1)

        scene2 = _make_scene()
        changes = differ.diff(scene2)
        assert any("object_left" in c for c in changes)

    def test_region_change(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(objects=[_make_obj(1, region="top_left")],
                                classes=["person"]))

        scene2 = _make_scene(
            objects=[_make_obj(1, region="middle_center")],
            classes=["person"],
        )
        changes = differ.diff(scene2)
        assert any("region_change" in c for c in changes)
        assert any("top_left" in c and "middle_center" in c for c in changes)

    def test_risk_change(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(risk="low"))

        changes = differ.diff(_make_scene(risk="high"))
        assert any("risk_change" in c for c in changes)
        assert any("low" in c and "high" in c for c in changes)

    def test_object_approaching(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(
            objects=[_make_obj(1, rel_size=0.05)], classes=["person"]
        ))

        changes = differ.diff(_make_scene(
            objects=[_make_obj(1, rel_size=0.07)], classes=["person"]
        ))
        assert any("object_approaching" in c for c in changes)

    def test_motion_start(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(
            objects=[_make_obj(1, moving=False)], classes=["person"]
        ))

        changes = differ.diff(_make_scene(
            objects=[_make_obj(1, moving=True, direction="left")],
            classes=["person"],
        ))
        assert any("motion_start" in c for c in changes)

    def test_new_class(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(classes=["person"]))

        changes = differ.diff(_make_scene(classes=["person", "car"]))
        assert any("new_class" in c and "car" in c for c in changes)

    def test_center_occupied(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(center=False))

        changes = differ.diff(_make_scene(center=True))
        assert any("center_occupied" in c for c in changes)

    def test_object_retreating(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(
            objects=[_make_obj(1, rel_size=0.10)], classes=["person"]
        ))

        changes = differ.diff(_make_scene(
            objects=[_make_obj(1, rel_size=0.07)], classes=["person"]
        ))
        assert any("object_retreating" in c for c in changes)

    def test_motion_stop(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(
            objects=[_make_obj(1, moving=True, direction="left")],
            classes=["person"],
        ))

        changes = differ.diff(_make_scene(
            objects=[_make_obj(1, moving=False)], classes=["person"]
        ))
        assert any("motion_stop" in c for c in changes)

    def test_class_gone(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(classes=["person", "car"]))

        changes = differ.diff(_make_scene(classes=["person"]))
        assert any("class_gone" in c and "car" in c for c in changes)

    def test_center_cleared(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(center=True))

        changes = differ.diff(_make_scene(center=False))
        assert any("center_cleared" in c for c in changes)

    def test_reset(self):
        differ = SceneDiffer()
        differ.diff(_make_scene(objects=[_make_obj(1)], classes=["person"]))
        differ.reset()
        # After reset, next diff should treat as first scene
        changes = differ.diff(_make_scene())
        assert changes == []
