"""Tests for Visualizer — OpenCV overlay drawing."""

import time

import cv2
import numpy as np
import pytest

from visualizer import Visualizer


def _make_scene(objects=None, risk="low", classes=None, center=False,
                moving_count=0):
    """Build a minimal scene dict for visualizer tests."""
    if objects is None:
        objects = []
    if classes is None:
        classes = [o["class"] for o in objects] if objects else []
    return {
        "objects": objects,
        "scene": {
            "object_count": len(objects),
            "risk_level": risk,
            "classes_present": classes,
            "center_occupied": center,
            "moving_count": moving_count,
        },
    }


def _make_obj(track_id=1, cls="person", region="middle_center",
              direction="stationary", moving=False, depth=None):
    obj = {
        "track_id": track_id,
        "class": cls,
        "bbox_px": {"x1": 100, "y1": 100, "x2": 300, "y2": 400},
        "position": {"region": region},
        "motion": {"direction": direction, "moving": moving},
    }
    if depth is not None:
        obj["depth"] = depth
    return obj


class TestVisualizerDraw:
    def test_draw_returns_frame(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene()
        result = viz.draw(frame, scene)
        assert isinstance(result, np.ndarray)
        assert result.shape == (720, 1280, 3)

    def test_draw_modifies_frame(self):
        """Draw should add overlay content (frame should not be all zeros)."""
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene(classes=["person"])
        result = viz.draw(frame, scene)
        assert result.sum() > 0  # something was drawn

    def test_draw_with_objects(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene(
            objects=[_make_obj(1, "person", moving=True, direction="left")],
            classes=["person"],
            moving_count=1,
        )
        result = viz.draw(frame, scene)
        assert result.sum() > 0

    def test_draw_with_depth(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene(
            objects=[_make_obj(1, depth={"value": 0.3, "label": "near"})],
            classes=["person"],
        )
        result = viz.draw(frame, scene)
        assert result.sum() > 0

    def test_draw_all_risk_levels(self):
        viz = Visualizer()
        for risk in ("clear", "low", "medium", "high"):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            scene = _make_scene(risk=risk, classes=["person"])
            result = viz.draw(frame, scene)
            assert result.shape == (720, 1280, 3)

    def test_draw_multiple_objects(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objs = [
            _make_obj(1, "person", moving=True, direction="right"),
            _make_obj(2, "car", moving=False),
            _make_obj(3, "dog", moving=True, direction="up"),
        ]
        scene = _make_scene(objects=objs, classes=["person", "car", "dog"],
                            moving_count=2)
        result = viz.draw(frame, scene)
        assert result.sum() > 0

    def test_fps_tracking(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene()
        # Draw several times to populate fps_history
        for _ in range(5):
            viz.draw(frame.copy(), scene)
            time.sleep(0.01)
        assert len(viz.fps_history) == 5

    def test_draw_empty_scene(self):
        viz = Visualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        scene = _make_scene(risk="clear")
        result = viz.draw(frame, scene)
        assert result.shape == (720, 1280, 3)

    def test_small_frame(self):
        """Should not crash on small frames."""
        viz = Visualizer()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        scene = _make_scene(objects=[_make_obj(1)], classes=["person"])
        result = viz.draw(frame, scene)
        assert result.shape == (100, 100, 3)

    def test_risk_colors_defined(self):
        assert "clear" in Visualizer.RISK_COLORS
        assert "low" in Visualizer.RISK_COLORS
        assert "medium" in Visualizer.RISK_COLORS
        assert "high" in Visualizer.RISK_COLORS
