"""Tests for ConfigGUI — logic methods only (no tkinter mainloop)."""

import pytest

from config_gui import ConfigGUI
from config import DEFAULT_CONFIG


class TestConfigGUILogic:
    """Test ConfigGUI methods that don't require a running tkinter mainloop."""

    def _make_gui(self):
        config = DEFAULT_CONFIG.copy()
        changes = []
        gui = ConfigGUI(config, lambda k, v: changes.append((k, v)))
        return gui, config, changes

    def test_construction(self):
        gui, config, changes = self._make_gui()
        assert gui._root is None
        assert gui._json_text is None

    def test_stop_without_start(self):
        gui, _, _ = self._make_gui()
        gui.stop()  # should not raise

    def test_set_status_no_root(self):
        gui, _, _ = self._make_gui()
        gui.set_status("Test")  # should not raise (no root)

    def test_update_json_no_root(self):
        gui, _, _ = self._make_gui()
        gui.update_json({"test": "data"})  # should not raise

    def test_apply_classes_empty(self):
        gui, config, changes = self._make_gui()
        gui._apply_classes("")
        assert config["filter_classes"] is None
        assert ("filter_classes", None) in changes

    def test_apply_classes_whitespace(self):
        gui, config, changes = self._make_gui()
        gui._apply_classes("  ,  ,  ")
        assert config["filter_classes"] is None

    def test_apply_classes_specific(self):
        gui, config, changes = self._make_gui()
        gui._apply_classes("person, car, dog")
        assert config["filter_classes"] == ["person", "car", "dog"]
        assert ("filter_classes", ["person", "car", "dog"]) in changes

    def test_apply_classes_single(self):
        gui, config, changes = self._make_gui()
        gui._apply_classes("bottle")
        assert config["filter_classes"] == ["bottle"]

    def test_switch_source_int(self):
        gui, _, changes = self._make_gui()
        gui._switch_source("0")
        assert ("source", 0) in changes

    def test_switch_source_string(self):
        gui, _, changes = self._make_gui()
        gui._switch_source("rtsp://192.168.1.1/stream")
        assert ("source", "rtsp://192.168.1.1/stream") in changes

    def test_switch_source_empty_ignored(self):
        gui, _, changes = self._make_gui()
        gui._switch_source("")
        assert len(changes) == 0

    def test_switch_source_whitespace_ignored(self):
        gui, _, changes = self._make_gui()
        gui._switch_source("   ")
        assert len(changes) == 0
