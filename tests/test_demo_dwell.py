"""Tests for WatchpointMonitor dwell tracking and object state detection."""

import time
from unittest.mock import patch

import pytest

from demo.watchpoint import WatchpointMonitor, Alert


def _make_scene(objects: list[dict]) -> dict:
    return {
        "objects": objects,
        "scene": {
            "object_count": len(objects),
            "classes_present": list({o["class"] for o in objects}),
            "risk_level": "clear",
        },
    }


def _make_obj(track_id: int, cls: str, x: float, y: float, moving: bool = False) -> dict:
    return {
        "track_id": track_id,
        "class": cls,
        "confidence": 0.9,
        "position": {
            "rel_x": x, "rel_y": y,
            "smoothed_x": x, "smoothed_y": y,
            "rel_size": 0.05, "region": "middle_center",
        },
        "motion": {
            "direction": "stationary", "speed": 0.0,
            "vx": 0.0, "vy": 0.0, "moving": moving, "reliable": True,
        },
    }


class TestZoneEntry:
    def test_object_entering_zone_generates_alert(self):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle", label="Table")
        alerts = wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        assert len(alerts) == 1
        assert alerts[0].event_type == "zone_entry"
        assert alerts[0].object_class == "bottle"

    def test_no_alert_for_wrong_class(self):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")
        alerts = wp.check_scene(_make_scene([_make_obj(1, "cup", 0.5, 0.5)]))
        assert len(alerts) == 0

    def test_object_outside_zone_no_alert(self):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 0.3, 0.3, target_class="bottle")
        alerts = wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.8, 0.8)]))
        assert len(alerts) == 0

    def test_any_class_matches_all(self):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="any")
        alerts = wp.check_scene(_make_scene([
            _make_obj(1, "bottle", 0.5, 0.5),
            _make_obj(2, "cup", 0.3, 0.3),
        ]))
        assert len(alerts) == 2  # One entry per class


class TestDwellTracking:
    """Test dwell time tracked by CLASS (not track_id) — survives YOLO flickering."""

    @patch("demo.watchpoint.time.time")
    def test_dwell_warning_after_threshold(self, mock_time):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # t=31: dwell_warning should fire
        mock_time.return_value = 1031.0
        alerts = wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        dwell = [a for a in alerts if a.event_type == "dwell_warning"]
        assert len(dwell) == 1
        assert dwell[0].severity == "warning"
        assert dwell[0].dwell_sec >= 30.0

    @patch("demo.watchpoint.time.time")
    def test_dwell_survives_track_id_change(self, mock_time):
        """Key test: dwell continues even when track_id changes (YOLO flickering)."""
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        # t=0: bottle enters with track_id=1
        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # t=15: same bottle, different track_id (YOLO flicker)
        mock_time.return_value = 1015.0
        wp.check_scene(_make_scene([_make_obj(99, "bottle", 0.5, 0.5)]))

        # t=31: dwell should still fire (based on class, not track_id)
        mock_time.return_value = 1031.0
        alerts = wp.check_scene(_make_scene([_make_obj(200, "bottle", 0.5, 0.5)]))
        dwell = [a for a in alerts if a.event_type == "dwell_warning"]
        assert len(dwell) == 1

    @patch("demo.watchpoint.time.time")
    def test_dwell_alert_after_60s(self, mock_time):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        mock_time.return_value = 1061.0
        alerts = wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        dwell = [a for a in alerts if a.event_type == "dwell_alert"]
        assert len(dwell) == 1
        assert dwell[0].severity == "alert"

    @patch("demo.watchpoint.time.time")
    def test_dwell_fires_only_once_per_level(self, mock_time):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        mock_time.return_value = 1031.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        mock_time.return_value = 1035.0
        alerts = wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        warning_alerts = [a for a in alerts if a.event_type == "dwell_warning"]
        assert len(warning_alerts) == 0

    @patch("demo.watchpoint.time.time")
    def test_grace_period_prevents_false_removal(self, mock_time):
        """Object disappearing briefly should NOT reset dwell."""
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # Simulate frames with bottle present (updates last_seen)
        mock_time.return_value = 1009.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # t=10: bottle disappears for 2s (within 5s grace)
        mock_time.return_value = 1010.0
        wp.check_scene(_make_scene([]))

        # t=12: bottle reappears (within grace period)
        mock_time.return_value = 1012.0
        wp.check_scene(_make_scene([_make_obj(5, "bottle", 0.5, 0.5)]))

        # t=31: dwell_warning should fire (counted from t=0, not t=12)
        mock_time.return_value = 1031.0
        alerts = wp.check_scene(_make_scene([_make_obj(5, "bottle", 0.5, 0.5)]))
        dwell = [a for a in alerts if a.event_type == "dwell_warning"]
        assert len(dwell) == 1


class TestObjectRemoval:
    @patch("demo.watchpoint.time.time")
    def test_object_removed_after_grace_period(self, mock_time):
        wp = WatchpointMonitor()
        wp.DEBOUNCE_SEC = 0
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # Update last_seen right before disappearance
        mock_time.return_value = 1004.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        # Object disappears
        mock_time.return_value = 1005.0
        wp.check_scene(_make_scene([]))

        # Within grace period (only 1s gone) — should NOT fire
        mock_time.return_value = 1006.0
        alerts = wp.check_scene(_make_scene([]))
        removal = [a for a in alerts if a.event_type == "object_removed"]
        assert len(removal) == 0

        # After grace period (16s since last seen at t=1004, grace=15s)
        mock_time.return_value = 1020.0
        alerts = wp.check_scene(_make_scene([]))
        removal = [a for a in alerts if a.event_type == "object_removed"]
        assert len(removal) == 1
        assert removal[0].severity == "warning"


class TestDwellInfo:
    @patch("demo.watchpoint.time.time")
    def test_get_dwell_times_by_class(self, mock_time):
        wp = WatchpointMonitor()
        zone_id = wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="any")

        mock_time.return_value = 1000.0
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))

        mock_time.return_value = 1020.0
        dwell = wp.get_dwell_times()
        assert zone_id in dwell
        assert len(dwell[zone_id]) == 1
        assert dwell[zone_id][0]["class"] == "bottle"
        assert dwell[zone_id][0]["dwell_sec"] == pytest.approx(20.0, abs=1.0)


class TestAlertSerialization:
    def test_alert_to_dict_has_all_fields(self):
        alert = Alert(
            timestamp=1000.0, zone_id="abc", zone_label="Table",
            object_class="bottle", track_id=1,
            message="bottle 在 Table 已停留 35s",
            event_type="dwell_warning", severity="warning", dwell_sec=35.2,
        )
        d = alert.to_dict()
        assert d["event_type"] == "dwell_warning"
        assert d["severity"] == "warning"
        assert d["dwell_sec"] == 35.2


class TestZoneCRUD:
    def test_clear_zones_resets_everything(self):
        wp = WatchpointMonitor()
        wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        wp.clear_zones()
        assert len(wp._class_dwell) == 0
        assert len(wp._dwell_alerted) == 0

    def test_remove_zone_cleans_up(self):
        wp = WatchpointMonitor()
        zone_id = wp.add_zone(0.0, 0.0, 1.0, 1.0, target_class="bottle")
        wp.check_scene(_make_scene([_make_obj(1, "bottle", 0.5, 0.5)]))
        wp.remove_zone(zone_id)
        assert zone_id not in wp._class_dwell
