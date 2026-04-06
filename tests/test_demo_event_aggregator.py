"""Tests for EventAggregator: event parsing, severity, compound detection."""

import time
from unittest.mock import patch

import pytest

from demo.event_aggregator import EventAggregator, SceneEvent
from demo.watchpoint import Alert


class TestSceneChangeIngestion:
    """Test parsing of SceneDiffer change strings."""

    def test_parse_object_entered(self):
        agg = EventAggregator()
        events = agg.ingest_scene_changes(
            ["object_entered: bottle #5 appeared in middle_center"],
            timestamp=1000.0,
        )
        assert len(events) == 1
        e = events[0]
        assert e.event_type == "object_entered"
        assert e.severity == "info"
        assert e.source == "perception"
        assert e.track_id == 5
        assert e.object_class == "bottle"
        assert e.region == "middle_center"

    def test_object_left_filtered_from_ui(self):
        """object_left is too noisy (YOLO flickering) — stored but not returned."""
        agg = EventAggregator()
        events = agg.ingest_scene_changes(
            ["object_left: cup #3 disappeared from top_left"],
            timestamp=1000.0,
        )
        assert len(events) == 0
        # Still in history for LLM
        assert len(agg.get_recent()) == 1
        assert agg.get_recent()[0].event_type == "object_left"

    def test_noisy_events_filtered_from_return(self):
        """region_change, risk_change etc. are stored in history but NOT returned."""
        agg = EventAggregator()
        events = agg.ingest_scene_changes(
            ["region_change: laptop #7 moved from top_left to middle_center"],
            timestamp=1000.0,
        )
        # Filtered from returned events (too noisy for UI)
        assert len(events) == 0
        # But still stored in history for LLM context
        assert len(agg.get_recent()) == 1
        assert agg.get_recent()[0].event_type == "region_change"

    def test_risk_change_filtered_from_return(self):
        agg = EventAggregator()
        events = agg.ingest_scene_changes(
            ["risk_change: clear → medium"],
            timestamp=1000.0,
        )
        assert len(events) == 0
        assert len(agg.get_recent()) == 1

    def test_significant_events_returned_noisy_filtered(self):
        """Only significant events (object_entered, new_class) are returned."""
        agg = EventAggregator()
        events = agg.ingest_scene_changes([
            "object_entered: bottle #1 appeared in middle_center",
            "new_class: bottle detected for first time",
            "center_occupied: object entered center region",
        ], timestamp=1000.0)
        # Only object_entered and new_class are significant
        assert len(events) == 2
        types = {e.event_type for e in events}
        assert types == {"object_entered", "new_class"}

    def test_parse_cell_phone_class(self):
        agg = EventAggregator()
        events = agg.ingest_scene_changes(
            ["object_entered: cell phone #12 appeared in top_right"],
            timestamp=1000.0,
        )
        assert len(events) == 1
        assert events[0].object_class == "cell phone"
        assert events[0].track_id == 12

    def test_empty_changes(self):
        agg = EventAggregator()
        events = agg.ingest_scene_changes([], timestamp=1000.0)
        assert events == []

    def test_invalid_change_string(self):
        agg = EventAggregator()
        events = agg.ingest_scene_changes(["no colon here"], timestamp=1000.0)
        assert events == []


class TestWatchpointAlertIngestion:
    """Test conversion of watchpoint Alerts to SceneEvents."""

    def test_ingest_zone_entry_alert(self):
        agg = EventAggregator()
        alert = Alert(
            timestamp=1000.0,
            zone_id="abc",
            zone_label="Table",
            object_class="bottle",
            track_id=1,
            message="bottle detected at Table",
            event_type="zone_entry",
            severity="info",
        )
        events = agg.ingest_watchpoint_alerts([alert])
        assert len(events) == 1
        e = events[0]
        assert e.event_type == "zone_entry"
        assert e.source == "watchpoint"
        assert e.details["zone_label"] == "Table"

    def test_ingest_dwell_warning(self):
        agg = EventAggregator()
        alert = Alert(
            timestamp=1000.0,
            zone_id="abc",
            zone_label="Table",
            object_class="bottle",
            track_id=1,
            message="bottle at Table for 31s",
            event_type="dwell_warning",
            severity="warning",
            dwell_sec=31.0,
        )
        events = agg.ingest_watchpoint_alerts([alert])
        assert events[0].severity == "warning"
        assert events[0].details["dwell_sec"] == 31.0

    def test_ingest_object_removed(self):
        agg = EventAggregator()
        alert = Alert(
            timestamp=1000.0,
            zone_id="abc",
            zone_label="Table",
            object_class="cell phone",
            track_id=5,
            message="cell phone removed from Table",
            event_type="object_removed",
            severity="warning",
            dwell_sec=45.0,
        )
        events = agg.ingest_watchpoint_alerts([alert])
        assert events[0].event_type == "object_removed"
        assert events[0].object_class == "cell phone"


class TestEventHistory:
    """Test event history and retrieval."""

    def test_get_recent_respects_count(self):
        agg = EventAggregator()
        for i in range(10):
            agg.ingest_scene_changes(
                [f"object_entered: bottle #{i} appeared in middle_center"],
                timestamp=1000.0 + i,
            )
        recent = agg.get_recent(count=5)
        assert len(recent) == 5

    def test_history_max_size(self):
        agg = EventAggregator(max_history=10)
        for i in range(20):
            agg.ingest_scene_changes(
                [f"object_entered: bottle #{i} appeared in middle_center"],
                timestamp=1000.0 + i,
            )
        assert len(agg.get_recent(count=100)) == 10


class TestLLMSummary:
    """Test get_summary_for_llm."""

    @patch("demo.event_aggregator.time.time")
    def test_summary_within_window(self, mock_time):
        agg = EventAggregator()
        mock_time.return_value = 1000.0

        agg.ingest_watchpoint_alerts([
            Alert(
                timestamp=980.0, zone_id="a", zone_label="T",
                object_class="bottle", track_id=1,
                message="dwell", event_type="dwell_warning",
                severity="warning", dwell_sec=31.0,
            ),
        ])

        summary = agg.get_summary_for_llm(window_sec=60.0)
        assert summary["total_events"] == 1
        assert summary["highest_severity"] == "warning"
        assert summary["active_dwell_count"] == 1

    @patch("demo.event_aggregator.time.time")
    def test_summary_excludes_old_events(self, mock_time):
        agg = EventAggregator()

        # Old event
        agg.ingest_scene_changes(
            ["object_entered: bottle #1 appeared in middle_center"],
            timestamp=500.0,
        )

        mock_time.return_value = 1000.0
        summary = agg.get_summary_for_llm(window_sec=60.0)
        assert summary["total_events"] == 0

    def test_summary_empty(self):
        agg = EventAggregator()
        summary = agg.get_summary_for_llm()
        assert summary["total_events"] == 0
        assert summary["highest_severity"] == "info"


class TestCompoundEvents:
    """Test compound event detection."""

    @patch("demo.event_aggregator.time.time")
    def test_dwell_then_removal_is_suspicious(self, mock_time):
        agg = EventAggregator()
        mock_time.return_value = 1000.0

        # Dwell alert
        agg.ingest_watchpoint_alerts([
            Alert(
                timestamp=990.0, zone_id="abc", zone_label="Table",
                object_class="bottle", track_id=1,
                message="dwell", event_type="dwell_alert",
                severity="alert", dwell_sec=62.0,
            ),
        ])

        # Removal
        agg.ingest_watchpoint_alerts([
            Alert(
                timestamp=995.0, zone_id="abc", zone_label="Table",
                object_class="bottle", track_id=1,
                message="removed", event_type="object_removed",
                severity="warning", dwell_sec=67.0,
            ),
        ])

        compounds = agg.detect_compound_events()
        assert len(compounds) == 1
        assert compounds[0].event_type == "suspicious_removal"
        assert compounds[0].severity == "critical"


class TestSceneEventSerialization:
    """Test SceneEvent.to_dict."""

    def test_to_dict_contains_all_fields(self):
        event = SceneEvent(
            event_id="abc",
            timestamp=1000.0,
            event_type="dwell_warning",
            severity="warning",
            source="watchpoint",
            track_id=1,
            object_class="bottle",
            region="middle_center",
            description="bottle at zone for 30s",
            details={"dwell_sec": 30.0},
        )
        d = event.to_dict()
        assert d["event_id"] == "abc"
        assert d["event_type"] == "dwell_warning"
        assert d["severity"] == "warning"
        assert d["source"] == "watchpoint"
        assert d["track_id"] == 1
        assert d["object_class"] == "bottle"
        assert "time_str" in d
