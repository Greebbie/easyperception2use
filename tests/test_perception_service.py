"""Tests for PerceptionService lifecycle and API (without real camera/YOLO)."""

import threading
import time

import pytest

from perception_service import PerceptionService
from config import DEFAULT_CONFIG


class TestPerceptionServiceInit:
    def test_default_config(self):
        svc = PerceptionService()
        assert svc.config is not None
        assert svc._running is False

    def test_custom_config(self):
        cfg = DEFAULT_CONFIG.copy()
        cfg["min_confidence"] = 0.8
        svc = PerceptionService(cfg)
        assert svc.config["min_confidence"] == 0.8

    def test_initial_state(self):
        svc = PerceptionService()
        assert svc.get_latest_scene() is None
        assert svc.get_status()["state"] == "initializing"


class TestPerceptionServiceSubscribe:
    def test_subscribe_returns_id(self):
        svc = PerceptionService()
        sub_id = svc.subscribe(lambda s: None)
        assert isinstance(sub_id, str)
        assert len(sub_id) == 8

    def test_unsubscribe(self):
        svc = PerceptionService()
        sub_id = svc.subscribe(lambda s: None)
        svc.unsubscribe(sub_id)
        assert sub_id not in svc._subscribers

    def test_notify_subscribers(self):
        svc = PerceptionService()
        received = []
        svc.subscribe(received.append)
        scene = {"objects": [], "timestamp": 1.0}
        svc._notify_subscribers(scene)
        assert len(received) == 1
        assert received[0] is scene

    def test_notify_with_filter(self):
        svc = PerceptionService()
        received = []
        svc.subscribe(received.append, filter_fn=lambda s: len(s["objects"]) > 0)
        svc._notify_subscribers({"objects": [], "timestamp": 1.0})
        assert len(received) == 0
        svc._notify_subscribers({"objects": [{"id": 1}], "timestamp": 2.0})
        assert len(received) == 1

    def test_notify_handles_subscriber_error(self):
        svc = PerceptionService()

        def bad_callback(s):
            raise RuntimeError("boom")

        svc.subscribe(bad_callback)
        # Should not raise
        svc._notify_subscribers({"objects": [], "timestamp": 1.0})

    def test_multiple_subscribers(self):
        svc = PerceptionService()
        r1, r2 = [], []
        svc.subscribe(r1.append)
        svc.subscribe(r2.append)
        svc._notify_subscribers({"objects": [], "timestamp": 1.0})
        assert len(r1) == 1
        assert len(r2) == 1


class TestPerceptionServiceConfig:
    def test_set_config(self):
        svc = PerceptionService()
        svc.set_config("min_confidence", 0.9)
        assert svc.config["min_confidence"] == 0.9

    def test_set_ego_motion_no_builder(self):
        """set_ego_motion should not crash when builder is not yet initialized."""
        svc = PerceptionService()
        svc.set_ego_motion(True, 0.1, 0.0)  # should not raise

    def test_switch_source_queued(self):
        svc = PerceptionService()
        assert svc.switch_source("rtsp://test") is True
        assert not svc._source_switch_queue.empty()

    def test_switch_source_queue_full(self):
        svc = PerceptionService()
        svc.switch_source("first")
        assert svc.switch_source("second") is False  # queue full


class TestPerceptionServiceLifecycle:
    def test_stop_without_start(self):
        """stop() on an unstarted service should not crash."""
        svc = PerceptionService()
        svc.stop()  # should not raise

    def test_double_start(self):
        """start() twice should be idempotent (only one thread)."""
        svc = PerceptionService()
        svc._running = True  # simulate already started
        svc.start()  # should return immediately
        assert svc._thread is None  # no new thread created

    def test_cleanup_nulls_resources(self):
        """_cleanup should null out references to prevent double-release."""
        svc = PerceptionService()
        svc._grabber = None
        svc._output_handler = None
        svc._cleanup()  # should not raise
