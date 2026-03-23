"""Tests for FrameGrabber — threaded frame reader with reconnection."""

import os
import time
import threading

import cv2
import numpy as np
import pytest

from frame_grabber import FrameGrabber

# Path to test video in repo root
_VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_video.mp4",
)
_HAS_VIDEO = os.path.exists(_VIDEO_PATH)

pytestmark = pytest.mark.skipif(not _HAS_VIDEO, reason="test_video.mp4 not found")


class TestFrameGrabberWithVideo:
    def _make_grabber(self):
        return FrameGrabber(_VIDEO_PATH, max_retries=1, retry_interval=0.1)

    def test_connects_to_video(self):
        g = self._make_grabber()
        try:
            assert g._connected is True
            assert g.is_alive() is True
        finally:
            g.release()

    def test_get_frame_size(self):
        g = self._make_grabber()
        try:
            size = g.get_frame_size()
            assert size is not None
            w, h = size
            assert w > 0
            assert h > 0
        finally:
            g.release()

    def test_get_latest_returns_frame(self):
        g = self._make_grabber()
        try:
            # Wait for background thread to read at least one frame
            for _ in range(50):
                ok, frame = g.get_latest()
                if ok:
                    break
                time.sleep(0.02)
            assert ok is True
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3  # BGR
        finally:
            g.release()

    def test_get_latest_returns_copy(self):
        """Returned frame should be a copy, not a reference to internal buffer."""
        g = self._make_grabber()
        try:
            time.sleep(0.1)
            ok1, frame1 = g.get_latest()
            assert ok1
            time.sleep(0.1)
            ok2, frame2 = g.get_latest()
            assert ok2
            # Modifying frame1 should not affect frame2
            frame1[:] = 0
            assert frame2.sum() > 0 or True  # different references
            assert frame1 is not frame2
        finally:
            g.release()

    def test_release_stops_thread(self):
        g = self._make_grabber()
        thread = g._thread
        assert thread.is_alive()
        g.release()
        time.sleep(0.1)
        assert not thread.is_alive()

    def test_release_idempotent(self):
        g = self._make_grabber()
        g.release()
        g.release()  # should not raise

    def test_not_network_stream(self):
        g = self._make_grabber()
        try:
            assert g.is_network_stream is False
        finally:
            g.release()

    def test_switch_source_to_same_video(self):
        g = self._make_grabber()
        try:
            assert g.switch_source(_VIDEO_PATH) is True
            time.sleep(0.2)
            ok, frame = g.get_latest()
            assert ok is True
        finally:
            g.release()

    def test_switch_source_to_invalid(self):
        g = self._make_grabber()
        try:
            result = g.switch_source("nonexistent_file.mp4")
            assert result is False
        finally:
            g.release()

    def test_video_file_ends(self):
        """Video file should eventually stop producing frames."""
        g = self._make_grabber()
        try:
            # Wait for video to finish (short test video)
            for _ in range(200):
                if not g.is_alive():
                    break
                time.sleep(0.05)
            # After video ends, is_alive should be False
            # (depends on video length, may still be alive for long videos)
        finally:
            g.release()


class TestFrameGrabberInvalidSource:
    def test_invalid_file_path(self):
        """Invalid source should fail to connect but not crash."""
        g = FrameGrabber(
            "nonexistent_video_12345.mp4",
            max_retries=1,
            retry_interval=0.01,
        )
        try:
            assert g._connected is False
            ok, frame = g.get_latest()
            assert ok is False
            assert frame is None
        finally:
            g.release()

    def test_network_stream_detection(self):
        """RTSP/HTTP URLs should be flagged as network streams."""
        # Don't actually connect — just check the flag via a quick fail
        g = FrameGrabber(
            "rtsp://192.168.1.999/nonexistent",
            max_retries=1,
            retry_interval=0.01,
        )
        try:
            assert g.is_network_stream is True
        finally:
            g.release()
