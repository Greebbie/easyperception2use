"""MJPEG video streamer for browser display.

Reads annotated frames from a shared file written by the perception pipeline,
ensuring perfect sync between bounding boxes and video.
"""

import os
import sys
import time
import threading
from typing import Optional

import cv2
import numpy as np

# Add parent dir for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHARED_FRAME_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_demo_frame.jpg",
)


class MJPEGStreamer:
    """Reads annotated frames from the pipeline's shared file and yields MJPEG data."""

    def __init__(self, fps: int = 12, jpeg_quality: int = 75):
        self._fps = fps
        self._jpeg_quality = jpeg_quality
        self._watchpoint_zones: list[dict] = []
        self._alert_active: bool = False
        self._alert_expire: float = 0.0
        self._lock = threading.Lock()
        self._frame_path = SHARED_FRAME_PATH

    def set_watchpoint_zones(self, zones: list[dict]) -> None:
        with self._lock:
            self._watchpoint_zones = zones

    def trigger_alert(self, duration: float = 2.0) -> None:
        with self._lock:
            self._alert_active = True
            self._alert_expire = time.time() + duration

    def generate_frames(self):
        """Yield MJPEG frames as bytes."""
        interval = 1.0 / self._fps
        last_data = None

        while True:
            start = time.time()

            # Read the shared annotated frame from pipeline
            frame_data = self._read_shared_frame()
            if frame_data is not None:
                last_data = frame_data
            elif last_data is None:
                # No frame available yet — send a placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    placeholder, "Waiting for pipeline...",
                    (120, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (100, 100, 100), 2, cv2.LINE_AA,
                )
                _, jpeg = cv2.imencode(".jpg", placeholder)
                last_data = jpeg.tobytes()
                time.sleep(0.5)

            # Decode frame to draw watchpoint zones if needed
            with self._lock:
                zones = list(self._watchpoint_zones)
                if self._alert_active and time.time() > self._alert_expire:
                    self._alert_active = False
                alert = self._alert_active

            if zones and last_data is not None:
                frame = cv2.imdecode(
                    np.frombuffer(last_data, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    self._draw_zones(frame, zones, alert)
                    _, jpeg = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                    )
                    last_data = jpeg.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                last_data +
                b"\r\n"
            )

            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _read_shared_frame(self) -> Optional[bytes]:
        """Read the latest annotated frame JPEG from the shared file."""
        try:
            if os.path.exists(self._frame_path):
                with open(self._frame_path, "rb") as f:
                    data = f.read()
                if len(data) > 100:  # valid JPEG
                    return data
        except (IOError, OSError):
            pass
        return None

    def _draw_zones(self, frame, zones: list[dict], alert: bool) -> None:
        pass

    def release(self) -> None:
        pass
