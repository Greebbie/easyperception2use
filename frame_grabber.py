"""Independent-thread frame grabber with network stream reconnection."""

import cv2
import threading
import time
from typing import Optional

import numpy as np


class FrameGrabber:
    """
    Independent thread frame reader — keeps only the latest frame.

    Solves the camera buffer accumulation problem, especially critical
    for network streams where YOLO inference time causes buffer backlog.
    """

    def __init__(
        self,
        source: int | str,
        max_retries: int = 5,
        retry_interval: float = 3.0,
    ):
        """
        Args:
            source: 0 (USB), "rtsp://..." (network stream), "video.mp4" (local file)
            max_retries: max reconnect attempts for network streams
            retry_interval: seconds between reconnect attempts
        """
        self.source = source
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.is_network_stream = isinstance(source, str) and (
            source.startswith("rtsp://")
            or source.startswith("http://")
            or source.startswith("https://")
        )

        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        self._connect()
        self._start_thread()

    def _connect(self) -> bool:
        """Open camera/video source. Retries on failure for network streams."""
        retries = 0
        while retries <= self.max_retries:
            self._cap = cv2.VideoCapture(self.source)
            if self.is_network_stream:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._cap.isOpened():
                self._connected = True
                print(f"[FrameGrabber] Connected: {self.source}")
                return True
            retries += 1
            if retries <= self.max_retries:
                print(
                    f"[FrameGrabber] Connection failed, retrying in "
                    f"{self.retry_interval}s ({retries}/{self.max_retries})"
                )
                time.sleep(self.retry_interval)
        print("[FrameGrabber] Connection failed, max retries reached")
        self._connected = False
        return False

    def _start_thread(self) -> None:
        """Start the background frame-reading thread."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        """Background thread: continuously read frames, keep only the latest."""
        consecutive_failures = 0
        while self._running:
            if not self._connected or self._cap is None or not self._cap.isOpened():
                if self.is_network_stream:
                    print("[FrameGrabber] Connection lost, attempting reconnect...")
                    if self._connect():
                        consecutive_failures = 0
                        continue
                    else:
                        break
                else:
                    break

            ok, frame = self._cap.read()
            if ok:
                consecutive_failures = 0
                with self._frame_lock:
                    self._frame = frame
            else:
                consecutive_failures += 1
                if not self.is_network_stream:
                    # Local file ended or USB disconnected
                    self._running = False
                    break
                if consecutive_failures > 30:
                    self._connected = False
                    consecutive_failures = 0

    def get_latest(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame (non-blocking).

        Returns:
            (success, frame): frame may be None if no frame is available yet.
        """
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get_frame_size(self) -> tuple[int, int] | None:
        """Get frame dimensions (width, height), or None if not connected."""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return None

    def is_alive(self) -> bool:
        """Check if the grabber is still running and connected."""
        return self._running and self._connected

    def switch_source(self, new_source: int | str) -> bool:
        """
        Switch to a new video source without creating a new FrameGrabber.

        Thread-safe: stops the read thread, swaps the capture, restarts.

        Args:
            new_source: new source (camera ID, RTSP URL, or file path)

        Returns:
            True if the new source connected successfully.
        """
        print(f"[FrameGrabber] Switching source to: {new_source}")

        # Stop the read thread
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        # Release old capture
        if self._cap:
            self._cap.release()

        # Update source info
        self.source = new_source
        self.is_network_stream = isinstance(new_source, str) and (
            new_source.startswith("rtsp://")
            or new_source.startswith("http://")
            or new_source.startswith("https://")
        )

        # Clear stale frame
        with self._frame_lock:
            self._frame = None

        # Connect to new source
        if self._connect():
            self._start_thread()
            return True

        print(f"[FrameGrabber] Failed to switch to: {new_source}")
        self._connected = False
        return False

    def release(self) -> None:
        """Release all resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        print("[FrameGrabber] Released")
