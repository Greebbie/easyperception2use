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
            source: "auto" (auto-detect), 0 (USB), "rtsp://..." (network), "video.mp4" (file)
            max_retries: max reconnect attempts for network streams
            retry_interval: seconds between reconnect attempts
        """
        # Auto-detect camera if source is "auto"
        if source == "auto":
            detected = self._auto_detect_source()
            if detected is not None:
                source = detected
                print(f"[FrameGrabber] Auto-detected camera at index {source}")
            else:
                print("[FrameGrabber] Auto-detect failed, falling back to source=0")
                source = 0

        self.source = source
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.is_network_stream = isinstance(source, str) and (
            source.startswith("rtsp://")
            or source.startswith("http://")
            or source.startswith("https://")
        )
        self.is_video_file = isinstance(source, str) and (
            source.endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))
            and not self.is_network_stream
        )

        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        self._connect()
        self._start_thread()

    @staticmethod
    def _auto_detect_source() -> Optional[int]:
        """Scan camera indices 0-3 and return the first available device."""
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.release()
                return idx
            cap.release()
        return None

    def _connect(self) -> bool:
        """Open camera/video source. Retries on failure for network streams."""
        retries = 0
        while retries < self.max_retries:
            self._cap = cv2.VideoCapture(self.source)
            if self.is_network_stream:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._cap.isOpened():
                self._connected = True
                print(f"[FrameGrabber] Connected: {self.source}")
                return True
            retries += 1
            if retries < self.max_retries:
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
        # For video files, throttle to native FPS for real-time playback
        video_delay = 0.0
        if self.is_video_file and self._cap is not None:
            native_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if native_fps > 0:
                video_delay = 1.0 / native_fps

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
                if video_delay > 0:
                    time.sleep(video_delay)
            else:
                if self.is_video_file:
                    # Loop video files from the beginning
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    continue
                consecutive_failures += 1
                if consecutive_failures > 30:
                    if self.is_network_stream:
                        self._connected = False
                        consecutive_failures = 0
                    else:
                        # USB camera — give up after sustained failures
                        self._running = False
                        break
                else:
                    # Brief pause before retry (helps USB camera recovery)
                    time.sleep(0.05)

    def get_latest(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Get the latest frame (non-blocking).

        Returns a copy to prevent race with the background read thread.

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
