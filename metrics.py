"""Pipeline performance metrics and health monitoring."""

import time
from collections import deque
from threading import Lock


class PipelineMetrics:
    """
    Tracks pipeline latency, FPS, and health state.

    Thread-safe: can be read from WebSocket/API threads while
    being written from the main processing loop.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: number of recent frames to keep for statistics
        """
        self._lock = Lock()
        self._window_size = window_size
        self._frame_times: deque[float] = deque(maxlen=window_size)
        self._latencies: dict[str, deque[float]] = {}
        self._start_time = time.time()
        self._frame_count = 0
        self._state = "initializing"
        self._degraded_modules: list[str] = []

    def record_frame(self, latency_breakdown: dict[str, float]) -> None:
        """
        Record metrics for one processed frame.

        Args:
            latency_breakdown: dict of stage_name -> duration_ms
                e.g., {"grab_to_detect": 12.5, "detect_to_depth": 45.0}
        """
        now = time.time()
        with self._lock:
            self._frame_times.append(now)
            self._frame_count += 1
            for stage, ms in latency_breakdown.items():
                if stage not in self._latencies:
                    self._latencies[stage] = deque(maxlen=self._window_size)
                self._latencies[stage].append(ms)

    def set_state(self, state: str) -> None:
        """Set pipeline state: initializing / running / degraded / error / stopped."""
        with self._lock:
            self._state = state

    def add_degraded_module(self, module: str) -> None:
        """Mark a module as degraded (e.g., 'depth')."""
        with self._lock:
            if module not in self._degraded_modules:
                self._degraded_modules.append(module)

    def remove_degraded_module(self, module: str) -> None:
        """Remove a module from the degraded list."""
        with self._lock:
            if module in self._degraded_modules:
                self._degraded_modules.remove(module)

    def get_fps(self) -> float:
        """Calculate current FPS from recent frame timestamps."""
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt <= 0:
                return 0.0
            return (len(self._frame_times) - 1) / dt

    def get_latency_p50(self, stage: str = "total") -> float:
        """Get median latency for a stage in ms."""
        return self._get_percentile(stage, 0.5)

    def get_latency_p95(self, stage: str = "total") -> float:
        """Get 95th percentile latency for a stage in ms."""
        return self._get_percentile(stage, 0.95)

    def _get_percentile(self, stage: str, p: float) -> float:
        """Calculate percentile for a latency stage."""
        with self._lock:
            if stage not in self._latencies or not self._latencies[stage]:
                return 0.0
            values = sorted(self._latencies[stage])
            idx = min(int(len(values) * p), len(values) - 1)
            return round(values[idx], 2)

    def get_health(self) -> dict:
        """Get comprehensive health status."""
        with self._lock:
            state = self._state
            degraded = list(self._degraded_modules)
            uptime = time.time() - self._start_time
            frame_count = self._frame_count

        fps = self.get_fps()
        return {
            "state": state,
            "degraded_modules": degraded,
            "uptime_sec": round(uptime, 1),
            "fps": round(fps, 1),
            "total_frames": frame_count,
            "latency_p50_ms": self.get_latency_p50("total"),
            "latency_p95_ms": self.get_latency_p95("total"),
        }

    def get_pipeline_info(self) -> dict:
        """Get pipeline info for inclusion in scene JSON output."""
        with self._lock:
            return {
                "state": self._state,
                "degraded_modules": list(self._degraded_modules),
                "uptime_sec": round(time.time() - self._start_time, 1),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._frame_times.clear()
            self._latencies.clear()
            self._frame_count = 0
            self._start_time = time.time()
