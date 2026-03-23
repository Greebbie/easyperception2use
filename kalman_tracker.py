"""Simple 2D Kalman filter for object position and velocity smoothing.

No external dependencies — hand-rolled using numpy only.
"""

import numpy as np


class KalmanFilter2D:
    """
    Single-object 2D Kalman filter.

    State vector: [x, y, vx, vy]
    Observation: [x, y]

    Smooths noisy detections and estimates velocity.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
    ):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self.R = np.eye(2, dtype=np.float64) * measurement_noise**2
        self._process_noise = process_noise
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self._initialized = False
        self._last_ts: float = 0.0

    def initialize(self, x: float, y: float, ts: float) -> None:
        """Set initial state from first observation."""
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.1
        self._last_ts = ts
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def predict(self, ts: float) -> tuple[float, float, float, float]:
        """Run prediction step only (no measurement update)."""
        if not self._initialized:
            return 0.0, 0.0, 0.0, 0.0

        dt = max(ts - self._last_ts, 0.001)
        self._last_ts = ts

        F = self._make_F(dt)
        Q = self._make_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        return (
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
            float(self.x[3]),
        )

    def predict_and_update(
        self, x_obs: float, y_obs: float, ts: float
    ) -> tuple[float, float, float, float]:
        """
        Run one Kalman cycle: predict, then update with observation.

        Args:
            x_obs: observed x position (camera-compensated, normalized 0-1)
            y_obs: observed y position (camera-compensated, normalized 0-1)
            ts: timestamp

        Returns:
            (smoothed_x, smoothed_y, vx, vy)
        """
        if not self._initialized:
            self.initialize(x_obs, y_obs, ts)
            return x_obs, y_obs, 0.0, 0.0

        dt = max(ts - self._last_ts, 0.001)
        self._last_ts = ts

        F = self._make_F(dt)
        Q = self._make_Q(dt)

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        z = np.array([x_obs, y_obs], dtype=np.float64)
        y_res = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        try:
            K = P_pred @ self.H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            self.x = x_pred
            self.P = P_pred
            return (
                float(self.x[0]), float(self.x[1]),
                float(self.x[2]), float(self.x[3]),
            )

        self.x = x_pred + K @ y_res
        # Joseph stabilized form (numerically stable, keeps P symmetric PSD)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        return (
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
            float(self.x[3]),
        )

    def get_position_confidence(self) -> float:
        """Position confidence from covariance (0-1, higher = more certain)."""
        if not self._initialized:
            return 0.0
        pos_trace = self.P[0, 0] + self.P[1, 1]
        return float(max(0.0, min(1.0, 1.0 - pos_trace * 10)))

    def get_velocity_confidence(self) -> float:
        """Velocity confidence from covariance (0-1, higher = more certain)."""
        if not self._initialized:
            return 0.0
        vel_trace = self.P[2, 2] + self.P[3, 3]
        return float(max(0.0, min(1.0, 1.0 - vel_trace * 5)))

    def predict_next_position(self, dt: float = 0.1) -> tuple[float, float]:
        """Predict position dt seconds into the future without modifying state."""
        if not self._initialized:
            return 0.0, 0.0
        nx = float(self.x[0] + self.x[2] * dt)
        ny = float(self.x[1] + self.x[3] * dt)
        return nx, ny

    def _make_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _make_Q(self, dt: float) -> np.ndarray:
        q = self._process_noise
        return np.array([
            [q * dt**3 / 3, 0, q * dt**2 / 2, 0],
            [0, q * dt**3 / 3, 0, q * dt**2 / 2],
            [q * dt**2 / 2, 0, q * dt, 0],
            [0, q * dt**2 / 2, 0, q * dt],
        ], dtype=np.float64)


class KalmanTracker:
    """
    Manages per-object Kalman filters for all tracked objects.

    Supports grace period: filters are kept alive for `lost_timeout` seconds
    after a track disappears, using predict-only to maintain state continuity.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
        lost_timeout: float = 1.0,
    ):
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._lost_timeout = lost_timeout
        self._filters: dict[int, KalmanFilter2D] = {}
        self._last_seen: dict[int, float] = {}
        self._update_count: dict[int, int] = {}

    def update(
        self, track_id: int, x: float, y: float, timestamp: float
    ) -> tuple[float, float, float, float]:
        """Update the Kalman filter for a tracked object."""
        if track_id not in self._filters:
            self._filters[track_id] = KalmanFilter2D(
                process_noise=self._process_noise,
                measurement_noise=self._measurement_noise,
            )
        self._last_seen[track_id] = timestamp
        self._update_count[track_id] = self._update_count.get(track_id, 0) + 1
        return self._filters[track_id].predict_and_update(x, y, timestamp)

    def predict(
        self, track_id: int, timestamp: float
    ) -> tuple[float, float, float, float] | None:
        """Run predict-only for a lost track. Returns None if no filter."""
        filt = self._filters.get(track_id)
        if filt is None or not filt.initialized:
            return None
        return filt.predict(timestamp)

    def get_confidence(self, track_id: int) -> tuple[float, float]:
        """Get (position_confidence, velocity_confidence) for a track."""
        filt = self._filters.get(track_id)
        if filt is None:
            return 0.0, 0.0
        return filt.get_position_confidence(), filt.get_velocity_confidence()

    def get_track_age(self, track_id: int) -> int:
        """Get number of updates for a track (track age in frames)."""
        return self._update_count.get(track_id, 0)

    def predict_next(self, track_id: int, dt: float = 0.1) -> tuple[float, float] | None:
        """Predict position dt seconds ahead without modifying state."""
        filt = self._filters.get(track_id)
        if filt is None or not filt.initialized:
            return None
        return filt.predict_next_position(dt)

    def remove(self, track_id: int) -> None:
        """Remove the Kalman filter for a lost track."""
        self._filters.pop(track_id, None)
        self._last_seen.pop(track_id, None)
        self._update_count.pop(track_id, None)

    def reset(self) -> None:
        """Clear all filters (e.g., on source switch)."""
        self._filters.clear()
        self._last_seen.clear()
        self._update_count.clear()

    def cleanup(self, active_ids: set[int], timestamp: float) -> None:
        """Remove filters for tracks not seen within the grace period."""
        expired = [
            tid for tid, last in self._last_seen.items()
            if tid not in active_ids
            and (timestamp - last) > self._lost_timeout
        ]
        for tid in expired:
            self._filters.pop(tid, None)
            self._last_seen.pop(tid, None)
            self._update_count.pop(tid, None)
