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
        # State: [x, y, vx, vy]
        self.x = np.zeros(4, dtype=np.float64)
        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 1.0
        # Measurement noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise**2
        # Process noise base
        self._process_noise = process_noise
        # Observation matrix: we observe [x, y]
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

    def predict_and_update(
        self, x_obs: float, y_obs: float, ts: float
    ) -> tuple[float, float, float, float]:
        """
        Run one Kalman cycle: predict, then update with observation.

        Args:
            x_obs: observed x position (normalized 0-1)
            y_obs: observed y position (normalized 0-1)
            ts: timestamp

        Returns:
            (smoothed_x, smoothed_y, vx, vy)
        """
        if not self._initialized:
            self.initialize(x_obs, y_obs, ts)
            return x_obs, y_obs, 0.0, 0.0

        dt = max(ts - self._last_ts, 0.001)
        self._last_ts = ts

        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Process noise (scaled by dt)
        q = self._process_noise
        Q = np.array([
            [q * dt**3 / 3, 0, q * dt**2 / 2, 0],
            [0, q * dt**3 / 3, 0, q * dt**2 / 2],
            [q * dt**2 / 2, 0, q * dt, 0],
            [0, q * dt**2 / 2, 0, q * dt],
        ], dtype=np.float64)

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Update
        z = np.array([x_obs, y_obs], dtype=np.float64)
        y_res = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return (
            float(self.x[0]),
            float(self.x[1]),
            float(self.x[2]),
            float(self.x[3]),
        )


class KalmanTracker:
    """
    Manages per-object Kalman filters for all tracked objects.

    Creates a new filter when a new track_id appears,
    removes it when the track is lost.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.05,
    ):
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._filters: dict[int, KalmanFilter2D] = {}

    def update(
        self, track_id: int, x: float, y: float, timestamp: float
    ) -> tuple[float, float, float, float]:
        """
        Update the Kalman filter for a tracked object.

        Args:
            track_id: object track ID
            x: observed normalized x position
            y: observed normalized y position
            timestamp: current time

        Returns:
            (smoothed_x, smoothed_y, smooth_vx, smooth_vy)
        """
        if track_id not in self._filters:
            self._filters[track_id] = KalmanFilter2D(
                process_noise=self._process_noise,
                measurement_noise=self._measurement_noise,
            )
        return self._filters[track_id].predict_and_update(x, y, timestamp)

    def remove(self, track_id: int) -> None:
        """Remove the Kalman filter for a lost track."""
        self._filters.pop(track_id, None)

    def reset(self) -> None:
        """Clear all filters (e.g., on source switch)."""
        self._filters.clear()

    def cleanup(self, active_ids: set[int]) -> None:
        """Remove filters for tracks no longer active."""
        lost = [tid for tid in self._filters if tid not in active_ids]
        for tid in lost:
            del self._filters[tid]
