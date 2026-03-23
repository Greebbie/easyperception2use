"""Tests for Kalman filter position/velocity smoothing."""

import pytest
from kalman_tracker import KalmanFilter2D, KalmanTracker


class TestKalmanFilter2D:
    def test_initialization(self):
        kf = KalmanFilter2D()
        sx, sy, vx, vy = kf.predict_and_update(0.5, 0.5, 1.0)
        assert sx == 0.5
        assert sy == 0.5
        assert vx == 0.0
        assert vy == 0.0

    def test_smoothing_reduces_noise(self):
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
        # First observation
        kf.predict_and_update(0.5, 0.5, 0.0)

        # Noisy observations around (0.5, 0.5)
        deviations = []
        for i in range(1, 20):
            noise_x = 0.5 + (0.02 if i % 2 == 0 else -0.02)
            noise_y = 0.5 + (0.01 if i % 3 == 0 else -0.01)
            sx, sy, _, _ = kf.predict_and_update(noise_x, noise_y, i * 0.1)
            deviations.append(abs(sx - 0.5) + abs(sy - 0.5))

        # Later observations should be smoother (closer to 0.5)
        early_avg = sum(deviations[:5]) / 5
        late_avg = sum(deviations[-5:]) / 5
        assert late_avg <= early_avg

    def test_velocity_estimation(self):
        kf = KalmanFilter2D(process_noise=0.1, measurement_noise=0.01)
        # Object moving right at constant speed
        for i in range(20):
            x = 0.1 + i * 0.01
            kf.predict_and_update(x, 0.5, i * 0.1)

        _, _, vx, vy = kf.predict_and_update(0.3, 0.5, 2.0)
        # vx should be positive (moving right)
        assert vx > 0
        # vy should be near zero
        assert abs(vy) < 0.05

    def test_stationary_object(self):
        kf = KalmanFilter2D()
        for i in range(10):
            sx, sy, vx, vy = kf.predict_and_update(0.5, 0.5, i * 0.1)

        assert abs(sx - 0.5) < 0.01
        assert abs(sy - 0.5) < 0.01
        assert abs(vx) < 0.01
        assert abs(vy) < 0.01


class TestKalmanTracker:
    def test_multiple_tracks(self):
        tracker = KalmanTracker()
        # Track 1 at (0.2, 0.3)
        sx1, sy1, _, _ = tracker.update(1, 0.2, 0.3, 0.0)
        # Track 2 at (0.8, 0.7)
        sx2, sy2, _, _ = tracker.update(2, 0.8, 0.7, 0.0)

        assert abs(sx1 - 0.2) < 0.01
        assert abs(sx2 - 0.8) < 0.01

    def test_remove_track(self):
        tracker = KalmanTracker()
        tracker.update(1, 0.5, 0.5, 0.0)
        tracker.remove(1)
        # After removal, updating track 1 should create new filter
        sx, sy, _, _ = tracker.update(1, 0.3, 0.3, 1.0)
        assert abs(sx - 0.3) < 0.01

    def test_cleanup(self):
        tracker = KalmanTracker(lost_timeout=1.0)
        tracker.update(1, 0.1, 0.1, 0.0)
        tracker.update(2, 0.5, 0.5, 0.0)
        tracker.update(3, 0.9, 0.9, 0.0)

        # Track 2 removed after grace period expires
        tracker.cleanup(active_ids={1, 3}, timestamp=2.0)
        assert 2 not in tracker._filters
        assert 1 in tracker._filters
        assert 3 in tracker._filters

    def test_cleanup_grace_period(self):
        tracker = KalmanTracker(lost_timeout=1.0)
        tracker.update(1, 0.1, 0.1, 0.0)
        tracker.update(2, 0.5, 0.5, 0.0)

        # Within grace period — track 2 should survive
        tracker.cleanup(active_ids={1}, timestamp=0.5)
        assert 2 in tracker._filters

        # After grace period — track 2 should be removed
        tracker.cleanup(active_ids={1}, timestamp=1.5)
        assert 2 not in tracker._filters

    def test_reset(self):
        tracker = KalmanTracker()
        tracker.update(1, 0.5, 0.5, 0.0)
        tracker.update(2, 0.5, 0.5, 0.0)
        tracker.reset()
        assert len(tracker._filters) == 0

    def test_predict_only_for_lost_track(self):
        """Grace-period predict-only should return a prediction, not None."""
        tracker = KalmanTracker()
        tracker.update(1, 0.5, 0.5, 0.0)
        tracker.update(1, 0.52, 0.5, 0.1)
        result = tracker.predict(1, 0.2)
        assert result is not None
        sx, sy, vx, vy = result
        # Position should have drifted from Kalman prediction
        assert sx != 0.0

    def test_predict_returns_none_for_unknown(self):
        tracker = KalmanTracker()
        assert tracker.predict(999, 1.0) is None

    def test_confidence_increases_with_observations(self):
        tracker = KalmanTracker()
        tracker.update(1, 0.5, 0.5, 0.0)
        early_pos, early_vel = tracker.get_confidence(1)
        for i in range(20):
            tracker.update(1, 0.5, 0.5, (i + 1) * 0.1)
        late_pos, late_vel = tracker.get_confidence(1)
        assert late_pos >= early_pos
        assert late_vel >= early_vel

    def test_predict_next_position(self):
        tracker = KalmanTracker()
        for i in range(10):
            tracker.update(1, 0.3 + i * 0.01, 0.5, i * 0.1)
        pred = tracker.predict_next(1, dt=0.1)
        assert pred is not None
        px, py = pred
        # Should predict slightly ahead of last position
        assert px > 0.38


class TestKalmanNumericalStability:
    def test_long_run_covariance_stays_psd(self):
        """After many iterations, P should remain positive semi-definite (no NaN/negative eigenvalues)."""
        import numpy as np
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.05)
        for i in range(1000):
            kf.predict_and_update(0.5 + 0.001 * (i % 10), 0.5, i * 0.1)

        eigenvalues = np.linalg.eigvalsh(kf.P)
        assert all(ev >= -1e-10 for ev in eigenvalues), f"Non-PSD covariance: {eigenvalues}"
        assert not np.any(np.isnan(kf.x)), f"NaN in state: {kf.x}"

    def test_confidence_stays_in_range(self):
        """Position and velocity confidence should always be in [0, 1]."""
        kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.05)
        for i in range(500):
            kf.predict_and_update(0.5 + 0.01 * (i % 5 - 2), 0.5, i * 0.1)
            pos_conf = kf.get_position_confidence()
            vel_conf = kf.get_velocity_confidence()
            assert 0.0 <= pos_conf <= 1.0, f"pos_conf={pos_conf} at iteration {i}"
            assert 0.0 <= vel_conf <= 1.0, f"vel_conf={vel_conf} at iteration {i}"
