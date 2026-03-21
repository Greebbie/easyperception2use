"""Default configuration for the Perception Pipeline."""

DEFAULT_CONFIG: dict = {
    # =========================================================================
    # CORE — Always-active 2D pipeline
    # Detection + Tracking + Kalman + Output + WebSocket
    # =========================================================================

    # === Input ===
    "source": 0,                        # 0=USB, "rtsp://...", "video.mp4"

    # === Detection ===
    "model_path": "yolov8n.pt",         # or yolo11n.pt
    "min_confidence": 0.3,              # drop detections below this
    "filter_classes": None,             # None=all, or ["person", "car", "dog"]

    # === Frame Rate Control ===
    "process_fps": 10,                  # main loop detection frame rate

    # === Tracking ===
    "tracker": "bytetrack.yaml",        # or "botsort.yaml"
    "track_history_len": 10,            # how many frames of track history to keep
    "track_lost_timeout": 2.0,          # seconds before clearing lost track
    "motion_speed_threshold": 0.02,     # normalized speed threshold (rel/sec)

    # === Kalman Filter ===
    "kalman_process_noise": 0.01,       # process noise (higher = trust measurements more)
    "kalman_measurement_noise": 0.05,   # measurement noise (higher = smoother but laggier)

    # === Output Control ===
    "output_strategy": "hybrid",        # "every_frame" / "interval" / "on_change" / "hybrid"
    "output_interval_sec": 1.0,         # interval for interval/hybrid mode
    "output_change_threshold": 0.01,    # change threshold for on_change/hybrid mode
    "output_method": "print",           # "print" / "file" / "callback"
    "output_file_path": "scene_output.jsonl",

    # === Scene Analysis ===
    "risk_thresholds": {
        "high": 0.15,                   # center object >15% of frame = high risk
        "medium": 0.05,                 # >5% = medium risk
    },
    "center_region_x": (0.33, 0.67),    # center region x range
    "center_region_y": (0.4, 0.7),      # center region y range

    # === Network Stream ===
    "max_retries": 5,                   # reconnect attempts for network streams
    "retry_interval": 3.0,             # reconnect interval (seconds)

    # === WebSocket Server ===
    "ws_enabled": False,                # opt-in, off by default
    "ws_host": "0.0.0.0",              # bind address
    "ws_port": 18790,                   # port (avoids OpenClaw Gateway 18789)

    # =========================================================================
    # ENHANCEMENT — Opt-in modules (not part of core 2D pipeline)
    # =========================================================================

    # === Depth Estimation (--depth flag) ===
    # Pluggable module. Provides relative depth only (not metric distance).
    # Useful for coarse near/far ordering; not reliable for closed-loop grasping.
    # Requires: pip install -r requirements-depth.txt
    "depth_enabled": False,             # off by default, enable with --depth
    "depth_model_size": "small",        # "small" / "base" / "large"
    "depth_device": "auto",             # "auto" / "cuda" / "cpu"

    # =========================================================================
    # DEBUG — Development tools (not for production)
    # =========================================================================

    # === Visualization (--no-viz to disable) ===
    "show_visualization": True,
    "viz_window_name": "Perception Pipeline",
    "viz_scale": 1.0,

    # === Config GUI (--gui flag) ===
    # Debug panel for runtime parameter tuning during development.
    "show_gui": False,                  # debug tool, not for production
}
