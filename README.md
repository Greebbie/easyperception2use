# Perception Pipeline v3.1

Camera video stream → structured scene JSON for downstream LLM/Claw decision-making.

## Quick Start

```bash
pip install -r requirements.txt

# USB camera
python main.py

# Full feature (visualization + GUI + WebSocket + depth)
python main.py --gui --ws --depth

# No camera — synthetic data for testing
python main.py --dry-run
```

## Architecture

```
Camera (USB/RTSP/file)
  → FrameGrabber        Daemon thread, keeps latest frame only, auto-reconnect for network streams
    → YOLOv8 + ByteTrack  Detection + tracking, persistent track IDs
      → Kalman Filter     2D Kalman smoothing on position and velocity, removes detection jitter
        → SceneBuilder    Detection results → normalized scene JSON (coords/velocity/regions/risk)
          → SceneDiffer   Frame-to-frame diff → changes list (LLM reads this directly)
            → OutputController  Output frequency control (4 strategies)
              → OutputHandler   Output to print / file / callback
              → WebSocket       JSON-RPC 2.0 push to Claw/Agent
            → Visualizer        OpenCV overlay (bboxes/grid/FPS/risk panel)
  → DepthEstimator       Depth Anything v2 monocular depth estimation (optional)
```

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Detection | YOLOv8n (ultralytics) | Lightweight, runs on CPU or GPU |
| Tracking | ByteTrack | Built into ultralytics, persistent track IDs |
| Depth | Depth Anything v2 Small (HuggingFace transformers) | Monocular relative depth, lazy-loaded, ~100MB download on first use |
| Smoothing | Hand-rolled 2D Kalman Filter (numpy) | State vector [x, y, vx, vy], zero extra dependencies |
| Video I/O | OpenCV VideoCapture | USB (int), RTSP (str), local file (str) |
| WebSocket | websockets 16 + JSON-RPC 2.0 | Compatible with OpenClaw Gateway protocol |
| GUI | tkinter | Python built-in, runs in daemon thread |

## Modules

```
├── main.py                 Entry point, CLI args, graceful shutdown (SIGINT/SIGTERM)
├── config.py               DEFAULT_CONFIG (31 config keys, all centralized here)
├── frame_grabber.py        Threaded frame reader + reconnection + runtime source switching
├── scene_builder.py        YOLO results → normalized JSON (integrates Kalman + Depth)
├── kalman_tracker.py       Per-object 2D Kalman filter instances
├── scene_differ.py         Frame-to-frame diff → changes list for LLM consumption
├── depth_estimator.py      Depth Anything v2 wrapper, 30s load timeout, auto-disable on failure
├── output_controller.py    4 output strategies: every_frame / interval / on_change / hybrid
├── output_handler.py       print / file (JSONL) / callback output
├── visualizer.py           OpenCV overlay (bboxes / 3x3 grid / FPS / risk panel)
├── metrics.py              Latency P50/P95, FPS, pipeline state machine
├── perception_service.py   Programmatic API (start/stop/subscribe/get_latest_scene)
├── ws_server.py            WebSocket JSON-RPC 2.0 server
├── config_gui.py           tkinter runtime config panel
├── dry_run.py              Synthetic data generator (no camera/model needed)
├── requirements.txt
└── tests/                  pytest suite (37 tests)
```

## JSON Output Schema

```json
{
  "frame_id": 1234,
  "schema_version": "3.1",
  "timestamp": 1711234567.123,
  "frame_size": {"w": 1280, "h": 720},
  "pipeline": {
    "state": "running",
    "degraded_modules": [],
    "uptime_sec": 456.7
  },
  "latency_ms": {
    "grab_to_detect": 12,
    "detect_to_depth": 45,
    "total": 60
  },
  "objects": [
    {
      "track_id": 3,
      "class": "person",
      "confidence": 0.912,
      "position": {
        "rel_x": 0.45,
        "rel_y": 0.62,
        "smoothed_x": 0.447,
        "smoothed_y": 0.618,
        "rel_size": 0.087,
        "region": "middle_center"
      },
      "bbox_px": {"x1": 412, "y1": 305, "x2": 668, "y2": 610},
      "motion": {
        "direction": "left",
        "speed": 0.032,
        "vx": -0.028,
        "vy": 0.015,
        "moving": true
      },
      "depth": {"value": 0.35, "label": "near"}
    }
  ],
  "changes": [
    "region_change: person #3 moved from top_center to middle_center",
    "risk_change: low -> medium"
  ],
  "scene": {
    "object_count": 1,
    "center_occupied": true,
    "dominant_object": {"class": "person", "track_id": 3, "rel_size": 0.087},
    "risk_level": "medium",
    "classes_present": ["person"],
    "moving_count": 1,
    "region_summary": {"middle_center": ["person"]},
    "nearest_object": {"class": "person", "track_id": 3, "depth": 0.35},
    "depth_ordering": [3]
  },
  "meta": {
    "active_tracks": 1,
    "total_tracks_in_memory": 3,
    "dropped_by_confidence": 0
  }
}
```

**Key fields:**
- `position.rel_x/rel_y` — Normalized coordinates (0-1), resolution-independent
- `position.smoothed_x/y` — Kalman-smoothed coordinates for hardware control
- `motion.vx/vy` — Normalized velocity vector (rel units/sec)
- `depth.value` — 0=nearest, 1=farthest (relative depth, requires `--depth`)
- `changes` — Human-readable diff from previous frame, LLM reads this directly for decisions
- `pipeline.state` — running / degraded / error

## WebSocket API (JSON-RPC 2.0)

Port 18790, enabled with `--ws`.

```
scene/latest    → Get latest scene JSON
scene/subscribe → Subscribe to scene push updates
config/set      → Update config  {"key": "min_confidence", "value": 0.5}
source/switch   → Switch video source  {"source": "rtsp://..."}
status/health   → Health status (fps, latency, state)
```

## CLI Arguments

```
--source PATH       Video source (0=USB, rtsp://..., video.mp4)
--model PATH        YOLO model path (default: yolov8n.pt)
--process-fps N     Detection frame rate (default: 10)
--strategy NAME     Output strategy: every_frame / interval / on_change / hybrid
--interval SEC      Output interval in seconds
--output METHOD     Output method: print / file / callback
--classes A B C     Only detect these classes
--depth             Enable depth estimation
--depth-model SIZE  Depth model size: small / base / large
--gui               Open config GUI panel
--ws                Start WebSocket server
--ws-port PORT      WebSocket port (default: 18790)
--no-viz            Disable visualization window
--dry-run           Synthetic data mode (no camera needed)
```

## Tests

```bash
python -m pytest tests/ -v
```
