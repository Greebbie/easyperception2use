# Perception Pipeline v3.2

Camera video stream → structured scene JSON for downstream LLM/robot decision-making.

## Quick Start

```bash
pip install -r requirements.txt

# Auto-detect USB camera
python main.py

# With GUI debug panel
python main.py --gui

# Production mode (stable snapshots + compact JSON)
python main.py --strategy stable --compact

# Full features (GUI + WebSocket + depth)
python main.py --gui --ws --depth

# No camera — synthetic data for testing
python main.py --dry-run
```

## Architecture

```
Camera (auto-detect / USB / RTSP / file)
  → FrameGrabber          Daemon thread, latest-frame-only, auto-reconnect
    → YOLOv8 + ByteTrack    Detection + tracking (persistent track IDs)
      → Optical Flow         Camera egomotion estimation (foreground-masked)
        → Kalman Filter      2D smoothing on compensated coordinates
          → SceneBuilder     Results → scene JSON (trust model + ego state)
            → SceneDiffer    Frame-to-frame diff → changes list (LLM-readable)
              → OutputController  5 strategies (every_frame/interval/on_change/hybrid/stable)
                → OutputHandler   print / file (JSONL) / callback / compact mode
                → WebSocket       JSON-RPC 2.0 push (port 18790)
              → Visualizer        OpenCV overlay (bboxes / grid / FPS / risk)
  → DepthEstimator          Depth Anything v2 (optional, --depth)
```

## Key Design Decisions

### Trust Model (actionable / trust)

Every JSON frame has a top-level `actionable` flag and per-category `trust` object. This tells the downstream LLM exactly what it can rely on:

| ego_state | actionable | detection | position | motion | scene |
|-----------|-----------|-----------|----------|--------|-------|
| `stopped` | **true** | trust | trust | trust | trust |
| `moving` | **false** | trust | **no** | **no** | **no** |
| `settling` | **false** | trust | **no** | **no** | **no** |

**Why**: When the robot moves, all objects shift in the frame. A stationary cup goes from `middle_center` to `middle_left` even though it didn't move. Only `class` + `track_id` + `confidence` remain valid.

### LLM Integration Pattern

```python
scene = get_latest_scene()

if scene["actionable"]:
    # Full decision: position, motion, risk all reliable
    plan_action(scene)
else:
    # Robot moving — only update inventory (what exists + track IDs)
    update_inventory(scene["objects"])  # class + track_id only
    wait_for_actionable()
```

### Ego Motion State Machine

```
STOPPED  →  (set_ego_motion(True))  →  MOVING
MOVING   →  (set_ego_motion(False)) →  SETTLING
SETTLING →  (0.5s elapsed)          →  STOPPED
```

Three ways to provide ego motion data:
1. **WebSocket**: `{"method": "ego/motion", "params": {"moving": true}}`
2. **Python API**: `svc.set_ego_motion(moving=True)`
3. **Auto-infer**: When `ego_motion_source="optical_flow"`, low flow confidence auto-triggers moving state

### Camera Motion Compensation

- Compensated coordinates fed to Kalman filter (not raw)
- Dense optical flow on 160x120 downsampled frame (~2ms)
- Foreground objects masked from flow computation (prevents moving objects from corrupting estimate)
- Automatic degradation: confidence < 0.3 disables compensation
- Works for: stationary camera, small pan/tilt
- Does NOT work for: robot driving forward (radial expansion), large rotation → use `ego_motion_source="external"`

### Track Stability

- ByteTrack handles low-confidence recovery internally (Stage 2: 0.1-0.25)
- No conf parameter passed to YOLO track() — lets ByteTrack work optimally
- Confidence filtering done post-ByteTrack in SceneBuilder (min_confidence=0.45)
- Debounce aligned with ByteTrack: confirm=1 frame, lost=10 frames
- Kalman grace period: filters survive 2s after track loss

## JSON Output Schema (v3.2)

```json
{
  "frame_id": 1234,
  "schema_version": "3.2",
  "timestamp": 1711234567.123,
  "frame_size": {"w": 640, "h": 480},

  "actionable": true,
  "trust": {
    "detection": true,
    "position": true,
    "motion": true,
    "scene": true
  },

  "camera_motion": {
    "tx": 0.003,
    "ty": -0.001,
    "compensated": true,
    "confidence": 0.87,
    "ego_state": "stopped"
  },

  "objects": [
    {
      "track_id": 3,
      "class": "person",
      "confidence": 0.912,
      "track_age": 42,
      "position_confidence": 0.93,
      "velocity_confidence": 0.7,
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
        "moving": true,
        "reliable": true
      },
      "predicted_next": {"cx": 0.444, "cy": 0.62},
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
    "stable": true,
    "time_since_change_sec": 2.3,
    "snapshot_quality": 0.92
  },

  "pipeline": {"state": "running", "degraded_modules": [], "uptime_sec": 456.7},
  "latency_ms": {"grab_to_detect": 12, "detect_to_depth": 0, "total": 17},
  "meta": {"active_tracks": 1, "total_tracks_in_memory": 3, "dropped_by_confidence": 0}
}
```

### Field Reference

**Top-level control fields:**
| Field | Type | Description |
|-------|------|-------------|
| `actionable` | bool | **LLM reads this first.** true = safe to act on all data |
| `trust.detection` | bool | class + track_id + confidence reliable (always true) |
| `trust.position` | bool | coordinates + region reliable (false when robot moving) |
| `trust.motion` | bool | velocity + direction reliable (false when robot moving) |
| `trust.scene` | bool | risk + center_occupied reliable (false when robot moving) |

**Per-object fields (new in v3.2):**
| Field | Type | Description |
|-------|------|-------------|
| `track_age` | int | Frames this track has been alive (higher = more trustworthy) |
| `position_confidence` | float | 0-1 from Kalman covariance (higher = more certain) |
| `velocity_confidence` | float | 0-1 from Kalman velocity covariance |
| `predicted_next` | {cx, cy} | Kalman-predicted position 0.1s ahead |
| `motion.reliable` | bool | false when ego is moving/settling |

**Scene-level fields (new in v3.2):**
| Field | Type | Description |
|-------|------|-------------|
| `scene.stable` | bool | No object changes for `stable_window_sec` |
| `scene.time_since_change_sec` | float | Seconds since last object enter/leave |
| `scene.snapshot_quality` | float | 0-1 composite (avg position_confidence × stability) |

**Camera motion:**
| Field | Type | Description |
|-------|------|-------------|
| `camera_motion.ego_state` | str | `stopped` / `moving` / `settling` |
| `camera_motion.confidence` | float | 0-1 optical flow reliability |
| `camera_motion.compensated` | bool | Whether coordinates are motion-compensated |

### Compact JSON (--compact)

~400 bytes per frame for bandwidth-constrained downstream:

```json
{
  "ts": 1711234567.12,
  "actionable": true,
  "objects": [
    {"id": 3, "cls": "person", "cx": 0.447, "cy": 0.618, "size": 0.087,
     "region": "middle_center", "conf": 0.912, "age": 42, "pos_conf": 0.93,
     "vx": -0.028, "vy": 0.015, "moving": true, "reliable": true,
     "pred": {"cx": 0.444, "cy": 0.62}}
  ],
  "scene": {"count": 1, "risk": "medium", "center": true, "moving": 1,
            "stable": true, "quality": 0.92},
  "camera": {"tx": 0.003, "ty": -0.001, "conf": 0.87, "ego": "stopped"},
  "changes": ["person #3 entered middle_center"]
}
```

## WebSocket API (JSON-RPC 2.0)

Port 18790, enabled with `--ws`.

| Method | Params | Description |
|--------|--------|-------------|
| `scene/latest` | — | Get latest scene JSON |
| `scene/subscribe` | — | Subscribe to scene push updates |
| `config/set` | `{"key": "...", "value": ...}` | Update config at runtime |
| `source/switch` | `{"source": "rtsp://..."}` | Switch video source |
| `status/health` | — | Health metrics (fps, latency, state) |
| `ego/motion` | `{"moving": true, "vx": 0.1}` | Set robot ego-motion state |

## CLI Arguments

```
--source PATH       Video source ("auto" / 0 / rtsp://... / video.mp4)
--model PATH        YOLO model path (default: yolov8n.pt)
--process-fps N     Detection frame rate (default: 10)
--strategy NAME     Output: every_frame / interval / on_change / hybrid / stable
--compact           Output minimal JSON for downstream LLM/Claw
--interval SEC      Output interval in seconds
--output METHOD     Output method: print / file / callback
--classes A B C     Only detect these classes
--depth             Enable depth estimation
--depth-model SIZE  Depth model: small / base / large
--gui               Open config GUI panel (with live JSON viewer)
--ws                Start WebSocket server
--ws-port PORT      WebSocket port (default: 18790)
--no-viz            Disable visualization window
--dry-run           Synthetic data mode (no camera needed)
```

## Configuration

All defaults in `config.py`. Key settings:

```python
# Video source
"source": "auto"                    # auto-detect USB, or specify 0, "rtsp://..."

# Detection
"min_confidence": 0.45              # SceneBuilder output filter (ByteTrack uses its own thresholds)

# Tracking stability
"track_confirm_frames": 1           # Frames to confirm new track (1 = trust ByteTrack)
"track_lost_frames": 10             # Frames before removing lost track

# Ego motion
"ego_motion_source": "optical_flow" # "optical_flow" / "external" / "none"
"ego_settle_sec": 0.5               # Settling → stopped transition time
"ego_auto_detect": True             # Auto-infer moving from optical flow quality

# Output
"output_strategy": "hybrid"         # or "stable" for LLM consumption
"output_compact": False             # --compact flag
"stable_window_sec": 1.0            # Stable mode: seconds of no-change before output
```

## Modules

```
├── main.py                 Entry point, CLI args, graceful shutdown
├── config.py               Centralized config defaults
├── frame_grabber.py        Threaded frame reader, auto-detect, reconnection
├── scene_builder.py        YOLO → scene JSON (Kalman + motion compensation + trust model)
├── kalman_tracker.py       Per-object 2D Kalman (confidence + prediction)
├── scene_differ.py         Frame diff → changes list (with cooldown debounce)
├── output_controller.py    5 output strategies (including stable mode)
├── output_handler.py       print / file / callback / compact
├── depth_estimator.py      Depth Anything v2 (optional, lazy-loaded)
├── visualizer.py           OpenCV overlay
├── metrics.py              Latency/FPS tracking
├── perception_service.py   Programmatic API
├── ws_server.py            WebSocket JSON-RPC 2.0 server
├── config_gui.py           tkinter debug panel (collapsible settings + live JSON)
├── dry_run.py              Synthetic data generator
└── tests/                  pytest suite (38 tests)
```

## Tests

```bash
python -m pytest tests/ -v
```
