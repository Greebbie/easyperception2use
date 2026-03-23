**English** | **[中文](README_CN.md)**

# Perception Pipeline v3.2

Camera feed → structured scene JSON for downstream robot / LLM decision-making. This is the **eyes**, not the brain.

All computation happens here (detection, tracking, filtering, compensation, scene analysis). The central controller receives ready-to-use data — no further processing needed.

## Responsibility Boundary

| Perception Module (this project) | Central Controller (separate project) |
|---|---|
| All "computing": model inference, filtering, compensation, scene analysis | All "thinking": what to grab, priority, motion planning |
| Output: clean structured JSON | Input: our JSON, used directly without further calculation |
| Tells the controller "what I see, where it is, how it moves, whether to trust the data" | Decides "whether to grab, how to grab, what to grab first" |

**Design principle: Perception and decision-making are separated, iterated independently.** The perception module doesn't need to know what the controller will do.

## Requirements

- **Python** 3.10+
- **Core dependencies** (`requirements.txt`): ultralytics (YOLOv8), opencv-python, numpy, websockets, pytest
- **Depth optional** (`requirements-depth.txt`): transformers, torch (~2GB, only needed with `--depth`)
- **GPU**: Optional. YOLO and Depth work on CPU; GPU (CUDA) is faster

```bash
pip install -r requirements.txt                  # Core (required)
pip install -r requirements-depth.txt            # Depth enhancement (optional)
```

## Quick Start

```bash
python main.py                     # Auto-detect USB camera
python main.py --gui               # Camera + GUI debug panel + visualization
python main.py --ws --compact      # Production: WebSocket push + compact JSON
python main.py --dry-run           # No camera, synthetic data for testing
```

## Architecture

```
Camera (auto-detect / USB / RTSP / file)
  → FrameGrabber          Daemon thread, latest-frame-only, auto-reconnect
    → YOLOv8 + ByteTrack    Detection + tracking (persistent track IDs)
      → Optical Flow         Camera egomotion estimation (foreground-masked)
        → Kalman Filter      2D smoothing on compensated coordinates
          → SceneBuilder     Results → scene JSON (trust model + ego state)
            → SceneDiffer    Frame-to-frame diff → changes list
              → OutputController  5 strategies (every_frame/interval/on_change/hybrid/stable)
                → OutputHandler   print / file (JSONL) / callback / compact mode
                → WebSocket       JSON-RPC 2.0 push (port 18790)
              → Visualizer        OpenCV overlay (bboxes / grid / FPS / risk)
  → DepthEstimator          Depth Anything v2 (optional, --depth)
```

## Supported Video Sources

| Type | Example | Notes |
|------|---------|-------|
| USB Camera | `--source 0` or `--source auto` | Auto-detects USB/UVC cameras |
| RTSP Stream | `--source rtsp://192.168.1.100:554/stream` | Wireless IP cameras, WiFi cameras |
| HTTP Stream | `--source http://192.168.1.100:8080/video` | ESP32-CAM, phone IP camera apps |
| Video File | `--source video.mp4` | Playback, offline testing |
| Synthetic | `--dry-run` | No hardware needed, for development |

**Wireless cameras**: Any camera supporting RTSP/HTTP streaming works out of the box. FrameGrabber has built-in auto-reconnect (up to 5 retries, 3s interval).

**Runtime switching**: Switch video sources at runtime via WebSocket `source/switch` or the GUI panel — no restart required.

## Key Design Decisions

### 1. Why 2D First, Not 3D

**Decision: 2D pipeline is the core. Depth is an optional enhancement. Camera calibration is a reserved interface.**

What 2D can do accurately:
- Object presence: is there a cup in the frame?
- Relative position: cup is left-of-center (normalized coordinates 0-1)
- Motion direction/speed: cup is moving right
- Scene changes: a person entered, the cup disappeared
- Collision risk: large object in center region

What 2D cannot do:
- "How far is the cup?" — normalized coordinates are not physical distance
- "Which cup is closer?" — can only guess by size, unreliable
- "How far should the arm extend?" — needs real depth + camera calibration

**Why this choice**: For high-level decisions ("grab that cup", "stop, there's a person"), 2D is sufficient. Monocular depth is relative-only, unusable for closed-loop grasping. The 2D pipeline is easiest to debug — you can tell whether it's a detector miss, tracker drift, or control logic error.

### 2. What Depth Can and Cannot Do

Depth Anything v2 outputs **relative depth** (0-1 normalized, per-frame ranking), not absolute distance.

| Depth Can Do | Depth Cannot Do |
|-------------|-----------------|
| Cup is closer than bottle (relative ordering) | Cup is 28.3cm away (absolute distance) |
| near / mid / far labels | Millimeter-level 3D localization |
| Depth ordering for controller reference | Closed-loop servo grasping |

**For precise grasping**: Requires real depth hardware (stereo / ToF / RealSense) + camera calibration → `calibration_enabled` interface is reserved, to be activated when hardware is ready.

### 3. Trust Model: Why the `actionable` Flag

**Decision: Every frame is tagged with data reliability. Auto-degrades when robot is moving.**

| ego_state | actionable | Detection | Position | Motion | Scene |
|-----------|-----------|-----------|----------|--------|-------|
| `stopped` | **true** | ✅ | ✅ | ✅ | ✅ |
| `moving` | **false** | ✅ | ❌ | ❌ | ❌ |
| `settling` | **false** | ✅ | ❌ | ❌ | ❌ |

**Why**: When the robot moves, all objects shift in the frame. A stationary cup appears to "move". Only class + track_id + confidence remain reliable. The controller should pause position-dependent decisions when `actionable=false`.

### 4. Pluggable Module Design

**Decision: All enhancements are optional plugins, off by default, no core dependency increase.**

| Module | Default | Enable | Dependency |
|--------|---------|--------|------------|
| Depth Anything v2 | Off | `--depth` | transformers + torch |
| Camera Calibration | Off | `calibration_enabled` (config) | To be implemented |
| WebSocket Server | Off | `--ws` | websockets |
| GUI Debug Panel | Off | `--gui` | tkinter (stdlib) |

**Why**: Core pipeline depends only on ultralytics + opencv + numpy. Production deploys only what's needed. Failures auto-degrade (depth load timeout → auto-disable, pipeline continues).

### 5. Why Joseph Form for Kalman

**Decision: Joseph stabilized form instead of the standard (I-KH)P update.**

The standard form loses positive-definiteness of the covariance matrix over long runs (numerical drift), causing negative confidence values or NaN. Joseph form `(I-KH)P(I-KH)^T + KRK^T` guarantees the covariance stays positive-definite. Verified: eigenvalues remain positive after 2000 iterations.

### 6. Output Strategy Selection

| Strategy | Use Case | Controller Latency |
|----------|---------|-------------------|
| `every_frame` | Debug | 0ms (every frame, high volume) |
| `interval` | Teleoperation | Fixed interval (predictable) |
| `on_change` | Event-driven | Output only on changes (irregular) |
| **`hybrid` (recommended)** | **Most robots** | **Interval + burst on events** |
| `stable` | LLM decisions | Output after scene stabilizes (highest latency) |

### 7. Performance

| Metric | Value | Condition |
|--------|-------|-----------|
| YOLO first frame | ~300ms | Model loading |
| YOLO steady-state | ~20ms | YOLOv8n, 640x480 |
| Kalman + scene building | ~3ms | |
| Optical flow | ~2ms | 160x120 downsampled |
| Depth first frame | ~3.5s | CPU, after model download |
| Depth steady-state | ~300-400ms | CPU; faster with GPU |
| **End-to-end latency** | **~25ms (no depth)** | |
| **End-to-end latency** | **~350ms (with depth, CPU)** | |

## Capabilities

| Capability | Status | Details |
|-----------|--------|---------|
| Object detection + tracking | ✅ | YOLOv8n + ByteTrack, persistent track ID |
| Position smoothing + velocity | ✅ | Kalman 2D (Joseph form), position/velocity confidence |
| Position prediction | ✅ | Kalman 0.1s lookahead, `predicted_next` |
| Camera motion compensation | ✅ | Optical flow + foreground masking, auto-degrade |
| Scene semantic analysis | ✅ | risk_level / center_occupied / stable / snapshot_quality |
| Change events | ✅ | entered / left / approaching / retreating / region_change / risk_change |
| Depth estimation | ✅ Optional | Depth Anything v2 (relative depth, near/mid/far) |
| Camera calibration + world coords | 🔧 Reserved | Config interface ready, awaiting hardware deployment |

## Output Data

### Key Fields for the Controller

```json
{
  "actionable": true,          // Can the data be trusted? (true when robot stopped)
  "objects": [{
    "track_id": 3,             // Persistent tracking ID (stable across frames)
    "class": "cup",            // YOLO class
    "confidence": 0.91,        // Detection confidence
    "track_age": 42,           // Frames tracked (higher = more trustworthy)
    "position_confidence": 0.93, // Kalman position confidence (0-1)
    "velocity_confidence": 0.70, // Kalman velocity confidence (0-1)
    "smoothed_x": 0.45,       // Kalman-smoothed normalized coords (0=left, 1=right)
    "smoothed_y": 0.62,       // (0=top, 1=bottom)
    "vx": -0.028,             // Normalized velocity (units/sec, camera-compensated)
    "vy": 0.015,
    "moving": true,
    "reliable": true,          // Is motion data trustworthy?
    "region": "middle_center", // 3x3 semantic region
    "predicted_next": {"cx": 0.44, "cy": 0.62},  // Predicted position 0.1s ahead
    "depth": {"value": 0.35, "label": "near"}     // Optional, requires --depth
  }],
  "scene": {
    "risk_level": "medium",    // clear / low / medium / high
    "center_occupied": true,   // Object in center region (collision risk)
    "stable": true,            // No objects entering/leaving
    "snapshot_quality": 0.92   // Overall data quality (0-1)
  },
  "camera_motion": {
    "ego_state": "stopped",    // stopped / moving / settling
    "confidence": 0.87,        // Optical flow compensation reliability
    "compensated": true        // Are coordinates camera-motion compensated?
  },
  "changes": [                 // Change events (for event-driven controllers)
    "object_entered: cup #3 appeared in middle_center",
    "risk_change: low → medium"
  ]
}
```

### Compact Mode (--compact)

~400 bytes/frame for bandwidth-constrained links:

```json
{"ts": 1711234567.12, "actionable": true,
 "objects": [{"id": 3, "cls": "cup", "cx": 0.45, "cy": 0.62,
   "vx": -0.028, "vy": 0.015, "moving": true, "conf": 0.91,
   "region": "middle_center", "reliable": true,
   "pred": {"cx": 0.44, "cy": 0.62}}],
 "scene": {"risk": "medium", "center": true, "stable": true, "quality": 0.92},
 "camera": {"tx": 0.003, "ty": -0.001, "conf": 0.87, "ego": "stopped"},
 "changes": ["object_entered: cup #3 appeared in middle_center"]}
```

## Controller Integration

### Option 1: WebSocket (JSON-RPC 2.0)

```bash
python main.py --ws --compact --strategy hybrid
```

Port 18790, default bind 127.0.0.1.

| Method | Params | Description |
|--------|--------|-------------|
| `scene/latest` | — | Get latest scene |
| `scene/subscribe` | — | Subscribe to real-time push (`scene/update`) |
| `ego/motion` | `{"moving": true, "vx": 0.1}` | Tell perception "I'm moving" (**must integrate**) |
| `config/set` | `{"key": "min_confidence", "value": 0.6}` | Runtime parameter tuning |
| `source/switch` | `{"source": 0}` | Switch video source |
| `status/health` | — | Pipeline health (state / fps / latency / degraded_modules) |

**Security**: config/set has an allowlist (only perception params, cannot modify model_path etc.), max 32 connections, 64KB message limit, error responses don't leak internals.

### Option 2: Python API (In-Process)

```python
from perception_service import PerceptionService

svc = PerceptionService(config)
svc.start()

# Subscribe to scene updates (only receive trusted data)
svc.subscribe(on_scene, filter_fn=lambda s: s["actionable"])

# Tell perception: robot arm is moving
svc.set_ego_motion(moving=True, vx=0.1, vy=0.0)

# Get latest scene
scene = svc.get_latest_scene()

# Runtime config
svc.set_config("min_confidence", 0.6)

# Health check
health = svc.get_status()  # {"state": "running", "fps": 10, ...}

svc.stop()
```

### Ego Motion Interface (Controller Must Integrate)

The controller calls this before/after arm movement so perception can tag data reliability:

```
Controller: ego/motion {"moving": true}   → Perception: marks position/motion/scene as untrusted
Controller: ego/motion {"moving": false}  → Perception: enters settling, trusted again after 0.5s
```

Three ego sources (config `ego_motion_source`):
- `"external"` — Controller tells perception via RPC/API (**recommended for production**)
- `"optical_flow"` — Perception auto-infers from video (default, for development)
- `"none"` — No motion compensation

**Why external is recommended**: Optical flow has detection delay and false positives. The controller knows "I'm about to move" more accurately than perception can guess.

## Configuration

All defaults in `config.py`. Key parameters:

```python
"min_confidence": 0.45              # Detection confidence threshold (lower = more detections, more false positives)
"process_fps": 10                   # Processing frame rate (30 = lower latency, higher CPU)
"output_strategy": "hybrid"         # hybrid (recommended) / stable / interval / on_change / every_frame
"output_compact": False             # --compact for minimal JSON output
"ego_motion_source": "optical_flow" # "external" (recommended production) / "optical_flow" / "none"
"ego_settle_sec": 0.5               # moving→stopped transition time
"kalman_process_noise": 0.01        # Higher = trust observations more; Lower = smoother
"kalman_measurement_noise": 0.05    # Higher = smoother but laggier
"depth_enabled": False              # --depth to enable depth estimation
"ws_enabled": False                 # --ws to enable WebSocket
"ws_host": "127.0.0.1"             # WebSocket bind address (0.0.0.0 for network access)
"ws_port": 18790                    # WebSocket port
```

## CLI

```
--source PATH       Video source ("auto" / 0 / rtsp://... / video.mp4)
--model PATH        YOLO model (default: yolov8n.pt)
--process-fps N     Processing frame rate (default: 10)
--strategy NAME     Output strategy: every_frame / interval / on_change / hybrid / stable
--compact           Compact JSON output
--interval SEC      Output interval (seconds)
--output METHOD     Output method: print / file / callback
--classes A B C     Only detect these classes
--depth             Enable depth estimation (requires pip install -r requirements-depth.txt)
--depth-model SIZE  Depth model: small / base / large (default: small)
--gui               Open GUI debug panel (Detection / Output / Plugins / Source + Live JSON)
--ws                Enable WebSocket server
--ws-port PORT      WebSocket port (default: 18790)
--no-viz            Disable visualization window
--dry-run           Synthetic data mode (no camera needed)
```

## Modules

```
├── main.py                 Entry point, CLI, graceful shutdown
├── config.py               Configuration defaults
├── frame_grabber.py        Threaded frame reader, auto-detect, reconnection
├── scene_builder.py        YOLO → scene JSON (Kalman + motion compensation + trust)
├── kalman_tracker.py       2D Kalman (Joseph form, confidence + prediction)
├── scene_differ.py         Frame diff → change events (with debounce)
├── output_controller.py    5 output strategies
├── output_handler.py       print / file / callback / compact
├── depth_estimator.py      Depth Anything v2 (optional, lazy-load, auto-degrade)
├── visualizer.py           OpenCV overlay
├── metrics.py              Latency / FPS / health monitoring
├── perception_service.py   Programmatic API (subscribe / set_ego_motion / get_status)
├── ws_server.py            WebSocket JSON-RPC 2.0 (thread-safe + allowlist + connection limit)
├── config_gui.py           tkinter debug panel (4 setting groups + Live JSON + metrics)
├── dry_run.py              Synthetic data generator (schema matches real pipeline)
└── tests/                  246 tests, 14 test files, 14/14 modules covered
```

## Coordinate System

```
(0,0) ────────────────── (1,0)
  │    top_left │ top_center │ top_right
  │  ───────────┼────────────┼──────────  y=0.4
  │  mid_left   │ mid_center │ mid_right
  │  ───────────┼────────────┼──────────  y=0.7
  │  btm_left   │ btm_center │ btm_right
(0,1) ────────────────── (1,1)
         x=0.33      x=0.67
```

- **Origin** `(0, 0)` = top-left corner of the frame
- **Normalization**: `rel_x = pixel_x / frame_width`, range 0-1
- **smoothed_x/y** = Kalman-filtered normalized coordinates (camera-motion compensated)
- **vx/vy** = Normalized velocity (units/sec), positive = right/down
- **3x3 region boundaries**: x at 0.33 / 0.67, y at 0.4 / 0.7 (configurable)

## YOLO Model Selection

| Model | Parameters | Speed (CPU) | Accuracy (mAP) | Recommended For |
|-------|-----------|-------------|-----------------|-----------------|
| **yolov8n.pt** (current default) | 3.2M | ~20ms | 37.3 | Real-time priority, embedded/CPU |
| yolov8s.pt | 11.2M | ~50ms | 44.9 | Balance of speed and accuracy |
| yolov8m.pt | 25.9M | ~120ms | 50.2 | High accuracy, requires GPU |
| yolo11n.pt | 2.6M | ~18ms | 39.5 | Latest architecture, worth trying |

Switch models: `python main.py --model yolov8s.pt` or modify config `"model_path"`. Models auto-download.

## Track ID Behavior

- `track_id` is assigned by ByteTrack, **stable across frames**: same object keeps the same ID in consecutive frames
- Object occluded < 2 seconds: Kalman predicts state, ID preserved on re-appearance
- Object gone > 2 seconds (`track_lost_timeout`): ID released, re-appearance gets a new ID
- **New object confirmation**: Default 1 frame (`track_confirm_frames=1`, trusts ByteTrack)
- **Loss buffer**: Must be missing for 10 consecutive frames before removal (`track_lost_frames=10`)
- `track_age` indicates how many frames the ID has been tracked — higher = more trustworthy

## Degradation & Fault Tolerance

| Fault | Behavior | Pipeline State |
|-------|----------|---------------|
| Camera disconnected | FrameGrabber auto-reconnects (network: up to 5 retries), USB/file exits | `state: error` |
| YOLO inference failure | Skips frame, continues next, outputs last valid scene | `degraded_modules: ["yolo"]` |
| Depth load timeout (30s) | Auto-disables depth, core pipeline continues | `degraded_modules: ["depth"]` |
| Depth inference failure | Returns None, no depth data for that frame, rest normal | Normal |
| Low optical flow confidence | Auto-stops motion compensation (`compensated: false`) | Normal |
| WebSocket client disconnects | Auto-cleanup, no impact on other clients or pipeline | Normal |
| Invalid output file path | Exits on startup (RuntimeError) | Does not start |

**Design principle: Core 2D pipeline always runs. Enhancement modules auto-degrade on failure.** `pipeline.degraded_modules` tells the controller which capabilities are impaired.

## Known Limitations

- **Small objects**: < 32x32 pixels are easily missed by YOLO
- **Occlusion**: Kalman can only predict ~2 seconds after full occlusion, then lost
- **Similar appearance**: Multiple same-class objects close together may confuse ByteTrack IDs
- **Fast motion**: Objects moving > 1/3 frame width between frames may break tracking
- **Lighting changes**: Sudden lighting changes (lights on/off) briefly affect detection and optical flow
- **Monocular depth**: Depth Anything v2 is relative depth only, not absolute distance, and inconsistent across frames
- **CPU latency**: Depth on CPU is ~350ms/frame, limiting real-time use; GPU recommended

## Tests

```bash
python -m pytest tests/ -v    # 246 tests, ~11s
```

Covers all 14 modules: scene_builder (81) / ws_server (20) / perception_service (16) / kalman (15) / output_controller (15) / scene_differ (15) / depth (14) / dry_run (13) / frame_grabber (12) / config_gui (12) / visualizer (10) / output_handler (8) / metrics (8) / config (7)
