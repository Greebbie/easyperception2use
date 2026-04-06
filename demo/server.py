"""FastAPI demo server — three-panel dashboard with event streaming and LLM interpretation."""

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent dir for core imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.perception_client import PerceptionClient
from demo.mjpeg_streamer import MJPEGStreamer
from demo.llm_middleware import LLMMiddleware
from demo.watchpoint import WatchpointMonitor
from demo.event_aggregator import EventAggregator

# Global state
_perception: Optional[PerceptionClient] = None
_streamer: Optional[MJPEGStreamer] = None
_llm: Optional[LLMMiddleware] = None
_watchpoint: Optional[WatchpointMonitor] = None
_aggregator: Optional[EventAggregator] = None
_event_websockets: list[WebSocket] = []
_scene_task: Optional[asyncio.Task] = None
_llm_task: Optional[asyncio.Task] = None

# Config from environment
DRY_RUN = os.environ.get("DEMO_DRY_RUN", "0") == "1"
_raw_source = os.environ.get("DEMO_VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE = int(_raw_source)
except ValueError:
    VIDEO_SOURCE = _raw_source
WS_URI = os.environ.get("DEMO_WS_URI", "ws://127.0.0.1:18790")

# LLM interpretation interval (seconds)
LLM_INTERPRET_INTERVAL = 5.0


async def _broadcast_event(msg: dict) -> None:
    """Broadcast a message to all connected /ws/events clients."""
    text = json.dumps(msg)
    stale: list[WebSocket] = []
    for ws in _event_websockets:
        try:
            await ws.send_text(text)
        except Exception:
            stale.append(ws)
    for ws in stale:
        _event_websockets.remove(ws)


async def _scene_listener() -> None:
    """Background task: subscribe to scene updates, run event pipeline."""
    global _perception, _streamer, _watchpoint, _aggregator
    try:
        async for scene in _perception.subscribe_scenes():
            # 1. Watchpoint checks → alerts
            wp_alerts = []
            if _watchpoint:
                wp_alerts = _watchpoint.check_scene(scene)

            # 2. Ingest SceneDiffer changes (for LLM context only, NOT pushed to UI)
            changes = scene.get("changes", [])
            timestamp = scene.get("timestamp", time.time())
            if _aggregator and changes:
                _aggregator.ingest_scene_changes(changes, timestamp)

            # 3. Ingest watchpoint alerts → these ARE pushed to UI (stable events)
            wp_events = []
            if _aggregator and wp_alerts:
                wp_events = _aggregator.ingest_watchpoint_alerts(wp_alerts)

            # 4. Check compound events
            compound_events = []
            if _aggregator and wp_events:
                compound_events = _aggregator.detect_compound_events()

            # 5. Broadcast ONLY watchpoint events (stable, class-based)
            for event in wp_events + compound_events:
                await _broadcast_event({"type": "event", "data": event.to_dict()})

            # 6. Broadcast dwell update (always, so frontend clears when empty)
            if _watchpoint:
                dwell_times = _watchpoint.get_dwell_times()
                await _broadcast_event({
                    "type": "dwell_update",
                    "data": {"dwell_times": dwell_times},
                })

            # 7. Broadcast stats — exclusively from watchpoint (stable)
            wp_dwell = _watchpoint.get_dwell_times() if _watchpoint else {}
            stable_track_count = sum(len(items) for items in wp_dwell.values())
            await _broadcast_event({
                "type": "stats_update",
                "data": {"tracks": stable_track_count},
            })

            # 8. Broadcast raw JSON sample (for JSON preview panel)
            scene_data = scene.get("scene", {})
            compact = {
                "schema": scene.get("schema_version", "3.2"),
                "timestamp": scene.get("timestamp"),
                "actionable": scene.get("actionable"),
                "objects": [
                    {
                        "track_id": o.get("track_id"),
                        "class": o.get("class"),
                        "confidence": o.get("confidence"),
                        "region": o.get("position", {}).get("region"),
                        "moving": o.get("motion", {}).get("moving"),
                    }
                    for o in scene.get("objects", [])[:5]
                ],
                "scene": {
                    "object_count": scene_data.get("object_count"),
                    "risk_level": scene_data.get("risk_level"),
                    "stable": scene_data.get("stable"),
                },
                "changes": scene.get("changes", [])[:3],
            }
            await _broadcast_event({"type": "raw_json", "data": compact})

            # 8. Visual alert on MJPEG stream
            if _streamer and wp_alerts:
                _streamer.trigger_alert(2.0)

            # 9. Update streamer zone overlays
            if _streamer and _watchpoint:
                status = _watchpoint.get_status()
                _streamer.set_watchpoint_zones(status.get("zones", []))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[SceneListener] Error: {e}")


async def _llm_interpreter_loop() -> None:
    """Background task: periodically call LLM to interpret events."""
    global _llm, _aggregator, _watchpoint
    last_interpret = 0.0

    while True:
        await asyncio.sleep(1.0)
        now = time.time()

        if not _llm or not _aggregator:
            continue

        # Check if there are new events worth interpreting
        summary = _aggregator.get_summary_for_llm(window_sec=60.0)
        if summary["total_events"] == 0:
            continue

        # Interpret on interval or immediately for high severity
        should_interpret = (now - last_interpret) >= LLM_INTERPRET_INTERVAL
        has_high_severity = summary["highest_severity"] in ("alert", "critical")

        if not should_interpret and not has_high_severity:
            continue

        last_interpret = now

        try:
            # Build scene summary
            scene_summary = None
            if _watchpoint:
                scene_summary = _watchpoint.get_status()

            result = await _llm.interpret_events(summary, scene_summary)
            await _broadcast_event({"type": "llm_update", "data": result})
        except Exception as e:
            print(f"[LLMInterpreter] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _perception, _streamer, _llm, _watchpoint, _aggregator, _scene_task, _llm_task

    _perception = PerceptionClient(uri=WS_URI)
    try:
        await _perception.connect(retries=15, delay=2.0)
    except ConnectionError as e:
        print(f"[DemoServer] WARNING: {e} — running without perception")
        _perception = None

    _streamer = MJPEGStreamer(fps=12)
    _llm = LLMMiddleware()
    _watchpoint = WatchpointMonitor()
    _aggregator = EventAggregator()

    if _perception:
        _scene_task = asyncio.create_task(_scene_listener())
    _llm_task = asyncio.create_task(_llm_interpreter_loop())

    print("[DemoServer] Ready at http://localhost:8080")
    yield

    if _scene_task:
        _scene_task.cancel()
    if _llm_task:
        _llm_task.cancel()
    if _perception:
        await _perception.close()
    if _streamer:
        _streamer.release()


app = FastAPI(title="EasyPerception Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Pages ──

@app.get("/")
async def index():
    index_path = os.path.join(_static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── Video ──

@app.get("/stream")
async def video_stream():
    if not _streamer:
        return JSONResponse({"error": "Streamer not ready"}, status_code=503)
    return StreamingResponse(
        _streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Scene API ──

@app.get("/api/scene")
async def get_scene():
    if not _perception:
        return JSONResponse({"error": "Not connected to perception"}, status_code=503)
    scene = await _perception.get_latest()
    return JSONResponse(scene)


# ── LLM Query ──

@app.post("/api/query")
async def query_llm(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "Missing 'text' field"}, status_code=400)
    if not _llm or not _perception:
        return JSONResponse({"error": "LLM or perception not ready"}, status_code=503)
    result = await _llm.process_query(text, _perception)
    return JSONResponse(result)


# ── Context-Aware Chat ──

@app.post("/api/chat")
async def chat(request: Request):
    """Context-aware chat — LLM answers with event history, dwell info, and scene state."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)
    if not _llm:
        return JSONResponse({"error": "LLM not ready"}, status_code=503)

    # Gather context
    context_parts = []

    # Current scene
    if _perception:
        try:
            scene = await _perception.get_latest()
            scene_info = scene.get("scene", {})
            context_parts.append(
                f"当前场景：{scene_info.get('object_count', 0)} 个物体，"
                f"类别 {scene_info.get('classes_present', [])}，"
                f"风险等级 {scene_info.get('risk_level', 'clear')}"
            )
        except Exception:
            pass

    # Dwell info
    if _watchpoint:
        dwell = _watchpoint.get_dwell_times()
        for zone_id, items in dwell.items():
            for item in items:
                context_parts.append(
                    f"{item['class']} #{item['track_id']} 已在监控区停留 {item['dwell_sec']:.0f} 秒"
                )

    # Recent events
    if _aggregator:
        summary = _aggregator.get_summary_for_llm(window_sec=120.0)
        if summary["total_events"] > 0:
            context_parts.append(f"最近2分钟共 {summary['total_events']} 个事件")
            for ev in summary["events"][-5:]:
                context_parts.append(f"  [{ev['time']}] {ev['description']}")

    # Watchpoint status
    if _watchpoint:
        status = _watchpoint.get_status()
        if status["alert_count"] > 0:
            context_parts.append(f"累计告警 {status['alert_count']} 次")

    result = await _llm.context_chat(text, "\n".join(context_parts))
    return JSONResponse(result)


# ── Watch Command (conversational) ──

@app.post("/api/watch-command")
async def watch_command(request: Request):
    body = await request.json()
    text = body.get("text", "")
    zone = body.get("zone")
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)
    if not _llm or not _perception or not _watchpoint:
        return JSONResponse({"error": "Not ready"}, status_code=503)
    result = await _llm.process_watch_command(text, zone, _perception, _watchpoint)
    return JSONResponse(result)


# ── Zone Setup (simplified one-click) ──

@app.post("/api/zone/setup")
async def zone_setup(request: Request):
    """One-click zone setup for the demo dashboard."""
    body = await request.json()
    zone = body.get("zone")
    target_class = body.get("target_class", "bottle")
    label = body.get("label", "Watched Zone")

    if not zone or not _watchpoint:
        return JSONResponse({"error": "Missing zone or not ready"}, status_code=400)

    zone_id = _watchpoint.add_zone(
        x1=zone["x1"], y1=zone["y1"],
        x2=zone["x2"], y2=zone["y2"],
        target_class=target_class,
        label=label,
    )

    if _perception:
        # Only detect common objects — eliminates toilet/bench/cat false positives
        await _perception.set_config("filter_classes", [
            "person", "bottle", "cup", "cell phone", "laptop", "book",
            "remote", "keyboard", "mouse", "scissors", "clock", "vase",
            "backpack", "handbag", "suitcase",
        ])
        await _perception.set_config("min_confidence", 0.35)
        await _perception.set_config("track_lost_frames", 50)

    return JSONResponse({"zone_id": zone_id, "target_class": target_class, "status": "monitoring"})


# ── Watchpoint CRUD ──

@app.post("/api/watchpoint")
async def set_watchpoint(request: Request):
    body = await request.json()
    if not _watchpoint:
        return JSONResponse({"error": "Watchpoint not ready"}, status_code=503)

    action = body.get("action", "add")

    if action == "add":
        zone_id = _watchpoint.add_zone(
            x1=body.get("x1", 0),
            y1=body.get("y1", 0),
            x2=body.get("x2", 1),
            y2=body.get("y2", 1),
            target_class=body.get("target_class", "person"),
            label=body.get("label", "Watched Zone"),
        )
        if _perception:
            await _perception.set_config("filter_classes", [])
        return JSONResponse({"zone_id": zone_id, "status": "added"})

    elif action == "remove":
        zone_id = body.get("zone_id", "")
        removed = _watchpoint.remove_zone(zone_id)
        return JSONResponse({"zone_id": zone_id, "removed": removed})

    elif action == "clear":
        _watchpoint.clear_zones()
        if _perception:
            await _perception.set_config("filter_classes", [])
        return JSONResponse({"status": "cleared"})

    return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)


# ── Reset API (called on page refresh) ──

@app.post("/api/reset")
async def reset_state():
    """Clear all events, alerts, dwell data. Called on page load."""
    if _watchpoint:
        _watchpoint.clear_zones()
    if _aggregator:
        _aggregator._history.clear()
    return JSONResponse({"status": "cleared"})


# ── Stats API ──

@app.get("/api/stats")
async def get_stats():
    """Return current dashboard stats for page init."""
    event_count = len(_aggregator.get_recent(count=9999)) if _aggregator else 0
    alert_count = _watchpoint.get_status()["alert_count"] if _watchpoint else 0
    return JSONResponse({
        "events": event_count,
        "alerts": alert_count,
    })


# ── Events API ──

@app.get("/api/events/recent")
async def get_recent_events():
    if not _aggregator:
        return JSONResponse({"events": []})
    events = _aggregator.get_recent_significant(count=50)
    return JSONResponse({"events": [e.to_dict() for e in events]})


@app.get("/api/watchpoint/status")
async def watchpoint_status():
    if not _watchpoint:
        return JSONResponse({"error": "Not ready"}, status_code=503)
    return JSONResponse(_watchpoint.get_status())


# ── Source Switch ──

@app.post("/api/source")
async def switch_source(request: Request):
    body = await request.json()
    source = body.get("source")
    if source is None:
        return JSONResponse({"error": "Missing 'source'"}, status_code=400)
    if not _perception:
        return JSONResponse({"error": "Not connected"}, status_code=503)
    result = await _perception.send_rpc("source/switch", {"source": source})
    return JSONResponse(result)


# ── WebSocket: Events Stream ──

@app.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    await websocket.accept()
    _event_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _event_websockets:
            _event_websockets.remove(websocket)


# ── WebSocket: Alerts (backwards compat) ──

@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    await websocket.accept()
    _event_websockets.append(websocket)  # Share the same list
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _event_websockets:
            _event_websockets.remove(websocket)


# ── Health ──

@app.get("/api/health")
async def health():
    if not _perception:
        return JSONResponse({"status": "no_perception", "connected": False})
    try:
        h = await _perception.get_health()
        return JSONResponse({"status": "ok", "connected": True, **h})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "connected": False})
