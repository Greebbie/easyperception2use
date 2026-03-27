"""FastAPI demo server for investor presentation."""

import asyncio
import os
import sys
import json
import threading
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

# Global state
_perception: Optional[PerceptionClient] = None
_streamer: Optional[MJPEGStreamer] = None
_llm: Optional[LLMMiddleware] = None
_watchpoint: Optional[WatchpointMonitor] = None
_alert_websockets: list[WebSocket] = []
_scene_task: Optional[asyncio.Task] = None

# Config from environment
DRY_RUN = os.environ.get("DEMO_DRY_RUN", "0") == "1"
_raw_source = os.environ.get("DEMO_VIDEO_SOURCE", "0")
try:
    VIDEO_SOURCE = int(_raw_source)
except ValueError:
    VIDEO_SOURCE = _raw_source
WS_URI = os.environ.get("DEMO_WS_URI", "ws://127.0.0.1:18790")


async def _scene_listener():
    """Background task: subscribe to scene updates, feed streamer + watchpoint."""
    global _perception, _streamer, _watchpoint
    try:
        async for scene in _perception.subscribe_scenes():
            if _watchpoint:
                alerts = _watchpoint.check_scene(scene)
                if alerts:
                    if _streamer:
                        _streamer.trigger_alert(2.0)
                    for alert in alerts:
                        msg = json.dumps({"type": "alert", "data": alert.to_dict()})
                        stale = []
                        for ws in _alert_websockets:
                            try:
                                await ws.send_text(msg)
                            except Exception:
                                stale.append(ws)
                        for ws in stale:
                            _alert_websockets.remove(ws)

                status = _watchpoint.get_status()
                zones = status.get("zones", [])
                if _streamer:
                    _streamer.set_watchpoint_zones(zones)
    except Exception as e:
        print(f"[SceneListener] Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _perception, _streamer, _llm, _watchpoint, _scene_task

    _perception = PerceptionClient(uri=WS_URI)
    try:
        await _perception.connect(retries=15, delay=2.0)
    except ConnectionError as e:
        print(f"[DemoServer] WARNING: {e} — running without perception")
        _perception = None

    _streamer = MJPEGStreamer(fps=12)
    _llm = LLMMiddleware()
    _watchpoint = WatchpointMonitor()

    if _perception:
        _scene_task = asyncio.create_task(_scene_listener())

    print("[DemoServer] Ready at http://localhost:8080")
    yield

    if _scene_task:
        _scene_task.cancel()
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


@app.get("/")
async def index():
    index_path = os.path.join(_static_dir, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/stream")
async def video_stream():
    if not _streamer:
        return JSONResponse({"error": "Streamer not ready"}, status_code=503)
    return StreamingResponse(
        _streamer.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/scene")
async def get_scene():
    if not _perception:
        return JSONResponse({"error": "Not connected to perception"}, status_code=503)
    scene = await _perception.get_latest()
    return JSONResponse(scene)


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


@app.post("/api/watch-command")
async def watch_command(request: Request):
    """Conversational watch command — LLM interprets user intent for monitoring."""
    body = await request.json()
    text = body.get("text", "")
    zone = body.get("zone")  # {x1, y1, x2, y2} or null
    if not text:
        return JSONResponse({"error": "Missing 'text'"}, status_code=400)
    if not _llm or not _perception or not _watchpoint:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    result = await _llm.process_watch_command(text, zone, _perception, _watchpoint)
    return JSONResponse(result)


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
            # Detect all objects (so phone/items stay visible on video)
            # but only alert on target_class entering the zone
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


@app.post("/api/source")
async def switch_source(request: Request):
    """Switch the pipeline's video source (camera ID or file path)."""
    body = await request.json()
    source = body.get("source")
    if source is None:
        return JSONResponse({"error": "Missing 'source'"}, status_code=400)
    if not _perception:
        return JSONResponse({"error": "Not connected"}, status_code=503)
    result = await _perception.send_rpc("source/switch", {"source": source})
    return JSONResponse(result)


@app.get("/api/watchpoint/status")
async def watchpoint_status():
    if not _watchpoint:
        return JSONResponse({"error": "Not ready"}, status_code=503)
    return JSONResponse(_watchpoint.get_status())


@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    await websocket.accept()
    _alert_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _alert_websockets:
            _alert_websockets.remove(websocket)


@app.get("/api/health")
async def health():
    if not _perception:
        return JSONResponse({"status": "no_perception", "connected": False})
    try:
        h = await _perception.get_health()
        return JSONResponse({"status": "ok", "connected": True, **h})
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e), "connected": False})
