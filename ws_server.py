"""WebSocket server using JSON-RPC 2.0 protocol (compatible with OpenClaw Gateway)."""

import asyncio
import json
import threading
from typing import Optional

from perception_service import PerceptionService


class WebSocketServer:
    """
    JSON-RPC 2.0 WebSocket server for the perception pipeline.

    Methods:
        scene/latest   — get the latest scene JSON
        scene/subscribe — subscribe to scene updates (push)
        config/set     — update a config value
        source/switch  — switch video source
        status/health  — get pipeline health metrics
    """

    def __init__(
        self,
        service: PerceptionService,
        host: str = "0.0.0.0",
        port: int = 18790,
    ):
        self.service = service
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._subscribers: set = set()  # connected WebSocket clients subscribed to scene

    def start(self) -> None:
        """Start the WebSocket server in a daemon thread."""
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_server(self) -> None:
        """Run the asyncio event loop for the WS server."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve_and_run())

    async def _serve_and_run(self) -> None:
        """Start serving WebSocket connections."""
        try:
            from websockets.asyncio.server import serve
        except ImportError:
            print("[WSServer] websockets not installed. Run: pip install websockets")
            return

        async with serve(self._handle_client, self.host, self.port):
            print(f"[WSServer] Listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _handle_client(self, websocket) -> None:
        """Handle a single WebSocket client connection."""
        print(f"[WSServer] Client connected: {websocket.remote_address}")
        self._subscribers.add(websocket)
        try:
            async for message in websocket:
                response = self._handle_rpc(message)
                await websocket.send(json.dumps(response, ensure_ascii=False))
        except Exception as e:
            print(f"[WSServer] Client error: {e}")
        finally:
            self._subscribers.discard(websocket)
            print(f"[WSServer] Client disconnected: {websocket.remote_address}")

    def _handle_rpc(self, raw_message: str) -> dict:
        """
        Parse and handle a JSON-RPC 2.0 request.

        Returns a JSON-RPC 2.0 response dict.
        """
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            return self._rpc_error(None, -32700, "Parse error")

        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        if not method:
            return self._rpc_error(req_id, -32600, "Invalid request: missing method")

        handler_map = {
            "scene/latest": self._handle_scene_latest,
            "scene/subscribe": self._handle_scene_subscribe,
            "config/set": self._handle_config_set,
            "source/switch": self._handle_source_switch,
            "status/health": self._handle_status_health,
            "ego/motion": self._handle_ego_motion,
        }

        handler = handler_map.get(method)
        if not handler:
            return self._rpc_error(req_id, -32601, f"Method not found: {method}")

        try:
            result = handler(params)
            return self._rpc_result(req_id, result)
        except Exception as e:
            return self._rpc_error(req_id, -32000, str(e))

    def _handle_scene_latest(self, params: dict) -> dict:
        scene = self.service.get_latest_scene()
        if scene is None:
            return {"scene": None, "message": "No scene available yet"}
        return scene

    def _handle_scene_subscribe(self, params: dict) -> dict:
        return {"subscribed": True, "message": "You will receive scene updates"}

    def _handle_config_set(self, params: dict) -> dict:
        key = params.get("key")
        value = params.get("value")
        if not key:
            raise ValueError("Missing 'key' parameter")
        self.service.set_config(key, value)
        return {"key": key, "value": value, "applied": True}

    def _handle_source_switch(self, params: dict) -> dict:
        source = params.get("source")
        if source is None:
            raise ValueError("Missing 'source' parameter")
        # Try to parse as int (camera ID)
        try:
            source = int(source)
        except (ValueError, TypeError):
            pass
        queued = self.service.switch_source(source)
        return {"source": source, "queued": queued}

    def _handle_status_health(self, params: dict) -> dict:
        return self.service.get_status()

    def _handle_ego_motion(self, params: dict) -> dict:
        """Set robot ego-motion state from external source (odometry/IMU).

        Params: {"moving": bool, "vx": float, "vy": float}
        """
        moving = bool(params.get("moving", False))
        vx = float(params.get("vx", 0.0))
        vy = float(params.get("vy", 0.0))
        self.service.set_ego_motion(moving, vx, vy)
        return {"moving": moving, "vx": vx, "vy": vy, "applied": True}

    def _on_scene_update(self, scene_json: dict) -> None:
        """Push scene updates to all subscribed WebSocket clients."""
        if not self._subscribers or not self._loop:
            return
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "scene/update",
            "params": scene_json,
        }, ensure_ascii=False)

        for ws in list(self._subscribers):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(msg), self._loop)
            except Exception:
                self._subscribers.discard(ws)

    @staticmethod
    def _rpc_result(req_id, result: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }

    @staticmethod
    def _rpc_error(req_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }
