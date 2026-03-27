"""Async WebSocket client for the perception pipeline's JSON-RPC 2.0 server."""

import asyncio
import json
from typing import AsyncGenerator, Optional

import websockets


class PerceptionClient:
    """Two persistent connections: RPC (no push) and subscription (push only)."""

    def __init__(self, uri: str = "ws://127.0.0.1:18790"):
        self.uri = uri
        self._ws_rpc: Optional[websockets.ClientConnection] = None
        self._ws_sub: Optional[websockets.ClientConnection] = None
        self._req_id = 0
        self._latest_scene: Optional[dict] = None
        self._rpc_lock = asyncio.Lock()

    async def connect(self, retries: int = 10, delay: float = 1.0) -> None:
        for attempt in range(retries):
            try:
                self._ws_rpc = await websockets.connect(self.uri, max_size=256 * 1024)
                self._ws_sub = await websockets.connect(self.uri, max_size=256 * 1024)
                print(f"[PerceptionClient] Connected to {self.uri}")
                return
            except (ConnectionRefusedError, OSError) as e:
                if attempt < retries - 1:
                    print(f"[PerceptionClient] Attempt {attempt + 1} failed, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Cannot connect to {self.uri}") from e

    async def close(self) -> None:
        for ws in (self._ws_rpc, self._ws_sub):
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
        self._ws_rpc = None
        self._ws_sub = None

    async def send_rpc(self, method: str, params: Optional[dict] = None) -> dict:
        """Send RPC on the dedicated RPC connection (receives no push messages)."""
        if not self._ws_rpc:
            raise ConnectionError("Not connected")
        async with self._rpc_lock:
            self._req_id += 1
            req_id = self._req_id
            request = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
            await self._ws_rpc.send(json.dumps(request))
            raw = await asyncio.wait_for(self._ws_rpc.recv(), timeout=15.0)
            response = json.loads(raw)
            if "error" in response:
                raise RuntimeError(f"RPC error: {response['error']}")
            return response.get("result", {})

    async def get_latest(self) -> dict:
        return await self.send_rpc("scene/latest")

    async def set_config(self, key: str, value) -> dict:
        return await self.send_rpc("config/set", {"key": key, "value": value})

    async def get_health(self) -> dict:
        return await self.send_rpc("status/health")

    async def subscribe_scenes(self) -> AsyncGenerator[dict, None]:
        """Subscribe via the dedicated sub connection (receives push messages)."""
        if not self._ws_sub:
            raise ConnectionError("Not connected")
        self._req_id += 1
        sub_req = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": "scene/subscribe",
            "params": {},
        }
        await self._ws_sub.send(json.dumps(sub_req))
        await self._ws_sub.recv()  # consume subscribe response

        while True:
            try:
                raw = await self._ws_sub.recv()
                msg = json.loads(raw)
                if msg.get("method") == "scene/update":
                    scene = msg.get("params", {})
                    self._latest_scene = scene
                    yield scene
            except websockets.ConnectionClosed:
                print("[PerceptionClient] Subscription lost, reconnecting...")
                try:
                    self._ws_sub = await websockets.connect(self.uri, max_size=256 * 1024)
                    await self._ws_sub.send(json.dumps(sub_req))
                    await self._ws_sub.recv()
                except Exception:
                    await asyncio.sleep(2)

    @property
    def latest_scene(self) -> Optional[dict]:
        return self._latest_scene
