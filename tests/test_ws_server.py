"""Tests for WebSocket server JSON-RPC handling and security hardening."""

import json
import pytest

from ws_server import WebSocketServer, _SETTABLE_KEYS, _MAX_CONNECTIONS


class MockService:
    """Minimal mock of PerceptionService for WS server tests."""

    def __init__(self):
        self._config = {}
        self._latest_scene = None
        self._ego_state = {}

    def get_latest_scene(self):
        return self._latest_scene

    def set_config(self, key, value):
        self._config[key] = value

    def switch_source(self, source):
        return True

    def get_status(self):
        return {"state": "running", "fps": 10.0}

    def set_ego_motion(self, moving, vx=0.0, vy=0.0):
        self._ego_state = {"moving": moving, "vx": vx, "vy": vy}


@pytest.fixture
def server():
    return WebSocketServer(MockService(), host="127.0.0.1", port=0)


# =========================================================================
# JSON-RPC parsing
# =========================================================================

class TestRpcParsing:
    def test_invalid_json(self, server):
        resp = server._handle_rpc("not json{")
        assert resp["error"]["code"] == -32700

    def test_missing_method(self, server):
        resp = server._handle_rpc('{"id": 1}')
        assert resp["error"]["code"] == -32600

    def test_unknown_method(self, server):
        resp = server._handle_rpc('{"id": 1, "method": "foo/bar"}')
        assert resp["error"]["code"] == -32601

    def test_notification_no_response(self, server):
        """JSON-RPC 2.0: requests without 'id' are notifications — no response."""
        resp = server._handle_rpc('{"method": "status/health"}')
        assert resp is None

    def test_valid_request(self, server):
        resp = server._handle_rpc('{"id": 1, "method": "status/health"}')
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["state"] == "running"


# =========================================================================
# Scene methods
# =========================================================================

class TestSceneMethods:
    def test_scene_latest_no_scene(self, server):
        resp = server._handle_rpc('{"id": 1, "method": "scene/latest"}')
        assert resp["result"]["scene"] is None

    def test_scene_latest_with_scene(self, server):
        server.service._latest_scene = {"objects": [], "timestamp": 1.0}
        resp = server._handle_rpc('{"id": 1, "method": "scene/latest"}')
        assert resp["result"]["timestamp"] == 1.0

    def test_scene_subscribe(self, server):
        resp = server._handle_rpc('{"id": 1, "method": "scene/subscribe"}')
        assert resp["result"]["subscribed"] is True


# =========================================================================
# Config security (allowlist)
# =========================================================================

class TestConfigSecurity:
    def test_set_allowed_key(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"key": "min_confidence", "value": 0.6}
        }))
        assert resp["result"]["applied"] is True
        assert server.service._config["min_confidence"] == 0.6

    def test_set_blocked_key_model_path(self, server):
        """model_path must NOT be settable via RPC."""
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"key": "model_path", "value": "/etc/malicious"}
        }))
        assert "error" in resp
        assert "model_path" not in server.service._config

    def test_set_blocked_key_output_file_path(self, server):
        """output_file_path must NOT be settable via RPC (path traversal risk)."""
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"key": "output_file_path", "value": "../../etc/cron.d/x"}
        }))
        assert "error" in resp

    def test_set_blocked_key_ws_host(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"key": "ws_host", "value": "0.0.0.0"}
        }))
        assert "error" in resp

    def test_set_missing_key(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"value": 0.5}
        }))
        assert "error" in resp

    def test_error_response_no_internal_details(self, server):
        """Error messages should not leak internal exception details."""
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "config/set",
            "params": {"key": "model_path", "value": "x"}
        }))
        assert resp["error"]["message"] == "Internal error"


# =========================================================================
# Ego motion
# =========================================================================

class TestEgoMotion:
    def test_set_ego_motion(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "ego/motion",
            "params": {"moving": True, "vx": 0.1, "vy": -0.05}
        }))
        assert resp["result"]["applied"] is True
        assert server.service._ego_state["moving"] is True

    def test_ego_motion_defaults(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "ego/motion",
            "params": {}
        }))
        assert resp["result"]["moving"] is False
        assert resp["result"]["vx"] == 0.0


# =========================================================================
# Source switch
# =========================================================================

class TestSourceSwitch:
    def test_switch_source(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "source/switch",
            "params": {"source": "0"}
        }))
        assert resp["result"]["queued"] is True
        assert resp["result"]["source"] == 0  # parsed as int

    def test_switch_source_missing(self, server):
        resp = server._handle_rpc(json.dumps({
            "id": 1, "method": "source/switch",
            "params": {}
        }))
        assert "error" in resp


# =========================================================================
# Thread safety: subscriber management
# =========================================================================

class TestSubscriberManagement:
    def test_on_scene_update_no_loop(self, server):
        """_on_scene_update should not crash when no event loop is running."""
        server._loop = None
        server._on_scene_update({"objects": []})  # should not raise

    def test_on_scene_update_no_subscribers(self, server):
        """_on_scene_update should be a no-op with no subscribers."""
        server._loop = None
        server._on_scene_update({"objects": []})  # should not raise
