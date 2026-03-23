"""Tests for OutputHandler resource management and output methods."""

import json
import os
import tempfile

import pytest

from output_handler import OutputHandler


class TestOutputHandlerPrint:
    def test_print_method(self, capsys):
        handler = OutputHandler("print", {})
        handler({"test": "data"})
        captured = capsys.readouterr()
        assert '"test": "data"' in captured.out

    def test_print_compact(self, capsys):
        handler = OutputHandler("print", {"output_compact": True})
        handler.set_compact_fn(lambda s: {"ts": s.get("timestamp", 0)})
        handler({"timestamp": 1.0, "objects": []})
        captured = capsys.readouterr()
        assert '"ts": 1.0' in captured.out


class TestOutputHandlerFile:
    def test_file_method(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False) as f:
            path = f.name
        try:
            handler = OutputHandler("file", {"output_file_path": path})
            handler({"frame": 1})
            handler({"frame": 2})
            handler.close()

            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["frame"] == 1
            assert json.loads(lines[1])["frame"] == 2
        finally:
            os.unlink(path)

    def test_file_invalid_path_raises(self):
        with pytest.raises(RuntimeError, match="Cannot open output file"):
            OutputHandler("file", {
                "output_file_path": "/nonexistent/dir/output.jsonl"
            })

    def test_close_idempotent(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False) as f:
            path = f.name
        try:
            handler = OutputHandler("file", {"output_file_path": path})
            handler.close()
            handler.close()  # second close should not raise
        finally:
            os.unlink(path)

    def test_context_manager(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                          delete=False) as f:
            path = f.name
        try:
            with OutputHandler("file", {"output_file_path": path}) as handler:
                handler({"frame": 1})
            # File should be closed after exiting context
            with open(path, "r") as f:
                assert len(f.readlines()) == 1
        finally:
            os.unlink(path)


class TestOutputHandlerCallback:
    def test_callback_method(self):
        received = []
        handler = OutputHandler("callback", {})
        handler.set_callback(received.append)
        handler({"frame": 1})
        handler({"frame": 2})
        assert len(received) == 2
        assert received[0]["frame"] == 1

    def test_callback_not_set(self):
        handler = OutputHandler("callback", {})
        handler({"frame": 1})  # should not raise
