"""Output handler with resource management for scene JSON."""

import json
from typing import Callable, Optional


class OutputHandler:
    """
    Output processor with lifecycle management.

    Supports print, file, and callback output methods.
    Must call close() on exit, or use as a context manager.
    """

    def __init__(self, method: str, config: dict):
        """
        Args:
            method: "print" / "file" / "callback"
            config: dict containing output_file_path etc.
        """
        self.method = method
        self._file = None
        self._callback: Optional[Callable] = None

        if method == "file":
            path = config.get("output_file_path", "scene_output.jsonl")
            self._file = open(path, "a", encoding="utf-8")
            print(f"[OutputHandler] Writing to file: {path}")

    def __call__(self, scene_json: dict) -> None:
        """Output the scene JSON using the configured method."""
        if self.method == "print":
            print(json.dumps(scene_json, ensure_ascii=False, indent=2))
        elif self.method == "file" and self._file:
            self._file.write(json.dumps(scene_json, ensure_ascii=False) + "\n")
            self._file.flush()
        elif self.method == "callback" and self._callback:
            self._callback(scene_json)

    def set_callback(self, fn: Callable) -> None:
        """Register a callback function for callback output mode."""
        self._callback = fn

    def close(self) -> None:
        """Release resources. Must be called on program exit."""
        if self._file:
            self._file.close()
            self._file = None
            print("[OutputHandler] File closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
