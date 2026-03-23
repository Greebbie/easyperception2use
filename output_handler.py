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
            config: dict containing output_file_path, output_compact, etc.
        """
        self.method = method
        self._compact = config.get("output_compact", False)
        self._compact_fn = None
        self._file = None
        self._callback: Optional[Callable] = None

        if method == "file":
            path = config.get("output_file_path", "scene_output.jsonl")
            try:
                self._file = open(path, "a", encoding="utf-8")
            except OSError as e:
                raise RuntimeError(
                    f"[OutputHandler] Cannot open output file '{path}': {e}"
                ) from e
            print(f"[OutputHandler] Writing to file: {path}")

    def set_compact_fn(self, fn: Callable) -> None:
        """Set the compact transform function (SceneBuilder.compact)."""
        self._compact_fn = fn

    def __call__(self, scene_json: dict) -> None:
        """Output the scene JSON using the configured method."""
        output = scene_json
        if self._compact and self._compact_fn:
            output = self._compact_fn(scene_json)

        if self.method == "print":
            print(json.dumps(output, ensure_ascii=False, indent=2))
        elif self.method == "file" and self._file:
            self._file.write(json.dumps(output, ensure_ascii=False) + "\n")
            self._file.flush()
        elif self.method == "callback" and self._callback:
            self._callback(output)

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
