"""PerceptionService — programmatic API wrapping the entire pipeline."""

import queue
import threading
import time
import uuid
from typing import Any, Callable, Optional

import cv2

from config import DEFAULT_CONFIG
from metrics import PipelineMetrics
from output_controller import OutputController
from output_handler import OutputHandler
from scene_differ import SceneDiffer
from visualizer import Visualizer


class PerceptionService:
    """
    Programmatic API for the perception pipeline.

    Encapsulates the full pipeline lifecycle:
    FrameGrabber → YOLO → Depth → SceneBuilder → OutputController → subscribers

    Usage:
        svc = PerceptionService(config)
        svc.start()
        scene = svc.get_latest_scene()
        sub_id = svc.subscribe(my_callback)
        svc.stop()
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self.metrics = PipelineMetrics()
        self._latest_scene: Optional[dict] = None
        self._scene_lock = threading.Lock()
        self._subscribers: dict[str, Callable] = {}
        self._sub_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._source_switch_queue: queue.Queue = queue.Queue(maxsize=1)

        # Components (initialized in start)
        self._grabber = None
        self._model = None
        self._builder = None
        self._depth_estimator = None
        self._differ = SceneDiffer(
            cooldown_sec=self.config.get("differ_cooldown_sec", 0.5),
        )
        self._output_ctrl = None
        self._output_handler = None

    def start(self) -> None:
        """Start the pipeline in a background thread."""
        if self._running:
            return
        self._running = True
        self.metrics.set_state("initializing")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._running = False
        self.metrics.set_state("stopped")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._grabber:
            self._grabber.release()
        if self._output_handler:
            self._output_handler.close()

    def get_latest_scene(self) -> Optional[dict]:
        """Get the most recent scene JSON (non-blocking)."""
        with self._scene_lock:
            return self._latest_scene

    def subscribe(
        self,
        callback: Callable[[dict], None],
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> str:
        """
        Subscribe to scene updates.

        Args:
            callback: called with scene_json on each output
            filter_fn: optional filter — callback only fires if filter returns True

        Returns:
            subscription ID for unsubscribe
        """
        sub_id = str(uuid.uuid4())[:8]
        with self._sub_lock:
            self._subscribers[sub_id] = (callback, filter_fn)
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        """Remove a subscription."""
        with self._sub_lock:
            self._subscribers.pop(sub_id, None)

    def set_config(self, key: str, value: Any) -> None:
        """Update a config value at runtime."""
        self.config[key] = value
        if key == "source":
            try:
                self._source_switch_queue.put_nowait(value)
            except queue.Full:
                pass

    def set_ego_motion(self, moving: bool, vx: float = 0.0, vy: float = 0.0) -> None:
        """Set robot ego-motion state (from odometry/IMU)."""
        if self._builder:
            self._builder.set_ego_motion(moving, vx, vy)

    def switch_source(self, source: int | str) -> bool:
        """Request a video source switch (async, returns immediately)."""
        try:
            self._source_switch_queue.put_nowait(source)
            return True
        except queue.Full:
            return False

    def get_status(self) -> dict:
        """Get pipeline health status."""
        return self.metrics.get_health()

    def _run(self) -> None:
        """Main pipeline loop (runs in background thread)."""
        try:
            self._init_components()
            self.metrics.set_state("running")
            self._main_loop()
        except Exception as e:
            print(f"[PerceptionService] Fatal error: {e}")
            self.metrics.set_state("error")
        finally:
            self._cleanup()

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        from frame_grabber import FrameGrabber
        from scene_builder import SceneBuilder

        # YOLO model
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.config["model_path"])
        except Exception as e:
            print(f"[PerceptionService] Failed to load YOLO model: {e}")
            raise

        # Frame grabber
        self._grabber = FrameGrabber(
            source=self.config["source"],
            max_retries=self.config["max_retries"],
            retry_interval=self.config["retry_interval"],
        )

        # Wait for first frame
        frame_size = None
        for _ in range(100):
            frame_size = self._grabber.get_frame_size()
            if frame_size:
                break
            time.sleep(0.03)
        if not frame_size:
            raise RuntimeError("Could not get frame size from source")

        w, h = frame_size
        print(f"[PerceptionService] Frame size: {w}x{h}")

        self._builder = SceneBuilder(w, h, self.config)

        # Depth (optional)
        if self.config.get("depth_enabled"):
            from depth_estimator import DepthEstimator
            self._depth_estimator = DepthEstimator(
                model_size=self.config.get("depth_model_size", "small"),
                device=self.config.get("depth_device", "auto"),
                enabled=True,
            )

        self._output_ctrl = OutputController(
            strategy=self.config["output_strategy"],
            interval_sec=self.config["output_interval_sec"],
            change_threshold=self.config["output_change_threshold"],
            stable_window_sec=self.config.get("stable_window_sec", 1.0),
        )
        self._output_handler = OutputHandler(
            self.config["output_method"], self.config
        )

    def _main_loop(self) -> None:
        """Core processing loop."""
        last_process_time = 0.0

        while self._running:
            # Handle source switch
            self._handle_source_switch()

            if not self._grabber.is_alive():
                print("[PerceptionService] Source disconnected")
                self.metrics.set_state("error")
                break

            ok, frame = self._grabber.get_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            fps = max(self.config.get("process_fps", 10), 1)
            frame_interval = 1.0 / fps
            if (now - last_process_time) < frame_interval:
                continue

            last_process_time = now
            t_start = time.time()

            # YOLO inference
            try:
                results = self._model.track(
                    frame,
                    persist=True,
                    tracker=self.config["tracker"],
                    verbose=False,
                )
            except Exception as e:
                print(f"[PerceptionService] YOLO error: {e}")
                self.metrics.add_degraded_module("yolo")
                continue

            t_detect = time.time()

            # Depth
            depth_map = None
            if (self._depth_estimator
                    and self._depth_estimator.enabled
                    and self.config.get("depth_enabled")):
                depth_map = self._depth_estimator.estimate(frame)
                if depth_map is None and self._depth_estimator._load_failed:
                    self.metrics.add_degraded_module("depth")

            t_depth = time.time()

            # Build scene
            latency_ms = {
                "grab_to_detect": round((t_detect - t_start) * 1000, 1),
                "detect_to_depth": round((t_depth - t_detect) * 1000, 1),
            }

            # Build depth_fn from depth_map if available
            depth_fn = None
            if depth_map is not None:
                from depth_estimator import DepthEstimator
                depth_fn = lambda bbox_px, _dm=depth_map: (
                    DepthEstimator.get_object_depth(_dm, bbox_px)
                )

            scene_json = self._builder.build(
                results, now, depth_fn=depth_fn, latency_ms=None, frame=frame
            )

            # Add pipeline info and latency
            t_output = time.time()
            latency_ms["depth_to_output"] = round((t_output - t_depth) * 1000, 1)
            latency_ms["total"] = round((t_output - t_start) * 1000, 1)
            scene_json["latency_ms"] = latency_ms
            scene_json["pipeline"] = self.metrics.get_pipeline_info()

            # Scene diff
            changes = self._differ.diff(scene_json)
            scene_json["changes"] = changes

            # Record metrics
            self.metrics.record_frame(latency_ms)

            # Store latest
            with self._scene_lock:
                self._latest_scene = scene_json

            # Output + notify subscribers
            if self._output_ctrl.should_output(scene_json):
                self._output_handler(scene_json)
                self._notify_subscribers(scene_json)

    def _handle_source_switch(self) -> None:
        """Check and execute pending source switches."""
        try:
            new_source = self._source_switch_queue.get_nowait()
        except queue.Empty:
            return

        if self._grabber.switch_source(new_source):
            new_size = None
            for _ in range(50):
                new_size = self._grabber.get_frame_size()
                if new_size:
                    break
                time.sleep(0.05)
            if new_size:
                self._builder.update_frame_size(*new_size)
                self._differ.reset()
                print(f"[PerceptionService] Switched to: {new_source}")
        else:
            print(f"[PerceptionService] Failed to switch to: {new_source}")

    def _notify_subscribers(self, scene_json: dict) -> None:
        """Notify all subscribers of a new scene."""
        with self._sub_lock:
            subs = list(self._subscribers.items())
        for sub_id, (callback, filter_fn) in subs:
            try:
                if filter_fn is None or filter_fn(scene_json):
                    callback(scene_json)
            except Exception as e:
                print(f"[PerceptionService] Subscriber {sub_id} error: {e}")

    def _cleanup(self) -> None:
        """Release all resources."""
        if self._grabber:
            self._grabber.release()
        if self._output_handler:
            self._output_handler.close()
        print("[PerceptionService] Stopped")
