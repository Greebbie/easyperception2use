"""
Perception Pipeline v3.1 — Main entry point.

Processes camera video streams into structured scene JSON
for downstream LLM/robot decision-making.

Usage:
    python main.py                          # USB camera, default settings
    python main.py --source video.mp4       # Local video file
    python main.py --source rtsp://...      # RTSP stream
    python main.py --dry-run                # Synthetic data, no camera needed
    python main.py --no-viz --output file   # Headless, output to file
    python main.py --depth                  # Enable depth estimation
    python main.py --gui                    # Open config GUI panel
    python main.py --ws                     # Start WebSocket server (JSON-RPC 2.0)
    python main.py --gui --depth --ws       # Full feature mode
"""

import queue
import signal
import sys
import time
import argparse

import cv2

from config import DEFAULT_CONFIG
from metrics import PipelineMetrics
from output_controller import OutputController
from output_handler import OutputHandler
from scene_differ import SceneDiffer
from visualizer import Visualizer


# === Graceful shutdown ===
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print("\n[Main] Shutdown signal received, cleaning up...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main() -> None:
    args = parse_args()
    config = load_config(args)

    print("=" * 60)
    print("  Perception Pipeline v3.1")
    print("=" * 60)
    print(f"  Source:       {config['source']}")
    print(f"  Model:        {config['model_path']}")
    print(f"  Process FPS:  {config['process_fps']} FPS")
    print(f"  Output:       {config['output_strategy']}"
          f" (interval {config['output_interval_sec']}s)")
    print(f"  Method:       {config['output_method']}")
    print(f"  Classes:      {config['filter_classes'] or 'all'}")
    print(f"  Visualization:{'on' if config['show_visualization'] else 'off'}")
    print(f"  Depth:        {'on' if config['depth_enabled'] else 'off'}")
    print(f"  GUI:          {'on' if config['show_gui'] else 'off'}")
    print(f"  WebSocket:    {'on' if config['ws_enabled'] else 'off'}")
    if args.dry_run:
        print("  Mode:         DRY RUN (synthetic data)")
    print("=" * 60)
    print("  Press 'q' or Ctrl+C to exit")
    print()

    if args.dry_run:
        _run_dry_run(config)
    else:
        _run_live(config)


def _run_live(config: dict) -> None:
    """Run the pipeline with a real camera/video source and YOLO model."""
    from frame_grabber import FrameGrabber
    from scene_builder import SceneBuilder

    metrics = PipelineMetrics()
    metrics.set_state("initializing")
    differ = SceneDiffer()

    # Load YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO(config["model_path"])
    except Exception as e:
        print(f"[Main] Failed to load YOLO model: {e}")
        sys.exit(1)

    grabber = FrameGrabber(
        source=config["source"],
        max_retries=config["max_retries"],
        retry_interval=config["retry_interval"],
    )

    # Wait for first frame to get dimensions
    frame_size = None
    for _ in range(100):
        frame_size = grabber.get_frame_size()
        if frame_size:
            break
        time.sleep(0.03)
    if not frame_size:
        print("[Main] Error: could not get frame size")
        grabber.release()
        sys.exit(1)

    w, h = frame_size
    print(f"[Main] Frame size: {w}x{h}")

    builder = SceneBuilder(w, h, config)
    output_ctrl = OutputController(
        strategy=config["output_strategy"],
        interval_sec=config["output_interval_sec"],
        change_threshold=config["output_change_threshold"],
    )
    output_handler = OutputHandler(config["output_method"], config)
    viz = Visualizer() if config["show_visualization"] else None

    # Depth estimator
    depth_estimator = None
    if config.get("depth_enabled"):
        from depth_estimator import DepthEstimator
        depth_estimator = DepthEstimator(
            model_size=config.get("depth_model_size", "small"),
            device=config.get("depth_device", "auto"),
            enabled=True,
        )

    # GUI + source switch queue
    gui = None
    source_switch_queue: queue.Queue = queue.Queue(maxsize=1)

    if config.get("show_gui"):
        from config_gui import ConfigGUI

        def _on_config_change(key: str, value) -> None:
            config[key] = value
            if key == "source":
                try:
                    source_switch_queue.put_nowait(value)
                except queue.Full:
                    pass
            elif key in ("output_strategy", "output_interval_sec",
                         "output_change_threshold"):
                nonlocal output_ctrl
                output_ctrl = OutputController(
                    strategy=config["output_strategy"],
                    interval_sec=config["output_interval_sec"],
                    change_threshold=config["output_change_threshold"],
                )
            elif key == "depth_enabled":
                nonlocal depth_estimator
                if value and depth_estimator is None:
                    from depth_estimator import DepthEstimator
                    depth_estimator = DepthEstimator(
                        model_size=config.get("depth_model_size", "small"),
                        device=config.get("depth_device", "auto"),
                        enabled=True,
                    )
                elif not value and depth_estimator is not None:
                    depth_estimator.enabled = False

        gui = ConfigGUI(config, _on_config_change)
        gui.start()

    # WebSocket server
    ws_server = None
    if config.get("ws_enabled"):
        from perception_service import PerceptionService
        from ws_server import WebSocketServer

        # Create a lightweight service wrapper for WS
        svc = PerceptionService(config)
        ws_server = WebSocketServer(
            svc,
            host=config.get("ws_host", "0.0.0.0"),
            port=config.get("ws_port", 18790),
        )
        ws_server.start()

    metrics.set_state("running")
    last_process_time = 0.0
    last_scene_json = None

    try:
        while not _shutdown_requested:
            # Handle source switch
            try:
                new_source = source_switch_queue.get_nowait()
                if grabber.switch_source(new_source):
                    new_size = None
                    for _ in range(50):
                        new_size = grabber.get_frame_size()
                        if new_size:
                            break
                        time.sleep(0.05)
                    if new_size:
                        nw, nh = new_size
                        builder.update_frame_size(nw, nh)
                        differ.reset()
                        last_scene_json = None
                        print(f"[Main] Switched to: {new_source} ({nw}x{nh})")
                        if gui:
                            gui.set_status(f"Switched to: {new_source}")
                else:
                    print(f"[Main] Failed to switch to: {new_source}")
                    if gui:
                        gui.set_status(f"Failed: {new_source}")
            except queue.Empty:
                pass

            if not grabber.is_alive():
                print("[Main] Video source disconnected")
                metrics.set_state("error")
                break

            ok, frame = grabber.get_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            fps = max(config.get("process_fps", 10), 1)
            frame_interval = 1.0 / fps
            if (now - last_process_time) < frame_interval:
                continue

            last_process_time = now
            t_start = time.time()

            # YOLO inference with error handling
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=config["tracker"],
                    verbose=False,
                )
            except Exception as e:
                print(f"[Main] YOLO inference error: {e}")
                metrics.add_degraded_module("yolo")
                continue

            t_detect = time.time()

            # Depth estimation
            depth_map = None
            if (depth_estimator
                    and depth_estimator.enabled
                    and config.get("depth_enabled")):
                depth_map = depth_estimator.estimate(frame)
                if depth_map is None and depth_estimator._load_failed:
                    metrics.add_degraded_module("depth")

            t_depth = time.time()

            # Build scene JSON
            latency_ms = {
                "grab_to_detect": round((t_detect - t_start) * 1000, 1),
                "detect_to_depth": round((t_depth - t_detect) * 1000, 1),
            }

            scene_json = builder.build(
                results, now, depth_map=depth_map, latency_ms=None
            )

            t_output = time.time()
            latency_ms["depth_to_output"] = round((t_output - t_depth) * 1000, 1)
            latency_ms["total"] = round((t_output - t_start) * 1000, 1)
            scene_json["latency_ms"] = latency_ms
            scene_json["pipeline"] = metrics.get_pipeline_info()

            # Scene diff
            changes = differ.diff(scene_json)
            scene_json["changes"] = changes

            # Record metrics
            metrics.record_frame(latency_ms)

            last_scene_json = scene_json

            if output_ctrl.should_output(scene_json):
                output_handler(scene_json)

            # Visualization
            if config.get("show_visualization") and viz and last_scene_json is not None:
                viz_frame = viz.draw(frame.copy(), last_scene_json)
                scale = config.get("viz_scale", 1.0)
                if scale != 1.0:
                    viz_frame = cv2.resize(viz_frame, None, fx=scale, fy=scale)
                cv2.imshow(config["viz_window_name"], viz_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        print("[Main] Cleaning up resources...")
        metrics.set_state("stopped")
        if gui:
            gui.stop()
        if ws_server:
            ws_server.stop()
        grabber.release()
        output_handler.close()
        cv2.destroyAllWindows()
        print("[Main] Exited")


def _run_dry_run(config: dict) -> None:
    """Run the pipeline with synthetic data (no camera or YOLO needed)."""
    from dry_run import DryRunGenerator

    metrics = PipelineMetrics()
    metrics.set_state("running")
    differ = SceneDiffer()

    generator = DryRunGenerator(frame_w=1280, frame_h=720, num_objects=5)
    output_ctrl = OutputController(
        strategy=config["output_strategy"],
        interval_sec=config["output_interval_sec"],
        change_threshold=config["output_change_threshold"],
    )
    output_handler = OutputHandler(config["output_method"], config)
    viz = Visualizer() if config["show_visualization"] else None

    # GUI in dry-run
    gui = None
    if config.get("show_gui"):
        from config_gui import ConfigGUI

        def _on_config_change(key: str, value) -> None:
            config[key] = value
            if key in ("output_strategy", "output_interval_sec",
                       "output_change_threshold"):
                nonlocal output_ctrl
                output_ctrl = OutputController(
                    strategy=config["output_strategy"],
                    interval_sec=config["output_interval_sec"],
                    change_threshold=config["output_change_threshold"],
                )

        gui = ConfigGUI(config, _on_config_change)
        gui.start()

    # WebSocket server (dry-run uses a simple standalone server)
    ws_server = None
    latest_scene_holder = {"scene": None}

    if config.get("ws_enabled"):
        from ws_server import WebSocketServer
        from perception_service import PerceptionService

        # Create a minimal service stub for WS in dry-run mode
        svc = PerceptionService.__new__(PerceptionService)
        svc.config = config
        svc.metrics = metrics
        svc._latest_scene = None
        svc._scene_lock = __import__("threading").Lock()
        svc._subscribers = {}
        svc._sub_lock = __import__("threading").Lock()
        svc._source_switch_queue = queue.Queue(maxsize=1)

        def _get_latest():
            return latest_scene_holder["scene"]
        svc.get_latest_scene = _get_latest
        svc.get_status = metrics.get_health
        svc.set_config = lambda k, v: config.__setitem__(k, v)
        svc.switch_source = lambda s: False

        ws_server = WebSocketServer(
            svc,
            host=config.get("ws_host", "0.0.0.0"),
            port=config.get("ws_port", 18790),
        )
        ws_server.start()

    last_process_time = 0.0

    try:
        while not _shutdown_requested:
            now = time.time()
            fps = max(config.get("process_fps", 10), 1)
            frame_interval = 1.0 / fps
            if (now - last_process_time) < frame_interval:
                time.sleep(0.005)
                continue

            last_process_time = now
            t_start = time.time()

            frame = generator.generate_frame()
            scene_json = generator.generate_scene_json(now)

            # Add production fields
            t_output = time.time()
            latency_ms = {"total": round((t_output - t_start) * 1000, 1)}
            scene_json["latency_ms"] = latency_ms
            scene_json["pipeline"] = metrics.get_pipeline_info()

            changes = differ.diff(scene_json)
            scene_json["changes"] = changes

            metrics.record_frame(latency_ms)

            # Update WS latest scene
            latest_scene_holder["scene"] = scene_json

            if output_ctrl.should_output(scene_json):
                output_handler(scene_json)

            if config.get("show_visualization") and viz:
                viz_frame = viz.draw(frame.copy(), scene_json)
                scale = config.get("viz_scale", 1.0)
                if scale != 1.0:
                    viz_frame = cv2.resize(viz_frame, None, fx=scale, fy=scale)
                cv2.imshow(config["viz_window_name"], viz_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        print("[Main] Cleaning up resources...")
        metrics.set_state("stopped")
        if gui:
            gui.stop()
        if ws_server:
            ws_server.stop()
        output_handler.close()
        cv2.destroyAllWindows()
        print("[Main] Exited")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perception Pipeline v3.1")
    parser.add_argument("--source", default=None, help="Camera ID or video path")
    parser.add_argument("--model", default=None, help="YOLO model path")
    parser.add_argument("--process-fps", type=int, default=None,
                        help="Detection processing frame rate")
    parser.add_argument("--output", default=None,
                        help="Output method: print / file / callback")
    parser.add_argument("--strategy", default=None,
                        help="Output strategy: every_frame / interval / "
                             "on_change / hybrid")
    parser.add_argument("--interval", type=float, default=None,
                        help="Output interval (seconds)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization window")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Only detect these classes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic data (no camera/model needed)")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth estimation (Depth Anything v2)")
    parser.add_argument("--depth-model", default=None,
                        choices=["small", "base", "large"],
                        help="Depth model size")
    parser.add_argument("--gui", action="store_true",
                        help="Show configuration GUI panel")
    parser.add_argument("--ws", action="store_true",
                        help="Start WebSocket server (JSON-RPC 2.0)")
    parser.add_argument("--ws-port", type=int, default=None,
                        help="WebSocket server port (default: 18790)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    """Merge default config with command-line arguments."""
    config = DEFAULT_CONFIG.copy()
    if args.source is not None:
        try:
            config["source"] = int(args.source)
        except ValueError:
            config["source"] = args.source
    if args.model:
        config["model_path"] = args.model
    if args.process_fps:
        config["process_fps"] = args.process_fps
    if args.output:
        config["output_method"] = args.output
    if args.strategy:
        config["output_strategy"] = args.strategy
    if args.interval:
        config["output_interval_sec"] = args.interval
    if args.no_viz:
        config["show_visualization"] = False
    if args.classes:
        config["filter_classes"] = args.classes
    if args.depth:
        config["depth_enabled"] = True
    if args.depth_model:
        config["depth_model_size"] = args.depth_model
    if args.gui:
        config["show_gui"] = True
    if args.ws:
        config["ws_enabled"] = True
    if args.ws_port:
        config["ws_port"] = args.ws_port
    return config


if __name__ == "__main__":
    main()
