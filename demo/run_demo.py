"""Launcher script for the investor demo."""

import argparse
import os
import subprocess
import sys
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="EasyPerception Investor Demo")
    parser.add_argument("--source", default="0", help="Video source (camera ID or file path)")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data (no camera)")
    parser.add_argument("--port", type=int, default=8080, help="Demo server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    demo_dir = os.path.dirname(os.path.abspath(__file__))

    # Build perception pipeline command
    pipeline_cmd = [
        sys.executable, os.path.join(project_root, "main.py"),
        "--ws", "--no-viz",
        "--strategy", "hybrid",
    ]
    share_frame_path = os.path.join(project_root, "_demo_frame.jpg")
    if args.dry_run:
        pipeline_cmd.append("--dry-run")
    else:
        pipeline_cmd.extend(["--source", str(args.source)])
    pipeline_cmd.extend(["--share-frame", share_frame_path])

    print("=" * 60)
    print("  EasyPerception — Investor Demo Launcher")
    print("=" * 60)
    print()

    # Start perception pipeline
    print("[1/3] Starting perception pipeline...")
    print(f"      Command: {' '.join(pipeline_cmd)}")
    pipeline_proc = subprocess.Popen(
        pipeline_cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for pipeline to be ready
    print("[2/3] Waiting for perception WebSocket server...")
    ready = False
    for i in range(20):
        time.sleep(1)
        if pipeline_proc.poll() is not None:
            print("      ERROR: Pipeline process exited unexpectedly")
            sys.exit(1)
        try:
            import websockets
            import asyncio
            async def _check():
                ws = await websockets.connect("ws://127.0.0.1:18790")
                await ws.close()
            asyncio.run(_check())
            ready = True
            print("      Connected!")
            break
        except Exception:
            print(f"      Attempt {i + 1}/20...")

    if not ready:
        print("      ERROR: Could not connect to perception WS server")
        pipeline_proc.terminate()
        sys.exit(1)

    # Set environment variables for demo server
    env = os.environ.copy()
    if args.dry_run:
        env["DEMO_DRY_RUN"] = "1"
    env["DEMO_VIDEO_SOURCE"] = str(args.source)

    # Start demo server
    print(f"[3/3] Starting demo server on port {args.port}...")
    server_cmd = [
        sys.executable, "-m", "uvicorn",
        "demo.server:app",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--log-level", "warning",
    ]
    server_proc = subprocess.Popen(
        server_cmd,
        cwd=project_root,
        env=env,
    )

    time.sleep(2)

    url = f"http://localhost:{args.port}"
    print()
    print("=" * 60)
    print(f"  Demo ready at: {url}")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_proc.terminate()
        pipeline_proc.terminate()
        print("Done.")


if __name__ == "__main__":
    main()
