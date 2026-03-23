"""tkinter configuration GUI for runtime parameter adjustment."""

import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Any, Callable


class ConfigGUI:
    """
    Runtime configuration panel using tkinter.

    Runs in a daemon thread alongside the main OpenCV loop.
    Changes are applied immediately via the on_change callback.
    """

    def __init__(self, config: dict, on_change: Callable[[str, Any], None]):
        self._config = config
        self._on_change = on_change
        self._root: tk.Tk | None = None
        self._thread: threading.Thread | None = None
        self._status_var: tk.StringVar | None = None
        self._json_text: scrolledtext.ScrolledText | None = None
        self._metrics_var: tk.StringVar | None = None

    def start(self) -> None:
        """Launch the GUI in a daemon thread."""
        self._thread = threading.Thread(target=self._build_and_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Close the GUI window (thread-safe)."""
        root = self._root
        if root:
            try:
                # Schedule quit on the tkinter thread to avoid cross-thread deadlock
                root.after_idle(root.quit)
            except tk.TclError:
                pass
            # Wait briefly for mainloop to exit
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            # Now safe to destroy
            try:
                root.destroy()
            except tk.TclError:
                pass
            self._root = None

    def set_status(self, text: str) -> None:
        """Update the status bar text (thread-safe)."""
        if self._status_var and self._root:
            try:
                self._root.after(0, lambda: self._status_var.set(text))
            except tk.TclError:
                pass

    def update_json(self, scene_json: dict) -> None:
        """Update the live JSON preview and metrics display (thread-safe)."""
        if not self._root:
            return

        text = json.dumps(scene_json, ensure_ascii=False, indent=2)
        try:
            self._root.after(0, lambda t=text: self._set_json_text(t))
        except tk.TclError:
            pass

        # Update metrics line
        if self._metrics_var:
            latency = scene_json.get("latency_ms", {})
            pipeline = scene_json.get("pipeline", {})
            cam = scene_json.get("camera_motion", {})
            obj_count = len(scene_json.get("objects", []))
            total_ms = latency.get("total", 0)
            state = pipeline.get("state", "?")
            ego = cam.get("ego_state", "?")
            cam_conf = cam.get("confidence", 0)
            degraded = pipeline.get("degraded_modules", [])
            actionable = scene_json.get("actionable", False)

            line = (
                f"Objects: {obj_count}  |  Latency: {total_ms:.0f}ms  |  "
                f"State: {state}  |  Ego: {ego}  |  "
                f"Cam conf: {cam_conf:.2f}  |  "
                f"Actionable: {'YES' if actionable else 'NO'}"
            )
            if degraded:
                line += f"  |  Degraded: {', '.join(degraded)}"
            try:
                self._root.after(0, lambda l=line: self._metrics_var.set(l))
            except tk.TclError:
                pass

    def _set_json_text(self, text: str) -> None:
        if self._json_text:
            self._json_text.config(state=tk.NORMAL)
            self._json_text.delete("1.0", tk.END)
            self._json_text.insert(tk.END, text)
            self._json_text.config(state=tk.DISABLED)
            self._json_text.see(tk.END)

    # -----------------------------------------------------------------
    # Build GUI
    # -----------------------------------------------------------------

    def _build_and_run(self) -> None:
        root = tk.Tk()
        root.title("Perception Pipeline v3.2 — Debug Panel")
        root.geometry("620x900")
        root.resizable(True, True)
        self._root = root

        main_frame = ttk.Frame(root, padding=6)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Live Metrics Bar ===
        self._metrics_var = tk.StringVar(value="Waiting for first frame...")
        metrics_label = ttk.Label(
            main_frame, textvariable=self._metrics_var,
            font=("Consolas", 9), foreground="#007ACC",
        )
        metrics_label.pack(fill=tk.X, pady=(0, 6))

        # === Settings (collapsible) ===
        self._settings_visible = tk.BooleanVar(value=True)
        toggle_btn = ttk.Button(
            main_frame, text="[ - ] Settings",
            command=lambda: self._toggle_settings(settings_frame, toggle_btn),
        )
        toggle_btn.pack(fill=tk.X, pady=(0, 4))

        settings_frame = ttk.Frame(main_frame)
        settings_frame.pack(fill=tk.X, pady=(0, 4))

        # Anchor for re-packing
        self._json_anchor = ttk.Frame(main_frame)
        self._json_anchor.pack(fill=tk.X)

        self._build_detection_section(settings_frame)
        self._build_output_section(settings_frame)
        self._build_plugins_section(settings_frame)
        self._build_source_section(settings_frame)

        # === Live JSON Preview ===
        json_frame = ttk.LabelFrame(main_frame, text="Live JSON (Claw Output)", padding=4)
        json_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self._json_text = scrolledtext.ScrolledText(
            json_frame, wrap=tk.NONE, font=("Consolas", 9),
            state=tk.DISABLED, bg="#1E1E1E", fg="#D4D4D4",
            insertbackground="#D4D4D4",
        )
        self._json_text.pack(fill=tk.BOTH, expand=True)

        # === Status Bar ===
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 0))

        self._settings_frame = settings_frame
        root.protocol("WM_DELETE_WINDOW", self.stop)
        root.mainloop()

    # -----------------------------------------------------------------
    # Detection section
    # -----------------------------------------------------------------

    def _build_detection_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Detection", padding=4)
        frame.pack(fill=tk.X, pady=(0, 4))

        # Confidence slider
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Confidence:").pack(side=tk.LEFT)
        conf_var = tk.DoubleVar(value=self._config.get("min_confidence", 0.45))
        conf_label = ttk.Label(row, text=f"{conf_var.get():.2f}", width=5)
        conf_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.05, to=0.95, variable=conf_var,
            command=lambda v: self._on_slider("min_confidence", float(v), conf_label, "{:.2f}"),
        ).pack(fill=tk.X)

        # FPS slider
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Process FPS:").pack(side=tk.LEFT)
        fps_var = tk.IntVar(value=self._config.get("process_fps", 10))
        fps_label = ttk.Label(row2, text=str(fps_var.get()), width=5)
        fps_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=1, to=30, variable=fps_var,
            command=lambda v: self._on_slider("process_fps", int(float(v)), fps_label, "{}"),
        ).pack(fill=tk.X)

        # Classes filter
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row3, text="Classes:").pack(side=tk.LEFT)
        classes_entry = ttk.Entry(row3, width=20)
        current_classes = self._config.get("filter_classes")
        if current_classes:
            classes_entry.insert(0, ", ".join(current_classes))
        classes_entry.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        ttk.Button(
            row3, text="Apply", width=6,
            command=lambda: self._apply_classes(classes_entry.get()),
        ).pack(side=tk.RIGHT)

    # -----------------------------------------------------------------
    # Output section
    # -----------------------------------------------------------------

    def _build_output_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Output", padding=4)
        frame.pack(fill=tk.X, pady=(0, 4))

        # Strategy
        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Strategy:").pack(side=tk.LEFT)
        strategy_var = tk.StringVar(value=self._config.get("output_strategy", "hybrid"))
        combo = ttk.Combobox(
            row, textvariable=strategy_var, state="readonly", width=14,
            values=["every_frame", "interval", "on_change", "hybrid", "stable"],
        )
        combo.pack(side=tk.RIGHT)
        combo.bind("<<ComboboxSelected>>",
                    lambda e: self._on_change("output_strategy", strategy_var.get()))

        # Interval
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X)
        ttk.Label(row2, text="Interval:").pack(side=tk.LEFT)
        interval_var = tk.DoubleVar(value=self._config.get("output_interval_sec", 1.0))
        interval_label = ttk.Label(row2, text=f"{interval_var.get():.1f}s", width=5)
        interval_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.1, to=10.0, variable=interval_var,
            command=lambda v: self._on_slider(
                "output_interval_sec", round(float(v), 1), interval_label, "{:.1f}s"),
        ).pack(fill=tk.X)

        # Stable window
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X)
        ttk.Label(row3, text="Stable window:").pack(side=tk.LEFT)
        stable_var = tk.DoubleVar(value=self._config.get("stable_window_sec", 1.0))
        stable_label = ttk.Label(row3, text=f"{stable_var.get():.1f}s", width=5)
        stable_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.1, to=5.0, variable=stable_var,
            command=lambda v: self._on_slider(
                "stable_window_sec", round(float(v), 1), stable_label, "{:.1f}s"),
        ).pack(fill=tk.X)

        # Compact toggle
        compact_var = tk.BooleanVar(value=self._config.get("output_compact", False))
        ttk.Checkbutton(
            frame, text="Compact JSON output", variable=compact_var,
            command=lambda: self._on_change("output_compact", compact_var.get()),
        ).pack(anchor=tk.W, pady=(2, 0))

    # -----------------------------------------------------------------
    # Plugins section (Depth, Ego Motion, Kalman)
    # -----------------------------------------------------------------

    def _build_plugins_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Plugins & Tuning", padding=4)
        frame.pack(fill=tk.X, pady=(0, 4))

        # --- Depth ---
        depth_row = ttk.Frame(frame)
        depth_row.pack(fill=tk.X, pady=1)
        depth_var = tk.BooleanVar(value=self._config.get("depth_enabled", False))
        ttk.Checkbutton(
            depth_row, text="Depth Estimation", variable=depth_var,
            command=lambda: self._on_change("depth_enabled", depth_var.get()),
        ).pack(side=tk.LEFT)

        depth_model_var = tk.StringVar(value=self._config.get("depth_model_size", "small"))
        depth_combo = ttk.Combobox(
            depth_row, textvariable=depth_model_var, state="readonly", width=8,
            values=["small", "base", "large"],
        )
        depth_combo.pack(side=tk.RIGHT)
        ttk.Label(depth_row, text="Model:").pack(side=tk.RIGHT, padx=(0, 4))
        depth_combo.bind("<<ComboboxSelected>>",
                         lambda e: self._on_change("depth_model_size", depth_model_var.get()))

        # --- Ego Motion ---
        ego_row = ttk.Frame(frame)
        ego_row.pack(fill=tk.X, pady=1)
        ttk.Label(ego_row, text="Ego motion:").pack(side=tk.LEFT)
        ego_var = tk.StringVar(value=self._config.get("ego_motion_source", "optical_flow"))
        ego_combo = ttk.Combobox(
            ego_row, textvariable=ego_var, state="readonly", width=14,
            values=["optical_flow", "external", "none"],
        )
        ego_combo.pack(side=tk.RIGHT)
        ego_combo.bind("<<ComboboxSelected>>",
                        lambda e: self._on_change("ego_motion_source", ego_var.get()))

        # Ego settle time
        settle_row = ttk.Frame(frame)
        settle_row.pack(fill=tk.X, pady=1)
        ttk.Label(settle_row, text="Settle time:").pack(side=tk.LEFT)
        settle_var = tk.DoubleVar(value=self._config.get("ego_settle_sec", 0.5))
        settle_label = ttk.Label(settle_row, text=f"{settle_var.get():.1f}s", width=5)
        settle_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.1, to=3.0, variable=settle_var,
            command=lambda v: self._on_slider(
                "ego_settle_sec", round(float(v), 1), settle_label, "{:.1f}s"),
        ).pack(fill=tk.X)

        # --- Kalman ---
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        kalman_row1 = ttk.Frame(frame)
        kalman_row1.pack(fill=tk.X, pady=1)
        ttk.Label(kalman_row1, text="Kalman process noise:").pack(side=tk.LEFT)
        kp_var = tk.DoubleVar(value=self._config.get("kalman_process_noise", 0.01))
        kp_label = ttk.Label(kalman_row1, text=f"{kp_var.get():.3f}", width=6)
        kp_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.001, to=0.1, variable=kp_var,
            command=lambda v: self._on_slider(
                "kalman_process_noise", round(float(v), 3), kp_label, "{:.3f}"),
        ).pack(fill=tk.X)

        kalman_row2 = ttk.Frame(frame)
        kalman_row2.pack(fill=tk.X, pady=1)
        ttk.Label(kalman_row2, text="Kalman measurement noise:").pack(side=tk.LEFT)
        km_var = tk.DoubleVar(value=self._config.get("kalman_measurement_noise", 0.05))
        km_label = ttk.Label(kalman_row2, text=f"{km_var.get():.3f}", width=6)
        km_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.001, to=0.2, variable=km_var,
            command=lambda v: self._on_slider(
                "kalman_measurement_noise", round(float(v), 3), km_label, "{:.3f}"),
        ).pack(fill=tk.X)

        # --- Motion threshold ---
        motion_row = ttk.Frame(frame)
        motion_row.pack(fill=tk.X, pady=1)
        ttk.Label(motion_row, text="Motion threshold:").pack(side=tk.LEFT)
        motion_var = tk.DoubleVar(value=self._config.get("motion_speed_threshold", 0.02))
        motion_label = ttk.Label(motion_row, text=f"{motion_var.get():.3f}", width=6)
        motion_label.pack(side=tk.RIGHT)
        ttk.Scale(
            frame, from_=0.005, to=0.1, variable=motion_var,
            command=lambda v: self._on_slider(
                "motion_speed_threshold", round(float(v), 3), motion_label, "{:.3f}"),
        ).pack(fill=tk.X)

    # -----------------------------------------------------------------
    # Source section
    # -----------------------------------------------------------------

    def _build_source_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Source", padding=4)
        frame.pack(fill=tk.X, pady=(0, 4))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Source:").pack(side=tk.LEFT)
        src_entry = ttk.Entry(row, width=20)
        src_entry.insert(0, str(self._config.get("source", 0)))
        src_entry.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        ttk.Button(
            row, text="Switch", width=6,
            command=lambda: self._switch_source(src_entry.get()),
        ).pack(side=tk.RIGHT)

        # Visualization toggle
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=(2, 0))
        viz_var = tk.BooleanVar(value=self._config.get("show_visualization", True))
        ttk.Checkbutton(
            row2, text="Show visualization window", variable=viz_var,
            command=lambda: self._on_change("show_visualization", viz_var.get()),
        ).pack(side=tk.LEFT)

    # -----------------------------------------------------------------
    # Toggle / callbacks
    # -----------------------------------------------------------------

    def _toggle_settings(self, settings_frame: ttk.Frame, btn: ttk.Button) -> None:
        visible = not self._settings_visible.get()
        self._settings_visible.set(visible)
        if visible:
            settings_frame.pack(fill=tk.X, pady=(0, 4), before=self._json_anchor)
            btn.config(text="[ - ] Settings")
        else:
            settings_frame.pack_forget()
            btn.config(text="[ + ] Settings")

    def _on_slider(self, key: str, value: Any, label: ttk.Label, fmt: str) -> None:
        label.config(text=fmt.format(value))
        self._config[key] = value
        self._on_change(key, value)

    def _apply_classes(self, text: str) -> None:
        text = text.strip()
        if not text:
            classes = None
        else:
            classes = [c.strip() for c in text.split(",") if c.strip()]
            if not classes:
                classes = None
        self._config["filter_classes"] = classes
        self._on_change("filter_classes", classes)
        self.set_status(f"Classes: {classes or 'all'}")

    def _switch_source(self, source_str: str) -> None:
        source_str = source_str.strip()
        if not source_str:
            self.set_status("Error: empty source")
            return
        try:
            source = int(source_str)
        except ValueError:
            source = source_str
        self.set_status(f"Switching to: {source}...")
        self._on_change("source", source)
