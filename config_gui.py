"""tkinter configuration GUI for runtime parameter adjustment."""

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable


class ConfigGUI:
    """
    Runtime configuration panel using tkinter.

    Runs in a daemon thread alongside the main OpenCV loop.
    Changes are applied immediately via the on_change callback.
    """

    def __init__(self, config: dict, on_change: Callable[[str, Any], None]):
        """
        Args:
            config: shared config dict (read for initial values, written on change)
            on_change: callback(key, new_value) fired when a setting changes
        """
        self._config = config
        self._on_change = on_change
        self._root: tk.Tk | None = None
        self._thread: threading.Thread | None = None
        self._status_var: tk.StringVar | None = None

    def start(self) -> None:
        """Launch the GUI in a daemon thread."""
        self._thread = threading.Thread(target=self._build_and_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Close the GUI window."""
        if self._root:
            try:
                self._root.quit()
                self._root.destroy()
            except tk.TclError:
                pass

    def set_status(self, text: str) -> None:
        """Update the status bar text (thread-safe)."""
        if self._status_var and self._root:
            try:
                self._root.after(0, lambda: self._status_var.set(text))
            except tk.TclError:
                pass

    def _build_and_run(self) -> None:
        """Build the GUI and start the tkinter mainloop."""
        root = tk.Tk()
        root.title("Perception Pipeline - Settings")
        root.geometry("420x680")
        root.resizable(False, True)
        self._root = root

        main_frame = ttk.Frame(root, padding=8)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Detection Section ===
        det_frame = ttk.LabelFrame(main_frame, text="Detection", padding=6)
        det_frame.pack(fill=tk.X, pady=(0, 6))

        # Confidence
        ttk.Label(det_frame, text="Min Confidence:").pack(anchor=tk.W)
        conf_var = tk.DoubleVar(value=self._config.get("min_confidence", 0.3))
        conf_label = ttk.Label(det_frame, text=f"{conf_var.get():.2f}")
        conf_label.pack(anchor=tk.E)
        conf_slider = ttk.Scale(
            det_frame, from_=0.05, to=0.95, variable=conf_var,
            command=lambda v: self._on_slider(
                "min_confidence", float(v), conf_label, "{:.2f}"
            ),
        )
        conf_slider.pack(fill=tk.X)

        # Process FPS
        ttk.Label(det_frame, text="Process FPS:").pack(anchor=tk.W)
        fps_var = tk.IntVar(value=self._config.get("process_fps", 10))
        fps_label = ttk.Label(det_frame, text=str(fps_var.get()))
        fps_label.pack(anchor=tk.E)
        fps_slider = ttk.Scale(
            det_frame, from_=1, to=30, variable=fps_var,
            command=lambda v: self._on_slider(
                "process_fps", int(float(v)), fps_label, "{}"
            ),
        )
        fps_slider.pack(fill=tk.X)

        # Filter classes
        ttk.Label(det_frame, text="Filter Classes (comma-sep, empty=all):").pack(
            anchor=tk.W
        )
        classes_entry = ttk.Entry(det_frame)
        current_classes = self._config.get("filter_classes")
        if current_classes:
            classes_entry.insert(0, ", ".join(current_classes))
        classes_entry.pack(fill=tk.X)
        ttk.Button(
            det_frame, text="Apply Classes",
            command=lambda: self._apply_classes(classes_entry.get()),
        ).pack(anchor=tk.E, pady=2)

        # === Output Section ===
        out_frame = ttk.LabelFrame(main_frame, text="Output", padding=6)
        out_frame.pack(fill=tk.X, pady=(0, 6))

        # Strategy
        ttk.Label(out_frame, text="Strategy:").pack(anchor=tk.W)
        strategy_var = tk.StringVar(
            value=self._config.get("output_strategy", "hybrid")
        )
        strategy_combo = ttk.Combobox(
            out_frame, textvariable=strategy_var, state="readonly",
            values=["every_frame", "interval", "on_change", "hybrid"],
        )
        strategy_combo.pack(fill=tk.X)
        strategy_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_change("output_strategy", strategy_var.get()),
        )

        # Interval
        ttk.Label(out_frame, text="Interval (sec):").pack(anchor=tk.W)
        interval_var = tk.DoubleVar(
            value=self._config.get("output_interval_sec", 1.0)
        )
        interval_label = ttk.Label(out_frame, text=f"{interval_var.get():.1f}")
        interval_label.pack(anchor=tk.E)
        ttk.Scale(
            out_frame, from_=0.1, to=10.0, variable=interval_var,
            command=lambda v: self._on_slider(
                "output_interval_sec", round(float(v), 1), interval_label, "{:.1f}"
            ),
        ).pack(fill=tk.X)

        # Change threshold
        ttk.Label(out_frame, text="Change Threshold:").pack(anchor=tk.W)
        thresh_var = tk.DoubleVar(
            value=self._config.get("output_change_threshold", 0.01)
        )
        thresh_label = ttk.Label(out_frame, text=f"{thresh_var.get():.3f}")
        thresh_label.pack(anchor=tk.E)
        ttk.Scale(
            out_frame, from_=0.001, to=0.1, variable=thresh_var,
            command=lambda v: self._on_slider(
                "output_change_threshold", round(float(v), 3),
                thresh_label, "{:.3f}"
            ),
        ).pack(fill=tk.X)

        # === Video Source Section ===
        src_frame = ttk.LabelFrame(main_frame, text="Video Source", padding=6)
        src_frame.pack(fill=tk.X, pady=(0, 6))

        current_src = str(self._config.get("source", 0))
        ttk.Label(src_frame, text=f"Current: {current_src}").pack(anchor=tk.W)

        src_entry = ttk.Entry(src_frame)
        src_entry.insert(0, current_src)
        src_entry.pack(fill=tk.X, pady=2)

        ttk.Button(
            src_frame, text="Switch Source",
            command=lambda: self._switch_source(src_entry.get()),
        ).pack(anchor=tk.E, pady=2)

        # === Visualization Section ===
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=6)
        viz_frame.pack(fill=tk.X, pady=(0, 6))

        viz_var = tk.BooleanVar(
            value=self._config.get("show_visualization", True)
        )
        ttk.Checkbutton(
            viz_frame, text="Show Visualization", variable=viz_var,
            command=lambda: self._on_change(
                "show_visualization", viz_var.get()
            ),
        ).pack(anchor=tk.W)

        ttk.Label(viz_frame, text="Scale:").pack(anchor=tk.W)
        scale_var = tk.DoubleVar(value=self._config.get("viz_scale", 1.0))
        scale_label = ttk.Label(viz_frame, text=f"{scale_var.get():.1f}")
        scale_label.pack(anchor=tk.E)
        ttk.Scale(
            viz_frame, from_=0.5, to=2.0, variable=scale_var,
            command=lambda v: self._on_slider(
                "viz_scale", round(float(v), 1), scale_label, "{:.1f}"
            ),
        ).pack(fill=tk.X)

        # === Depth Section ===
        depth_frame = ttk.LabelFrame(main_frame, text="Depth Estimation", padding=6)
        depth_frame.pack(fill=tk.X, pady=(0, 6))

        depth_var = tk.BooleanVar(
            value=self._config.get("depth_enabled", False)
        )
        ttk.Checkbutton(
            depth_frame, text="Enable Depth", variable=depth_var,
            command=lambda: self._on_change("depth_enabled", depth_var.get()),
        ).pack(anchor=tk.W)

        ttk.Label(depth_frame, text="Model Size:").pack(anchor=tk.W)
        depth_model_var = tk.StringVar(
            value=self._config.get("depth_model_size", "small")
        )
        depth_combo = ttk.Combobox(
            depth_frame, textvariable=depth_model_var, state="readonly",
            values=["small", "base", "large"],
        )
        depth_combo.pack(fill=tk.X)
        depth_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_change("depth_model_size", depth_model_var.get()),
        )

        # === Motion Section ===
        motion_frame = ttk.LabelFrame(main_frame, text="Motion & Risk", padding=6)
        motion_frame.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(motion_frame, text="Motion Speed Threshold:").pack(anchor=tk.W)
        motion_var = tk.DoubleVar(
            value=self._config.get("motion_speed_threshold", 0.02)
        )
        motion_label = ttk.Label(motion_frame, text=f"{motion_var.get():.3f}")
        motion_label.pack(anchor=tk.E)
        ttk.Scale(
            motion_frame, from_=0.005, to=0.1, variable=motion_var,
            command=lambda v: self._on_slider(
                "motion_speed_threshold", round(float(v), 3),
                motion_label, "{:.3f}"
            ),
        ).pack(fill=tk.X)

        # === Status Bar ===
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W,
        ).pack(fill=tk.X, pady=(6, 0))

        root.protocol("WM_DELETE_WINDOW", self.stop)
        root.mainloop()

    def _on_slider(
        self, key: str, value: Any, label: ttk.Label, fmt: str
    ) -> None:
        """Handle slider change: update config, label, and fire callback."""
        label.config(text=fmt.format(value))
        self._config[key] = value
        self._on_change(key, value)

    def _apply_classes(self, text: str) -> None:
        """Parse comma-separated class names and update config."""
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
        """Request a video source switch."""
        source_str = source_str.strip()
        if not source_str:
            self.set_status("Error: empty source")
            return

        # Try to parse as int (camera ID)
        try:
            source = int(source_str)
        except ValueError:
            source = source_str

        self.set_status(f"Switching to: {source}...")
        self._on_change("source", source)
