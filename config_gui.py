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
        self._json_text: scrolledtext.ScrolledText | None = None

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

    def update_json(self, scene_json: dict) -> None:
        """Update the live JSON preview (thread-safe)."""
        if self._json_text and self._root:
            text = json.dumps(scene_json, ensure_ascii=False, indent=2)
            try:
                self._root.after(0, lambda t=text: self._set_json_text(t))
            except tk.TclError:
                pass

    def _set_json_text(self, text: str) -> None:
        """Replace JSON text widget content (must run on tk thread)."""
        if self._json_text:
            self._json_text.config(state=tk.NORMAL)
            self._json_text.delete("1.0", tk.END)
            self._json_text.insert(tk.END, text)
            self._json_text.config(state=tk.DISABLED)
            self._json_text.see(tk.END)

    def _build_and_run(self) -> None:
        """Build the GUI and start the tkinter mainloop."""
        root = tk.Tk()
        root.title("Perception Pipeline - Settings")
        root.geometry("600x1100")
        root.resizable(True, True)
        self._root = root

        main_frame = ttk.Frame(root, padding=4)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Collapsible Settings Section ===
        self._settings_visible = tk.BooleanVar(value=False)
        toggle_btn = ttk.Checkbutton(
            main_frame, text="Settings", variable=self._settings_visible,
            style="Toolbutton",
            command=lambda: self._toggle_settings(settings_frame),
        )
        toggle_btn.pack(fill=tk.X, pady=(0, 4))

        settings_frame = ttk.Frame(main_frame)
        # Start collapsed — don't pack

        # --- Detection ---
        det_frame = ttk.LabelFrame(settings_frame, text="Detection", padding=4)
        det_frame.pack(fill=tk.X, pady=(0, 4))

        row_det1 = ttk.Frame(det_frame)
        row_det1.pack(fill=tk.X)
        ttk.Label(row_det1, text="Confidence:").pack(side=tk.LEFT)
        conf_var = tk.DoubleVar(value=self._config.get("min_confidence", 0.3))
        conf_label = ttk.Label(row_det1, text=f"{conf_var.get():.2f}", width=5)
        conf_label.pack(side=tk.RIGHT)
        ttk.Scale(
            det_frame, from_=0.05, to=0.95, variable=conf_var,
            command=lambda v: self._on_slider(
                "min_confidence", float(v), conf_label, "{:.2f}"
            ),
        ).pack(fill=tk.X)

        row_det2 = ttk.Frame(det_frame)
        row_det2.pack(fill=tk.X)
        ttk.Label(row_det2, text="FPS:").pack(side=tk.LEFT)
        fps_var = tk.IntVar(value=self._config.get("process_fps", 10))
        fps_label = ttk.Label(row_det2, text=str(fps_var.get()), width=5)
        fps_label.pack(side=tk.RIGHT)
        ttk.Scale(
            det_frame, from_=1, to=30, variable=fps_var,
            command=lambda v: self._on_slider(
                "process_fps", int(float(v)), fps_label, "{}"
            ),
        ).pack(fill=tk.X)

        row_det3 = ttk.Frame(det_frame)
        row_det3.pack(fill=tk.X)
        ttk.Label(row_det3, text="Classes:").pack(side=tk.LEFT)
        classes_entry = ttk.Entry(row_det3, width=20)
        current_classes = self._config.get("filter_classes")
        if current_classes:
            classes_entry.insert(0, ", ".join(current_classes))
        classes_entry.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        ttk.Button(
            row_det3, text="Apply", width=6,
            command=lambda: self._apply_classes(classes_entry.get()),
        ).pack(side=tk.RIGHT)

        # --- Output ---
        out_frame = ttk.LabelFrame(settings_frame, text="Output", padding=4)
        out_frame.pack(fill=tk.X, pady=(0, 4))

        row_out1 = ttk.Frame(out_frame)
        row_out1.pack(fill=tk.X)
        ttk.Label(row_out1, text="Strategy:").pack(side=tk.LEFT)
        strategy_var = tk.StringVar(
            value=self._config.get("output_strategy", "hybrid")
        )
        strategy_combo = ttk.Combobox(
            row_out1, textvariable=strategy_var, state="readonly", width=14,
            values=["every_frame", "interval", "on_change", "hybrid", "stable"],
        )
        strategy_combo.pack(side=tk.RIGHT)
        strategy_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._on_change("output_strategy", strategy_var.get()),
        )

        row_out2 = ttk.Frame(out_frame)
        row_out2.pack(fill=tk.X)
        ttk.Label(row_out2, text="Interval:").pack(side=tk.LEFT)
        interval_var = tk.DoubleVar(
            value=self._config.get("output_interval_sec", 1.0)
        )
        interval_label = ttk.Label(row_out2, text=f"{interval_var.get():.1f}s", width=5)
        interval_label.pack(side=tk.RIGHT)
        ttk.Scale(
            out_frame, from_=0.1, to=10.0, variable=interval_var,
            command=lambda v: self._on_slider(
                "output_interval_sec", round(float(v), 1), interval_label, "{:.1f}s"
            ),
        ).pack(fill=tk.X)

        # --- Source / Viz / Depth / Motion in compact rows ---
        misc_frame = ttk.LabelFrame(settings_frame, text="Source & Options", padding=4)
        misc_frame.pack(fill=tk.X, pady=(0, 4))

        row_src = ttk.Frame(misc_frame)
        row_src.pack(fill=tk.X, pady=1)
        ttk.Label(row_src, text="Source:").pack(side=tk.LEFT)
        src_entry = ttk.Entry(row_src, width=16)
        src_entry.insert(0, str(self._config.get("source", 0)))
        src_entry.pack(side=tk.LEFT, padx=4, expand=True, fill=tk.X)
        ttk.Button(
            row_src, text="Switch", width=6,
            command=lambda: self._switch_source(src_entry.get()),
        ).pack(side=tk.RIGHT)

        row_opts = ttk.Frame(misc_frame)
        row_opts.pack(fill=tk.X, pady=1)
        viz_var = tk.BooleanVar(
            value=self._config.get("show_visualization", True)
        )
        ttk.Checkbutton(
            row_opts, text="Viz", variable=viz_var,
            command=lambda: self._on_change("show_visualization", viz_var.get()),
        ).pack(side=tk.LEFT)
        depth_var = tk.BooleanVar(
            value=self._config.get("depth_enabled", False)
        )
        ttk.Checkbutton(
            row_opts, text="Depth", variable=depth_var,
            command=lambda: self._on_change("depth_enabled", depth_var.get()),
        ).pack(side=tk.LEFT, padx=8)

        row_motion = ttk.Frame(misc_frame)
        row_motion.pack(fill=tk.X, pady=1)
        ttk.Label(row_motion, text="Motion threshold:").pack(side=tk.LEFT)
        motion_var = tk.DoubleVar(
            value=self._config.get("motion_speed_threshold", 0.02)
        )
        motion_label = ttk.Label(row_motion, text=f"{motion_var.get():.3f}", width=5)
        motion_label.pack(side=tk.RIGHT)
        ttk.Scale(
            misc_frame, from_=0.005, to=0.1, variable=motion_var,
            command=lambda v: self._on_slider(
                "motion_speed_threshold", round(float(v), 3),
                motion_label, "{:.3f}"
            ),
        ).pack(fill=tk.X)

        # === Live JSON Preview (main area) ===
        json_frame = ttk.LabelFrame(main_frame, text="Live JSON (Claw Output)", padding=4)
        json_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self._json_text = scrolledtext.ScrolledText(
            json_frame, wrap=tk.NONE, font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self._json_text.pack(fill=tk.BOTH, expand=True)

        # === Status Bar ===
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            main_frame, textvariable=self._status_var,
            relief=tk.SUNKEN, anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 0))

        # Store settings_frame ref for toggle
        self._settings_frame = settings_frame

        root.protocol("WM_DELETE_WINDOW", self.stop)
        root.mainloop()

    def _toggle_settings(self, settings_frame: ttk.Frame) -> None:
        """Show/hide the settings panel."""
        if self._settings_visible.get():
            settings_frame.pack(fill=tk.X, pady=(0, 4), before=self._json_text.master)
        else:
            settings_frame.pack_forget()

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
