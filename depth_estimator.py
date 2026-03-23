"""
Monocular depth estimation using Depth Anything v2.

ENHANCEMENT MODULE — not part of the core 2D pipeline.
This module is opt-in (--depth flag) and gracefully degrades
if transformers/torch are not installed.
"""

import threading
import numpy as np
from typing import Optional

try:
    import torch
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class DepthEstimator:
    """
    Wraps Depth Anything v2 for per-frame depth estimation.

    The model is loaded lazily on first estimate() call.
    Loading has a 30s timeout — if it fails, depth is auto-disabled.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        enabled: bool = True,
        load_timeout: float = 30.0,
    ):
        """
        Args:
            model_size: "small" / "base" / "large"
            device: "auto" / "cuda" / "cpu"
            enabled: if False, estimate() always returns None
            load_timeout: max seconds to wait for model loading
        """
        self.model_size = model_size
        self.device = device
        self.enabled = enabled
        self.load_timeout = load_timeout
        self._model = None
        self._loaded = False
        self._load_failed = False

    def _load_model(self) -> bool:
        """
        Load the Depth Anything v2 model with timeout.

        Returns True on success, False on failure.
        On failure, self.enabled is set to False (auto-disable).
        """
        result = {"success": False}

        def _do_load():
            try:
                if not _HAS_TRANSFORMERS:
                    raise ImportError(
                        "transformers and/or torch not installed. "
                        "Install with: pip install -r requirements-depth.txt"
                    )

                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

                model_map = {
                    "small": "depth-anything/Depth-Anything-V2-Small-hf",
                    "base": "depth-anything/Depth-Anything-V2-Base-hf",
                    "large": "depth-anything/Depth-Anything-V2-Large-hf",
                }
                model_id = model_map.get(self.model_size, model_map["small"])

                print(f"[DepthEstimator] Loading {model_id} on {self.device}...")

                self._model = hf_pipeline(
                    "depth-estimation",
                    model=model_id,
                    device=self.device,
                )
                result["success"] = True
                print("[DepthEstimator] Model loaded")
            except Exception as e:
                print(f"[DepthEstimator] Load failed: {e}")

        thread = threading.Thread(target=_do_load, daemon=True)
        thread.start()
        thread.join(timeout=self.load_timeout)

        if thread.is_alive():
            print(
                f"[DepthEstimator] Loading timed out after {self.load_timeout}s, "
                "depth disabled"
            )
            self._load_failed = True
            self.enabled = False
            return False

        if not result["success"]:
            self._load_failed = True
            self.enabled = False
            print("[DepthEstimator] Depth auto-disabled due to load failure")
            return False

        self._loaded = True
        return True

    @property
    def load_failed(self) -> bool:
        """Whether model loading has failed (read-only)."""
        return self._load_failed

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run depth estimation on a BGR frame.

        Args:
            frame: HxWx3 BGR uint8 image

        Returns:
            HxW float32 depth map normalized to 0.0-1.0 (0=nearest, 1=farthest),
            or None if disabled or failed.
        """
        if not self.enabled or self._load_failed:
            return None

        if not self._loaded:
            if not self._load_model():
                return None

        try:
            from PIL import Image
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            result = self._model(pil_image)
            depth_map = np.array(result["depth"], dtype=np.float32)

            h, w = frame.shape[:2]
            if depth_map.shape != (h, w):
                depth_map = cv2.resize(
                    depth_map, (w, h), interpolation=cv2.INTER_LINEAR
                )

            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 1e-6:
                depth_map = (depth_map - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map

        except Exception as e:
            print(f"[DepthEstimator] Inference error: {e}")
            return None

    @staticmethod
    def get_object_depth(depth_map: np.ndarray, bbox_px: dict) -> dict:
        """
        Extract depth info for a detected object from the depth map.

        Samples the center 50% of the bounding box to avoid edge noise.

        Args:
            depth_map: HxW float32 normalized depth map
            bbox_px: {"x1": int, "y1": int, "x2": int, "y2": int}

        Returns:
            {"value": float, "label": "near"/"mid"/"far"}
        """
        h, w = depth_map.shape[:2]
        x1 = max(0, bbox_px["x1"])
        y1 = max(0, bbox_px["y1"])
        x2 = min(w, bbox_px["x2"])
        y2 = min(h, bbox_px["y2"])

        if x2 <= x1 or y2 <= y1:
            return {"value": 0.5, "label": "mid"}

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        qw = max(1, (x2 - x1) // 4)
        qh = max(1, (y2 - y1) // 4)

        roi = depth_map[
            max(0, cy - qh): min(h, cy + qh),
            max(0, cx - qw): min(w, cx + qw),
        ]

        if roi.size == 0:
            return {"value": 0.5, "label": "mid"}

        value = float(np.mean(roi))

        if value < 0.33:
            label = "near"
        elif value < 0.66:
            label = "mid"
        else:
            label = "far"

        return {"value": round(value, 3), "label": label}
