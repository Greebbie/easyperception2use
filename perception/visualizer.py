"""Demo visualization overlay for the perception pipeline."""

import cv2
import time
from collections import deque


class Visualizer:
    """Draws detection results, region grid, and scene status on the frame."""

    RISK_COLORS: dict[str, tuple[int, int, int]] = {
        "clear": (0, 255, 0),
        "low": (0, 200, 100),
        "medium": (0, 200, 255),
        "high": (0, 0, 255),
    }

    def __init__(self):
        self.fps_history: deque[float] = deque(maxlen=30)
        self.last_time: float = time.time()

    def draw(self, frame, scene_json: dict):
        """
        Draw visualization overlay on the frame.

        Args:
            frame: BGR image (will be modified in place)
            scene_json: scene state dict from SceneBuilder

        Returns:
            The annotated frame
        """
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.fps_history.append(1.0 / dt)
        self.last_time = now
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        h, w = frame.shape[:2]

        # Detection boxes + labels
        for obj in scene_json["objects"]:
            b = obj["bbox_px"]
            is_moving = obj["motion"]["moving"]
            color = (0, 200, 255) if is_moving else (0, 255, 0)
            thickness = 3 if is_moving else 2
            cv2.rectangle(
                frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness
            )

            label = f'{obj["class"]} #{obj["track_id"]}'
            depth_info = obj.get("depth")
            depth_str = f' | D:{depth_info["label"]}' if depth_info else ""
            sublabel = (
                f'{obj["position"]["region"]} | '
                f'{obj["motion"]["direction"]}{depth_str}'
            )
            cv2.putText(
                frame, label, (b["x1"], b["y1"] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, sublabel, (b["x1"], b["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )

        # 3x3 region grid lines
        for x_ratio in [0.33, 0.67]:
            x = int(w * x_ratio)
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
        for y_ratio in [0.4, 0.7]:
            y = int(h * y_ratio)
            cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)

        # Top-left: scene summary panel
        scene = scene_json["scene"]
        risk_color = self.RISK_COLORS.get(scene["risk_level"], (255, 255, 255))

        info_lines = [
            f'FPS: {avg_fps:.1f} | Objects: {scene["object_count"]}'
            f' | Moving: {scene["moving_count"]}',
            f'Risk: {scene["risk_level"].upper()}'
            f' | Center: {"YES" if scene["center_occupied"] else "no"}',
            f'Classes: {", ".join(scene["classes_present"])}',
        ]

        # Semi-transparent black background
        panel_h = 25 + 22 * len(info_lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (520, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(info_lines):
            color = risk_color if i == 1 else (220, 220, 220)
            cv2.putText(
                frame, line, (10, 25 + 22 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        return frame
