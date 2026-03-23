"""Synthetic data generator for --dry-run mode (no camera required)."""

import math
import random
import time

import cv2
import numpy as np


# Simulated object classes with typical sizes (relative to frame)
FAKE_CLASSES = [
    {"name": "person", "rel_w": (0.08, 0.20), "rel_h": (0.15, 0.45)},
    {"name": "car", "rel_w": (0.10, 0.30), "rel_h": (0.08, 0.20)},
    {"name": "dog", "rel_w": (0.05, 0.12), "rel_h": (0.04, 0.10)},
    {"name": "chair", "rel_w": (0.04, 0.10), "rel_h": (0.06, 0.15)},
    {"name": "bottle", "rel_w": (0.02, 0.05), "rel_h": (0.04, 0.10)},
]


class FakeObject:
    """A simulated moving object for dry-run mode."""

    def __init__(self, track_id: int, frame_w: int, frame_h: int):
        self.track_id = track_id
        self.frame_w = frame_w
        self.frame_h = frame_h

        cls_info = random.choice(FAKE_CLASSES)
        self.cls_name = cls_info["name"]
        self.rel_w = random.uniform(*cls_info["rel_w"])
        self.rel_h = random.uniform(*cls_info["rel_h"])
        self.confidence = random.uniform(0.5, 0.98)

        # Position (normalized 0-1)
        self.x = random.uniform(0.1, 0.9)
        self.y = random.uniform(0.1, 0.9)

        # Velocity (normalized units/sec)
        speed = random.uniform(0.0, 0.08)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

    def update(self, dt: float) -> None:
        """Update position based on velocity, bounce off edges."""
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off edges
        if self.x < 0.05 or self.x > 0.95:
            self.vx = -self.vx
            self.x = max(0.05, min(0.95, self.x))
        if self.y < 0.05 or self.y > 0.95:
            self.vy = -self.vy
            self.y = max(0.05, min(0.95, self.y))

        # Occasionally change direction
        if random.random() < 0.01:
            angle = random.uniform(0, 2 * math.pi)
            speed = (self.vx**2 + self.vy**2) ** 0.5
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)

    def get_bbox_px(self) -> dict:
        """Get bounding box in pixel coordinates."""
        w_px = self.rel_w * self.frame_w
        h_px = self.rel_h * self.frame_h
        cx_px = self.x * self.frame_w
        cy_px = self.y * self.frame_h
        return {
            "x1": round(cx_px - w_px / 2),
            "y1": round(cy_px - h_px / 2),
            "x2": round(cx_px + w_px / 2),
            "y2": round(cy_px + h_px / 2),
        }


class DryRunGenerator:
    """
    Generates synthetic frames and fake detection results for dry-run mode.

    Simulates the full pipeline without requiring a camera or YOLO model.
    """

    def __init__(
        self,
        frame_w: int = 1280,
        frame_h: int = 720,
        num_objects: int = 5,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.last_time = time.time()
        self._frame_id = 0

        self.objects = [
            FakeObject(i + 1, frame_w, frame_h) for i in range(num_objects)
        ]

    def generate_frame(self) -> np.ndarray:
        """Generate a synthetic frame with colored rectangles for fake objects."""
        frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # Dark gray background

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        for obj in self.objects:
            obj.update(dt)
            bbox = obj.get_bbox_px()
            color = (
                random.randint(80, 255),
                random.randint(80, 255),
                random.randint(80, 255),
            )
            cv2.rectangle(
                frame,
                (bbox["x1"], bbox["y1"]),
                (bbox["x2"], bbox["y2"]),
                color,
                -1,
            )
            cv2.putText(
                frame,
                f'{obj.cls_name} #{obj.track_id}',
                (bbox["x1"], bbox["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Add "DRY RUN" watermark
        cv2.putText(
            frame,
            "DRY RUN MODE",
            (self.frame_w // 2 - 120, self.frame_h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 200),
            2,
            cv2.LINE_AA,
        )

        return frame

    def generate_scene_json(self, timestamp: float) -> dict:
        """
        Generate a fake scene JSON matching the real pipeline's output schema.

        This bypasses YOLO and SceneBuilder, directly producing the JSON
        so the OutputController and OutputHandler can be tested.
        """
        objects = []
        for obj in self.objects:
            bbox = obj.get_bbox_px()
            rel_size = obj.rel_w * obj.rel_h
            speed = (obj.vx**2 + obj.vy**2) ** 0.5
            moving = speed > 0.02

            if not moving:
                direction = "stationary"
            elif abs(obj.vx) > abs(obj.vy):
                direction = "right" if obj.vx > 0 else "left"
            else:
                direction = "down" if obj.vy > 0 else "up"

            # Region calculation (matching SceneBuilder logic)
            if obj.x < 0.33:
                col = "left"
            elif obj.x < 0.67:
                col = "center"
            else:
                col = "right"

            if obj.y < 0.4:
                row = "top"
            elif obj.y < 0.7:
                row = "middle"
            else:
                row = "bottom"

            objects.append({
                "track_id": obj.track_id,
                "class": obj.cls_name,
                "confidence": round(obj.confidence, 3),
                "track_age": self._frame_id + 1,
                "position_confidence": 0.9,
                "velocity_confidence": 0.5,
                "position": {
                    "rel_x": round(obj.x, 3),
                    "rel_y": round(obj.y, 3),
                    "smoothed_x": round(obj.x, 3),
                    "smoothed_y": round(obj.y, 3),
                    "rel_size": round(rel_size, 4),
                    "region": f"{row}_{col}",
                },
                "bbox_px": bbox,
                "motion": {
                    "direction": direction,
                    "speed": round(speed, 4),
                    "vx": round(obj.vx, 4),
                    "vy": round(obj.vy, 4),
                    "moving": moving,
                    "reliable": True,
                },
            })

        # Build scene summary
        center_objects = [o for o in objects if "center" in o["position"]["region"]]
        moving_objects = [o for o in objects if o["motion"]["moving"]]
        largest = max(objects, key=lambda o: o["position"]["rel_size"]) if objects else None

        risk = "clear"
        if center_objects:
            max_size = max(o["position"]["rel_size"] for o in center_objects)
            if max_size > 0.15:
                risk = "high"
            elif max_size > 0.05:
                risk = "medium"
            else:
                risk = "low"

        region_summary: dict[str, list[str]] = {}
        for obj in objects:
            region = obj["position"]["region"]
            if region not in region_summary:
                region_summary[region] = []
            region_summary[region].append(obj["class"])

        self._frame_id += 1
        return {
            "frame_id": self._frame_id,
            "schema_version": "3.2",
            "timestamp": timestamp,
            "frame_size": {"w": self.frame_w, "h": self.frame_h},
            "actionable": True,
            "trust": {
                "detection": True,
                "position": True,
                "motion": True,
                "scene": True,
            },
            "camera_motion": {
                "tx": 0.0,
                "ty": 0.0,
                "compensated": True,
                "confidence": 1.0,
                "ego_state": "stopped",
            },
            "objects": objects,
            "scene": {
                "object_count": len(objects),
                "center_occupied": len(center_objects) > 0,
                "dominant_object": {
                    "class": largest["class"],
                    "track_id": largest["track_id"],
                    "rel_size": largest["position"]["rel_size"],
                } if largest else None,
                "risk_level": risk,
                "classes_present": list(set(o["class"] for o in objects)),
                "moving_count": len(moving_objects),
                "region_summary": region_summary,
                "stable": True,
                "time_since_change_sec": 999.0,
                "snapshot_quality": 0.9,
            },
            "meta": {
                "active_tracks": len(objects),
                "total_tracks_in_memory": len(objects),
                "dropped_by_confidence": 0,
            },
        }
