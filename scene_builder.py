"""Converts YOLO + ByteTrack detection results into structured scene JSON."""

from collections import deque
from typing import Optional

import numpy as np

from kalman_tracker import KalmanTracker


class SceneBuilder:
    """Translates YOLO + ByteTrack detection results into scene semantic JSON."""

    SCHEMA_VERSION = "3.1"

    def __init__(self, frame_width: int, frame_height: int, config: dict):
        """
        Args:
            frame_width: frame width in pixels
            frame_height: frame height in pixels
            config: pipeline configuration dict
        """
        self.fw = frame_width
        self.fh = frame_height
        self.config = config
        self.prev_positions: dict[int, deque] = {}
        self.track_history_len: int = config.get("track_history_len", 10)
        self.track_lost_timeout: float = config.get("track_lost_timeout", 2.0)
        self._dropped_by_confidence: int = 0
        self._frame_id: int = 0
        self._kalman = KalmanTracker(
            process_noise=config.get("kalman_process_noise", 0.01),
            measurement_noise=config.get("kalman_measurement_noise", 0.05),
        )

    def update_frame_size(self, width: int, height: int) -> None:
        """
        Update frame dimensions after a video source switch.

        Clears all track history and Kalman state since positions from
        the old resolution are meaningless.
        """
        self.fw = width
        self.fh = height
        self.prev_positions.clear()
        self._kalman.reset()
        print(f"[SceneBuilder] Frame size updated to {width}x{height}")

    def build(
        self,
        results,
        timestamp: float,
        depth_map: Optional[np.ndarray] = None,
        latency_ms: Optional[dict[str, float]] = None,
    ) -> dict:
        """
        Process one frame's detection + tracking results into structured scene JSON.

        Args:
            results: Ultralytics model.track() return value
            timestamp: current frame timestamp (time.time())
            depth_map: optional HxW float32 depth map (0=near, 1=far)
            latency_ms: optional latency breakdown dict

        Returns:
            Scene state JSON dict
        """
        self._frame_id += 1
        objects: list[dict] = []
        active_ids: set[int] = set()
        self._dropped_by_confidence = 0

        if results[0].boxes is None or len(results[0].boxes) == 0:
            self._cleanup_lost_tracks(active_ids, timestamp)
            return self._make_output(
                objects, active_ids, timestamp, latency_ms
            )

        for box in results[0].boxes:
            track_id: Optional[int] = None
            if box.id is not None:
                track_id = int(box.id)

            cls_id = int(box.cls)
            if cls_id not in results[0].names:
                continue
            cls_name = results[0].names[cls_id]

            # Class filter
            filter_classes = self.config.get("filter_classes")
            if filter_classes and cls_name not in filter_classes:
                continue

            # Confidence filter
            conf = float(box.conf)
            if conf < self.config.get("min_confidence", 0.3):
                self._dropped_by_confidence += 1
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # Normalized coordinates (0-1)
            rel_x = cx / self.fw
            rel_y = cy / self.fh
            rel_size = (w * h) / (self.fw * self.fh)

            # Kalman smoothing + velocity
            smoothed_x, smoothed_y = rel_x, rel_y
            kalman_vx, kalman_vy = 0.0, 0.0
            if track_id is not None:
                smoothed_x, smoothed_y, kalman_vx, kalman_vy = (
                    self._kalman.update(track_id, rel_x, rel_y, timestamp)
                )

            # Motion (using raw coordinates for backward compat,
            # Kalman velocity used for vx/vy)
            motion = self._calc_motion(
                track_id, rel_x, rel_y, timestamp, kalman_vx, kalman_vy
            )
            region = self._get_region(smoothed_x, smoothed_y)

            if track_id is not None:
                active_ids.add(track_id)

            bbox_px = {
                "x1": round(x1),
                "y1": round(y1),
                "x2": round(x2),
                "y2": round(y2),
            }

            obj_dict: dict = {
                "track_id": track_id,
                "class": cls_name,
                "confidence": round(conf, 3),
                "position": {
                    "rel_x": round(rel_x, 3),
                    "rel_y": round(rel_y, 3),
                    "smoothed_x": round(smoothed_x, 3),
                    "smoothed_y": round(smoothed_y, 3),
                    "rel_size": round(rel_size, 4),
                    "region": region,
                },
                "bbox_px": bbox_px,
                "motion": motion,
            }

            # Add depth info if depth map is available
            if depth_map is not None:
                from depth_estimator import DepthEstimator
                obj_dict["depth"] = DepthEstimator.get_object_depth(
                    depth_map, bbox_px
                )

            objects.append(obj_dict)

        self._cleanup_lost_tracks(active_ids, timestamp)
        self._kalman.cleanup(active_ids)
        return self._make_output(objects, active_ids, timestamp, latency_ms)

    def _make_output(
        self,
        objects: list,
        active_ids: set,
        timestamp: float,
        latency_ms: Optional[dict[str, float]] = None,
    ) -> dict:
        """Assemble the final output JSON."""
        scene_state = self._build_scene_summary(objects)
        output = {
            "frame_id": self._frame_id,
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": timestamp,
            "frame_size": {"w": self.fw, "h": self.fh},
            "objects": objects,
            "scene": scene_state,
            "meta": {
                "active_tracks": len(active_ids),
                "total_tracks_in_memory": len(self.prev_positions),
                "dropped_by_confidence": self._dropped_by_confidence,
            },
        }
        if latency_ms is not None:
            output["latency_ms"] = latency_ms
        return output

    def _get_region(self, rel_x: float, rel_y: float) -> str:
        """
        Map a frame position to a semantic region name.

        Uses a 3x3 grid with configurable thresholds.
        """
        cx_range = self.config.get("center_region_x", (0.33, 0.67))
        cy_range = self.config.get("center_region_y", (0.4, 0.7))

        if rel_x < cx_range[0]:
            col = "left"
        elif rel_x < cx_range[1]:
            col = "center"
        else:
            col = "right"

        if rel_y < cy_range[0]:
            row = "top"
        elif rel_y < cy_range[1]:
            row = "middle"
        else:
            row = "bottom"

        return f"{row}_{col}"

    def _calc_motion(
        self,
        track_id: Optional[int],
        rel_x: float,
        rel_y: float,
        ts: float,
        kalman_vx: float = 0.0,
        kalman_vy: float = 0.0,
    ) -> dict:
        """
        Calculate object motion direction and speed.

        Speed unit: normalized coordinates/second (i.e., frame proportion/sec).
        speed=0.1 means "moves 10% of frame width per second".
        """
        if track_id is None:
            return {
                "direction": "unknown",
                "speed": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "moving": False,
            }

        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = deque(maxlen=self.track_history_len)

        history = self.prev_positions[track_id]

        if len(history) == 0:
            history.append((rel_x, rel_y, ts))
            return {
                "direction": "stationary",
                "speed": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "moving": False,
            }

        prev_x, prev_y, prev_ts = history[-1]
        history.append((rel_x, rel_y, ts))

        dt = max(ts - prev_ts, 0.001)
        dx = rel_x - prev_x
        dy = rel_y - prev_y
        speed = (dx**2 + dy**2) ** 0.5 / dt

        # Use Kalman-derived velocities for vx/vy output
        vx = kalman_vx if kalman_vx != 0.0 else dx / dt
        vy = kalman_vy if kalman_vy != 0.0 else dy / dt

        threshold = self.config.get("motion_speed_threshold", 0.02)
        moving = speed > threshold

        if not moving:
            direction = "stationary"
        elif abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"

        return {
            "direction": direction,
            "speed": round(speed, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "moving": moving,
        }

    def _cleanup_lost_tracks(self, active_ids: set, timestamp: float) -> None:
        """Remove track history for objects not seen within the timeout period."""
        lost = [
            tid
            for tid, history in self.prev_positions.items()
            if tid not in active_ids
            and len(history) > 0
            and (timestamp - history[-1][2]) > self.track_lost_timeout
        ]
        for tid in lost:
            del self.prev_positions[tid]

    def _build_scene_summary(self, objects: list) -> dict:
        """Build scene-level semantic summary — the most useful part for LLM/Claw."""
        if not objects:
            return {
                "object_count": 0,
                "center_occupied": False,
                "dominant_object": None,
                "risk_level": "clear",
                "classes_present": [],
                "moving_count": 0,
                "region_summary": {},
            }

        center_objects = [o for o in objects if "center" in o["position"]["region"]]
        largest = max(objects, key=lambda o: o["position"]["rel_size"])
        moving_objects = [o for o in objects if o["motion"]["moving"]]

        risk = "clear"
        if center_objects:
            max_size = max(o["position"]["rel_size"] for o in center_objects)
            thresholds = self.config.get(
                "risk_thresholds", {"high": 0.15, "medium": 0.05}
            )
            if max_size > thresholds["high"]:
                risk = "high"
            elif max_size > thresholds["medium"]:
                risk = "medium"
            else:
                risk = "low"

        region_summary: dict[str, list[str]] = {}
        for obj in objects:
            region = obj["position"]["region"]
            if region not in region_summary:
                region_summary[region] = []
            region_summary[region].append(obj["class"])

        summary: dict = {
            "object_count": len(objects),
            "center_occupied": len(center_objects) > 0,
            "dominant_object": {
                "class": largest["class"],
                "track_id": largest["track_id"],
                "rel_size": largest["position"]["rel_size"],
            },
            "risk_level": risk,
            "classes_present": list(set(o["class"] for o in objects)),
            "moving_count": len(moving_objects),
            "region_summary": region_summary,
        }

        # Add depth ordering if depth info is available
        objects_with_depth = [o for o in objects if "depth" in o]
        if objects_with_depth:
            sorted_by_depth = sorted(
                objects_with_depth, key=lambda o: o["depth"]["value"]
            )
            summary["depth_ordering"] = [
                o["track_id"] for o in sorted_by_depth
            ]
            nearest = sorted_by_depth[0]
            summary["nearest_object"] = {
                "class": nearest["class"],
                "track_id": nearest["track_id"],
                "depth": nearest["depth"]["value"],
            }

        return summary
