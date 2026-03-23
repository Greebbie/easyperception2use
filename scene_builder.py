"""Converts YOLO + ByteTrack detection results into structured scene JSON."""

import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from kalman_tracker import KalmanTracker


class SceneBuilder:
    """Translates YOLO + ByteTrack detection results into scene semantic JSON."""

    SCHEMA_VERSION = "3.2"

    def __init__(self, frame_width: int, frame_height: int, config: dict):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(
                f"Invalid frame dimensions: {frame_width}x{frame_height}"
            )
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
            lost_timeout=config.get("track_lost_timeout", 2.0),
        )

        # Object debounce state
        self._confirm_frames = config.get("track_confirm_frames", 2)
        self._lost_frames_threshold = config.get("track_lost_frames", 3)
        self._pending_count: dict[int, int] = {}
        self._confirmed: set[int] = set()
        self._missing_count: dict[int, int] = {}

        # Camera motion estimation
        self._prev_gray: Optional[np.ndarray] = None
        self._global_dx: float = 0.0
        self._global_dy: float = 0.0
        self._camera_confidence: float = 1.0
        self._ego_motion_source: str = config.get("ego_motion_source", "optical_flow")

        # Ego motion state machine: stopped → moving → settling → stopped
        self._ego_moving: bool = False
        self._ego_vx: float = 0.0
        self._ego_vy: float = 0.0
        self._ego_state: str = "stopped"  # "stopped" / "moving" / "settling"
        self._ego_stop_ts: float = 0.0
        self._ego_settle_sec: float = config.get("ego_settle_sec", 0.5)
        self._ego_auto_detect: bool = config.get("ego_auto_detect", True)
        self._low_confidence_streak: int = 0  # for auto-detect

        # Scene stability tracking
        self._last_change_ts: float = 0.0
        self._prev_object_set: set[int] = set()

    def update_frame_size(self, width: int, height: int) -> None:
        """Update frame dimensions after a video source switch."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame dimensions: {width}x{height}")
        self.fw = width
        self.fh = height
        self.prev_positions.clear()
        self._kalman.reset()
        self._pending_count.clear()
        self._confirmed.clear()
        self._missing_count.clear()
        self._prev_gray = None
        self._prev_object_set.clear()
        print(f"[SceneBuilder] Frame size updated to {width}x{height}")

    def set_ego_motion(self, moving: bool, vx: float = 0.0, vy: float = 0.0) -> None:
        """Set robot ego-motion state from external source (odometry/IMU).

        State machine: stopped → moving → settling → stopped
        """
        self._ego_vx = vx
        self._ego_vy = vy

        if moving and not self._ego_moving:
            # Transition: stopped/settling → moving
            self._ego_state = "moving"
        elif not moving and self._ego_moving:
            # Transition: moving → settling
            self._ego_state = "settling"
            self._ego_stop_ts = time.time()

        self._ego_moving = moving

    def build(
        self,
        results,
        timestamp: float,
        depth_fn=None,
        latency_ms: Optional[dict[str, float]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> dict:
        """Process one frame's detection + tracking results into structured scene JSON."""
        self._frame_id += 1
        objects: list[dict] = []
        active_ids: set[int] = set()
        self._dropped_by_confidence = 0

        # Estimate camera motion from optical flow
        current_bboxes: list[tuple] = []
        if frame is not None:
            # We'll collect bboxes first, then do optical flow with foreground masking
            pass

        # Validate YOLO results
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            if frame is not None:
                self._estimate_global_motion(frame, [], timestamp)
            self._update_debounce(active_ids)
            self._track_stability(active_ids, timestamp)
            self._cleanup_lost_tracks(active_ids, timestamp)
            self._kalman.cleanup(active_ids, timestamp)
            return self._make_output(objects, active_ids, timestamp, latency_ms)

        # Collect raw detections
        raw_detections: list[dict] = []
        for box in results[0].boxes:
            track_id: Optional[int] = None
            if box.id is not None:
                track_id = int(box.id)

            cls_id = int(box.cls)
            if cls_id not in results[0].names:
                continue
            cls_name = results[0].names[cls_id]

            filter_classes = self.config.get("filter_classes")
            if filter_classes and cls_name not in filter_classes:
                continue

            conf = float(box.conf)
            if conf < self.config.get("min_confidence", 0.45):
                self._dropped_by_confidence += 1
                continue

            xyxy = box.xyxy[0].tolist()
            raw_detections.append({
                "track_id": track_id,
                "cls_name": cls_name,
                "conf": conf,
                "xyxy": xyxy,
            })
            # Collect bboxes for foreground masking in optical flow
            current_bboxes.append((
                int(xyxy[0] * 160 / self.fw),
                int(xyxy[1] * 120 / self.fh),
                int(xyxy[2] * 160 / self.fw),
                int(xyxy[3] * 120 / self.fh),
            ))

        # Compute camera motion with foreground masking
        if frame is not None:
            self._estimate_global_motion(frame, current_bboxes, timestamp)

        # Process detections with debouncing
        seen_this_frame: set[int] = set()
        for det in raw_detections:
            if det["track_id"] is not None:
                seen_this_frame.add(det["track_id"])

        self._update_debounce(seen_this_frame)

        for det in raw_detections:
            track_id = det["track_id"]
            if track_id is None:
                continue
            if track_id not in self._confirmed:
                continue

            x1, y1, x2, y2 = det["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            rel_x = cx / self.fw
            rel_y = cy / self.fh
            rel_size = (w * h) / (self.fw * self.fh)

            # Camera-compensated coordinates for Kalman
            comp_x = rel_x - self._global_dx
            comp_y = rel_y - self._global_dy

            # Kalman smoothing on compensated coordinates
            smoothed_x, smoothed_y, kalman_vx, kalman_vy = (
                self._kalman.update(track_id, comp_x, comp_y, timestamp)
            )

            # Motion (Kalman already compensated, use directly)
            motion = self._calc_motion(
                track_id, comp_x, comp_y, timestamp, kalman_vx, kalman_vy
            )
            region = self._get_region(smoothed_x, smoothed_y)
            active_ids.add(track_id)

            bbox_px = {
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2),
            }

            # Confidence and prediction from Kalman
            pos_conf, vel_conf = self._kalman.get_confidence(track_id)
            track_age = self._kalman.get_track_age(track_id)
            predicted = self._kalman.predict_next(track_id, dt=0.1)

            obj_dict: dict = {
                "track_id": track_id,
                "class": det["cls_name"],
                "confidence": round(det["conf"], 3),
                "track_age": track_age,
                "position_confidence": round(pos_conf, 2),
                "velocity_confidence": round(vel_conf, 2),
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

            if predicted is not None:
                obj_dict["predicted_next"] = {
                    "cx": round(predicted[0], 3),
                    "cy": round(predicted[1], 3),
                }

            if depth_fn is not None:
                obj_dict["depth"] = depth_fn(bbox_px)

            objects.append(obj_dict)

        self._track_stability(active_ids, timestamp)
        self._cleanup_lost_tracks(active_ids, timestamp)
        self._kalman.cleanup(active_ids, timestamp)
        return self._make_output(objects, active_ids, timestamp, latency_ms)

    def _update_ego_state(self, timestamp: float) -> None:
        """Advance the ego state machine (settling → stopped after settle time)."""
        if self._ego_state == "settling":
            elapsed = timestamp - self._ego_stop_ts
            if elapsed >= self._ego_settle_sec:
                self._ego_state = "stopped"

    def _estimate_global_motion(
        self, frame: np.ndarray, fg_bboxes: list[tuple], timestamp: float
    ) -> None:
        """
        Estimate camera egomotion. Three modes:
        - "none": no compensation
        - "external": use set_ego_motion() state machine
        - "optical_flow": dense flow + optional auto-infer of ego state
        """
        self._update_ego_state(timestamp)

        if self._ego_motion_source == "none":
            self._global_dx = 0.0
            self._global_dy = 0.0
            self._camera_confidence = 1.0
            self._ego_state = "stopped"
            return

        if self._ego_motion_source == "external":
            if self._ego_state == "moving":
                self._global_dx = 0.0
                self._global_dy = 0.0
                self._camera_confidence = 0.0
            elif self._ego_state == "settling":
                # Run optical flow but with reduced trust
                self._optical_flow_estimate(frame, fg_bboxes)
                settle_progress = min(1.0, (timestamp - self._ego_stop_ts) / max(self._ego_settle_sec, 0.01))
                self._camera_confidence *= settle_progress
            else:
                # stopped — full optical flow
                self._optical_flow_estimate(frame, fg_bboxes)
            return

        # Default: optical_flow mode
        self._optical_flow_estimate(frame, fg_bboxes)

        # Auto-detect ego motion from optical flow quality
        if self._ego_auto_detect:
            if self._camera_confidence < 0.3:
                self._low_confidence_streak += 1
                if self._low_confidence_streak >= 3 and self._ego_state == "stopped":
                    self._ego_state = "moving"
            else:
                if self._low_confidence_streak >= 3 and self._ego_state == "moving":
                    self._ego_state = "settling"
                    self._ego_stop_ts = timestamp
                self._low_confidence_streak = 0

        # Auto-degrade: if flow is unreliable, don't compensate
        if self._camera_confidence < 0.3:
            self._global_dx = 0.0
            self._global_dy = 0.0

    def _optical_flow_estimate(
        self, frame: np.ndarray, fg_bboxes: list[tuple]
    ) -> None:
        """Dense optical flow with foreground masking."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._global_dx = 0.0
            self._global_dy = 0.0
            self._camera_confidence = 1.0
            return

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        self._prev_gray = gray

        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

        # Background mask (exclude detected foreground)
        mask = np.ones((120, 160), dtype=bool)
        for bx1, by1, bx2, by2 in fg_bboxes:
            bx1 = max(0, min(159, bx1))
            by1 = max(0, min(119, by1))
            bx2 = max(0, min(160, bx2))
            by2 = max(0, min(120, by2))
            mask[by1:by2, bx1:bx2] = False

        bg_pixels = mask.sum()
        if bg_pixels < 100:
            self._global_dx = 0.0
            self._global_dy = 0.0
            self._camera_confidence = 0.0
            return

        bg_flow_x = flow[..., 0][mask]
        bg_flow_y = flow[..., 1][mask]
        med_dx = float(np.median(bg_flow_x))
        med_dy = float(np.median(bg_flow_y))

        std_x = float(np.std(bg_flow_x))
        std_y = float(np.std(bg_flow_y))
        flow_std = (std_x + std_y) / 2
        bg_ratio = bg_pixels / (120 * 160)
        self._camera_confidence = float(
            max(0.0, min(1.0, bg_ratio * max(0.0, 1.0 - flow_std * 2)))
        )

        self._global_dx = med_dx / 160.0
        self._global_dy = med_dy / 120.0

    def _update_debounce(self, seen_ids: set[int]) -> None:
        """Update object confirmation/loss debounce counters."""
        for tid in seen_ids:
            self._pending_count[tid] = self._pending_count.get(tid, 0) + 1
            self._missing_count.pop(tid, None)
            if self._pending_count[tid] >= self._confirm_frames:
                self._confirmed.add(tid)

        confirmed_copy = set(self._confirmed)
        for tid in confirmed_copy:
            if tid not in seen_ids:
                self._missing_count[tid] = self._missing_count.get(tid, 0) + 1
                if self._missing_count[tid] >= self._lost_frames_threshold:
                    self._confirmed.discard(tid)
                    self._pending_count.pop(tid, None)
                    self._missing_count.pop(tid, None)

        stale_pending = [
            tid for tid in self._pending_count
            if tid not in seen_ids and tid not in self._confirmed
        ]
        for tid in stale_pending:
            del self._pending_count[tid]

    def _track_stability(self, active_ids: set[int], timestamp: float) -> None:
        """Track when the scene last changed (for stability indicator)."""
        if active_ids != self._prev_object_set:
            self._last_change_ts = timestamp
        self._prev_object_set = set(active_ids)

    def _make_output(
        self,
        objects: list,
        active_ids: set,
        timestamp: float,
        latency_ms: Optional[dict[str, float]] = None,
    ) -> dict:
        """Assemble the final output JSON."""
        # Determine what the downstream can trust in this frame
        ego = self._ego_state
        actionable = ego == "stopped"

        scene_state = self._build_scene_summary(objects, timestamp)

        # Mark per-field reliability based on ego state
        trust = {
            "detection": True,         # class, track_id, confidence — always ok
            "position": actionable,    # rel_x/y, region — only when camera stable
            "motion": actionable,      # vx/vy, moving, direction
            "scene": actionable,       # risk, center_occupied, region_summary
        }

        output = {
            "frame_id": self._frame_id,
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": timestamp,
            "frame_size": {"w": self.fw, "h": self.fh},
            "actionable": actionable,
            "trust": trust,
            "camera_motion": {
                "tx": round(self._global_dx, 5),
                "ty": round(self._global_dy, 5),
                "compensated": self._camera_confidence > 0.3,
                "confidence": round(self._camera_confidence, 2),
                "ego_state": ego,
            },
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
        """Map a frame position to a semantic region name (3x3 grid)."""
        cx_range = self.config.get("center_region_x", (0.33, 0.67))
        cy_range = self.config.get("center_region_y", (0.4, 0.7))

        col = "left" if rel_x < cx_range[0] else ("center" if rel_x < cx_range[1] else "right")
        row = "top" if rel_y < cy_range[0] else ("middle" if rel_y < cy_range[1] else "bottom")
        return f"{row}_{col}"

    def _calc_motion(
        self,
        track_id: int,
        comp_x: float,
        comp_y: float,
        ts: float,
        kalman_vx: float = 0.0,
        kalman_vy: float = 0.0,
    ) -> dict:
        """
        Calculate object motion from camera-compensated coordinates.

        Since Kalman receives compensated coords, its velocity IS the true
        object velocity. No post-hoc compensation needed.
        """
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = deque(maxlen=self.track_history_len)

        history = self.prev_positions[track_id]

        if len(history) == 0:
            history.append((comp_x, comp_y, ts))
            return {
                "direction": "stationary",
                "speed": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "moving": False,
            }

        prev_x, prev_y, prev_ts = history[-1]
        history.append((comp_x, comp_y, ts))

        dt = max(ts - prev_ts, 0.001)

        # Use Kalman velocity directly (already compensated)
        vx = kalman_vx
        vy = kalman_vy

        # Fallback to raw compensated delta only on first frame
        if vx == 0.0 and vy == 0.0 and self._kalman.get_track_age(track_id) <= 1:
            vx = (comp_x - prev_x) / dt
            vy = (comp_y - prev_y) / dt

        speed = (vx**2 + vy**2) ** 0.5

        threshold = self.config.get("motion_speed_threshold", 0.02)
        moving = speed > threshold

        if not moving:
            direction = "stationary"
        elif abs(vx) > abs(vy):
            direction = "right" if vx > 0 else "left"
        else:
            direction = "down" if vy > 0 else "up"

        # Motion is reliable only when ego is stopped and compensation is trustworthy
        motion_reliable = self._ego_state == "stopped" and self._camera_confidence > 0.3

        return {
            "direction": direction,
            "speed": round(speed, 4),
            "vx": round(vx, 4),
            "vy": round(vy, 4),
            "moving": moving,
            "reliable": motion_reliable,
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

    def _build_scene_summary(self, objects: list, timestamp: float) -> dict:
        """Build scene-level semantic summary for LLM/Claw."""
        stable_window = self.config.get("stable_window_sec", 1.0)
        time_since_change = timestamp - self._last_change_ts if self._last_change_ts > 0 else 999.0
        is_stable = time_since_change >= stable_window

        base = {
            "object_count": 0,
            "center_occupied": False,
            "dominant_object": None,
            "risk_level": "clear",
            "classes_present": [],
            "moving_count": 0,
            "region_summary": {},
            "stable": is_stable,
            "time_since_change_sec": round(time_since_change, 1),
        }

        if not objects:
            # Compute snapshot quality even with no objects
            base["snapshot_quality"] = 1.0 if is_stable else 0.5
            return base

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

        # Snapshot quality: avg position confidence × stability
        avg_conf = sum(o.get("position_confidence", 0.5) for o in objects) / len(objects)
        quality = avg_conf * (0.8 if is_stable else 0.4) + (0.2 if is_stable else 0.0)

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
            "stable": is_stable,
            "time_since_change_sec": round(time_since_change, 1),
            "snapshot_quality": round(min(1.0, quality), 2),
        }

        objects_with_depth = [o for o in objects if "depth" in o]
        if objects_with_depth:
            sorted_by_depth = sorted(
                objects_with_depth, key=lambda o: o["depth"]["value"]
            )
            summary["depth_ordering"] = [o["track_id"] for o in sorted_by_depth]
            nearest = sorted_by_depth[0]
            summary["nearest_object"] = {
                "class": nearest["class"],
                "track_id": nearest["track_id"],
                "depth": nearest["depth"]["value"],
            }

        return summary

    @staticmethod
    def compact(scene_json: dict) -> dict:
        """Compress to minimal JSON for downstream consumers (~300 bytes)."""
        objects = []
        for o in scene_json.get("objects", []):
            pos = o["position"]
            mot = o["motion"]
            obj = {
                "id": o["track_id"],
                "cls": o["class"],
                "cx": pos["smoothed_x"],
                "cy": pos["smoothed_y"],
                "size": pos["rel_size"],
                "region": pos["region"],
                "conf": o["confidence"],
                "age": o.get("track_age", 0),
                "pos_conf": o.get("position_confidence", 0.0),
                "vx": mot["vx"],
                "vy": mot["vy"],
                "moving": mot["moving"],
                "reliable": mot.get("reliable", True),
            }
            if "predicted_next" in o:
                obj["pred"] = o["predicted_next"]
            if "depth" in o:
                obj["depth"] = o["depth"]["value"]
            objects.append(obj)

        s = scene_json.get("scene", {})
        cam = scene_json.get("camera_motion", {})
        return {
            "ts": round(scene_json["timestamp"], 2),
            "actionable": scene_json.get("actionable", True),
            "objects": objects,
            "scene": {
                "count": s.get("object_count", 0),
                "risk": s.get("risk_level", "clear"),
                "center": s.get("center_occupied", False),
                "moving": s.get("moving_count", 0),
                "stable": s.get("stable", False),
                "quality": s.get("snapshot_quality", 0.0),
            },
            "camera": {
                "tx": cam.get("tx", 0.0),
                "ty": cam.get("ty", 0.0),
                "conf": cam.get("confidence", 0.0),
                "ego": cam.get("ego_state", "stopped"),
            },
            "changes": scene_json.get("changes", []),
        }
