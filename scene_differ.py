"""Scene change detection and natural-language diff generation for LLM consumption."""

import time
from typing import Optional


class SceneDiffer:
    """
    Compares two scene JSON outputs and generates human-readable change descriptions.

    Includes per-track cooldown to suppress rapid enter/leave event spam.
    """

    SIZE_CHANGE_THRESHOLD = 0.15

    def __init__(self, cooldown_sec: float = 0.5):
        self._prev_scene: Optional[dict] = None
        self._cooldown_sec = cooldown_sec
        # track_id -> last event timestamp (for enter/leave debounce)
        self._last_event_time: dict[int, float] = {}

    def diff(self, scene_json: dict) -> list[str]:
        """
        Compare current scene with previous scene and generate change descriptions.

        Returns:
            List of change description strings
        """
        changes: list[str] = []
        now = time.time()

        if self._prev_scene is None:
            self._prev_scene = scene_json
            if scene_json["scene"]["object_count"] > 0:
                classes = ", ".join(scene_json["scene"]["classes_present"])
                changes.append(
                    f"scene_start: initial detection with "
                    f"{scene_json['scene']['object_count']} objects ({classes})"
                )
            return changes

        prev = self._prev_scene
        curr = scene_json

        prev_objs = {
            o["track_id"]: o for o in prev["objects"] if o["track_id"] is not None
        }
        curr_objs = {
            o["track_id"]: o for o in curr["objects"] if o["track_id"] is not None
        }

        prev_ids = set(prev_objs.keys())
        curr_ids = set(curr_objs.keys())

        # Objects entered (with cooldown)
        for tid in curr_ids - prev_ids:
            if self._is_cooled_down(tid, now):
                obj = curr_objs[tid]
                changes.append(
                    f"object_entered: {obj['class']} #{tid} appeared "
                    f"in {obj['position']['region']}"
                )
                self._last_event_time[tid] = now

        # Objects left (with cooldown)
        for tid in prev_ids - curr_ids:
            if self._is_cooled_down(tid, now):
                obj = prev_objs[tid]
                changes.append(
                    f"object_left: {obj['class']} #{tid} disappeared "
                    f"from {obj['position']['region']}"
                )
                self._last_event_time[tid] = now

        # Objects that persisted — check for changes
        for tid in curr_ids & prev_ids:
            curr_obj = curr_objs[tid]
            prev_obj = prev_objs[tid]

            # Region change
            curr_region = curr_obj["position"]["region"]
            prev_region = prev_obj["position"]["region"]
            if curr_region != prev_region:
                changes.append(
                    f"region_change: {curr_obj['class']} #{tid} moved "
                    f"from {prev_region} to {curr_region}"
                )

            # Size change
            curr_size = curr_obj["position"]["rel_size"]
            prev_size = prev_obj["position"]["rel_size"]
            if prev_size > 0:
                size_ratio = (curr_size - prev_size) / prev_size
                if size_ratio > self.SIZE_CHANGE_THRESHOLD:
                    pct = round(size_ratio * 100)
                    changes.append(
                        f"object_approaching: {curr_obj['class']} #{tid} "
                        f"size increased by {pct}%"
                    )
                elif size_ratio < -self.SIZE_CHANGE_THRESHOLD:
                    pct = round(abs(size_ratio) * 100)
                    changes.append(
                        f"object_retreating: {curr_obj['class']} #{tid} "
                        f"size decreased by {pct}%"
                    )

            # Motion state change
            curr_moving = curr_obj["motion"]["moving"]
            prev_moving = prev_obj["motion"]["moving"]
            if curr_moving and not prev_moving:
                direction = curr_obj["motion"]["direction"]
                changes.append(
                    f"motion_start: {curr_obj['class']} #{tid} "
                    f"started moving {direction}"
                )
            elif not curr_moving and prev_moving:
                changes.append(
                    f"motion_stop: {curr_obj['class']} #{tid} stopped moving"
                )

        # Scene-level changes
        prev_s = prev["scene"]
        curr_s = curr["scene"]

        if curr_s["risk_level"] != prev_s["risk_level"]:
            changes.append(
                f"risk_change: {prev_s['risk_level']} → {curr_s['risk_level']}"
            )

        prev_classes = set(prev_s.get("classes_present", []))
        curr_classes = set(curr_s.get("classes_present", []))
        new_classes = curr_classes - prev_classes
        if new_classes:
            changes.append(
                f"new_class: {', '.join(new_classes)} detected for first time"
            )
        gone_classes = prev_classes - curr_classes
        if gone_classes:
            changes.append(
                f"class_gone: {', '.join(gone_classes)} no longer detected"
            )

        if curr_s["center_occupied"] and not prev_s["center_occupied"]:
            changes.append("center_occupied: object entered center region")
        elif not curr_s["center_occupied"] and prev_s["center_occupied"]:
            changes.append("center_cleared: center region is now clear")

        self._prev_scene = scene_json

        # Cleanup old cooldown entries
        self._cleanup_cooldowns(now)

        return changes

    def _is_cooled_down(self, track_id: int, now: float) -> bool:
        """Check if enough time has passed since the last event for this track."""
        last = self._last_event_time.get(track_id)
        if last is None:
            return True
        return (now - last) >= self._cooldown_sec

    def _cleanup_cooldowns(self, now: float) -> None:
        """Remove stale cooldown entries."""
        max_age = self._cooldown_sec * 10
        # Get currently active track_ids from last scene
        active_ids = set()
        if self._prev_scene:
            active_ids = {
                o["track_id"] for o in self._prev_scene.get("objects", [])
                if o.get("track_id") is not None
            }
        stale = [
            tid for tid, ts in self._last_event_time.items()
            if (now - ts) > max_age or tid not in active_ids
        ]
        for tid in stale:
            del self._last_event_time[tid]

    def reset(self) -> None:
        """Reset diff state (e.g., on source switch)."""
        self._prev_scene = None
        self._last_event_time.clear()
