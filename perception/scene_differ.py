"""Scene change detection and natural-language diff generation for LLM consumption."""

from typing import Optional


class SceneDiffer:
    """
    Compares two scene JSON outputs and generates human-readable change descriptions.

    The 'changes' list is the most valuable field for LLM decision-making —
    the LLM can read changes directly without diffing raw JSON.
    """

    # Size change threshold (relative): 15% increase/decrease is significant
    SIZE_CHANGE_THRESHOLD = 0.15

    def __init__(self):
        self._prev_scene: Optional[dict] = None

    def diff(self, scene_json: dict) -> list[str]:
        """
        Compare current scene with previous scene and generate change descriptions.

        Args:
            scene_json: current scene JSON dict

        Returns:
            List of change description strings
        """
        changes: list[str] = []

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

        # Build track_id -> object maps
        prev_objs = {
            o["track_id"]: o for o in prev["objects"] if o["track_id"] is not None
        }
        curr_objs = {
            o["track_id"]: o for o in curr["objects"] if o["track_id"] is not None
        }

        prev_ids = set(prev_objs.keys())
        curr_ids = set(curr_objs.keys())

        # Objects entered
        for tid in curr_ids - prev_ids:
            obj = curr_objs[tid]
            changes.append(
                f"object_entered: {obj['class']} #{tid} appeared "
                f"in {obj['position']['region']}"
            )

        # Objects left
        for tid in prev_ids - curr_ids:
            obj = prev_objs[tid]
            changes.append(
                f"object_left: {obj['class']} #{tid} disappeared "
                f"from {obj['position']['region']}"
            )

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

            # Size change (approaching/retreating)
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

        # Risk level change
        if curr_s["risk_level"] != prev_s["risk_level"]:
            changes.append(
                f"risk_change: {prev_s['risk_level']} → {curr_s['risk_level']}"
            )

        # New classes appeared
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

        # Center occupation change
        if curr_s["center_occupied"] and not prev_s["center_occupied"]:
            changes.append("center_occupied: object entered center region")
        elif not curr_s["center_occupied"] and prev_s["center_occupied"]:
            changes.append("center_cleared: center region is now clear")

        # Save current as previous for next diff
        self._prev_scene = scene_json

        return changes

    def reset(self) -> None:
        """Reset diff state (e.g., on source switch)."""
        self._prev_scene = None
