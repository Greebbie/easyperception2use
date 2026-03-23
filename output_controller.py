"""Controls when scene state JSON is output to downstream consumers."""


class OutputController:
    """Controls when to output SceneState to downstream systems."""

    def __init__(
        self,
        strategy: str = "interval",
        interval_sec: float = 1.0,
        change_threshold: float = 0.01,
        stable_window_sec: float = 1.0,
    ):
        """
        Args:
            strategy:
                "every_frame" -- output every detection (debug)
                "interval"    -- fixed time interval output
                "on_change"   -- output only on significant scene change
                "hybrid"      -- interval + on_change (scheduled + burst)
                "stable"      -- output once after scene stabilizes (for LLM/Claw)
            interval_sec: output interval for interval/hybrid modes (seconds)
            change_threshold: threshold for scene change detection (normalized)
            stable_window_sec: how long scene must be stable before output (stable mode)
        """
        self.strategy = strategy
        self.interval_sec = interval_sec
        self.change_threshold = change_threshold
        self.stable_window_sec = stable_window_sec
        self.last_output_time: float = 0
        self.last_output_scene: dict | None = None
        # Stable mode state
        self._last_change_time: float = 0
        self._stable_emitted: bool = False

    def should_output(self, scene_json: dict) -> bool:
        """
        Determine whether the current scene should be output.

        Args:
            scene_json: current scene state dict

        Returns:
            True if the scene should be output
        """
        now = scene_json["timestamp"]

        if self.strategy == "every_frame":
            return True

        if self.strategy == "interval":
            if now - self.last_output_time >= self.interval_sec:
                self.last_output_time = now
                self.last_output_scene = scene_json
                return True
            return False

        if self.strategy == "on_change":
            if self._scene_changed(scene_json):
                self.last_output_time = now
                self.last_output_scene = scene_json
                return True
            return False

        if self.strategy == "hybrid":
            interval_hit = (now - self.last_output_time) >= self.interval_sec
            change_hit = self._scene_changed(scene_json)
            if interval_hit or change_hit:
                self.last_output_time = now
                self.last_output_scene = scene_json
                return True
            return False

        if self.strategy == "stable":
            changed = self._scene_changed(scene_json)
            if changed:
                self._last_change_time = now
                self._stable_emitted = False
                self.last_output_scene = scene_json
                return False
            # Scene hasn't changed — check if stable window elapsed
            if not self._stable_emitted and (now - self._last_change_time) >= self.stable_window_sec:
                self._stable_emitted = True
                self.last_output_time = now
                self.last_output_scene = scene_json
                return True
            return False

        return False

    def _scene_changed(self, scene_json: dict) -> bool:
        """
        Determine if the scene has changed significantly.

        Checks:
            1. Object count changed
            2. Risk level changed
            3. New class appeared
            4. Dominant object size changed significantly (approach/retreat)
            5. Any tracked object crossed region boundaries
        """
        if self.last_output_scene is None:
            return True

        prev = self.last_output_scene
        prev_s = prev["scene"]
        curr_s = scene_json["scene"]

        # 1. Object count
        if curr_s["object_count"] != prev_s["object_count"]:
            return True

        # 2. Risk level
        if curr_s["risk_level"] != prev_s["risk_level"]:
            return True

        # 3. New classes
        if set(curr_s.get("classes_present", [])) != set(
            prev_s.get("classes_present", [])
        ):
            return True

        # 4. Dominant object size change
        curr_dom = curr_s.get("dominant_object")
        prev_dom = prev_s.get("dominant_object")
        if curr_dom and prev_dom:
            size_delta = abs(curr_dom["rel_size"] - prev_dom["rel_size"])
            if size_delta > self.change_threshold:
                return True

        # 5. Any tracked object crossed region boundaries
        curr_regions = {
            o["track_id"]: o["position"]["region"]
            for o in scene_json["objects"]
            if o["track_id"] is not None
        }
        prev_regions = {
            o["track_id"]: o["position"]["region"]
            for o in prev["objects"]
            if o["track_id"] is not None
        }
        for tid, curr_region in curr_regions.items():
            prev_region = prev_regions.get(tid)
            if prev_region is not None and prev_region != curr_region:
                return True

        return False
