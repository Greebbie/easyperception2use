"""Event aggregation: unify perception changes + watchpoint alerts into a single event stream."""

import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from demo.watchpoint import Alert


@dataclass(frozen=True)
class SceneEvent:
    """Unified event from any source."""
    event_id: str
    timestamp: float
    event_type: str       # "object_entered", "object_left", "dwell_warning", "object_removed", ...
    severity: str         # "info" | "warning" | "alert" | "critical"
    source: str           # "perception" | "watchpoint"
    track_id: Optional[int]
    object_class: Optional[str]
    region: Optional[str]
    description: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "event_type": self.event_type,
            "severity": self.severity,
            "source": self.source,
            "track_id": self.track_id,
            "object_class": self.object_class,
            "region": self.region,
            "description": self.description,
            "details": self.details,
        }


# Severity ordering for comparison
_SEVERITY_ORDER = {"info": 0, "warning": 1, "alert": 2, "critical": 3}

# Map SceneDiffer event prefixes to (event_type, severity)
_CHANGE_PREFIX_MAP = {
    "scene_start": ("scene_start", "info"),
    "object_entered": ("object_entered", "info"),
    "object_left": ("object_left", "info"),
    "region_change": ("region_change", "info"),
    "object_approaching": ("object_approaching", "info"),
    "object_retreating": ("object_retreating", "info"),
    "motion_start": ("motion_start", "info"),
    "motion_stop": ("motion_stop", "info"),
    "risk_change": ("risk_change", "warning"),
    "new_class": ("new_class", "info"),
    "class_gone": ("class_gone", "info"),
    "center_occupied": ("center_occupied", "info"),
    "center_cleared": ("center_cleared", "info"),
}

# Events worth showing in the UI (the rest are too noisy for demo)
# object_left / class_gone are excluded — YOLO flickering causes constant
# false disappearances. Real removal is detected by watchpoint instead.
_SIGNIFICANT_PREFIXES = {
    "object_entered", "new_class", "scene_start",
}

# Per-type cooldown: suppress duplicate event types within this window (seconds)
_PERCEPTION_COOLDOWN_SEC = 8.0

# Regex to extract track_id and class from change strings
_TRACK_RE = re.compile(r"(\w[\w\s]*?)\s*#(\d+)")
_REGION_RE = re.compile(r"(?:in|from|to)\s+([\w_]+)")


class EventAggregator:
    """Aggregates events from SceneDiffer changes and WatchpointMonitor alerts."""

    def __init__(self, max_history: int = 200) -> None:
        self._history: deque[SceneEvent] = deque(maxlen=max_history)
        # Per-type cooldown: event_type -> last_timestamp
        self._last_perception_event: dict[str, float] = {}

    def ingest_scene_changes(
        self, changes: list[str], timestamp: float,
    ) -> list[SceneEvent]:
        """Parse SceneDiffer change strings into SceneEvents.

        Only significant events are returned (object enter/leave, new/gone class).
        Noisy events like risk_change, center_occupied are stored in history
        for LLM context but NOT pushed to the UI.
        """
        events: list[SceneEvent] = []
        for change in changes:
            event = self._parse_change(change, timestamp)
            if event is None:
                continue
            self._history.append(event)

            # Only push significant events to UI, with cooldown
            if event.event_type not in _SIGNIFICANT_PREFIXES:
                continue
            # Use class name for cooldown (not track_id) because
            # flickering detection reassigns track_id each time
            cooldown_key = f"{event.event_type}:{event.object_class or '_'}"
            last = self._last_perception_event.get(cooldown_key, 0)
            if timestamp - last < _PERCEPTION_COOLDOWN_SEC:
                continue
            self._last_perception_event[cooldown_key] = timestamp
            events.append(event)
        return events

    def ingest_watchpoint_alerts(self, alerts: list[Alert]) -> list[SceneEvent]:
        """Convert watchpoint Alerts to SceneEvents."""
        events: list[SceneEvent] = []
        for alert in alerts:
            event = SceneEvent(
                event_id=str(uuid.uuid4())[:8],
                timestamp=alert.timestamp,
                event_type=alert.event_type,
                severity=alert.severity,
                source="watchpoint",
                track_id=alert.track_id,
                object_class=alert.object_class,
                region=None,
                description=alert.message,
                details={
                    "zone_id": alert.zone_id,
                    "zone_label": alert.zone_label,
                    "dwell_sec": alert.dwell_sec,
                },
            )
            events.append(event)
            self._history.append(event)
        return events

    def detect_compound_events(self) -> list[SceneEvent]:
        """Check recent history for compound event patterns.

        Patterns detected:
        - Object removed shortly after dwell alert -> "suspicious_removal"
        - Multiple dwell alerts in same zone -> "zone_congestion"
        """
        now = time.time()
        window = [e for e in self._history if now - e.timestamp < 120.0]
        compounds: list[SceneEvent] = []

        # Pattern: dwell_alert + object_removed within 30s -> suspicious
        dwell_alerts = [e for e in window if e.event_type in ("dwell_alert", "dwell_critical")]
        removals = [e for e in window if e.event_type == "object_removed"]

        for removal in removals:
            for dwell in dwell_alerts:
                same_zone = (
                    removal.details.get("zone_id") == dwell.details.get("zone_id")
                )
                time_close = abs(removal.timestamp - dwell.timestamp) < 30.0
                if same_zone and time_close:
                    compound = SceneEvent(
                        event_id=str(uuid.uuid4())[:8],
                        timestamp=now,
                        event_type="suspicious_removal",
                        severity="critical",
                        source="watchpoint",
                        track_id=removal.track_id,
                        object_class=removal.object_class,
                        region=removal.region,
                        description=(
                            f"{removal.object_class} 在长时间停留"
                            f"（{dwell.details.get('dwell_sec', 0):.0f}s）后"
                            f"从 {removal.details.get('zone_label', '区域')} 被移除"
                        ),
                        details={
                            "related_dwell_event": dwell.event_id,
                            "related_removal_event": removal.event_id,
                            "zone_id": removal.details.get("zone_id"),
                        },
                    )
                    # Avoid duplicate compound events
                    existing_ids = {
                        e.details.get("related_removal_event")
                        for e in self._history
                        if e.event_type == "suspicious_removal"
                    }
                    if removal.event_id not in existing_ids:
                        compounds.append(compound)
                        self._history.append(compound)

        return compounds

    def get_recent(self, count: int = 50) -> list[SceneEvent]:
        """Get most recent events."""
        items = list(self._history)
        return items[-count:]

    def get_summary_for_llm(self, window_sec: float = 60.0) -> dict:
        """Build a summary dict suitable for LLM interpretation.

        Returns:
            {
                "window_sec": 60.0,
                "total_events": 12,
                "highest_severity": "alert",
                "events": [
                    {"time": "14:32:05", "type": "dwell_warning", "severity": "warning",
                     "description": "bottle at Watched Zone for 31s"},
                    ...
                ],
                "active_dwell_count": 2,
                "removal_count": 1,
            }
        """
        now = time.time()
        window = [e for e in self._history if now - e.timestamp < window_sec]

        if not window:
            return {
                "window_sec": window_sec,
                "total_events": 0,
                "highest_severity": "info",
                "events": [],
                "active_dwell_count": 0,
                "removal_count": 0,
            }

        highest = max(window, key=lambda e: _SEVERITY_ORDER.get(e.severity, 0))
        dwell_count = sum(
            1 for e in window if e.event_type.startswith("dwell_")
        )
        removal_count = sum(
            1 for e in window if e.event_type == "object_removed"
        )

        return {
            "window_sec": window_sec,
            "total_events": len(window),
            "highest_severity": highest.severity,
            "events": [
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)),
                    "type": e.event_type,
                    "severity": e.severity,
                    "description": e.description,
                }
                for e in window[-20:]  # Last 20 events max for LLM context
            ],
            "active_dwell_count": dwell_count,
            "removal_count": removal_count,
        }

    def get_recent_significant(self, count: int = 50) -> list[SceneEvent]:
        """Get recent events, filtered to only significant types + watchpoint."""
        items = [
            e for e in self._history
            if e.source == "watchpoint" or e.event_type in _SIGNIFICANT_PREFIXES
        ]
        return items[-count:]

    # ── Internal ──

    # Class name -> Chinese label
    _CLASS_CN = {
        "person": "人", "bottle": "水瓶", "cup": "杯子",
        "cell phone": "手机", "laptop": "笔记本", "book": "书",
        "remote": "遥控器", "keyboard": "键盘", "mouse": "鼠标",
        "chair": "椅子", "backpack": "背包", "tv": "电视",
    }

    # Region -> Chinese label
    _REGION_CN = {
        "top_left": "左上", "top_center": "上方", "top_right": "右上",
        "middle_left": "左侧", "middle_center": "中央", "middle_right": "右侧",
        "bottom_left": "左下", "bottom_center": "下方", "bottom_right": "右下",
    }

    def _cn_class(self, cls: str) -> str:
        return self._CLASS_CN.get(cls, cls) if cls else "物体"

    def _cn_region(self, region: str) -> str:
        return self._REGION_CN.get(region, region) if region else "画面"

    def _parse_change(self, change: str, timestamp: float) -> Optional[SceneEvent]:
        """Parse a SceneDiffer change string into a SceneEvent."""
        parts = change.split(":", 1)
        if len(parts) < 2:
            return None

        prefix = parts[0].strip()

        event_type, severity = _CHANGE_PREFIX_MAP.get(
            prefix, ("unknown", "info")
        )

        # Extract track_id and class
        track_id = None
        object_class = None
        track_match = _TRACK_RE.search(change)
        if track_match:
            object_class = track_match.group(1).strip()
            track_id = int(track_match.group(2))

        # Extract region
        region = None
        region_match = _REGION_RE.search(change)
        if region_match:
            region = region_match.group(1)

        # Build Chinese description
        cn_cls = self._cn_class(object_class)
        cn_region = self._cn_region(region)
        tid_str = f" #{track_id}" if track_id else ""

        desc_map = {
            "object_entered": f"{cn_cls}{tid_str} 出现在{cn_region}",
            "object_left": f"{cn_cls}{tid_str} 从{cn_region}消失",
            "new_class": f"首次检测到 {cn_cls}",
            "class_gone": f"{cn_cls} 不再被检测到",
            "scene_start": f"场景初始化",
            "region_change": f"{cn_cls}{tid_str} 移动到{cn_region}",
            "risk_change": parts[1].strip() if len(parts) > 1 else "",
            "motion_start": f"{cn_cls}{tid_str} 开始移动",
            "motion_stop": f"{cn_cls}{tid_str} 停止移动",
            "center_occupied": "物体进入中央区域",
            "center_cleared": "中央区域已清空",
            "object_approaching": f"{cn_cls}{tid_str} 正在靠近",
            "object_retreating": f"{cn_cls}{tid_str} 正在远离",
        }
        description = desc_map.get(event_type, change)

        return SceneEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            source="perception",
            track_id=track_id,
            object_class=object_class,
            region=region,
            description=description,
        )
