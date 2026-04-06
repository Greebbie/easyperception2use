"""Zone monitoring: intrusion detection, dwell time tracking, object state changes."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Zone:
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    target_class: str = "person"
    label: str = "Watched Zone"
    active: bool = True
    dwell_threshold_sec: float = 30.0


@dataclass
class Alert:
    timestamp: float
    zone_id: str
    zone_label: str
    object_class: str
    track_id: int
    message: str
    event_type: str = "zone_entry"
    severity: str = "info"
    dwell_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "object_class": self.object_class,
            "track_id": self.track_id,
            "message": self.message,
            "event_type": self.event_type,
            "severity": self.severity,
            "dwell_sec": round(self.dwell_sec, 1),
        }


# Dwell severity thresholds (seconds -> severity)
_DWELL_LEVELS = [
    (120.0, "dwell_critical", "critical"),
    (60.0, "dwell_alert", "alert"),
    (30.0, "dwell_warning", "warning"),
]

# Grace period: how long a class can disappear before dwell resets (seconds)
# Set high because YOLO detection is often unstable (object flickers in/out)
_DWELL_GRACE_SEC = 15.0


_CLASS_CN = {
    "person": "人", "bottle": "水瓶", "cup": "杯子",
    "cell phone": "手机", "laptop": "笔记本", "book": "书",
    "remote": "遥控器", "keyboard": "键盘", "mouse": "鼠标",
    "chair": "椅子", "backpack": "背包", "tv": "电视",
    "toilet": "马桶", "bench": "长椅", "cat": "猫", "dog": "狗",
    "tie": "领带",
}


def _cn(cls: str) -> str:
    return _CLASS_CN.get(cls, cls)


class WatchpointMonitor:
    """Monitors zones for target objects: entry, dwell time, removal."""

    DEBOUNCE_SEC = 10.0

    def __init__(self) -> None:
        self._zones: dict[str, Zone] = {}
        self._alert_history: list[Alert] = []
        self._last_alert_time: dict[str, float] = {}

        # Class-based dwell: zone_id -> {class_name: first_seen_timestamp}
        self._class_dwell: dict[str, dict[str, float]] = {}
        # Classes present last frame: zone_id -> set of class names
        self._prev_classes: dict[str, set[str]] = {}
        # Last seen time per class: zone_id -> {class_name: last_seen_timestamp}
        self._class_last_seen: dict[str, dict[str, float]] = {}
        # Dwell alerts already fired: "zone_id:class:level"
        self._dwell_alerted: set[str] = set()

    # ── Zone CRUD ──

    def add_zone(
        self,
        x1: float, y1: float, x2: float, y2: float,
        target_class: str = "person",
        label: str = "Watched Zone",
        dwell_threshold_sec: float = 30.0,
    ) -> str:
        zone_id = str(uuid.uuid4())[:8]
        self._zones[zone_id] = Zone(
            id=zone_id, x1=min(x1, x2), y1=min(y1, y2),
            x2=max(x1, x2), y2=max(y1, y2),
            target_class=target_class, label=label,
            dwell_threshold_sec=dwell_threshold_sec,
        )
        self._class_dwell[zone_id] = {}
        self._prev_classes[zone_id] = set()
        self._class_last_seen[zone_id] = {}
        return zone_id

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self._zones:
            del self._zones[zone_id]
            self._class_dwell.pop(zone_id, None)
            self._prev_classes.pop(zone_id, None)
            self._class_last_seen.pop(zone_id, None)
            self._dwell_alerted = {
                k for k in self._dwell_alerted if not k.startswith(f"{zone_id}:")
            }
            return True
        return False

    def clear_zones(self) -> None:
        self._zones.clear()
        self._class_dwell.clear()
        self._prev_classes.clear()
        self._class_last_seen.clear()
        self._dwell_alerted.clear()

    # ── Core detection ──

    def check_scene(self, scene_json: dict) -> list[Alert]:
        """Check scene for zone events. Returns new alerts."""
        now = time.time()
        new_alerts: list[Alert] = []
        objects = scene_json.get("objects", [])

        for zone_id, zone in self._zones.items():
            if not zone.active:
                continue

            current_ids, current_objects = self._find_objects_in_zone(objects, zone)

            # Get current classes in zone
            current_classes: dict[str, int] = {}  # class -> any track_id
            for tid, obj in current_objects.items():
                cls = obj["class"]
                current_classes[cls] = tid

            current_class_set = set(current_classes.keys())
            prev_class_set = self._prev_classes.get(zone_id, set())
            dwell_times = self._class_dwell.setdefault(zone_id, {})
            last_seen = self._class_last_seen.setdefault(zone_id, {})

            # ── Entry detection (by class) ──
            for cls in current_class_set:
                last_seen[cls] = now

                if cls not in dwell_times:
                    # New class in zone — start dwell
                    dwell_times[cls] = now

                    # Debounce entry alert
                    debounce_key = f"{zone_id}:{cls}:entry"
                    last_time = self._last_alert_time.get(debounce_key, 0)
                    if now - last_time >= self.DEBOUNCE_SEC:
                        tid = current_classes[cls]
                        alert = Alert(
                            timestamp=now,
                            zone_id=zone_id,
                            zone_label=zone.label,
                            object_class=cls,
                            track_id=tid,
                            message=f"{_cn(cls)} 进入 {zone.label}",
                            event_type="zone_entry",
                            severity="info",
                        )
                        new_alerts.append(alert)
                        self._alert_history.append(alert)
                        self._last_alert_time[debounce_key] = now

            # ── Removal detection (by class, with grace period) ──
            for cls in list(dwell_times.keys()):
                if cls in current_class_set:
                    continue  # Still present

                cls_last = last_seen.get(cls, 0)
                gone_duration = now - cls_last

                if gone_duration < _DWELL_GRACE_SEC:
                    continue  # Within grace period, don't remove yet

                # Class truly gone — fire removal alert
                dwell = now - dwell_times[cls]

                debounce_key = f"{zone_id}:{cls}:removed"
                last_time = self._last_alert_time.get(debounce_key, 0)
                if now - last_time >= self.DEBOUNCE_SEC:
                    alert = Alert(
                        timestamp=now,
                        zone_id=zone_id,
                        zone_label=zone.label,
                        object_class=cls,
                        track_id=0,
                        message=f"{_cn(cls)} 从 {zone.label} 移除（停留 {dwell:.0f}s）",
                        event_type="object_removed",
                        severity="warning",
                        dwell_sec=dwell,
                    )
                    new_alerts.append(alert)
                    self._alert_history.append(alert)
                    self._last_alert_time[debounce_key] = now

                # Clean up
                del dwell_times[cls]
                last_seen.pop(cls, None)
                self._dwell_alerted = {
                    k for k in self._dwell_alerted
                    if not k.startswith(f"{zone_id}:{cls}:")
                }

            # ── Dwell time checks (by class) ──
            for cls, first_seen in dwell_times.items():
                dwell = now - first_seen

                for threshold, event_type, severity in _DWELL_LEVELS:
                    alert_key = f"{zone_id}:{cls}:{event_type}"
                    if dwell >= threshold and alert_key not in self._dwell_alerted:
                        self._dwell_alerted.add(alert_key)
                        tid = current_classes.get(cls, 0)
                        alert = Alert(
                            timestamp=now,
                            zone_id=zone_id,
                            zone_label=zone.label,
                            object_class=cls,
                            track_id=tid,
                            message=f"{_cn(cls)} 在 {zone.label} 已停留 {dwell:.0f}s",
                            event_type=event_type,
                            severity=severity,
                            dwell_sec=dwell,
                        )
                        new_alerts.append(alert)
                        self._alert_history.append(alert)
                        break  # Only fire the highest unalerted level

            self._prev_classes[zone_id] = current_class_set

        return new_alerts

    # ── Helpers ──

    def _find_objects_in_zone(
        self, objects: list[dict], zone: Zone,
    ) -> tuple[set[int], dict[int, dict]]:
        """Find objects of zone's target_class within zone bounds."""
        ids: set[int] = set()
        obj_map: dict[int, dict] = {}
        match_all = not zone.target_class or zone.target_class.lower() == "any"

        for obj in objects:
            if not match_all and obj["class"] != zone.target_class:
                continue

            pos = obj.get("position", {})
            ox = pos.get("smoothed_x", pos.get("rel_x", 0))
            oy = pos.get("smoothed_y", pos.get("rel_y", 0))

            if zone.x1 <= ox <= zone.x2 and zone.y1 <= oy <= zone.y2:
                tid = obj["track_id"]
                ids.add(tid)
                obj_map[tid] = obj

        return ids, obj_map

    def get_dwell_times(self) -> dict[str, list[dict]]:
        """Get current dwell times for all classes in all zones."""
        now = time.time()
        result: dict[str, list[dict]] = {}
        for zone_id, dwell_times in self._class_dwell.items():
            items = []
            for cls, first_seen in dwell_times.items():
                items.append({
                    "track_id": 0,
                    "class": cls,
                    "dwell_sec": round(now - first_seen, 1),
                })
            if items:
                result[zone_id] = items
        return result

    def get_status(self) -> dict:
        now = time.time()
        zones = []
        for z in self._zones.values():
            dwell_info = []
            for cls, first_seen in self._class_dwell.get(z.id, {}).items():
                dwell_info.append({
                    "class": cls,
                    "dwell_sec": round(now - first_seen, 1),
                })
            zones.append({
                "id": z.id, "label": z.label,
                "x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2,
                "target_class": z.target_class, "active": z.active,
                "objects_in_zone": len(self._class_dwell.get(z.id, {})),
                "dwell_threshold_sec": z.dwell_threshold_sec,
                "dwell_info": dwell_info,
            })
        return {
            "zones": zones,
            "alert_count": len(self._alert_history),
            "recent_alerts": [a.to_dict() for a in self._alert_history[-20:]],
        }
