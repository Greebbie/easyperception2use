"""Zone monitoring and alert generation for the door monitoring demo."""

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


@dataclass
class Alert:
    timestamp: float
    zone_id: str
    zone_label: str
    object_class: str
    track_id: int
    message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "time_str": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "zone_id": self.zone_id,
            "zone_label": self.zone_label,
            "object_class": self.object_class,
            "track_id": self.track_id,
            "message": self.message,
        }


class WatchpointMonitor:
    """Monitors zones for target objects and generates alerts."""

    DEBOUNCE_SEC = 10.0

    def __init__(self):
        self._zones: dict[str, Zone] = {}
        self._alert_history: list[Alert] = []
        self._active_in_zone: dict[str, set[int]] = {}
        self._last_alert_time: dict[str, float] = {}

    def add_zone(
        self,
        x1: float, y1: float, x2: float, y2: float,
        target_class: str = "person",
        label: str = "Watched Zone",
    ) -> str:
        zone_id = str(uuid.uuid4())[:8]
        self._zones[zone_id] = Zone(
            id=zone_id, x1=min(x1, x2), y1=min(y1, y2),
            x2=max(x1, x2), y2=max(y1, y2),
            target_class=target_class, label=label,
        )
        self._active_in_zone[zone_id] = set()
        return zone_id

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self._zones:
            del self._zones[zone_id]
            self._active_in_zone.pop(zone_id, None)
            return True
        return False

    def clear_zones(self) -> None:
        self._zones.clear()
        self._active_in_zone.clear()

    def check_scene(self, scene_json: dict) -> list[Alert]:
        """Check scene for objects entering watched zones. Returns new alerts."""
        now = time.time()
        new_alerts: list[Alert] = []
        objects = scene_json.get("objects", [])

        for zone_id, zone in self._zones.items():
            if not zone.active:
                continue

            current_ids: set[int] = set()
            for obj in objects:
                if obj["class"] != zone.target_class:
                    continue

                pos = obj.get("position", {})
                ox = pos.get("smoothed_x", pos.get("rel_x", 0))
                oy = pos.get("smoothed_y", pos.get("rel_y", 0))

                if zone.x1 <= ox <= zone.x2 and zone.y1 <= oy <= zone.y2:
                    current_ids.add(obj["track_id"])

            prev_ids = self._active_in_zone.get(zone_id, set())
            entered = current_ids - prev_ids

            for track_id in entered:
                debounce_key = f"{zone_id}:{track_id}"
                last_time = self._last_alert_time.get(debounce_key, 0)
                if now - last_time < self.DEBOUNCE_SEC:
                    continue

                alert = Alert(
                    timestamp=now,
                    zone_id=zone_id,
                    zone_label=zone.label,
                    object_class=zone.target_class,
                    track_id=track_id,
                    message=f"{zone.target_class.capitalize()} detected at {zone.label}",
                )
                new_alerts.append(alert)
                self._alert_history.append(alert)
                self._last_alert_time[debounce_key] = now

            self._active_in_zone[zone_id] = current_ids

        return new_alerts

    def get_status(self) -> dict:
        zones = []
        for z in self._zones.values():
            in_zone = len(self._active_in_zone.get(z.id, set()))
            zones.append({
                "id": z.id, "label": z.label,
                "x1": z.x1, "y1": z.y1, "x2": z.x2, "y2": z.y2,
                "target_class": z.target_class, "active": z.active,
                "objects_in_zone": in_zone,
            })
        return {
            "zones": zones,
            "alert_count": len(self._alert_history),
            "recent_alerts": [a.to_dict() for a in self._alert_history[-20:]],
        }
