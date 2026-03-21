# Perception Pipeline 实现计划 v3

## 目标

把普通摄像头的视频流处理成结构化 JSON，供 OpenClaw / LLM 层做决策推理。
Demo 导向：要能可视化中间状态，要好看，要稳。

---

## 整体架构

```
Camera (USB/RTSP/文件)
  → FrameGrabber (独立读帧线程, 只保留最新帧, 断线重连)
    → Detector + Tracker (YOLOv8/v11 + ByteTrack)
      → SceneBuilder (检测结果 → 场景语义 JSON)
        → OutputController (控制输出频率和策略)
          → OutputHandler (print/file/callback, 带资源管理)
            → 输出: SceneState JSON (给 Claw/LLM)
          → 输出: 可视化叠加帧 (给 Demo 展示)
```

---

## 关键工程约束

1. **FrameGrabber 必须用独立线程读帧**。主循环和读帧不能在同一个线程，否则 YOLO 推理期间摄像头缓冲区堆积，导致延迟越来越大。网络流场景下尤其致命。
2. **所有速度/运动量必须归一化**。不能用像素绝对值，否则换分辨率全部失效。
3. **网络流必须有断线重连**。Demo 现场网络不稳是常态。
4. **程序必须 graceful shutdown**。Ctrl+C 要能干净退出，释放摄像头、关闭文件、销毁窗口。

---

## 频率控制设计

整条链路涉及三个不同的频率，必须分开控制：

### 三层频率

| 层级 | 名称 | 默认值 | 说明 |
|------|------|--------|------|
| L1 | 摄像头原始帧率 | 30 FPS | 摄像头硬件决定，读帧线程全速读取 |
| L2 | 检测处理帧率 | 10 FPS | 主循环按此节奏从最新帧跑 YOLO |
| L3 | JSON 输出频率 | 按策略触发 | 给 Claw/LLM 的输出频率 |

### 推荐配置

| 场景 | strategy | interval_sec | process_fps | 说明 |
|------|----------|-------------|-------------|------|
| Demo 展示 | `hybrid` | 1.0 | 10 | 定时 1 秒输出 + 突发事件立刻输出 |
| Debug 调试 | `every_frame` | - | 5 | 每帧都输出，慢一点但看得清 |
| 接 LLM 推理 | `interval` | 2.0 | 10 | LLM 响应慢，2 秒一次够了 |
| 接规则引擎 | `hybrid` | 0.5 | 15 | 规则引擎快，可以更频繁 |
| 录像回放分析 | `every_frame` | - | 原始帧率 | 全量处理 |

---

## 核心模块设计

### 1. FrameGrabber（独立线程读帧 + 断线重连）

**核心要求：**
- 内部启动一个 daemon 线程持续读帧，只保留最新一帧
- 主线程调用 `get_latest()` 拿最新帧，无锁等待
- 网络流断线后自动重连，最多重试 N 次
- 支持 USB 摄像头 (int)、RTSP 流 (str)、本地视频文件 (str)

```python
import cv2
import threading
import time
from typing import Optional
import numpy as np


class FrameGrabber:
    """
    独立线程读帧，只保留最新帧。
    解决摄像头缓冲区积压问题 -- 尤其是网络流场景。
    """

    def __init__(
        self,
        source: int | str,
        max_retries: int = 5,
        retry_interval: float = 3.0,
    ):
        """
        Args:
            source: 0 (USB), "rtsp://..." (网络流), "video.mp4" (本地文件)
            max_retries: 网络流断线重连最大次数
            retry_interval: 重连间隔(秒)
        """
        self.source = source
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.is_network_stream = isinstance(source, str) and (
            source.startswith("rtsp://")
            or source.startswith("http://")
            or source.startswith("https://")
        )

        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False

        self._connect()
        self._start_thread()

    def _connect(self) -> bool:
        """打开摄像头/视频源。网络流失败时重试。"""
        retries = 0
        while retries <= self.max_retries:
            self._cap = cv2.VideoCapture(self.source)
            if self.is_network_stream:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._cap.isOpened():
                self._connected = True
                print(f"[FrameGrabber] 已连接: {self.source}")
                return True
            retries += 1
            if retries <= self.max_retries:
                print(
                    f"[FrameGrabber] 连接失败，{self.retry_interval}s 后重试 "
                    f"({retries}/{self.max_retries})"
                )
                time.sleep(self.retry_interval)
        print(f"[FrameGrabber] 连接失败，已达最大重试次数")
        self._connected = False
        return False

    def _start_thread(self):
        """启动后台读帧线程"""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """后台线程：持续读帧，只保留最新帧"""
        consecutive_failures = 0
        while self._running:
            if not self._connected or self._cap is None or not self._cap.isOpened():
                if self.is_network_stream:
                    print("[FrameGrabber] 连接丢失，尝试重连...")
                    if self._connect():
                        consecutive_failures = 0
                        continue
                    else:
                        break
                else:
                    break

            ok, frame = self._cap.read()
            if ok:
                consecutive_failures = 0
                with self._frame_lock:
                    self._frame = frame
            else:
                consecutive_failures += 1
                if not self.is_network_stream:
                    self._running = False
                    break
                if consecutive_failures > 30:
                    self._connected = False
                    consecutive_failures = 0

    def get_latest(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        获取最新帧（非阻塞）。

        Returns:
            (success, frame): frame 可能为 None
        """
        with self._frame_lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def get_frame_size(self) -> tuple[int, int] | None:
        """获取帧尺寸 (width, height)，未连接时返回 None"""
        if self._cap and self._cap.isOpened():
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return None

    def is_alive(self) -> bool:
        return self._running and self._connected

    def release(self):
        """释放资源"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
        print("[FrameGrabber] 已释放")
```

### 2. SceneBuilder（检测结果 → 场景语义 JSON）

**关键设计：**
- 运动速度用归一化坐标计算（rel 单位/秒），不依赖分辨率
- `_dropped_by_confidence` 正确初始化和计数
- 所有 import 完整

```python
from collections import deque
import time
from typing import Optional


class SceneBuilder:
    """将 YOLO + ByteTrack 的检测结果翻译成场景语义 JSON"""

    def __init__(self, frame_width: int, frame_height: int, config: dict):
        self.fw = frame_width
        self.fh = frame_height
        self.config = config
        self.prev_positions: dict[int, deque] = {}
        self.track_history_len: int = config.get("track_history_len", 10)
        self.track_lost_timeout: float = config.get("track_lost_timeout", 2.0)
        self._dropped_by_confidence: int = 0

    def build(self, results, timestamp: float) -> dict:
        """
        处理一帧的检测+跟踪结果，输出结构化场景 JSON。

        Args:
            results: Ultralytics model.track() 返回的结果
            timestamp: 当前帧的时间戳 (time.time())

        Returns:
            场景状态 JSON dict
        """
        objects = []
        active_ids: set[int] = set()
        self._dropped_by_confidence = 0

        if results[0].boxes is None or len(results[0].boxes) == 0:
            self._cleanup_lost_tracks(active_ids, timestamp)
            return self._make_output(objects, active_ids, timestamp)

        for box in results[0].boxes:
            track_id: Optional[int] = None
            if box.id is not None:
                track_id = int(box.id)

            cls_name = results[0].names[int(box.cls)]

            # 类别过滤
            filter_classes = self.config.get("filter_classes")
            if filter_classes and cls_name not in filter_classes:
                continue

            # 置信度过滤
            conf = float(box.conf)
            if conf < self.config.get("min_confidence", 0.3):
                self._dropped_by_confidence += 1
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            # 归一化坐标 (0-1)
            rel_x = cx / self.fw
            rel_y = cy / self.fh
            rel_size = (w * h) / (self.fw * self.fh)

            # 运动（使用归一化坐标计算，单位: rel/sec）
            motion = self._calc_motion(track_id, rel_x, rel_y, timestamp)
            region = self._get_region(rel_x, rel_y)

            if track_id is not None:
                active_ids.add(track_id)

            objects.append({
                "track_id": track_id,
                "class": cls_name,
                "confidence": round(conf, 3),
                "position": {
                    "rel_x": round(rel_x, 3),
                    "rel_y": round(rel_y, 3),
                    "rel_size": round(rel_size, 4),
                    "region": region,
                },
                "bbox_px": {
                    "x1": round(x1),
                    "y1": round(y1),
                    "x2": round(x2),
                    "y2": round(y2),
                },
                "motion": motion,
            })

        self._cleanup_lost_tracks(active_ids, timestamp)
        return self._make_output(objects, active_ids, timestamp)

    def _make_output(self, objects: list, active_ids: set, timestamp: float) -> dict:
        """组装最终输出 JSON"""
        scene_state = self._build_scene_summary(objects)
        return {
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

    def _get_region(self, rel_x: float, rel_y: float) -> str:
        """
        把画面位置映射到语义区域名。
        阈值从 config 读取。
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
        self, track_id: Optional[int], rel_x: float, rel_y: float, ts: float
    ) -> dict:
        """
        计算目标运动方向和速度。

        速度单位: 归一化坐标/秒 (即画面比例/秒)。
        speed=0.1 表示"每秒移动画面宽度的 10%"。
        这样不管分辨率是 640 还是 1920，阈值都通用。
        """
        if track_id is None:
            return {"direction": "unknown", "speed": 0.0, "moving": False}

        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = deque(maxlen=self.track_history_len)

        history = self.prev_positions[track_id]

        if len(history) == 0:
            history.append((rel_x, rel_y, ts))
            return {"direction": "stationary", "speed": 0.0, "moving": False}

        prev_x, prev_y, prev_ts = history[-1]
        history.append((rel_x, rel_y, ts))

        dt = max(ts - prev_ts, 0.001)
        dx = rel_x - prev_x
        dy = rel_y - prev_y
        speed = (dx**2 + dy**2) ** 0.5 / dt

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
            "moving": moving,
        }

    def _cleanup_lost_tracks(self, active_ids: set, timestamp: float):
        """清除超时未出现的 track 历史"""
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
        """场景级语义摘要 -- 给 LLM/Claw 最有用的部分"""
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

        return {
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
```

### 3. OutputController（输出频率和策略）

**增加 region 跨区域变化检测，防止漏掉位移事件。**

```python
class OutputController:
    """控制什么时候把 SceneState 输出给下游"""

    def __init__(
        self,
        strategy: str = "interval",
        interval_sec: float = 1.0,
        change_threshold: float = 0.01,
    ):
        """
        Args:
            strategy:
                "every_frame" -- 每次检测都输出（debug 用）
                "interval"    -- 固定时间间隔输出
                "on_change"   -- 场景有显著变化时才输出
                "hybrid"      -- interval + on_change，定时 + 突发立即
            interval_sec: interval/hybrid 模式的输出间隔（秒）
            change_threshold: 判断场景变化的阈值（归一化单位）
        """
        self.strategy = strategy
        self.interval_sec = interval_sec
        self.change_threshold = change_threshold
        self.last_output_time: float = 0
        self.last_output_scene: dict | None = None

    def should_output(self, scene_json: dict) -> bool:
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

        return False

    def _scene_changed(self, scene_json: dict) -> bool:
        """
        判断场景是否发生了显著变化。
        检查维度：
        1. 物体数量变化
        2. risk level 变化
        3. 出现新类别
        4. 主要物体大小显著变化（接近/远离）
        5. 任何物体跨区域移动（region 变化）
        """
        if self.last_output_scene is None:
            return True

        prev = self.last_output_scene
        prev_s = prev["scene"]
        curr_s = scene_json["scene"]

        # 1
        if curr_s["object_count"] != prev_s["object_count"]:
            return True

        # 2
        if curr_s["risk_level"] != prev_s["risk_level"]:
            return True

        # 3
        if set(curr_s.get("classes_present", [])) != set(
            prev_s.get("classes_present", [])
        ):
            return True

        # 4
        curr_dom = curr_s.get("dominant_object")
        prev_dom = prev_s.get("dominant_object")
        if curr_dom and prev_dom:
            size_delta = abs(curr_dom["rel_size"] - prev_dom["rel_size"])
            if size_delta > self.change_threshold:
                return True

        # 5: 任何被跟踪物体跨区域移动
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
```

### 4. OutputHandler（带资源管理）

```python
import json
from typing import Callable, Optional


class OutputHandler:
    """输出处理器，管理输出生命周期。必须在退出时调用 close()。"""

    def __init__(self, method: str, config: dict):
        """
        Args:
            method: "print" / "file" / "callback"
            config: 包含 output_file_path 等配置
        """
        self.method = method
        self._file = None
        self._callback: Optional[Callable] = None

        if method == "file":
            path = config.get("output_file_path", "scene_output.jsonl")
            self._file = open(path, "a", encoding="utf-8")
            print(f"[OutputHandler] 输出到文件: {path}")

    def __call__(self, scene_json: dict):
        if self.method == "print":
            print(json.dumps(scene_json, ensure_ascii=False, indent=2))
        elif self.method == "file" and self._file:
            self._file.write(json.dumps(scene_json, ensure_ascii=False) + "\n")
            self._file.flush()
        elif self.method == "callback" and self._callback:
            self._callback(scene_json)

    def set_callback(self, fn: Callable):
        """注册回调函数"""
        self._callback = fn

    def close(self):
        """释放资源。必须在程序退出时调用。"""
        if self._file:
            self._file.close()
            self._file = None
            print("[OutputHandler] 文件已关闭")
```

### 5. Visualizer（Demo 可视化叠加）

```python
import cv2
import time
from collections import deque


class Visualizer:
    """在画面上叠加检测结果、区域线、场景状态"""

    RISK_COLORS = {
        "clear": (0, 255, 0),
        "low": (0, 200, 100),
        "medium": (0, 200, 255),
        "high": (0, 0, 255),
    }

    def __init__(self):
        self.fps_history: deque[float] = deque(maxlen=30)
        self.last_time: float = time.time()

    def draw(self, frame, scene_json: dict):
        now = time.time()
        dt = max(now - self.last_time, 0.001)
        self.fps_history.append(1.0 / dt)
        self.last_time = now
        avg_fps = sum(self.fps_history) / len(self.fps_history)

        h, w = frame.shape[:2]

        # 检测框 + 标签
        for obj in scene_json["objects"]:
            b = obj["bbox_px"]
            is_moving = obj["motion"]["moving"]
            color = (0, 200, 255) if is_moving else (0, 255, 0)
            thickness = 3 if is_moving else 2
            cv2.rectangle(
                frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness
            )

            label = f'{obj["class"]} #{obj["track_id"]}'
            sublabel = f'{obj["position"]["region"]} | {obj["motion"]["direction"]}'
            cv2.putText(
                frame, label, (b["x1"], b["y1"] - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            cv2.putText(
                frame, sublabel, (b["x1"], b["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
            )

        # 九宫格区域线
        for x_ratio in [0.33, 0.67]:
            x = int(w * x_ratio)
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
        for y_ratio in [0.4, 0.7]:
            y = int(h * y_ratio)
            cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)

        # 左上: 场景摘要面板
        scene = scene_json["scene"]
        risk_color = self.RISK_COLORS.get(scene["risk_level"], (255, 255, 255))

        info_lines = [
            f'FPS: {avg_fps:.1f} | Objects: {scene["object_count"]} | Moving: {scene["moving_count"]}',
            f'Risk: {scene["risk_level"].upper()} | Center: {"YES" if scene["center_occupied"] else "no"}',
            f'Classes: {", ".join(scene["classes_present"])}',
        ]

        # 半透明黑底
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
```

### 6. 配置文件

```python
# config.py

DEFAULT_CONFIG = {
    # === 输入 ===
    "source": 0,                       # 0=USB, "rtsp://...", "video.mp4"

    # === 检测 ===
    "model_path": "yolov8n.pt",        # 或 yolo11n.pt
    "min_confidence": 0.3,             # 低于此置信度丢弃
    "filter_classes": None,            # None=全部, 或 ["person", "car", "dog"]

    # === 帧率控制 ===
    "process_fps": 10,                 # 主循环检测处理帧率

    # === 跟踪 ===
    "tracker": "bytetrack.yaml",       # 或 "botsort.yaml"
    "track_history_len": 10,           # 保留多少帧的轨迹历史
    "track_lost_timeout": 2.0,         # track 丢失多久后清除 (秒)
    "motion_speed_threshold": 0.02,    # 归一化速度阈值 (rel单位/秒), 低于此认为静止

    # === 输出控制 ===
    "output_strategy": "hybrid",       # "every_frame" / "interval" / "on_change" / "hybrid"
    "output_interval_sec": 1.0,        # interval/hybrid 模式的输出间隔
    "output_change_threshold": 0.01,   # on_change/hybrid 模式的变化阈值 (归一化单位)
    "output_method": "print",          # "print" / "file" / "callback"
    "output_file_path": "scene_output.jsonl",

    # === 场景分析 ===
    "risk_thresholds": {
        "high": 0.15,                  # 中心物体占画面 >15% = 高风险
        "medium": 0.05,                # >5% = 中风险
    },
    "center_region_x": (0.33, 0.67),   # 中心区域 x 范围
    "center_region_y": (0.4, 0.7),     # 中心区域 y 范围

    # === 网络流 ===
    "max_retries": 5,                  # 网络流断线重连次数
    "retry_interval": 3.0,             # 重连间隔 (秒)

    # === 可视化 ===
    "show_visualization": True,
    "viz_window_name": "Perception Pipeline",
    "viz_scale": 1.0,
}
```

---

## 主循环（带 Graceful Shutdown）

```python
# main.py

import signal
import sys
import time
import argparse

from ultralytics import YOLO
import cv2

from config import DEFAULT_CONFIG
from frame_grabber import FrameGrabber
from scene_builder import SceneBuilder
from output_controller import OutputController
from output_handler import OutputHandler
from visualizer import Visualizer


# === Graceful shutdown ===
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print("\n[Main] 收到退出信号，正在清理...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main():
    args = parse_args()
    config = load_config(args)

    print("=" * 60)
    print("  Perception Pipeline v3")
    print("=" * 60)
    print(f"  视频源:     {config['source']}")
    print(f"  模型:       {config['model_path']}")
    print(f"  处理帧率:   {config['process_fps']} FPS")
    print(f"  输出策略:   {config['output_strategy']} (间隔 {config['output_interval_sec']}s)")
    print(f"  输出方式:   {config['output_method']}")
    print(f"  类别过滤:   {config['filter_classes'] or '全部'}")
    print(f"  可视化:     {'开' if config['show_visualization'] else '关'}")
    print("=" * 60)
    print("  按 'q' 或 Ctrl+C 退出")
    print()

    # --- 初始化组件 ---
    model = YOLO(config["model_path"])

    grabber = FrameGrabber(
        source=config["source"],
        max_retries=config["max_retries"],
        retry_interval=config["retry_interval"],
    )

    # 等待第一帧以获取尺寸
    frame_size = None
    for _ in range(100):
        frame_size = grabber.get_frame_size()
        if frame_size:
            break
        time.sleep(0.03)
    if not frame_size:
        print("[Main] 错误：无法获取帧尺寸")
        grabber.release()
        sys.exit(1)

    w, h = frame_size
    print(f"[Main] 帧尺寸: {w}x{h}")

    builder = SceneBuilder(w, h, config)
    output_ctrl = OutputController(
        strategy=config["output_strategy"],
        interval_sec=config["output_interval_sec"],
        change_threshold=config["output_change_threshold"],
    )
    output_handler = OutputHandler(config["output_method"], config)
    viz = Visualizer() if config["show_visualization"] else None

    frame_interval = 1.0 / config["process_fps"]
    last_process_time = 0.0
    last_scene_json = None

    # --- 主循环 ---
    try:
        while not _shutdown_requested:
            if not grabber.is_alive():
                print("[Main] 视频源已断开")
                break

            ok, frame = grabber.get_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            should_process = (now - last_process_time) >= frame_interval

            if should_process:
                last_process_time = now

                results = model.track(
                    frame,
                    persist=True,
                    tracker=config["tracker"],
                    verbose=False,
                )

                scene_json = builder.build(results, now)
                last_scene_json = scene_json

                if output_ctrl.should_output(scene_json):
                    output_handler(scene_json)

            # 可视化
            if viz and last_scene_json is not None:
                viz_frame = viz.draw(frame.copy(), last_scene_json)

                scale = config.get("viz_scale", 1.0)
                if scale != 1.0:
                    viz_frame = cv2.resize(viz_frame, None, fx=scale, fy=scale)

                cv2.imshow(config["viz_window_name"], viz_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        print("[Main] 正在清理资源...")
        grabber.release()
        output_handler.close()
        cv2.destroyAllWindows()
        print("[Main] 已退出")


def parse_args():
    parser = argparse.ArgumentParser(description="Perception Pipeline v3")
    parser.add_argument("--source", default=None, help="摄像头ID或视频路径")
    parser.add_argument("--model", default=None, help="YOLO模型路径")
    parser.add_argument("--process-fps", type=int, default=None, help="检测处理帧率")
    parser.add_argument("--output", default=None, help="输出方式: print / file / callback")
    parser.add_argument("--strategy", default=None, help="输出策略: every_frame / interval / on_change / hybrid")
    parser.add_argument("--interval", type=float, default=None, help="输出间隔(秒)")
    parser.add_argument("--no-viz", action="store_true", help="关闭可视化窗口")
    parser.add_argument("--classes", nargs="+", default=None, help="只检测这些类别")
    parser.add_argument("--dry-run", action="store_true", help="不需要摄像头，用假数据验证全链路")
    return parser.parse_args()


def load_config(args) -> dict:
    """合并默认配置和命令行参数"""
    config = DEFAULT_CONFIG.copy()
    if args.source is not None:
        try:
            config["source"] = int(args.source)
        except ValueError:
            config["source"] = args.source
    if args.model:
        config["model_path"] = args.model
    if args.process_fps:
        config["process_fps"] = args.process_fps
    if args.output:
        config["output_method"] = args.output
    if args.strategy:
        config["output_strategy"] = args.strategy
    if args.interval:
        config["output_interval_sec"] = args.interval
    if args.no_viz:
        config["show_visualization"] = False
    if args.classes:
        config["filter_classes"] = args.classes
    return config


if __name__ == "__main__":
    main()
```

---

## 完整 JSON 输出示例

```json
{
  "timestamp": 1711234567.123,
  "frame_size": {"w": 1280, "h": 720},
  "objects": [
    {
      "track_id": 3,
      "class": "person",
      "confidence": 0.912,
      "position": {
        "rel_x": 0.45,
        "rel_y": 0.62,
        "rel_size": 0.087,
        "region": "middle_center"
      },
      "bbox_px": {"x1": 412, "y1": 305, "x2": 668, "y2": 610},
      "motion": {"direction": "left", "speed": 0.032, "moving": true}
    },
    {
      "track_id": 7,
      "class": "chair",
      "confidence": 0.847,
      "position": {
        "rel_x": 0.78,
        "rel_y": 0.71,
        "rel_size": 0.032,
        "region": "bottom_right"
      },
      "bbox_px": {"x1": 890, "y1": 420, "x2": 1020, "y2": 590},
      "motion": {"direction": "stationary", "speed": 0.0, "moving": false}
    }
  ],
  "scene": {
    "object_count": 2,
    "center_occupied": true,
    "dominant_object": {
      "class": "person",
      "track_id": 3,
      "rel_size": 0.087
    },
    "risk_level": "medium",
    "classes_present": ["person", "chair"],
    "moving_count": 1,
    "region_summary": {
      "middle_center": ["person"],
      "bottom_right": ["chair"]
    }
  },
  "meta": {
    "active_tracks": 2,
    "total_tracks_in_memory": 5,
    "dropped_by_confidence": 1
  }
}
```

---

## 文件结构

```
perception/
├── main.py                 # 主循环 + CLI + graceful shutdown
├── config.py               # DEFAULT_CONFIG
├── frame_grabber.py        # 独立线程取帧 + 断线重连
├── scene_builder.py        # 检测结果 → 场景 JSON
├── output_controller.py    # 输出频率和策略控制
├── output_handler.py       # 输出方式封装 + 资源管理
├── visualizer.py           # Demo 可视化叠加
├── requirements.txt
└── README.md
```

## requirements.txt

```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy
```

---

## 给 Claude Code 的指令

```
请按照 perception_pipeline_plan_v3.md 中的设计实现一个 Python 视频感知管道。

要求：
1. 严格按文件结构拆分模块，每个类一个文件
2. FrameGrabber 必须用独立 daemon 线程读帧，主线程只调用 get_latest() 获取最新帧
3. FrameGrabber 必须实现网络流断线重连逻辑 (max_retries + retry_interval)
4. SceneBuilder 的运动速度必须用归一化坐标计算 (rel单位/秒)，不能用像素绝对值
5. SceneBuilder 的 JSON 输出格式必须完全匹配文档中的 schema，包括 meta 字段
6. OutputController 必须实现全部四种策略，on_change/hybrid 必须包含 region 跨区域检测
7. OutputHandler 必须在 close() 中正确释放文件句柄
8. main.py 必须注册 SIGINT/SIGTERM handler，在 finally 块中清理所有资源
9. 所有函数加类型标注和 docstring
10. config.py 中 DEFAULT_CONFIG 包含文档中所有配置项，代码中不能有任何硬编码魔法数字
11. Visualizer 要有半透明黑底面板、FPS、九宫格线、运动/静止物体不同颜色和粗细
12. 加 --dry-run 模式：生成假数据 (随机物体、运动) 跑全链路，验证 JSON 格式和输出策略
13. README.md 包含：安装、使用示例 (USB/视频/RTSP)、配置说明、JSON schema 说明、--dry-run 说明
```

---

## v2 → v3 修正清单

| # | 问题 | 修正 |
|---|------|------|
| 1 | 单线程阻塞，缓冲区积压 | FrameGrabber 改为独立 daemon 线程读帧，主线程只取最新帧 |
| 2 | `_last_dropped_count` 未初始化 | 改为 `_dropped_by_confidence`，`__init__` 初始化，`build()` 正确计数 |
| 3 | 网络流断线无重连 | FrameGrabber 加 `_connect()` 重连，`_read_loop` 检测断连自动重试 |
| 4 | 运动速度用像素绝对值 | `_calc_motion` 改用 rel_x/rel_y，阈值从 `2.0` 改为 `0.02` |
| 5 | 文件句柄泄漏 | 输出封装为 `OutputHandler` 类，有 `close()`，main `finally` 中调用 |
| 6 | scene_changed 漏掉位移事件 | 增加 track_id → region 跨区域变化检测 |
| 7 | 缺少 import 声明 | 所有模块补全 import |
| 8 | 无 graceful shutdown | main.py 注册 SIGINT/SIGTERM + `finally` 清理 |

---

## 后续可扩展点（不要现在做）

- 区域自定义：支持用 JSON 定义任意多边形区域
- 事件系统：目标进入/离开区域、两目标接近、目标消失，输出 event JSON
- 深度估计：加 MiDaS / Depth Anything 补充距离
- LLM 摘要层：每 N 次输出聚合成自然语言描述
- WebSocket 推送 + 前端仪表盘
- 多摄像头支持
- 录像回放 + 标注工具
