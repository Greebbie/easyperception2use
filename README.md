# Perception Pipeline v3.1

摄像头视频流 → 结构化场景 JSON，给下游 LLM/Claw 做决策。

## 快速开始

```bash
pip install -r requirements.txt

# 接摄像头直接跑
python main.py

# 全功能（可视化 + GUI 调参 + WebSocket + 深度估计）
python main.py --gui --ws --depth

# 没摄像头用 dry-run 验证
python main.py --dry-run
```

## 架构

```
Camera (USB/RTSP/file)
  → FrameGrabber        独立 daemon 线程读帧，只保留最新帧，网络流断线自动重连
    → YOLOv8 + ByteTrack  检测 + 跟踪，输出带 track_id 的 bbox
      → Kalman Filter     2D 卡尔曼滤波，平滑位置和速度，去除检测抖动
        → SceneBuilder    检测结果 → 归一化场景 JSON（坐标/速度/区域/风险）
          → SceneDiffer   和上一帧对比，生成 changes 列表（给 LLM 直接读）
            → OutputController  控制输出频率（4 种策略）
              → OutputHandler   输出到 print / file / callback
              → WebSocket       JSON-RPC 2.0 推送给 Claw/Agent
            → Visualizer        OpenCV 叠加检测框/区域线/信息面板
  → DepthEstimator       Depth Anything v2 单目深度估计（可选）
```

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 检测 | YOLOv8n (ultralytics) | 轻量模型，CPU 也能跑 |
| 跟踪 | ByteTrack | ultralytics 内置，track_id 持续 |
| 深度 | Depth Anything v2 Small (HuggingFace transformers) | 单目相对深度，lazy load，首次下载 ~100MB |
| 平滑 | 手写 2D Kalman Filter (numpy) | 状态 [x, y, vx, vy]，无额外依赖 |
| 视频 | OpenCV VideoCapture | 支持 USB(int)、RTSP(str)、本地文件(str) |
| WS | websockets 16 + JSON-RPC 2.0 | 和 OpenClaw Gateway 协议一致 |
| GUI | tkinter | Python 内置，daemon 线程运行 |

## 模块说明

```
perception/
├── main.py                 入口，CLI 参数，graceful shutdown (SIGINT/SIGTERM)
├── config.py               DEFAULT_CONFIG（31 个配置项，全在这里）
├── frame_grabber.py        独立线程读帧 + 断线重连 + 运行时切换源
├── scene_builder.py        YOLO 结果 → 归一化 JSON（集成 Kalman + Depth）
├── kalman_tracker.py       每个 tracked object 一个 2D Kalman 实例
├── scene_differ.py         两帧对比 → changes 列表（LLM 直接消费）
├── depth_estimator.py      Depth Anything v2 封装，30s 超时，失败自动禁用
├── output_controller.py    4 种输出策略: every_frame / interval / on_change / hybrid
├── output_handler.py       print / file(JSONL) / callback 输出
├── visualizer.py           OpenCV 叠加（检测框/九宫格/FPS/风险面板）
├── metrics.py              延迟 P50/P95、FPS、pipeline 状态机
├── perception_service.py   编程 API（start/stop/subscribe/get_latest_scene）
├── ws_server.py            WebSocket JSON-RPC 2.0 服务端
├── config_gui.py           tkinter 运行时调参面板
├── dry_run.py              合成数据生成器（不需要摄像头/模型）
├── requirements.txt
└── tests/                  pytest 测试套件（37 tests）
```

## JSON 输出格式

```json
{
  "frame_id": 1234,
  "schema_version": "3.1",
  "timestamp": 1711234567.123,
  "frame_size": {"w": 1280, "h": 720},
  "pipeline": {
    "state": "running",
    "degraded_modules": [],
    "uptime_sec": 456.7
  },
  "latency_ms": {
    "grab_to_detect": 12,
    "detect_to_depth": 45,
    "total": 60
  },
  "objects": [
    {
      "track_id": 3,
      "class": "person",
      "confidence": 0.912,
      "position": {
        "rel_x": 0.45,
        "rel_y": 0.62,
        "smoothed_x": 0.447,
        "smoothed_y": 0.618,
        "rel_size": 0.087,
        "region": "middle_center"
      },
      "bbox_px": {"x1": 412, "y1": 305, "x2": 668, "y2": 610},
      "motion": {
        "direction": "left",
        "speed": 0.032,
        "vx": -0.028,
        "vy": 0.015,
        "moving": true
      },
      "depth": {"value": 0.35, "label": "near"}
    }
  ],
  "changes": [
    "region_change: person #3 moved from top_center to middle_center",
    "risk_change: low → medium"
  ],
  "scene": {
    "object_count": 1,
    "center_occupied": true,
    "dominant_object": {"class": "person", "track_id": 3, "rel_size": 0.087},
    "risk_level": "medium",
    "classes_present": ["person"],
    "moving_count": 1,
    "region_summary": {"middle_center": ["person"]},
    "nearest_object": {"class": "person", "track_id": 3, "depth": 0.35},
    "depth_ordering": [3]
  },
  "meta": {
    "active_tracks": 1,
    "total_tracks_in_memory": 3,
    "dropped_by_confidence": 0
  }
}
```

**关键字段：**
- `position.rel_x/rel_y` — 归一化坐标 (0-1)，分辨率无关
- `position.smoothed_x/y` — Kalman 平滑后坐标，给硬件控制用
- `motion.vx/vy` — 归一化速度向量 (rel/sec)
- `depth.value` — 0=最近 1=最远（相对深度，需要 `--depth`）
- `changes` — 和上一帧的差异描述，LLM 直接读这个做决策
- `pipeline.state` — running / degraded / error

## WebSocket API (JSON-RPC 2.0)

端口 18790，用 `--ws` 启动。

```
scene/latest    → 获取最新场景 JSON
scene/subscribe → 订阅场景推送
config/set      → 修改配置 {"key": "min_confidence", "value": 0.5}
source/switch   → 切换视频源 {"source": "rtsp://..."}
status/health   → 健康状态 (fps, latency, state)
```

## CLI 参数

```
--source PATH       视频源（0=USB, rtsp://..., video.mp4）
--model PATH        YOLO 模型路径（默认 yolov8n.pt）
--process-fps N     检测帧率（默认 10）
--strategy NAME     输出策略: every_frame / interval / on_change / hybrid
--interval SEC      输出间隔秒数
--output METHOD     输出方式: print / file / callback
--classes A B C     只检测这些类别
--depth             启用深度估计
--depth-model SIZE  深度模型: small / base / large
--gui               打开配置 GUI
--ws                启动 WebSocket 服务
--ws-port PORT      WS 端口（默认 18790）
--no-viz            关闭可视化窗口
--dry-run           合成数据模式（不需要摄像头）
```

## 测试

```bash
cd perception
python -m pytest tests/ -v
```
