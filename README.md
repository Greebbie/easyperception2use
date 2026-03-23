# Perception Pipeline v3.2 — OpenClaw 视觉感知模块

摄像头画面 → 结构化场景 JSON。是 OpenClaw 机械臂的**眼睛**，不是大脑。

所有计算在这里完成（检测、跟踪、滤波、补偿、场景分析）。中枢拿到数据直接用，不需要再算。

## Quick Start

```bash
pip install -r requirements.txt

python main.py                     # 自动检测 USB 摄像头
python main.py --gui               # 摄像头 + GUI 调试面板 + 可视化
python main.py --ws --compact      # 生产模式：WebSocket 推送 + 紧凑 JSON
python main.py --dry-run           # 无摄像头，合成数据测试
```

## 架构

```
Camera (auto-detect / USB / RTSP / file)
  → FrameGrabber          Daemon thread, latest-frame-only, auto-reconnect
    → YOLOv8 + ByteTrack    Detection + tracking (persistent track IDs)
      → Optical Flow         Camera egomotion estimation (foreground-masked)
        → Kalman Filter      2D smoothing on compensated coordinates
          → SceneBuilder     Results → scene JSON (trust model + ego state)
            → SceneDiffer    Frame-to-frame diff → changes list
              → OutputController  5 strategies (every_frame/interval/on_change/hybrid/stable)
                → OutputHandler   print / file (JSONL) / callback / compact mode
                → WebSocket       JSON-RPC 2.0 push (port 18790)
              → Visualizer        OpenCV overlay (bboxes / grid / FPS / risk)
  → DepthEstimator          Depth Anything v2 (optional, --depth)
```

## 感知能力

| 能力 | 状态 | 说明 |
|------|------|------|
| 目标检测 + 跟踪 | ✅ | YOLOv8n + ByteTrack，persistent track ID |
| 位置平滑 + 速度估计 | ✅ | Kalman 2D（Joseph form），position/velocity confidence |
| 位置预测 | ✅ | Kalman 0.1s 前瞻，`predicted_next` |
| 相机运动补偿 | ✅ | 光流 + 前景遮罩，自动降级 |
| 场景语义分析 | ✅ | risk_level / center_occupied / stable / snapshot_quality |
| 变化事件 | ✅ | entered / left / approaching / retreating / region_change / risk_change |
| 深度估计 | ✅ 可选 | Depth Anything v2（相对深度，near/mid/far）|
| 相机标定 + 世界坐标 | 🔧 预留 | config 接口已备，待硬件部署启用 |

## 输出数据

### 中枢最关心的字段

```json
{
  "actionable": true,          // 数据是否可信（robot stopped 时 true）
  "objects": [{
    "track_id": 3,             // 持久跟踪 ID
    "class": "cup",            // YOLO 类别
    "confidence": 0.91,        // 检测置信度
    "smoothed_x": 0.45,       // Kalman 平滑归一化坐标 (0-1)
    "smoothed_y": 0.62,
    "vx": -0.028,             // 归一化速度 (单位/秒)
    "vy": 0.015,
    "moving": true,
    "region": "middle_center", // 3x3 语义区域
    "predicted_next": {"cx": 0.44, "cy": 0.62}
  }],
  "scene": {
    "risk_level": "medium",    // clear / low / medium / high
    "center_occupied": true,   // 中心区域有物体
    "stable": true,            // 场景是否稳定
    "snapshot_quality": 0.92   // 数据质量 (0-1)
  },
  "changes": [                 // 变化事件
    "object_entered: cup #3 appeared in middle_center",
    "risk_change: low → medium"
  ]
}
```

### Trust Model

每帧都有 `actionable` 标志和 `trust` 对象，告诉中枢哪些数据可信：

| ego_state | actionable | 检测 | 位置 | 运动 | 场景 |
|-----------|-----------|------|------|------|------|
| `stopped` | **true** | ✅ | ✅ | ✅ | ✅ |
| `moving` | **false** | ✅ | ❌ | ❌ | ❌ |
| `settling` | **false** | ✅ | ❌ | ❌ | ❌ |

**为什么**：机器人移动时所有物体在画面中位移，静止的杯子看起来也在"移动"。只有 class + track_id + confidence 始终可靠。

### Compact 模式 (--compact)

~400 bytes/frame，带宽受限时使用：

```json
{"ts": 1711234567.12, "actionable": true,
 "objects": [{"id": 3, "cls": "cup", "cx": 0.45, "cy": 0.62,
   "vx": -0.028, "vy": 0.015, "moving": true, "conf": 0.91,
   "region": "middle_center", "reliable": true}],
 "scene": {"risk": "medium", "center": true, "stable": true}}
```

## 中枢接入

### 方式 1: WebSocket (JSON-RPC 2.0)

```bash
python main.py --ws --compact --strategy hybrid
```

端口 18790，默认绑定 127.0.0.1。

| 方法 | 参数 | 说明 |
|------|------|------|
| `scene/latest` | — | 获取最新场景 |
| `scene/subscribe` | — | 订阅实时推送 |
| `ego/motion` | `{"moving": true, "vx": 0.1}` | 告诉感知"我在动" |
| `config/set` | `{"key": "min_confidence", "value": 0.6}` | 运行时调参 |
| `source/switch` | `{"source": 0}` | 切换视频源 |
| `status/health` | — | 管线健康状态 |

安全：config/set 有 allowlist（只允许调感知参数，不能改 model_path 等敏感项），最多 32 连接，单消息 64KB 上限。

### 方式 2: Python API

```python
from perception_service import PerceptionService

svc = PerceptionService(config)
svc.start()

# 订阅场景更新（仅接收可信数据）
svc.subscribe(on_scene, filter_fn=lambda s: s["actionable"])

# 告诉感知模块：机械臂在移动
svc.set_ego_motion(moving=True, vx=0.1, vy=0.0)

# 获取最新场景
scene = svc.get_latest_scene()

svc.stop()
```

### Ego Motion 接口

中枢在机械臂运动前/后调用，感知模块据此判断数据可信度：

```
中枢: ego/motion {"moving": true}   → 感知: 标记数据不可信
中枢: ego/motion {"moving": false}  → 感知: 进入 settling，0.5s 后恢复可信
```

三种 ego 来源（config `ego_motion_source`）：
- `"external"` — 中枢通过 RPC/API 告知（推荐）
- `"optical_flow"` — 感知自动从画面推断（默认）
- `"none"` — 不做运动补偿

## 配置

所有默认值在 `config.py`。关键参数：

```python
"min_confidence": 0.45              # 检测置信度阈值
"process_fps": 10                   # 处理帧率（30 = 更低延迟）
"output_strategy": "hybrid"         # hybrid（推荐）/ stable / interval / on_change / every_frame
"output_compact": False             # --compact 紧凑输出
"ego_motion_source": "optical_flow" # "external"（推荐生产）/ "optical_flow" / "none"
"ego_settle_sec": 0.5               # moving→stopped 过渡时间
"depth_enabled": False              # --depth 启用深度估计
"ws_enabled": False                 # --ws 启用 WebSocket
```

## CLI

```
--source PATH       视频源 ("auto" / 0 / rtsp://... / video.mp4)
--model PATH        YOLO 模型 (default: yolov8n.pt)
--process-fps N     处理帧率 (default: 10)
--strategy NAME     输出策略: every_frame / interval / on_change / hybrid / stable
--compact           紧凑 JSON 输出
--interval SEC      输出间隔（秒）
--output METHOD     输出方式: print / file / callback
--classes A B C     只检测指定类别
--depth             启用深度估计
--gui               打开 GUI 调试面板（含 Live JSON 预览）
--ws                启用 WebSocket 服务
--ws-port PORT      WebSocket 端口 (default: 18790)
--no-viz            关闭可视化窗口
--dry-run           合成数据模式（不需要摄像头）
```

## 模块

```
├── main.py                 入口，CLI，graceful shutdown
├── config.py               配置默认值
├── frame_grabber.py        线程化帧读取，自动检测，断线重连
├── scene_builder.py        YOLO → 场景 JSON（Kalman + 运动补偿 + trust）
├── kalman_tracker.py       2D Kalman（Joseph form，置信度 + 预测）
├── scene_differ.py         帧间差分 → 变化事件（带去抖）
├── output_controller.py    5 种输出策略
├── output_handler.py       print / file / callback / compact
├── depth_estimator.py      Depth Anything v2（可选，lazy-load）
├── visualizer.py           OpenCV 叠加显示
├── metrics.py              延迟 / FPS 监控
├── perception_service.py   程序化 API
├── ws_server.py            WebSocket JSON-RPC 2.0（线程安全 + 安全加固）
├── config_gui.py           tkinter 调试面板（可折叠设置 + Live JSON）
├── dry_run.py              合成数据生成器
└── tests/                  246 tests, 14 test files, 14/14 modules covered
```

## 测试

```bash
python -m pytest tests/ -v    # 246 tests, ~11s
```

覆盖：scene_builder / kalman / depth / output / ws_server / frame_grabber / visualizer / config_gui / perception_service / dry_run / metrics / config / scene_differ / output_controller
