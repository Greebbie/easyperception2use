# Perception Pipeline v3.2 — OpenClaw 视觉感知模块

摄像头画面 → 结构化场景 JSON。是 OpenClaw 机械臂的**眼睛**，不是大脑。

所有计算在这里完成（检测、跟踪、滤波、补偿、场景分析）。中枢拿到数据直接用，不需要再算。

## 职责边界

| 感知模块（本项目） | 中枢（不在本项目） |
|---|---|
| 所有"算"的事：模型推理、滤波、补偿、场景分析 | 所有"想"的事：抓什么、优先级、运动规划 |
| 输出：干净的结构化 JSON | 输入：本模块的 JSON，直接用不需要再算 |
| 告诉中枢"我看到了什么、在哪、往哪动、数据能不能信" | 决定"要不要抓、怎么抓、先抓哪个" |

**设计原则：感知和决策分离，各自独立迭代。** 感知模块不需要知道中枢要干什么。

## 环境要求

- **Python** 3.10+
- **核心依赖**（`requirements.txt`）：ultralytics (YOLOv8)、opencv-python、numpy、websockets、pytest
- **Depth 可选依赖**（`requirements-depth.txt`）：transformers、torch（~2GB，仅在 `--depth` 时需要）
- **GPU**：可选。YOLO 和 Depth 在 CPU 上可用，GPU (CUDA) 会更快

```bash
pip install -r requirements.txt                  # 核心（必装）
pip install -r requirements-depth.txt            # Depth 增强（可选）
```

## Quick Start

```bash
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

## 支持的视频源

| 类型 | 示例 | 说明 |
|------|------|------|
| USB 摄像头 | `--source 0` 或 `--source auto` | 自动检测 USB/UVC 摄像头，当前开发用 |
| RTSP 流 | `--source rtsp://192.168.1.100:554/stream` | 无线 IP 摄像头、WiFi 摄像头、网络摄像头 |
| HTTP 流 | `--source http://192.168.1.100:8080/video` | ESP32-CAM、手机 IP 摄像头 |
| 视频文件 | `--source video.mp4` | 录像回放、离线测试 |
| 合成数据 | `--dry-run` | 无硬件测试，开发用 |

**无线摄像头**：任何支持 RTSP/HTTP 推流的无线摄像头（WiFi IP 摄像头、手机 IP Camera app 等）都可以直接接入，不需要改代码。FrameGrabber 内置断线自动重连（最多 5 次，间隔 3 秒）。

**运行时切换**：通过 WebSocket `source/switch` 或 GUI 面板可以在运行中切换视频源，不需要重启。

## 关键设计决策

### 1. 为什么 2D 优先，不是 3D

**决策：2D 管线是核心，Depth 是可选增强，相机标定是预留接口。**

2D 管线能准确做到的：
- 目标存在性：画面里有没有杯子
- 相对位置：杯子在画面中心偏左（归一化坐标 0-1）
- 运动方向/速度：杯子在往右移
- 场景变化：有人进来了、杯子消失了
- 碰撞风险：中心区域有大物体

2D 做不到的：
- "杯子离我多远" → 归一化坐标不是物理距离
- "两个杯子哪个更近" → 只能靠大小猜，不准
- "机械臂要伸多长" → 需要真实深度 + 相机标定

**为什么这样选**：对中枢做高层决策（"去抓那个杯子"、"停下来有人"）2D 完全够用。单目深度是相对值，闭环抓取用不了。2D 管线最容易调试——检测器漏了、跟踪漂了、还是控制逻辑有问题，一目了然。

### 2. Depth 能补什么、不能补什么

Depth Anything v2 输出**相对深度**（0-1 归一化，帧内排序），不是绝对距离。

| Depth 能做 | Depth 不能做 |
|-----------|-------------|
| 杯子比瓶子近（相对排序） | 杯子距离 28.3cm（绝对距离） |
| near / mid / far 标签 | 毫米级 3D 定位 |
| 深度排序给中枢决策参考 | 闭环伺服抓取控制 |

**精确抓取需要**：真实深度硬件（双目 / ToF / RealSense）+ 相机标定 → `calibration_enabled` 接口已预留，等硬件到位再开。

### 3. Trust Model：为什么需要 actionable 标志

**决策：每帧标记数据可信度，机器人移动时自动降级。**

| ego_state | actionable | 检测 | 位置 | 运动 | 场景 |
|-----------|-----------|------|------|------|------|
| `stopped` | **true** | ✅ | ✅ | ✅ | ✅ |
| `moving` | **false** | ✅ | ❌ | ❌ | ❌ |
| `settling` | **false** | ✅ | ❌ | ❌ | ❌ |

**为什么**：机器人移动时所有物体在画面中位移，静止的杯子看起来也在"移动"。只有 class + track_id + confidence 始终可靠。中枢应该在 `actionable=false` 时暂停位置相关决策。

### 4. Pluggable 模块设计

**决策：所有增强功能都是可选插件，默认关闭，不增加核心依赖。**

| 模块 | 默认 | 启用方式 | 依赖 |
|------|------|---------|------|
| Depth Anything v2 | 关 | `--depth` | transformers + torch |
| 相机标定 | 关 | `calibration_enabled` (config) | 待实现 |
| WebSocket 服务 | 关 | `--ws` | websockets |
| GUI 调试面板 | 关 | `--gui` | tkinter (stdlib) |

**为什么**：核心管线只依赖 ultralytics + opencv + numpy。生产部署时只装需要的。失败自动降级（depth 加载超时 → 自动关闭，管线继续跑）。

### 5. 为什么 Kalman 用 Joseph form

**决策：用 Joseph 稳定形式替代标准 (I-KH)P 更新。**

标准形式在长时间运行后协方差矩阵会丢失正定性（数值漂移），导致置信度计算出负数或 NaN。Joseph form `(I-KH)P(I-KH)^T + KRK^T` 保证协方差矩阵始终正定。已验证 2000 次迭代后特征值仍为正。

### 6. 输出策略选择

| 策略 | 适用场景 | 中枢延迟 |
|------|---------|---------|
| `every_frame` | 调试 | 0ms（每帧输出，量大） |
| `interval` | 遥操作 | 固定间隔（可预测） |
| `on_change` | 事件驱动 | 仅变化时输出（不规律） |
| **`hybrid`（推荐）** | **大多数机器人** | **间隔 + 突发事件都能响应** |
| `stable` | LLM 决策 | 场景稳定后输出（延迟最高） |

### 7. 性能预期

| 指标 | 数值 | 条件 |
|------|------|------|
| YOLO 首帧 | ~300ms | 模型加载 |
| YOLO 稳态 | ~20ms | YOLOv8n, 640x480 |
| Kalman + 场景构建 | ~3ms | |
| 光流补偿 | ~2ms | 160x120 降采样 |
| Depth 首帧 | ~3.5s | CPU, 模型下载后 |
| Depth 稳态 | ~300-400ms | CPU; GPU 更快 |
| **端到端总延迟** | **~25ms (无 depth)** | |
| **端到端总延迟** | **~350ms (有 depth, CPU)** | |

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
    "track_id": 3,             // 持久跟踪 ID（跨帧稳定，用于追踪同一物体）
    "class": "cup",            // YOLO 类别
    "confidence": 0.91,        // 检测置信度
    "track_age": 42,           // 跟踪了多少帧（越大越可信）
    "position_confidence": 0.93, // Kalman 位置置信度 (0-1)
    "velocity_confidence": 0.70, // Kalman 速度置信度 (0-1)
    "smoothed_x": 0.45,       // Kalman 平滑归一化坐标 (0=左, 1=右)
    "smoothed_y": 0.62,       // (0=上, 1=下)
    "vx": -0.028,             // 归一化速度 (单位/秒，已补偿相机运动)
    "vy": 0.015,
    "moving": true,
    "reliable": true,          // 运动数据是否可信
    "region": "middle_center", // 3x3 语义区域
    "predicted_next": {"cx": 0.44, "cy": 0.62},  // 0.1秒后预测位置
    "depth": {"value": 0.35, "label": "near"}     // 可选，需 --depth
  }],
  "scene": {
    "risk_level": "medium",    // clear / low / medium / high
    "center_occupied": true,   // 中心区域有物体（碰撞风险）
    "stable": true,            // 场景是否稳定（无物体进出）
    "snapshot_quality": 0.92   // 数据整体质量 (0-1)
  },
  "camera_motion": {
    "ego_state": "stopped",    // stopped / moving / settling
    "confidence": 0.87,        // 光流补偿可信度
    "compensated": true        // 坐标是否已补偿相机运动
  },
  "changes": [                 // 变化事件（中枢可用于事件驱动）
    "object_entered: cup #3 appeared in middle_center",
    "risk_change: low → medium"
  ]
}
```

### Compact 模式 (--compact)

~400 bytes/frame，带宽受限时使用：

```json
{"ts": 1711234567.12, "actionable": true,
 "objects": [{"id": 3, "cls": "cup", "cx": 0.45, "cy": 0.62,
   "vx": -0.028, "vy": 0.015, "moving": true, "conf": 0.91,
   "region": "middle_center", "reliable": true,
   "pred": {"cx": 0.44, "cy": 0.62}}],
 "scene": {"risk": "medium", "center": true, "stable": true, "quality": 0.92},
 "camera": {"tx": 0.003, "ty": -0.001, "conf": 0.87, "ego": "stopped"},
 "changes": ["object_entered: cup #3 appeared in middle_center"]}
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
| `scene/subscribe` | — | 订阅实时推送（感知主动推 `scene/update`）|
| `ego/motion` | `{"moving": true, "vx": 0.1}` | 告诉感知"我在动"（**必须接入**）|
| `config/set` | `{"key": "min_confidence", "value": 0.6}` | 运行时调参 |
| `source/switch` | `{"source": 0}` | 切换视频源 |
| `status/health` | — | 管线健康状态（state / fps / latency / degraded_modules）|

**安全**：config/set 有 allowlist（只允许调感知参数，不能改 model_path 等敏感项），最多 32 连接，单消息 64KB 上限，错误响应不泄露内部细节。

### 方式 2: Python API（同进程集成）

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

# 运行时调参
svc.set_config("min_confidence", 0.6)

# 健康检查
health = svc.get_status()  # {"state": "running", "fps": 10, ...}

svc.stop()
```

### Ego Motion 接口（中枢必须接入）

中枢在机械臂运动前/后调用，感知模块据此判断数据可信度：

```
中枢: ego/motion {"moving": true}   → 感知: 标记位置/运动/场景数据不可信
中枢: ego/motion {"moving": false}  → 感知: 进入 settling，0.5s 后恢复可信
```

三种 ego 来源（config `ego_motion_source`）：
- `"external"` — 中枢通过 RPC/API 告知（**生产推荐**）
- `"optical_flow"` — 感知自动从画面推断（默认，开发用）
- `"none"` — 不做运动补偿

**为什么 external 推荐**：光流推断有延迟和误判风险，中枢自己知道"我要动了"比感知猜更准确。

## 配置

所有默认值在 `config.py`。关键参数：

```python
"min_confidence": 0.45              # 检测置信度阈值（低 = 更多检测但更多误报）
"process_fps": 10                   # 处理帧率（30 = 更低延迟，更高 CPU）
"output_strategy": "hybrid"         # hybrid（推荐）/ stable / interval / on_change / every_frame
"output_compact": False             # --compact 紧凑输出
"ego_motion_source": "optical_flow" # "external"（推荐生产）/ "optical_flow" / "none"
"ego_settle_sec": 0.5               # moving→stopped 过渡时间
"kalman_process_noise": 0.01        # 越高 = 更信任观测，越低 = 更平滑
"kalman_measurement_noise": 0.05    # 越高 = 更平滑但更滞后
"depth_enabled": False              # --depth 启用深度估计
"ws_enabled": False                 # --ws 启用 WebSocket
"ws_host": "127.0.0.1"             # WebSocket 绑定地址（0.0.0.0 开放网络）
"ws_port": 18790                    # WebSocket 端口（避开 OpenClaw Gateway 18789）
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
--depth             启用深度估计（需 pip install -r requirements-depth.txt）
--depth-model SIZE  深度模型: small / base / large（default: small）
--gui               打开 GUI 调试面板（Detection / Output / Plugins / Source + Live JSON）
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
├── depth_estimator.py      Depth Anything v2（可选，lazy-load，失败自动降级）
├── visualizer.py           OpenCV 叠加显示
├── metrics.py              延迟 / FPS / 健康监控
├── perception_service.py   程序化 API（subscribe / set_ego_motion / get_status）
├── ws_server.py            WebSocket JSON-RPC 2.0（线程安全 + allowlist + 连接限制）
├── config_gui.py           tkinter 调试面板（4 组设置 + Live JSON + 实时指标）
├── dry_run.py              合成数据生成器（schema 与真实管线一致）
└── tests/                  246 tests, 14 test files, 14/14 modules covered
```

## 坐标系

```
(0,0) ────────────────── (1,0)
  │    top_left │ top_center │ top_right
  │  ───────────┼────────────┼──────────  y=0.4
  │  mid_left   │ mid_center │ mid_right
  │  ───────────┼────────────┼──────────  y=0.7
  │  btm_left   │ btm_center │ btm_right
(0,1) ────────────────── (1,1)
         x=0.33      x=0.67
```

- **原点** `(0, 0)` = 画面左上角
- **归一化** `rel_x = pixel_x / frame_width`，范围 0-1
- **smoothed_x/y** = Kalman 滤波后的归一化坐标（已补偿相机运动）
- **vx/vy** = 归一化速度（单位/秒），正值 = 向右/向下
- **3x3 区域边界**：x 方向 0.33 / 0.67，y 方向 0.4 / 0.7（可通过 config 调整）

## YOLO 模型选择

| 模型 | 参数量 | 速度 (CPU) | 精度 (mAP) | 推荐场景 |
|------|--------|-----------|-----------|---------|
| **yolov8n.pt**（当前默认）| 3.2M | ~20ms | 37.3 | 实时性优先，嵌入式/CPU |
| yolov8s.pt | 11.2M | ~50ms | 44.9 | 精度和速度平衡 |
| yolov8m.pt | 25.9M | ~120ms | 50.2 | 高精度，需 GPU |
| yolo11n.pt | 2.6M | ~18ms | 39.5 | 最新架构，推荐试用 |

切换模型：`python main.py --model yolov8s.pt` 或修改 config `"model_path"`。模型会自动下载。

## Track ID 行为

- `track_id` 由 ByteTrack 分配，**跨帧稳定**：同一物体在连续帧中保持相同 ID
- 物体被遮挡 < 2 秒：Kalman 预测维持状态，重新出现后 ID 保持不变
- 物体消失 > 2 秒（`track_lost_timeout`）：ID 释放，重新出现会分配新 ID
- **新物体确认**：默认 1 帧即确认（`track_confirm_frames=1`，信任 ByteTrack）
- **丢失缓冲**：连续 10 帧未检测到才真正移除（`track_lost_frames=10`）
- `track_age` 字段表示该 ID 被跟踪了多少帧，越大越可信

## 降级和容错

| 故障 | 行为 | 管线状态 |
|------|------|---------|
| 摄像头断开 | FrameGrabber 自动重连（网络流最多 5 次），USB/文件直接退出 | `state: error` |
| YOLO 推理失败 | 跳过该帧，继续下一帧，输出上一次有效场景 | `degraded_modules: ["yolo"]` |
| Depth 加载超时 (30s) | 自动关闭 depth，核心管线继续 | `degraded_modules: ["depth"]` |
| Depth 推理失败 | 返回 None，该帧无 depth 数据，其余正常 | 正常 |
| 光流置信度低 | 自动停止运动补偿（`compensated: false`）| 正常 |
| WebSocket 客户端断开 | 自动清理，不影响其他客户端和管线 | 正常 |
| 输出文件路径无效 | 启动时报错退出（RuntimeError） | 不启动 |

**设计原则：核心 2D 管线永远跑，增强模块失败自动降级。** `pipeline.degraded_modules` 告诉中枢哪些能力受损。

## 已知限制

- **小目标**：< 32x32 像素的物体 YOLO 容易漏检
- **遮挡**：物体被完全遮挡后 Kalman 只能预测 ~2 秒，之后丢失
- **相似外观**：多个相同类别物体紧挨着时 ByteTrack 可能混淆 ID
- **快速运动**：物体在帧间移动 > 画面宽度 1/3 时跟踪可能断裂
- **光照变化**：剧烈光照变化（开灯/关灯）会短暂影响检测和光流
- **单目深度**：Depth Anything v2 是相对深度，不是绝对距离，且帧间不一致
- **CPU 延迟**：Depth 在 CPU 上 ~350ms/帧，实时性受限；建议有 GPU 再开

## 测试

```bash
python -m pytest tests/ -v    # 246 tests, ~11s
```

覆盖全部 14 个模块：scene_builder (81) / ws_server (20) / perception_service (16) / kalman (15) / output_controller (15) / scene_differ (15) / depth (14) / dry_run (13) / frame_grabber (12) / config_gui (12) / visualizer (10) / output_handler (8) / metrics (8) / config (7)
