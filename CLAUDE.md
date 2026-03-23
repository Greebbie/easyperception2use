# easyperception2use — 通用视觉感知模块

## 定位
摄像头画面 → 结构化场景 JSON，供下游机器人/LLM 做决策。是"眼睛"，不是"大脑"。
- 所有"算"的事情在这里完成（检测、跟踪、滤波、补偿、场景分析）
- 不做任何决策（抓什么、优先级、规划 = 中枢的事）
- 中枢拿到我们的输出直接用，不需要再做计算

## 架构
2D 核心管线始终运行：Detection → Tracking → Kalman → Optical Flow → Scene JSON
- Depth Anything v2 是可选增强（--depth），不是默认依赖
- 相机标定接口已预留（calibration_enabled），待硬件部署时启用
- GUI（tkinter）是调试工具，不是生产界面

## 当前状态
- Schema v3.2，246 测试全通过（14/14 模块覆盖）
- 33+ 源码 bug 已修复（Kalman Joseph form、WS 线程安全、安全加固等）
- 摄像头实跑验证通过：640x480，稳态延迟 20ms

## 设计原则
1. 2D pipeline 优先，depth / calibration 是 pluggable 增强模块
2. Trust model：actionable=true 时才信任 position/motion/scene
3. Ego state machine：stopped → moving → settling → stopped
4. 感知只输出数据，决策是中枢的事
5. 新模块遵循 pluggable pattern（不增加核心依赖）

## 常用命令
```bash
python main.py --source 0 --gui                              # 开发调试（摄像头+GUI+可视化）
python main.py --source 0 --ws --compact --strategy hybrid    # 生产模式（WebSocket推送）
python main.py --dry-run                                      # 无摄像头测试
python -m pytest tests/ -v                                    # 运行 246 个测试
```

## 核心输出字段
中枢最关心的：`{track_id, class, smoothed_x, smoothed_y, vx, vy, confidence, region, actionable, risk_level, changes}`

## 文件结构
14 个 Python 模块 + 14 个测试文件，详见 README.md

## 禁止事项
- 不要把 depth / calibration 加为默认依赖
- 不要修改 scene_builder.py 的输出 schema 而不同步更新 dry_run.py 和测试
- 不要在 ws_server 的 _SETTABLE_KEYS 之外暴露 config key
- 不要在感知模块里加决策逻辑（那是中枢的事）
