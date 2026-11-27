# NDI AI Vision Control

Windows x64 应用。设备 A 提供 NDI 视频，设备 B 使用 TensorRT 或 CUDA 推理并显示跟踪结果。总览提供独立的「开始真实跟随」入口，支持已保存的按住式热键。真实发送默认关闭，Dry Run 只计算候选，现场可使用单步校准。

- [设备依赖、安装、构建和现场验收步骤](apps/vision-studio/README.md)
- [V1.7 Final 候选更新验收结果](docs/v1.7-final-candidate-validation.md)
- [90 帧录像的人工标注和回放对比](docs/v1.7-tracking-regression.json)
- [KMBox NET 静止长按专项验收](docs/kmbox-net-hold-validation.md)
- [开发交接与剩余验收](docs/AGENT_HANDOFF.md)

V1.7 跟随状态更新在 `release/v1.7`，保留 `v1.7.0-rc.3` 作为历史基线。模型、第三方运行库和 Windows 安装包需要单独准备。根目录的 Phase 1 任务书与 V1.1 HTML 是早期设计参考。
