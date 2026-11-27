# NDI AI Vision Control

Windows x64 应用。设备 A 提供 NDI 视频，设备 B 使用 TensorRT 或 CUDA 推理并显示跟踪结果。干运行只计算候选。实体控制默认关闭，现场可使用独立单步校准入口。

- [设备依赖、安装、构建和现场验收步骤](apps/vision-studio/README.md)
- [V1.7 Final 候选更新验收结果](docs/v1.7-final-candidate-validation.md)
- [90 帧录像的人工标注和回放对比](docs/v1.7-tracking-regression.json)

冻结版应用源码对应 Git 标签 `v1.7.0-rc.3`。模型、第三方运行库和 Windows 安装包需要单独准备。
