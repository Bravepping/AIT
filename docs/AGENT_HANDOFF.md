# NDI AI Vision Control 开发交接

更新时间为 2026-11-27。当前交付为 **V1.7 跟随状态更新**，源码基线 `2441c2c`，发布分支 `release/v1.7`，历史基线 `v1.7.0-rc.3`。

## 从哪里接手

- 设备 A 只推送 NDI，设备 B 运行 Windows x64 主程序。总览的「开始真实跟随」按保存配置接入 NDI、加载模型和启动 KMBox 监控。看到等待重新按下热键后，再按住配置的左键、右键或组合键。Dry Run 只计算候选。新方案可沿用同一 NDI 源、裁剪尺寸、盒子和桌面分辨率下的校准。
- [Tauri/Vue 界面](../apps/vision-studio/src/App.vue)负责配置和模式展示。[Rust 控制器](../apps/vision-studio/src-tauri/src/control.rs)管理物理热键、跟踪、控制状态和命令队列。[NDI/TensorRT 工作进程](../apps/vision-phase2/worker.py)提供视频和检测结果。[KMBox 侧车](../apps/vision-studio/sidecar/Monitor.cs)通过 [共享驱动](../apps/hardware-check/KmBox.cs)处理 UDP 报文。
- 真实发送默认关闭，只发送相对位移。物理按键状态保护为 `1200 ms`，NDI 和推理结果帧龄保护为 `150 ms`。命令心跳只用于确认连接。松键、F8、失焦、目标丢失、断流或断连会清空待发送命令，恢复后须重新开启真实跟随。现场由操作人员确认后测试实体位移。

## 构建与验证

在 Windows x64 上准备 Node.js、Rust MSVC 工具链和 Windows SDK。在 `apps/vision-studio` 执行 `npm ci`，从仓库根目录构建。`-RuntimeSeed` 指向便携 Python 3.13 运行时，`VISION_ONNX_MODEL` 指向有权使用的 `1013v26s_640.onnx`。运行依赖和首次启动见 [应用 README](../apps/vision-studio/README.md)。

```powershell
$env:VISION_ONNX_MODEL = 'D:\models\1013v26s_640.onnx'
.\scripts\build-vision-studio-v11.ps1 -PackageName VisionStudio-V1.7-Tracking-Status -KmCalibration -AutoTracking -RuntimeSeed 'D:\runtime\python313'
cargo test --release --manifest-path apps/vision-studio/src-tauri/Cargo.toml --features custom-protocol --target x86_64-pc-windows-msvc
```

- 最新本地 ZIP 位于 `dist/VisionStudio-V1.7-Tracking-Status-windows-x64.zip`，SHA-256 为 `ecc49ea42389c57a0a9fea6bd8b676764f14635e0e2ad319d6397990f07c7265`。源码仓库独立于本地构建产物、模型、运行库和现场日志。
- 最新包通过 Windows 启动自检、15 项 Rust 测试与 7 项模拟界面检查。覆盖左右单键触发与松键停止、只读隔离、启动取消、连接失败、匹配校准复用和裁剪变化。实机日志已确认 NDI + TensorRT FP16 检测和左键 Dry Run。90 帧人工标注回放及历史验证见 [候选验收记录](v1.7-final-candidate-validation.md)。

## 待确认

1. **真实位移**。在普通桌面先确认运行模式显示「真实跟随已开启」，再用方案 3 按住左键。核对每次盒子应答、设备 A 的实际位移和松键停止。最近一次现场日志停留在 Dry Run，已有候选，没有真实位移应答。当前控制以 NDI 参考点计算相对位移，程序未读取 Windows 光标绝对坐标。
2. **静止长按**。已有约 5 秒的实体左键、右键和组合键通过记录，后续仍观察到超过 `1200 ms` 的报文间隔。若真实跟随因此停止，保留超时保护和原始 UDP 时间戳继续定位。专项记录见 [KMBox NET 静止长按验证](kmbox-net-hold-validation.md)。
3. **真实多目标**。补充身份可辨且经过人工标注的多目标录像，再验证交叉、遮挡恢复和优先级。已有实录覆盖单目标，跨目标关联目前有模拟覆盖。
