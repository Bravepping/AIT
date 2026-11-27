# NDI AI Vision Control V1.7 Final 候选更新

设备 A 推送 NDI。设备 B 使用 Windows x64 安装包，或解压便携包后启动 `VisionStudio.exe`。更新版使用独立安装目录，同一 Windows 用户沿用原有配置。升级前关闭其他版本，避免同时改写同一份设置。

## 设备 B 首次安装

1. 确认已安装 **Microsoft Edge WebView2 Evergreen Runtime**。离线环境可使用 Microsoft 的 x64 独立安装程序。[Microsoft 安装说明](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution)
2. 安装 **NDI 6 Runtime x64**。安装后重新启动程序，使它读取新的运行库路径。[NDI 官方运行库说明](https://docs.ndi.video/all/developing-with-ndi/sdk/dynamic-loading-of-ndi-libraries)
3. 安装适用的 **NVIDIA 驱动**。若同一台设备上的旧版已实测运行，可以复制旧版 `vision-runtime\cuda-libraries` 和 `vision-runtime\trt-packages` 到新版同名位置。全新目录则先运行 `Start-Vision-CUDA.cmd` 准备 CUDA 和 cuDNN，再运行 `Install-TensorRT.cmd` 安装随包清单锁定的 TensorRT。运行库安装脚本需要联网。离线步骤见随包 `README-TensorRT.txt`。[NVIDIA TensorRT 10.16.1 说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.16.1.html)
4. 启动 `VisionStudio.exe`。在「总览」点「开始真实跟随」，程序按已保存设置加载模型、接入设备 A 的 NDI 并开启 KMBox 监控。多个 NDI 源可在总览选择。首次使用的盒子地址、端口和 UUID 在「设备连接」填写。「仅启动推理和监控」可单独准备这些连接。首次加载 TensorRT 可能构建缓存，后续可在「推理设置」查看缓存。

使用已保存的热键和校准时，看到「等待重新按下热键」后，按住方案中的按键即可跟随。支持左键、右键单独按住和多类别。真实发送默认关闭，按 F8 可停止。有同一 NDI 源、裁剪尺寸和盒子的校准时，桌面分辨率也一致即可点「沿用已保存的校准」。

安装包包含应用、Python 运行时、模型与独立单步校准工具。WebView2、NDI 6 Runtime、NVIDIA 驱动及 GPU 运行库分别准备。

## 配置迁移

程序读取原有热键、校准和设备偏好。在「热键管理」可导出 JSON、导入其他配置，也可恢复上次导入前的设置。文件校验通过后才应用。应用时停止跟随，真实控制需要重新武装。设备地址变化时按新地址重新连接。配对码和真实发送状态使用当前会话。

## 跟踪与算法设置

- 「目标识别」按热键方案设置类别阈值、优先级、偏移、换锁确认帧数和遮挡等待。动态范围从起始半径开始，按锁定目标的速度扩展到配置的最大半径。
- 「鼠标控制」可切换 P 或 PD、调整 X/Y 参数、开启自适应平滑和短时预测。结果区显示 Track ID、目标速度和控制范围。沿用旧配置时，先使用 P，遮挡等待初始为 0 ms。
- 检测缺失时清除候选。Tracker 对旧身份最多保留 320 ms，并以预测中心、检测框重叠和运动方向重新关联。旧身份保留与控制等待分开。真实模式丢失目标时锁止，须重新武装。

## 三种模式

- **Dry Run** 只计算和显示候选。事件保持方案可在盒子、视频和模型就绪后自动等待热键。
- **单步校准** 打开原有校准窗口，由现场人员逐次确认单轴 ±10 位移。测量桌面像素。关闭校准窗口，恢复 NDI 与 KMBox 监控后，选定热键方案，填实测 X/Y 和设备 A 桌面宽高并保存。旧现场数据为 2560×1440、中心裁剪 640×640 和两轴各 +13 px。以当次实测为准。桌面、裁剪或设备变更后重新校准。
- **真实跟随** 按配置使用检测框内的锁定点与 NDI 画面参考点的相对误差。需要按住式热键、有效物理按键报文和新鲜视频帧。保留设备 B 窗口焦点以便按 F8，失焦会停止。松键、断流、目标丢失、通信故障或手动停止后需重新武装。设备 A 只需 NDI。

## 现场顺序

1. 在受控桌面测试场景通过 Dry Run，检查类别、参考点、组合键和松键日志。确认物理按键进入设备 A 后的行为。
2. 完成单步移动和校准。保存会绑定当前 NDI 源、裁剪尺寸和 KMBox 设备标识。
3. 点击「开始真实跟随」。等待程序收到有效的松键报文，看到提示后按住所选方案的按键。
4. 验证按设置的热键开始、松键停止。另测目标消失、F8、NDI 断流和 KMBox 断连。每项测试恢复后应保持停止，直到手动重新开启。
5. 导出运行日志，核对后端停止时间、原因、NDI 和推理帧龄、队列是否清空。网络应答仅表示盒子确认命令，设备 A 的鼠标位移与实际效果用现场观察验收。

KMBox 固件在静止按住时可能停止上报，真实跟随会在物理状态超过 1200 ms 后锁止。等待下一次候选的 Dry Run 可按上次事件保持。单独校准支持小幅手动发送。真实跟随期间每次最多一条在途命令，当前候选替换待处理的旧命令。

构建使用 `scripts/build-vision-studio-v11.ps1 -PackageName VisionStudio-V1.7-Final-Candidate-Update -KmCalibration -AutoTracking -NsisCompiler <makensis.exe>`。
