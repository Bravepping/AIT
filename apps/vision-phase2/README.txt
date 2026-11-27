VisionHardwareCheck · NDI / ONNX · Windows x64

1. KMBox 按 README.txt 接线。只在设备 B 解压整个 ZIP，双击 VisionHardwareCheck.exe，点「NDI / ONNX 联调」。设备 A 只需推送 NDI。
2. 设备 B 使用 NDI® 6 Runtime。若尚未安装，从 https://ndi.link/NDIRedistV6 安装。NDI 官网 https://ndi.video/
3. 接收 A 的源，保持调试预览关闭。用 CPU 加载随包模型。日志每约 5 秒显示预览状态、累计输出数、接收和推理统计。预览关闭时累计输出数应持平，推理帧数应继续增长。停止或恢复推流可在日志里看到独立事件。
4. 点击「CUDA 环境自检」可查看驱动、显卡、CUDA/cuDNN 库和加载错误。CUDA 只有实际执行模型节点才会显示模型自检成功。依赖为适用的 NVIDIA 驱动、CUDA 12.8+（12.x）和 cuDNN 9。运行库可通过包内 CUDA 安装程序安装，也可从旧版复制 vision-runtime\cuda-libraries。若缺 Microsoft Visual C++ x64 运行库，可从 https://aka.ms/vs/17/release/vc_redist.x64.exe 安装。
5. 接收、模型、KMBox 和物理监控就绪后，手动启用「只读 Dry Run」。左右键同时按住时显示候选位移（画面像素）。松开一键、断流、断开 KMBox 或 F8 后应退出。联调窗口内锁止桌面位移按钮。候选只显示，未接入实际自动位移。
6. 加载模型后点「测试本地图片」，选择一张 PNG、JPEG 或 BMP（最多 1600 万像素）。无需接收 NDI。结果窗口显示检测框、类别编号、置信度及推理耗时，使用和实时推理相同的 0.55 置信度阈值。检测摘要会写入运行日志。
7. 用主窗口「导出日志」，回传测试环境、结果、实际使用的引擎以及日志。

HardwareCheck-03 支持 Auto、CUDA、TensorRT、TensorRT for RTX、CPU 五种选择。手动模式会直接显示所选引擎的错误。Auto 按 TensorRT → CUDA → CPU 尝试，日志会显示实际执行的引擎及前序失败原因。TensorRT 安装和缓存说明见 README-TensorRT.txt。

ONNX Runtime CUDA 环境说明 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
