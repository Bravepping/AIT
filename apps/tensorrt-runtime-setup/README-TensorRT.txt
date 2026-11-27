VisionHardwareCheck Windows x64 TensorRT 运行库

1. 关闭 VisionHardwareCheck。保留包内的 Install-TensorRT.cmd、setup-vision-tensorrt.py 和 tensorrt-packages.json。
2. 双击 Install-TensorRT.cmd。首次下载约 2.3 GB，建议至少预留 6 GB 可用空间。安装在 vision-runtime\trt-packages。
3. 已在同一台电脑装过本包的 TensorRT 运行库时，可复制整个 trt-packages 文件夹至新版的 vision-runtime。再运行 Install-TensorRT.cmd 可检查版本和文件大小。
4. 准备 CUDA/cuDNN 运行库后启动 VisionHardwareCheck.exe，进入「NDI / ONNX 联调」，选择 TensorRT 或 TensorRT for RTX，再加载模型。CUDA 安装步骤见 README-CUDA.txt。

离线安装时，将 tensorrt-packages.json 列出的四个原版 .whl 文件放入同一文件夹，再运行 Install-TensorRT.cmd --wheel-cache "该文件夹的完整路径"。需要重新安装时可添加 --repair。
