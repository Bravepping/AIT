VisionHardwareCheck Windows x64 CUDA 运行库

1. 在设备 B 关闭正在运行的 VisionHardwareCheck。
2. 如果程序目录已有 Start-Vision-CUDA.cmd，可直接使用。单独下载的补丁需要解压到 NDI / ONNX 版 VisionHardwareCheck.exe 的同一目录。升级时可将旧版已安装的 vision-runtime\cuda-libraries 文件夹复制到新版的 vision-runtime，再运行脚本复用运行库。
3. 双击 Start-Vision-CUDA.cmd。首次从 NVIDIA 发布的 PyPI 包下载并校验运行库，下载约 1.8 GB。安装和缓存共需要约 5 GB 可用空间，文件均保存在程序目录下。
4. 程序启动后，打开「NDI / ONNX 联调」，先点「CUDA 环境自检」，之后选择 CUDA 并加载模型。把结果从主窗口导出。

后续用 Start-Vision-CUDA.cmd 启动 CUDA 版。脚本只给本次启动设置运行库路径。不会更新 NVIDIA 驱动或改系统 PATH。
CUDA 环境自检通过仅说明依赖可加载。加载模型后还需要实际 CUDA 推理验证。

ONNX Runtime CUDA https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
NVIDIA Windows CUDA Wheels https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
