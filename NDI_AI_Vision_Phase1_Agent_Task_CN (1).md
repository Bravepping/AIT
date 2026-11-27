# NDI AI Vision Control — Phase 1 Windows 实机测试开发任务

## 目标与优先级

**立即开发并交付第一个可以在 Windows x64 双击运行的实机测试程序**，先验证 KMBox NET 物理鼠标按键监控及小幅移动，再加入 NDI 接收、YOLO26s 推理，最后做只读/干运行闭环。用户没有本地编译环境，只有源码不算完成交付。

## 双机拓扑（已经确定，不要再次询问）

- **设备 A（目标电脑）**：运行目标程序、发送 NDI 画面；KMBox NET 通过 USB 向 A 提供鼠标输入；A **不安装、不运行 AI 推理程序或热键监听程序**。
- **设备 B（AI 推理电脑）**：Windows + RTX 3060，运行我们开发的 Tauri v2 / Vue 3 / Rust 程序；接收 NDI、加载 `1013v26s_640.onnx`、读取 KMBox 物理鼠标按键、判断热键、通过局域网向盒子发送鼠标控制命令。
- 物理鼠标接在 **KMBox NET 的鼠标输入接口**，KMBox 的 USB 输出连接设备 A，KMBox 的网口与设备 B 在同一局域网。
- **热键规则**：左键 AND 右键同时处于按住状态才允许目标跟随；松开任意一键立即终止该触发。左右键按下顺序不限。后续可扩展侧键和多组配置。
- **视频预览默认关闭**：关闭只停止视频绘制和 UI 帧传递，不得关闭 NDI 接收或模型推理。调试时可开启并限制为 5/10/15/30 FPS。
- **自动控制默认关闭**，用户必须明确启用。启动识别 ≠ 启动鼠标自动控制。

## 第一优先交付物：Windows 可执行测试包

必须先做 `VisionHardwareCheck.exe`（名称可调整），提供 Windows x64 ZIP、必要运行库、`README.txt`、日志导出。若 Agent 没有本地 Windows 编译环境，创建 GitHub Actions `windows-latest` 构建流水线并上传 ZIP Artifact；没有真实产物、构建记录，不得宣称可执行包已经完成。

测试程序只需简洁界面，不需要复刻完整 V1.1 UI。最少包含以下功能：

1. **KMBox 连接**：输入真实 IP、命令端口、MAC/UUID，连接/断开及错误提示。
2. **监控设置**：单独设置设备 B 的本地 UDP 监控端口（例：12345，仅示例，不等于 KMBox 命令端口），开启/关闭物理键鼠监控。
3. **物理按键实时状态**：左/右/中/侧键1/侧键2；组合 `左键+右键` 状态和触发次数。初始必须为只读监控。
4. **手动鼠标小幅移动**：四个按钮 `(+10,0) (-10,0) (0,+10) (0,-10)`，只有用户明确点击才向盒子发送命令；不自动点击或屏蔽物理鼠标。
5. **NDI 源发现和接收**：显示源、分辨率、帧率、丢帧；预览默认关闭、可选打开。
6. **ONNX 模型测试**：模型加载、CPU/CUDA 手动选择、实际使用的 Execution Provider、推理耗时、检测框数量。CUDA 未真实执行不得标 PASS。
7. **安全控制**：仅显示组合热键是否成立的 Dry Run 模式、紧急停止（UI + 设备 B 的 F8）、断流/断连立即禁止新命令。
8. **日志**：按时间戳记录设备状态、监控事件、命令结果、模型/NDI 性能；支持一键导出。

如果第一包无法一次全部完成，**必须先交付 `HardwareCheck-01`（KMBox 真实监控 + 小幅移动）**，再交付 `HardwareCheck-02`（NDI + ONNX），不能因等待完整 UI 推迟第一包。

## KMBox SDK 已核查接口

官方 C++ Header：https://github.com/kvmaibox/kmboxnet/blob/main/c%2B%2B_demo/NetConfig/kmboxNet.h
官方 Python 说明：https://www.kmbox.top/wiki_doc/kmboxNet/site/python/

```cpp
int kmNet_init(char* ip, char* port, char* mac);
int kmNet_monitor(short port);
int kmNet_monitor_mouse_left();
int kmNet_monitor_mouse_right();
int kmNet_monitor_mouse_middle();
int kmNet_monitor_mouse_side1();
int kmNet_monitor_mouse_side2();
int kmNet_mouse_move(short x, short y);
```

注意：这里的 `kmNet_monitor(port)` 接受 **本机接收监控数据的 UDP 端口**，传 `0` 关闭监控。不要把其他 SDK 的 `monitor(1)` 示例直接套用。监控端口在设备 B 上监听，必要时只为可信局域网放行 Windows 防火墙入站 UDP。

优先使用官方 SDK，通过薄 C ABI 包装供 Rust 调用；不要猜测或自行发明设备协议。为 SDK 的共享状态和网络调用设置单独受控工作线程。检测监控断线和状态新鲜度：**不能只读取缓存的 `left=1/right=1` 就认定盒子一直在线**；先确定设备是否周期上报，再设计独立探活/失联保护。失联时按不触发处理。

## Checkpoint A — KMBox 实体监控（第一个包）

1. 实际连接 KMBox（参数从盒子屏幕读取，不能硬编码参考截图里的地址）。
2. 开启物理键鼠监控，显示按键状态和更新时间。
3. 左键单独按住、右键单独按住、先左后右、先右后左，各测试至少 10 轮；确认只有两键都按住才触发组合状态。
4. 松开任意一键，组合状态必须退出；统计实际观测延迟，不得虚构数字。
5. 鼠标在设备 A 的普通桌面上，使用四个小幅移动按钮测试。确认正常物理鼠标输入不受影响。
6. 拔掉 KMBox 网线/断开连接，验证故障状态、禁止新自动命令和队列清空。
7. 关闭应用时干净停止监控，应用只释放自己注入的按键状态，不扰乱物理按键。

真实硬件日志通过后才写：`KMBOX_CONNECT_PASS`、`KMBOX_MONITOR_LR_PASS`、`KMBOX_COMBO_AND_PASS`、`KMBOX_MOVE_PASS`、`KMBOX_DISCONNECT_SAFE_PASS`。

## Checkpoint B — NDI 实机接收

使用官方 NDI SDK 完成：源发现、`NDIlib_recv_create_v3`、`NDIlib_recv_capture_v3`、帧及时释放、断流/丢帧统计。视频流进入容量 1–2 的“仅最新帧”队列，过期帧直接丢弃。预览 OFF 不停止推理；ON 时才在 UI 绘制。实际连接 A 的 NDI 源并连续运行至少 3 分钟。

验收：`NDI_DISCOVER_PASS`、`NDI_RECEIVE_PASS`、`NDI_PREVIEW_OFF_PASS`。

## Checkpoint C — YOLO26s 推理

加载用户提供的 `1013v26s_640.onnx`，在运行时检查真实输入输出名称和形状（先前检查预期为输入 `images [1,3,640,640]`，输出 `output0 [1,300,6]`）。实现 NDI 帧转 RGB、Letterbox、归一化、检测框坐标逆变换。先通过 CPU，再配置匹配的 ONNX Runtime + CUDA/cuDNN 运行环境，验证 RTX 3060 实际执行。CPU/CUDA 切换不允许隐性回退后还显示 GPU 成功。记录平均耗时和实际检测数量。

验收：`ONNX_CPU_PASS`、`ONNX_CUDA_PASS`（真实 GPU 执行后）、`NDI_TO_ONNX_PASS`。

## Checkpoint D — 组合触发和安全干运行

在设备 B 实现以下门控（AND）：

```text
用户手动武装
AND NDI 视频有效
AND 模型推理有效
AND KMBox 网络连接有效
AND 物理按键监控有效
AND 左键按下
AND 右键按下
```

先只显示目标检测框、候选位移和当前门控结果（Dry Run），**不发送自动鼠标移动**。只有完成鼠标相对位移校准并经用户明确启用后，才进入受限真实控制测试。松开任意键、断开 NDI/KMBox、模型失败或紧急停止时，立即禁止新命令并清空待处理命令。自动命令只取最新结果，限制每次位移与发送速率。

验收：`COMBO_DRY_RUN_PASS`、`SAFETY_STOP_PASS`。真实自动控制另行记录独立测试结果。

## 推荐最小目录

```text
ndi-ai-vision/
  apps/hardware-check/         # 第一个 Windows GUI EXE
  crates/kmbox-net/            # 官方 SDK 适配层
  crates/hotkeys/              # 物理按键状态 + AND 组合
  crates/ndi-receiver/         # NDI 接收 + 最新帧队列
  crates/vision/               # ONNX 推理和预处理
  crates/control/              # 状态机 / Dry Run / 安全停机
  .github/workflows/windows-build.yml
  README.md
```

## 最终交付和报告

- Windows x64 已实际构建的 ZIP，包含 EXE、运行库、配置说明与日志导出能力；两台机器无需 Rust、Node 或 Visual Studio。
- 附上 CI / 构建日志、ZIP 校验和、真实测试的 PASS/FAIL/SKIP 表（实际没测过标 SKIP，不能标 PASS）。
- 若没有访问用户两台实体机的权限，交付可运行包及 7 步操作说明，让用户在设备 B 运行并上传日志；不得假装已测试双机硬件。
- 研发方式：写代码 + README 简要汇报，不写冗长 Plan/ADR/Contract/Evidence；完成一个 Checkpoint 直接推进下一个。

## 用户需要怎么运行第一个测试包

1. KMBox USB 输出接 A、物理鼠标接 KMBox，KMBox 网口与设备 B 在同一局域网，A 开启 NDI 推流。
2. 在 B 双击 EXE，填入盒子屏幕真实 IP、命令端口和 MAC，并设置一个未占用的本机 UDP 监控端口（如 12345）。
3. 点击“连接 KMBox”→“开始监控”，分别按左键、右键、同时按住两键，查看真实状态。
4. 在 A 的普通桌面执行四个小幅位移测试，确认真实鼠标移动方向。
5. 选择 NDI 视频源；先保持预览关闭验证 FPS，再开启预览调试。
6. 加载模型，分别运行 CPU 和 CUDA 测试，记录实际使用引擎及延迟。
7. 测试“停止/F8”和断网，导出日志给 Agent 修复和推进下一阶段。
