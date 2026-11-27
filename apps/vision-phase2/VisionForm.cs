using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using Timer = System.Windows.Forms.Timer;

namespace VisionHardwareCheck
{
    internal sealed class VisionForm : Form
    {
        readonly KmBox box;
        readonly Action<string> log;
        readonly Action stop;
        readonly ComboBox sources = new ComboBox(), provider = new ComboBox(), previewRate = new ComboBox();
        readonly Button discover = new Button(), connect = new Button(), model = new Button(), disconnect = new Button(), cudaCheck = new Button();
        readonly Button sample = new Button();
        readonly CheckBox arm = new CheckBox();
        readonly Label streamStatus = new Label(), modelStatus = new Label(), gateStatus = new Label(), offsetStatus = new Label();
        readonly Label previewStatus = new Label(), cudaStatus = new Label();
        readonly Label sampleStatus = new Label();
        readonly Label latencyStatus = new Label();
        readonly PictureBox preview = new PictureBox();
        readonly Timer timer = new Timer { Interval = 40 };
        readonly JavaScriptSerializer reader = new JavaScriptSerializer(), writer = new JavaScriptSerializer();
        readonly ConcurrentQueue<Dictionary<string, object>> messages = new ConcurrentQueue<Dictionary<string, object>>();
        readonly Stopwatch elapsed = Stopwatch.StartNew();
        Dictionary<string, object> latestPreview;
        Process process;
        Form sampleWindow;
        bool ready, modelReady, armed, gateWasOpen, cudaChecking, samplePending;
        int sourceId, modelId, previewId, previewFps, sampleId;
        double streamAt, inferredAt, streamSince;
        long nextLog;
        string modelPath, actualProvider;
        string candidate = "—";
        internal bool WorkerReady { get { return ready; } }
        internal bool WorkerModelReady { get { return modelReady; } }
        internal bool WorkerCudaChecked { get; private set; }
        internal bool SampleResultVisible { get { return sampleWindow != null && sampleWindow.Visible; } }

        sealed class NdiSource
        {
            internal string Name, Url;
            public override string ToString() { return Name; }
        }

        internal VisionForm(KmBox box, Action<string> log, Action stop)
        {
            this.box = box;
            this.log = log;
            this.stop = stop;
            Text = "VisionHardwareCheck · NDI / ONNX 联调";
            StartPosition = FormStartPosition.CenterParent;
            Font = new Font("Microsoft YaHei UI", 10);
            MinimumSize = new Size(794, 560);
            ClientSize = new Size(900, 650);
            var root = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(14), RowCount = 12, ColumnCount = 1 };
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            foreach (int height in new[] { 40, 32, 41, 33, 27, 27, 27, 41, 42, 31, 31 }) root.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            Controls.Add(root);

            modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "1013v26s_640.onnx");
            Configure(sources, 355);
            Configure(provider, 155, "Auto", "CUDA", "TensorRT", "TensorRT for RTX", "CPU");
            provider.SelectedItem = "CPU";
            Configure(previewRate, 90, "关闭", "5", "10", "15", "30");
            var ndiHome = new LinkLabel { Text = "ndi.video", AutoSize = true, Margin = new Padding(5, 7, 0, 0) };
            ndiHome.LinkClicked += delegate { Process.Start(new ProcessStartInfo("https://ndi.video/") { UseShellExecute = true }); };
            SetButton(discover, "发现 NDI 源", delegate { if (Send("discover")) discover.Enabled = false; });
            SetButton(connect, "接收", delegate
            {
                var source = sources.SelectedItem as NdiSource;
                if (source != null) ChangeStream(source);
            });
            SetButton(disconnect, "断开 NDI", delegate { ChangeStream(null); });
            root.Controls.Add(Row(discover, sources, connect, disconnect, ndiHome), 0, 0);
            streamStatus.Text = "启动接收组件…";
            root.Controls.Add(Fill(streamStatus), 0, 1);

            var pick = new Button();
            SetButton(pick, "选 ONNX 文件", delegate
            {
                using (var dialog = new OpenFileDialog { Filter = "ONNX (*.onnx)|*.onnx", CheckFileExists = true })
                    if (dialog.ShowDialog(this) == DialogResult.OK)
                    {
                        modelPath = dialog.FileName;
                        LoadModel();
                    }
            });
            SetButton(model, "加载 / 测试模型", LoadModel);
            SetButton(cudaCheck, "CUDA 环境自检", CheckCuda);
            provider.SelectedIndexChanged += delegate { if (ready && modelId != 0) LoadModel(); };
            previewRate.SelectedIndexChanged += delegate
            {
                int fps;
                int.TryParse((string)previewRate.SelectedItem, out fps);
                ClearPicture();
                Send("preview", "fps", fps, "preview", ++previewId);
            };
            var previewLabel = new Label { Text = "调试预览 FPS", AutoSize = true, Margin = new Padding(16, 7, 0, 0) };
            root.Controls.Add(Row(provider, model, pick, cudaCheck, previewLabel, previewRate), 0, 2);
            modelStatus.Text = "模型未加载";
            root.Controls.Add(Fill(modelStatus), 0, 3);
            root.Controls.Add(Fill(latencyStatus), 0, 4);
            cudaStatus.Text = "CUDA 环境尚未检查";
            root.Controls.Add(Fill(cudaStatus), 0, 5);
            previewStatus.Text = "预览关闭";
            root.Controls.Add(Fill(previewStatus), 0, 6);
            SetButton(sample, "测试本地图片", PickSample);
            sampleStatus.AutoSize = true;
            sampleStatus.Margin = new Padding(5, 8, 0, 0);
            sampleStatus.Text = "加载模型后可测试 PNG、JPEG 或 BMP";
            root.Controls.Add(Row(sample, sampleStatus), 0, 7);

            arm.Text = "启用只读 Dry Run";
            arm.AutoSize = true;
            arm.Margin = new Padding(0, 8, 16, 0);
            arm.CheckedChanged += delegate
            {
                armed = false;
                if (!arm.Checked) return;
                string reason = MissingPrerequisite(box.Current);
                if (reason != null) { Disarm(reason); return; }
                armed = true;
                log("用户启用只读 Dry Run");
            };
            var emergency = new Button();
            SetButton(emergency, "紧急停止 · F8", stop);
            emergency.ForeColor = Color.Firebrick;
            root.Controls.Add(Row(arm, emergency), 0, 8);
            root.Controls.Add(Fill(gateStatus), 0, 9);
            root.Controls.Add(Fill(offsetStatus), 0, 10);
            preview.Dock = DockStyle.Fill;
            preview.BackColor = Color.FromArgb(32, 39, 52);
            preview.SizeMode = PictureBoxSizeMode.Zoom;
            root.Controls.Add(preview, 0, 11);
            discover.Enabled = connect.Enabled = model.Enabled = arm.Enabled = cudaCheck.Enabled = previewRate.Enabled = sample.Enabled = false;
            Shown += delegate { StartWorker(); timer.Start(); };
            timer.Tick += delegate { RefreshStatus(); };
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData != Keys.F8) return base.ProcessCmdKey(ref msg, keyData);
            stop();
            return true;
        }

        void StartWorker()
        {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string runtime = Path.Combine(root, "vision-runtime");
            var info = new ProcessStartInfo(Path.Combine(runtime, "python.exe"), "-B -X utf8 -u \"" + Path.Combine(runtime, "worker.py") + "\"")
            {
                WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            };
            try
            {
                process = Process.Start(info);
                process.OutputDataReceived += (s, e) =>
                {
                    if (e.Data == null) return;
                    try
                    {
                        var message = reader.Deserialize<Dictionary<string, object>>(e.Data);
                        if (Value(message, "kind") == "preview") Interlocked.Exchange(ref latestPreview, message);
                        else messages.Enqueue(message);
                    }
                    catch (ArgumentException ex) { log("联调输出读取错误：" + ex.Message); }
                };
                process.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrWhiteSpace(e.Data)) log("联调组件：" + e.Data.Replace("\0", ""));
                };
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }
            catch (Exception ex)
            {
                streamStatus.Text = "联调组件启动失败：" + ex.Message;
                log(streamStatus.Text);
            }
        }

        void ChangeStream(NdiSource source)
        {
            Disarm("更换或断开 NDI 源");
            streamAt = inferredAt = streamSince = 0;
            latencyStatus.Text = "";
            candidate = "—";
            ClearPicture();
            streamStatus.Text = source == null ? "NDI 已断开" : "等待 NDI 视频 " + source.Name;
            log(source == null ? "用户断开 NDI" : "请求接收 NDI · " + source.Name);
            Send("stream", "source", ++sourceId, "name", source == null ? "" : source.Name, "url", source == null ? "" : source.Url);
        }

        internal void CheckCuda()
        {
            if (!ready || cudaChecking || !Send("cuda_check")) return;
            cudaChecking = true;
            WorkerCudaChecked = false;
            cudaStatus.Text = "正在检查 CUDA 环境…";
        }

        internal void LoadModel()
        {
            if (!ready) return;
            Disarm("重新加载模型");
            modelReady = false;
            latencyStatus.Text = "";
            samplePending = false;
            sampleStatus.Text = "加载模型后可测试 PNG、JPEG 或 BMP";
            inferredAt = 0;
            candidate = "—";
            ClearPicture();
            modelStatus.Text = "加载 " + provider.SelectedItem + " · " + Path.GetFileName(modelPath);
            Send("model", "model", ++modelId, "path", modelPath, "provider", provider.SelectedItem);
        }

        void PickSample()
        {
            if (!modelReady || samplePending) return;
            using (var dialog = new OpenFileDialog
            {
                Filter = "图片 (*.png;*.jpg;*.jpeg;*.bmp)|*.png;*.jpg;*.jpeg;*.bmp",
                CheckFileExists = true
            })
            {
                if (dialog.ShowDialog(this) == DialogResult.OK) TestSample(dialog.FileName);
            }
        }

        internal void TestSample(string path)
        {
            if (!modelReady || samplePending) return;
            samplePending = Send("sample", "sample", ++sampleId, "model", modelId, "path", path);
            sampleStatus.Text = samplePending ? "正在检测 " + Path.GetFileName(path) + "…" : "无法向联调组件提交图片";
        }

        bool Send(string command, params object[] items)
        {
            if (process == null || process.HasExited) return false;
            var message = new Dictionary<string, object> { { "cmd", command } };
            for (int i = 0; i < items.Length; i += 2) message[(string)items[i]] = items[i + 1];
            try
            {
                var json = new StringBuilder();
                foreach (char character in writer.Serialize(message))
                {
                    if (character > 127) json.Append("\\u").Append(((int)character).ToString("x4"));
                    else json.Append(character);
                }
                process.StandardInput.WriteLine(json.ToString());
                process.StandardInput.Flush();
                return true;
            }
            catch (IOException ex)
            {
                log("联调组件通信中断：" + ex.Message);
                return false;
            }
        }

        void RefreshStatus()
        {
            if (process != null && process.HasExited)
            {
                ready = modelReady = false;
                streamAt = inferredAt = streamSince = 0;
                latencyStatus.Text = "";
                cudaChecking = false;
                if (samplePending) sampleStatus.Text = "联调组件已退出，图片测试中止";
                samplePending = false;
                streamStatus.Text = "联调组件已退出 · " + process.ExitCode;
                log(streamStatus.Text);
                Disarm(streamStatus.Text);
                process.Dispose();
                process = null;
                discover.Enabled = false;
            }
            Dictionary<string, object> message;
            while (process != null && messages.TryDequeue(out message)) ReadMessage(message);
            var image = Interlocked.Exchange(ref latestPreview, null);
            if (image != null && streamSince > 0 && previewFps > 0 && Number(image, "source") == sourceId &&
                Number(image, "preview") == previewId && Number(image, "frame_at") >= streamSince)
                ShowPicture(Value(image, "image"));
            if (!Fresh(streamAt, 1800) && !Fresh(inferredAt, 700)) ClearPicture();

            KmBox.Snapshot km = box.Current;
            string reason = MissingPrerequisite(km);
            if (armed && reason != null) Disarm(reason);
            arm.Enabled = ready && reason == null;
            bool open = DryRunGate.IsOpen(armed, reason, km);
            gateStatus.Text = "门控 " + (open ? "成立 · 只读" : armed ? "等待物理左键 + 右键" : "锁止 · " + (reason ?? "尚未手动启用"));
            gateStatus.ForeColor = open ? Color.DarkGreen : Color.Firebrick;
            offsetStatus.Text = "候选位移（画面像素） " + (open ? candidate : "—");
            if (open != gateWasOpen) log(open ? "Dry Run 门控成立，仅显示候选" : "Dry Run 门控已退出");
            gateWasOpen = open;
            connect.Enabled = ready && sources.SelectedItem != null;
            disconnect.Enabled = model.Enabled = ready;
            cudaCheck.Enabled = ready && !cudaChecking;
            previewRate.Enabled = ready;
            sample.Enabled = modelReady && !samplePending;
        }

        void ReadMessage(Dictionary<string, object> message)
        {
            string error = Value(message, "error");
            switch (Value(message, "kind"))
            {
                case "ready":
                    ready = true;
                    streamStatus.Text = "请发现并选择 NDI 源";
                    discover.Enabled = true;
                    previewStatus.Text = PreviewText(Value(message, "preview_fps"), Value(message, "preview_frames"));
                    log("联调启动 · " + previewStatus.Text);
                    break;
                case "sources":
                    ShowSources(message);
                    break;
                case "stream":
                    if (Number(message, "source") == sourceId) ShowStream(message);
                    break;
                case "stream_state":
                    if (Number(message, "source") == sourceId) ShowStreamState(message);
                    break;
                case "preview_state":
                    if (Number(message, "preview") != previewId) break;
                    previewFps = (int)Number(message, "fps");
                    previewStatus.Text = PreviewText(Value(message, "fps"), Value(message, "frames"));
                    log(previewStatus.Text);
                    if (error != "")
                    {
                        log("预览错误：" + error);
                        previewRate.SelectedIndex = 0;
                    }
                    break;
                case "cuda_check":
                    ShowCudaCheck(message);
                    break;
                case "model":
                    if (Number(message, "model") != modelId) break;
                    modelReady = error == "";
                    actualProvider = Value(message, "provider");
                    if (!modelReady) { inferredAt = 0; latencyStatus.Text = ""; Disarm("模型失败"); }
                    string requested = Value(message, "requested");
                    modelStatus.Text = modelReady ? (requested == "Auto" ? "Auto → " : "") + actualProvider +
                        " 自检 " + Value(message, "elapsed") + " ms  •  检测框 " + Value(message, "count") +
                        "  •  " + Value(message, "metadata") : requested + " 模型失败 · " + error;
                    log(modelStatus.Text + (modelReady ? " · " + Path.GetFileName(modelPath) : ""));
                    if (modelReady) LogBackend(message);
                    break;
                case "inference":
                    if (streamSince > 0 && Number(message, "frame_at") >= streamSince &&
                        Number(message, "model") == modelId && Number(message, "source") == sourceId && modelReady)
                        ShowInference(message);
                    break;
                case "sample":
                    if (samplePending && Number(message, "sample") == sampleId && Number(message, "model") == modelId)
                        ShowSample(message, error);
                    break;
                case "log":
                    log(Value(message, "text"));
                    break;
            }
        }

        string MissingPrerequisite(KmBox.Snapshot km)
        {
            bool inferenceFresh = Fresh(inferredAt, 700);
            bool streamFresh = streamSince > 0 && (Fresh(streamAt, 1800) || inferenceFresh);
            return DryRunGate.BlockReason(km, streamFresh, modelReady && inferenceFresh);
        }

        static bool Fresh(double timestamp, int limit)
        {
            double age = (DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalMilliseconds - timestamp;
            return age >= 0 && age <= limit;
        }

        internal void Disarm(string reason)
        {
            if (armed || arm.Checked) log("Dry Run 已解除启用：" + reason);
            armed = false;
            arm.Checked = false;
            offsetStatus.Text = "候选位移（画面像素） —";
        }

        void ShowSources(Dictionary<string, object> message)
        {
            var previous = sources.SelectedItem as NdiSource;
            sources.Items.Clear();
            object items;
            if (message.TryGetValue("sources", out items))
                foreach (IList pair in (IEnumerable)items)
                {
                    int index = sources.Items.Add(new NdiSource { Name = (string)pair[0], Url = (string)pair[1] });
                    if (previous != null && previous.Name == (string)pair[0]) sources.SelectedIndex = index;
                }
            if (sources.SelectedIndex < 0 && sources.Items.Count > 0) sources.SelectedIndex = 0;
            string error = Value(message, "error");
            if (error != "") streamStatus.Text = error;
            else if (streamAt == 0) streamStatus.Text = "发现 " + sources.Items.Count + " 个 NDI 源";
            log("NDI 发现 " + sources.Items.Count + " 个源 " + error);
            discover.Enabled = true;
        }

        void ShowStream(Dictionary<string, object> message)
        {
            streamAt = Number(message, "frame_at");
            string error = Value(message, "error");
            if (error != "")
            {
                streamSince = inferredAt = 0;
                latencyStatus.Text = "";
                ClearPicture();
                Disarm("NDI 接收错误");
                streamStatus.Text = error;
                log("NDI 错误：" + error);
                return;
            }
            streamStatus.Text = (Fresh(streamAt, 1800) ? "NDI 接收" : "等待视频 / 断流") + "  " +
                Value(message, "width") + "×" + Value(message, "height") + "  •  " + Value(message, "fps") +
                " FPS（源 " + Value(message, "nominal") + "）  •  SDK 丢帧 " + Value(message, "dropped") +
                "  •  最新帧替换 " + Value(message, "replaced");
            previewStatus.Text = PreviewText(Value(message, "preview_fps"), Value(message, "preview_frames"));
            if (elapsed.ElapsedMilliseconds < nextLog) return;
            nextLog = elapsed.ElapsedMilliseconds + 5000;
            log(streamStatus.Text + "  •  " + previewStatus.Text);
            if (modelReady && Fresh(inferredAt, 700))
            {
                log(modelStatus.Text);
                if (latencyStatus.Text.Length > 0) log(latencyStatus.Text);
            }
        }

        void ShowStreamState(Dictionary<string, object> message)
        {
            string state = Value(message, "state");
            if (state == "lost")
            {
                streamSince = streamAt = inferredAt = 0;
                latencyStatus.Text = "";
                candidate = "—";
                ClearPicture();
                Disarm("NDI 断流");
                streamStatus.Text = "NDI 断流 · " + Value(message, "gap_ms") + " ms 未收到新视频帧";
                log(streamStatus.Text);
            }
            else if (state == "live" || state == "resumed")
            {
                streamSince = streamAt = Number(message, "frame_at");
                nextLog = 0;
                log(state == "resumed" ? "NDI 恢复收到视频帧" : "NDI 收到首帧");
            }
        }

        void ShowCudaCheck(Dictionary<string, object> message)
        {
            cudaChecking = false;
            WorkerCudaChecked = true;
            cudaStatus.Text = "CUDA 环境自检 · " + Value(message, "status");
            log(cudaStatus.Text);
            object items;
            if (message.TryGetValue("details", out items) && items is IList)
                foreach (var item in (IList)items) log("CUDA · " + item);
        }

        void LogBackend(Dictionary<string, object> message)
        {
            object raw;
            if (!message.TryGetValue("backend", out raw)) return;
            var backend = raw as Dictionary<string, object>;
            if (backend == null) return;
            string cache = Value(backend, "cache");
            string summary = "实际引擎 " + actualProvider + " " + Value(backend, "version") + " · " + Value(backend, "precision");
            if (cache != "") summary += " · Engine Cache " + cache;
            string jit = Value(backend, "jit_cache");
            if (jit != "") summary += " · RTX Runtime Cache " + jit;
            string layers = Value(backend, "fp16_layers");
            if (layers != "") summary += " · FP16 层 " + layers + "/" + Value(backend, "engine_layers");
            modelStatus.Text += " · " + (cache == "" ? Value(backend, "precision") : "Engine Cache " + cache);
            log(summary);
            if (backend.TryGetValue("previous_attempts", out raw) && raw is IList)
                foreach (object item in (IList)raw)
                {
                    var failure = item as Dictionary<string, object>;
                    if (failure != null) log("Auto 尝试 " + Value(failure, "provider") + " 失败：" + Value(failure, "error"));
                }
        }

        static string PreviewText(string fps, string frames)
        {
            return (fps == "0" ? "预览关闭" : "预览上限 " + fps + " FPS") + " · 预览累计输出 " + frames + " 帧";
        }

        void ShowInference(Dictionary<string, object> message)
        {
            inferredAt = Number(message, "frame_at");
            modelStatus.Text = actualProvider + " · " + Value(message, "elapsed") + " ms  •  均值 " +
                Value(message, "average") + " ms  •  检测框 " + Value(message, "count") + "  •  已推理 " + Value(message, "runs") + " 帧";
            latencyStatus.Text = Number(message, "warm_samples") < 2 ? "正在预热推理统计…" :
                "预热后均值 / P95 · 模型 " + Value(message, "model_avg") + " / " + Value(message, "model_p95") +
                " ms · NDI 接收到坐标 " + Value(message, "receiver_avg") + " / " + Value(message, "receiver_p95") +
                " ms · " + Value(message, "inference_fps") + " FPS";
            object values;
            var offset = message.TryGetValue("offset", out values) ? values as IList : null;
            candidate = offset != null && offset.Count == 2 ? "(" + offset[0] + ", " + offset[1] + ")" : "无候选";
        }

        void ShowSample(Dictionary<string, object> message, string error)
        {
            samplePending = false;
            if (error != "")
            {
                sampleStatus.Text = "图片检测失败，详情见运行日志";
                log("本地图片检测失败：" + error);
                return;
            }

            Bitmap image;
            try { image = DecodePicture(Value(message, "image")); }
            catch (ArgumentException ex)
            {
                sampleStatus.Text = "图片结果读取失败，详情见运行日志";
                log("本地图片结果解码失败：" + ex.Message);
                return;
            }

            string filename = Value(message, "filename");
            string description = Value(message, "provider") + " · " + Value(message, "width") + "×" + Value(message, "height") +
                " · " + Value(message, "elapsed") + " ms · 检测框 " + Value(message, "count") +
                " · 阈值 " + Value(message, "threshold");
            string score = Value(message, "max_score");
            if (score != "") description += " · 最高置信度 " + score;
            sampleStatus.Text = "图片检测完成 · " + Value(message, "count") + " 个检测框";
            log("本地图片检测 · " + filename + " · " + description);

            if (sampleWindow != null) sampleWindow.Close();
            sampleWindow = new Form
            {
                Text = "本地图片检测 · " + filename, StartPosition = FormStartPosition.CenterParent,
                ClientSize = new Size(760, 560), MinimumSize = new Size(480, 340), Font = Font
            };
            var picture = new PictureBox
            {
                Dock = DockStyle.Fill, Image = image, SizeMode = PictureBoxSizeMode.Zoom, BackColor = preview.BackColor
            };
            sampleWindow.Controls.Add(picture);
            sampleWindow.Controls.Add(new Label { Text = description, Dock = DockStyle.Top, Height = 30, AutoEllipsis = true });
            sampleWindow.FormClosed += delegate { image.Dispose(); sampleWindow = null; };
            sampleWindow.Show(this);
        }

        static Bitmap DecodePicture(string encoded)
        {
            byte[] data;
            try { data = Convert.FromBase64String(encoded); }
            catch (FormatException ex) { throw new ArgumentException("结果图片编码无效", ex); }
            using (var bytes = new MemoryStream(data))
            using (var frame = Image.FromStream(bytes))
                return new Bitmap(frame);
        }

        void ShowPicture(string encoded)
        {
            try
            {
                var image = DecodePicture(encoded);
                ClearPicture();
                preview.Image = image;
            }
            catch (ArgumentException ex) { log("预览解码失败：" + ex.Message); }
        }

        void ClearPicture()
        {
            var image = preview.Image;
            if (image == null) return;
            preview.Image = null;
            image.Dispose();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            Disarm("联调窗口已关闭");
            timer.Stop();
            timer.Dispose();
            ClearPicture();
            if (process != null)
            {
                Send("exit");
                process.StandardInput.Close();
                if (!process.WaitForExit(3500)) { process.Kill(); process.WaitForExit(); }
                process.Dispose();
                process = null;
            }
            base.OnFormClosed(e);
        }

        static string Value(Dictionary<string, object> message, string key)
        {
            object value;
            return message.TryGetValue(key, out value) ? Convert.ToString(value, CultureInfo.InvariantCulture) ?? "" : "";
        }

        static double Number(Dictionary<string, object> message, string key)
        {
            double number;
            return double.TryParse(Value(message, key), NumberStyles.Float, CultureInfo.InvariantCulture, out number) ? number : 0;
        }

        static void Configure(ComboBox combo, int width, params object[] items)
        {
            combo.Width = width;
            combo.DropDownStyle = ComboBoxStyle.DropDownList;
            combo.Items.AddRange(items);
            if (items.Length != 0) combo.SelectedIndex = 0;
        }

        static Control Fill(Label label) { label.Dock = DockStyle.Fill; label.AutoEllipsis = true; return label; }

        static void SetButton(Button button, string name, Action action)
        {
            button.Text = name;
            button.AutoSize = true;
            button.Height = 31;
            button.Click += delegate { action(); };
        }

        static Control Row(params Control[] controls)
        {
            var row = new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false };
            row.Controls.AddRange(controls);
            return row;
        }
    }
}
