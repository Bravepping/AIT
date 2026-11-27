using System;
using System.Collections.Concurrent;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
#if !KMBOX_CALIBRATION
using System.Runtime.InteropServices;
#endif
using System.Text;
using System.Windows.Forms;

namespace VisionHardwareCheck
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            if (args.Length == 2 && args[0] == "--self-test")
            {
                Environment.ExitCode = KmBoxSelfTest.Run(args[1]);
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            using (var form = new CheckForm())
            {
#if PHASE2
                if (args.Length == 2 && args[0] == "--verify-vision-start")
                    form.Shown += delegate { form.VerifyVisionStart(args[1]); };
#endif
                if (args.Length == 2 && args[0] == "--verify-start")
                {
                    form.Shown += delegate
                    {
                        form.BeginInvoke((MethodInvoker)delegate
                        {
#if KMBOX_CALIBRATION
                            if (!form.ManualMovementLocked)
                            {
                                File.WriteAllText(args[1], "MANUAL_MOVEMENT_SHOULD_BE_LOCKED_AT_STARTUP");
                                Environment.ExitCode = 1;
                                form.Close();
                                return;
                            }
#endif
                            File.WriteAllText(args[1], "WINDOWS_X64_GUI_STARTED");
                            form.Close();
                        });
                    };
                }
                Application.Run(form);
            }
        }
    }

    internal sealed class CheckForm : Form
    {
#if !KMBOX_CALIBRATION
        const int StopHotkey = 1;
        [DllImport("user32.dll", SetLastError = true)]
        static extern bool RegisterHotKey(IntPtr window, int id, uint modifiers, uint key);
        [DllImport("user32.dll")]
        static extern bool UnregisterHotKey(IntPtr window, int id);
        readonly Label f8 = new Label();
        bool hotkeyRegistered;
#endif

        readonly TextBox ip = new TextBox(), port = new TextBox(), mac = new TextBox();
        readonly NumericUpDown monitorPort = new NumericUpDown();
        readonly Label connection = new Label(), health = new Label(), stationary = new Label(), combo = new Label();
        readonly Label[] indicators = new Label[5];
        readonly Button[] moves = new Button[4];
        readonly Button connect = new Button(), disconnect = new Button(), monitor = new Button(), resume = new Button();
        readonly TextBox events = new TextBox();
        readonly Timer timer = new Timer();
        readonly ConcurrentQueue<string> pendingLogs = new ConcurrentQueue<string>();
        readonly KmBox box;
        bool disposed;
#if PHASE2
        VisionForm vision;
#endif
#if KMBOX_CALIBRATION
        readonly Button armMove = new Button();
        readonly Label moveStatus = new Label();
        long armedUntil;
        static long NowTicks { get { return System.Diagnostics.Stopwatch.GetTimestamp(); } }
        internal bool ManualMovementLocked
        {
            get { return !armMove.Enabled && Array.TrueForAll(moves, button => !button.Enabled); }
        }
#endif

        internal CheckForm()
        {
#if KMBOX_CALIBRATION
            Text = "VisionKmCalibration · V1.4 现场验收";
#elif PHASE2
            Text = "VisionHardwareCheck · HardwareCheck-02.1";
#else
            Text = "VisionHardwareCheck · HardwareCheck-01";
#endif
            Font = new Font("Microsoft YaHei UI", 10);
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(850, 665);
            MinimumSize = new Size(780, 620);
            BackColor = Color.FromArgb(243, 246, 250);
            ForeColor = Color.FromArgb(30, 42, 60);
            AutoScaleMode = AutoScaleMode.Dpi;
#if !KMBOX_CALIBRATION
            KeyPreview = true;
#endif
            box = new KmBox(Log);

            var root = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(18), ColumnCount = 1, RowCount = 6 };
            root.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
#if KMBOX_CALIBRATION
            foreach (int height in new[] { 58, 107, 133, 80, 49 }) root.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
#else
            foreach (int height in new[] { 58, 107, 133, 45, 49 }) root.RowStyles.Add(new RowStyle(SizeType.Absolute, height));
#endif
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            Controls.Add(root);
            var title = Caption("KMBox NET 硬件验收", 19);
            var header = Row();
            header.Controls.Add(title);
#if PHASE2
            AddButton(header, new Button(), "NDI / ONNX 联调", (s, e) => OpenVision());
#endif
            root.Controls.Add(header, 0, 0);

            var setup = Card(2);
            var parameters = Row();
            AddInput(parameters, "盒子 IPv4", ip, 136);
            AddInput(parameters, "命令端口", port, 86);
            AddInput(parameters, "MAC / UUID（8 位）", mac, 147);
            monitorPort.Minimum = 1024;
            monitorPort.Maximum = 49151;
            monitorPort.Value = 12345;
            AddInput(parameters, "本机监控 UDP", monitorPort, 125);
            setup.Controls.Add(parameters, 0, 0);
            var linking = Row();
            AddButton(linking, connect, "连接 KMBox", (s, e) => ConnectBox());
            AddButton(linking, disconnect, "断开", (s, e) =>
            {
#if KMBOX_CALIBRATION
                armedUntil = 0;
#endif
                box.Emergency(true);
            });
            AddButton(linking, monitor, "开始监控", (s, e) =>
            {
#if KMBOX_CALIBRATION
                armedUntil = 0;
#endif
                bool accepted = box.Current.Monitoring ? box.StopMonitor() : box.StartMonitor((int)monitorPort.Value);
                if (!accepted) Log("监控请求被拒绝：操作忙碌、断开或急停已锁止。");
            });
            linking.Controls.Add(Inline(connection, 242));
            setup.Controls.Add(linking, 0, 1);
            root.Controls.Add(setup, 0, 1);

            var status = Card(4);
            var keys = Row();
            string[] names = { "左键", "右键", "中键", "侧键 1", "侧键 2" };
            for (int i = 0; i < names.Length; i++)
            {
                var indicator = Caption(names[i], 10);
                indicator.Tag = names[i];
                indicator.Margin = new Padding(0, 2, 6, 2);
                indicator.Padding = new Padding(6, 1, 6, 1);
                indicator.BackColor = Color.FromArgb(230, 235, 241);
                indicators[i] = indicator;
                keys.Controls.Add(indicator);
            }
            status.Controls.Add(keys, 0, 0);
            status.Controls.Add(Inline(combo, 700), 0, 1);
            status.Controls.Add(Inline(health, 700), 0, 2);
            status.Controls.Add(Inline(stationary, 700), 0, 3);
            root.Controls.Add(status, 0, 2);

            var movement = Row();
#if KMBOX_CALIBRATION
            AddButton(movement, armMove, "设备 A 桌面就绪 · 启用单次", (s, e) =>
            {
                string blocked = box.Current.ManualArmBlockReason;
                if (blocked != null) { Log("启用单次位移被拒绝：" + blocked); return; }
                armedUntil = NowTicks + 10 * System.Diagnostics.Stopwatch.Frequency;
                Log("已手动启用单次校准。请保持设备 A 光标静止，在 10 秒内点击一个方向。");
                RefreshView();
            });
#else
            movement.Controls.Add(Caption("设备 A 桌面位移", 10));
#endif
            int[,] offsets = { { -10, 0 }, { 10, 0 }, { 0, -10 }, { 0, 10 } };
            string[] directions = { "← (−10,0)", "→ (+10,0)", "↑ (0,−10)", "↓ (0,+10)" };
            for (int i = 0; i < moves.Length; i++)
            {
                int x = offsets[i, 0], y = offsets[i, 1];
                moves[i] = new Button();
                AddButton(movement, moves[i], directions[i], (s, e) =>
                {
#if PHASE2
                    if (vision != null && !vision.IsDisposed && vision.Visible) return;
#endif
#if KMBOX_CALIBRATION
                    bool permitted = armedUntil > NowTicks;
                    armedUntil = 0;
                    if (!permitted) { Log("单次位移被拒绝：请先在验收窗口启用本次测试。"); return; }
                    bool accepted = box.MoveCalibrationStep(x, y);
                    string blocked = box.Current.ManualArmBlockReason;
#else
                    bool accepted = box.Move(x, y);
                    string blocked = box.Current.MoveBlockReason;
#endif
                    if (accepted) Log("已请求单次位移 (" + x + "," + y + ")，等待盒子应答。");
                    else Log("单次位移未排入：" + (blocked ?? "队列已被占用或控制状态刚刚改变"));
                    RefreshView();
                });
            }
#if KMBOX_CALIBRATION
            var movementArea = Card(2);
            movementArea.Controls.Add(movement, 0, 0);
            movementArea.Controls.Add(Inline(moveStatus, 690), 0, 1);
            root.Controls.Add(movementArea, 0, 3);
#else
            root.Controls.Add(movement, 0, 3);
#endif

            var actions = Row();
            var stop = new Button();
#if KMBOX_CALIBRATION
            AddButton(actions, stop, "紧急停止", (s, e) => StopAll());
#else
            AddButton(actions, stop, "紧急停止 · F8", (s, e) => StopAll());
#endif
            stop.ForeColor = Color.Firebrick;
            AddButton(actions, resume, "解除急停", (s, e) => { if (!box.Resume()) Log("解除急停被拒绝，请等待当前操作结束。"); });
            AddButton(actions, new Button(), "导出日志", (s, e) => Export());
#if !KMBOX_CALIBRATION
            actions.Controls.Add(Inline(f8, 360));
#endif
            root.Controls.Add(actions, 0, 4);

            events.Multiline = true;
            events.ReadOnly = true;
            events.ScrollBars = ScrollBars.Vertical;
            events.Dock = DockStyle.Fill;
            events.BackColor = Color.White;
            root.Controls.Add(events, 0, 5);

            timer.Interval = 35;
            timer.Tick += (s, e) => RefreshView();
            timer.Start();
            Log(Text + " 启动。物理按键只读；仅四个手动移动按钮发送位移。");
            RefreshView();
        }

#if !KMBOX_CALIBRATION
        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            hotkeyRegistered = RegisterHotKey(Handle, StopHotkey, 0x4000, 0x77);
            f8.Text = hotkeyRegistered ? "F8 已注册（设备 B 全局）" : "全局 F8 被占用；窗口内 F8 或按钮可停止";
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            if (hotkeyRegistered) UnregisterHotKey(Handle, StopHotkey);
            hotkeyRegistered = false;
            base.OnHandleDestroyed(e);
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == 0x0312 && m.WParam.ToInt32() == StopHotkey) StopAll();
            base.WndProc(ref m);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData != Keys.F8) return base.ProcessCmdKey(ref msg, keyData);
            StopAll();
            return true;
        }
#endif

        void StopAll()
        {
#if KMBOX_CALIBRATION
            armedUntil = 0;
#endif
            box.Emergency(false);
#if PHASE2
            if (vision != null && !vision.IsDisposed) vision.Disarm("紧急停止");
#endif
        }

#if PHASE2
        void OpenVision()
        {
            if (vision == null || vision.IsDisposed)
            {
                vision = new VisionForm(box, Log, StopAll);
                vision.Show(this);
            }
            else vision.Activate();
        }

        internal void VerifyVisionStart(string marker)
        {
            OpenVision();
            string sample = Path.ChangeExtension(marker, ".bmp");
            using (var image = new Bitmap(640, 360)) image.Save(sample, System.Drawing.Imaging.ImageFormat.Bmp);
            var verify = new Timer { Interval = 80 };
            var deadline = DateTime.UtcNow.AddSeconds(12);
            bool requested = false, sampleRequested = false;
            verify.Tick += delegate
            {
                if (vision.WorkerReady && !requested)
                {
                    requested = true;
                    vision.LoadModel();
                    vision.CheckCuda();
                }
                if (vision.WorkerModelReady && !sampleRequested)
                {
                    sampleRequested = true;
                    vision.TestSample(sample);
                }
                bool passed = vision.WorkerModelReady && vision.WorkerCudaChecked && vision.SampleResultVisible;
                if (!passed && DateTime.UtcNow < deadline) return;
                verify.Stop();
                verify.Dispose();
                if (passed) File.WriteAllText(marker, "WINDOWS_X64_VISION_SAMPLE_READY");
                else Environment.ExitCode = 1;
                vision.Close();
                File.Delete(sample);
                Close();
            };
            verify.Start();
        }
#endif

        protected override void Dispose(bool disposing)
        {
            if (disposing && !disposed)
            {
                disposed = true;
                timer.Stop();
                timer.Dispose();
                box.Dispose();
            }
            base.Dispose(disposing);
        }

        void ConnectBox()
        {
            IPEndPoint target;
            uint id;
            if (!KmBox.ParseAddress(ip.Text, port.Text, mac.Text, out target, out id))
            {
                Log("填写盒子屏幕上的 IPv4、命令端口（1–65535）以及 8 位十六进制 MAC / UUID。");
                return;
            }
            if (box.Connect(target, id)) Log("连接请求 " + target);
            else Log("连接请求被拒绝：操作队列忙碌或正在断开。");
        }

        void Log(string text)
        {
            pendingLogs.Enqueue(DateTimeOffset.Now.ToString("yyyy-MM-dd HH:mm:ss.fff zzz", CultureInfo.InvariantCulture) + "  " + text);
        }

        void FlushLogs()
        {
            var buffer = new StringBuilder();
            string line;
            while (pendingLogs.TryDequeue(out line)) buffer.AppendLine(line);
            if (buffer.Length != 0) events.AppendText(buffer.ToString());
        }

        void RefreshView()
        {
            KmBox.Snapshot state = box.Current;
            connection.Text = state.Connection + (state.Halted ? " · 已锁止" : "");
            combo.Text = "左键 + 右键  " + (state.Combo ? "成立" : "未成立") + "   累计触发 " + state.Count + " 次";
            combo.ForeColor = state.Combo ? Color.DarkGreen : ForeColor;
            for (int i = 0; i < indicators.Length; i++)
            {
                bool pressed = state.Buttons >= 0 && (state.Buttons & (1 << i)) != 0;
                var label = indicators[i];
                label.Text = label.Tag + (state.Buttons < 0 ? "  未知" : pressed ? "  按下" : "  松开");
                label.BackColor = pressed ? Color.FromArgb(183, 232, 204) : Color.FromArgb(230, 235, 241);
            }
            health.Text = state.Monitor + (state.Age < 0 ? "" : "  •  上次 " + state.LastReportAt.ToLocalTime().ToString("HH:mm:ss.fff") +
                "  •  " + state.Age + " ms 前  •  " + state.Reports + " 包  •  最大间隔 " + state.LongestGap + " ms");
            stationary.Text = "最近静止持键段  " + state.StationaryReports + " 包 / " + state.StationaryDuration +
                " ms  •  最大间隔 " + state.StationaryMaxGap + " ms  •  忽略 " + state.IgnoredReports + " 包";
            connect.Enabled = !state.Connected && !state.Busy;
            disconnect.Enabled = state.Connected || state.Busy;
            ip.Enabled = port.Enabled = mac.Enabled = connect.Enabled;
            monitor.Enabled = state.Connected && !state.Busy && !state.Halted;
            monitor.Text = state.Monitoring ? "停止监控" : "开始监控";
            monitorPort.Enabled = !state.Monitoring && !state.Busy;
            resume.Enabled = state.Halted && state.Connected && !state.Busy;
#if KMBOX_CALIBRATION
            if (!state.Connected || !state.Monitoring || state.Halted || armedUntil <= NowTicks) armedUntil = 0;
            armMove.Enabled = state.ManualArmBlockReason == null && armedUntil == 0;
            moveStatus.Text = state.ManualArmBlockReason ?? (armedUntil == 0 ? "先确认设备 A 普通桌面，再启用单次 ±10 位移。" :
                "已启用单次校准。保持设备 A 光标静止，手动点击一个方向。");
            bool movementAllowed = state.ManualArmBlockReason == null && armedUntil != 0;
#else
            bool movementAllowed = state.CanMove;
#endif
#if PHASE2
            movementAllowed &= vision == null || vision.IsDisposed || !vision.Visible;
#endif
            foreach (var button in moves) button.Enabled = movementAllowed;
            FlushLogs();
        }

        void Export()
        {
            using (var save = new SaveFileDialog { Filter = "文本日志 (*.txt)|*.txt", FileName = "KMBox-验收-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + ".txt" })
            {
                if (save.ShowDialog(this) != DialogResult.OK) return;
                FlushLogs();
                try { File.WriteAllText(save.FileName, events.Text, new UTF8Encoding(true)); }
                catch (Exception ex) { MessageBox.Show(this, ex.Message, "日志导出失败"); }
            }
        }

        static Label Caption(string text, float size)
        {
            return new Label { Text = text, AutoSize = true, Font = new Font("Microsoft YaHei UI", size, FontStyle.Bold),
                Margin = new Padding(0, 8, 10, 0), TextAlign = ContentAlignment.MiddleLeft };
        }

        static Label Inline(Label label, int width)
        {
            label.Width = width;
            label.Height = 29;
            label.TextAlign = ContentAlignment.MiddleLeft;
            label.Margin = new Padding(0);
            return label;
        }

        static TableLayoutPanel Card(int rows)
        {
            var card = new TableLayoutPanel { Dock = DockStyle.Fill, BackColor = Color.White,
                Padding = new Padding(10), ColumnCount = 1, RowCount = rows, Margin = new Padding(0, 0, 0, 8) };
            card.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            for (int i = 0; i < rows; i++) card.RowStyles.Add(new RowStyle(SizeType.Percent, 100f / rows));
            return card;
        }

        static FlowLayoutPanel Row()
        {
            return new FlowLayoutPanel { Dock = DockStyle.Fill, WrapContents = false, Margin = new Padding(0) };
        }

        static void AddInput(FlowLayoutPanel panel, string title, Control input, int width)
        {
            panel.Controls.Add(Caption(title, 9));
            input.Width = width;
            input.Margin = new Padding(0, 4, 11, 0);
            panel.Controls.Add(input);
        }

        static void AddButton(FlowLayoutPanel row, Button button, string title, EventHandler click)
        {
            button.Text = title;
            button.AutoSize = true;
            button.Height = 31;
            button.Padding = new Padding(4, 0, 4, 0);
            button.Margin = new Padding(0, 0, 7, 2);
            button.Click += click;
            row.Controls.Add(button);
        }
    }
}
