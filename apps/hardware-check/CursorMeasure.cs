using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace VisionHardwareCheck
{
    internal sealed class CursorMeasure : Form
    {
        const int MarkId = 1, MeasureId = 2;
        const int HotkeyMessage = 0x0312;

        [StructLayout(LayoutKind.Sequential)]
        struct PixelPoint { internal int X, Y; }

        [DllImport("user32.dll", SetLastError = true)]
        static extern bool GetPhysicalCursorPos(out PixelPoint point);
        [DllImport("user32.dll", SetLastError = true)]
        static extern bool RegisterHotKey(IntPtr window, int id, uint modifiers, uint key);
        [DllImport("user32.dll")]
        static extern bool UnregisterHotKey(IntPtr window, int id);

        readonly Label status = new Label();
        readonly TextBox results = new TextBox();
        readonly Button copy = new Button();
        PixelPoint? start;
        bool markRegistered, measureRegistered;

        [STAThread]
        static void Main(string[] args)
        {
            if (args.Length == 2 && args[0] == "--self-test")
            {
                var a = new PixelPoint { X = -400, Y = 160 };
                var b = new PixelPoint { X = -384, Y = 157 };
                bool passed = Difference(a, b) == "ΔX +16 px，ΔY -3 px" && Difference(b, a) == "ΔX -16 px，ΔY +3 px"
                    && Difference(a, a) == "ΔX 0 px，ΔY 0 px";
                File.WriteAllText(args[1], passed ? "CURSOR_DELTA_PASS" : "CURSOR_DELTA_FAIL");
                Environment.ExitCode = passed ? 0 : 1;
                return;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            using (var form = new CursorMeasure())
            {
                if (args.Length == 2 && args[0] == "--verify-start") form.Shown += delegate
                {
                    form.BeginInvoke((MethodInvoker)delegate
                    {
                        PixelPoint position;
                        bool readable = GetPhysicalCursorPos(out position);
                        File.WriteAllText(args[1], readable ? "CURSOR_WINDOW_AND_PHYSICAL_POSITION_READY" : "CURSOR_POSITION_UNAVAILABLE");
                        Environment.ExitCode = readable ? 0 : 1;
                        form.Close();
                    });
                };
                Application.Run(form);
            }
        }

        CursorMeasure()
        {
            Text = "设备 A 光标像素测量";
            Font = new Font("Microsoft YaHei UI", 10);
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(610, 315);
            MinimumSize = new Size(490, 300);
            AutoScaleMode = AutoScaleMode.Dpi;
            KeyPreview = true;

            var body = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 4, Padding = new Padding(16) };
            body.RowStyles.Add(new RowStyle(SizeType.Absolute, 62));
            body.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            body.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            body.RowStyles.Add(new RowStyle(SizeType.Absolute, 40));
            Controls.Add(body);
            body.Controls.Add(new Label { Dock = DockStyle.Fill, Text = "在设备 A 留出一块桌面空白区域。按 F9 记起点，在设备 B 手动做一次位移，再按 F10 读取像素差。测量期间请保持设备 A 的鼠标位置。" }, 0, 0);
            status.Dock = DockStyle.Fill;
            status.Text = "按 F9 记录起点。";
            body.Controls.Add(status, 0, 1);
            results.Dock = DockStyle.Fill;
            results.ReadOnly = true;
            results.Multiline = true;
            results.ScrollBars = ScrollBars.Vertical;
            body.Controls.Add(results, 0, 2);
            copy.Text = "复制测量结果";
            copy.AutoSize = true;
            copy.Enabled = false;
            copy.Click += delegate
            {
                try { Clipboard.SetText(results.Text); status.Text = "测量结果已复制。"; }
                catch (ExternalException) { status.Text = "剪贴板当前不可用，可选中测量结果后按 Ctrl+C。"; }
            };
            body.Controls.Add(copy, 0, 3);
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            markRegistered = RegisterHotKey(Handle, MarkId, 0x4000, (uint)Keys.F9);
            measureRegistered = RegisterHotKey(Handle, MeasureId, 0x4000, (uint)Keys.F10);
            if (!markRegistered || !measureRegistered) status.Text = "有全局快捷键被占用，请保持本窗口在前台后使用 F9 和 F10。";
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            if (markRegistered) UnregisterHotKey(Handle, MarkId);
            if (measureRegistered) UnregisterHotKey(Handle, MeasureId);
            markRegistered = measureRegistered = false;
            base.OnHandleDestroyed(e);
        }

        protected override void WndProc(ref Message m)
        {
            if (m.Msg == HotkeyMessage)
            {
                int id = m.WParam.ToInt32();
                if (id == MarkId || id == MeasureId) { CapturePosition(id == MarkId); return; }
            }
            base.WndProc(ref m);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == Keys.F9 && !markRegistered) { CapturePosition(true); return true; }
            if (keyData == Keys.F10 && !measureRegistered) { CapturePosition(false); return true; }
            return base.ProcessCmdKey(ref msg, keyData);
        }

        void CapturePosition(bool mark)
        {
            if (!mark && !start.HasValue) { status.Text = "请先按 F9 记录起点。"; return; }
            PixelPoint pixel;
            if (!GetPhysicalCursorPos(out pixel)) { status.Text = "无法读取光标坐标，请确认设备 A 当前停在正常桌面。"; return; }
            if (mark)
            {
                start = pixel;
                status.Text = "起点已记录 " + Position(pixel) + "。位移完成后按 F10。";
                return;
            }
            PixelPoint before = start.Value;
            start = null;
            string delta = Difference(before, pixel);
            results.AppendText(DateTimeOffset.Now.ToString("HH:mm:ss.fff zzz", CultureInfo.InvariantCulture) + "  " +
                Position(before) + " → " + Position(pixel) + "  " + delta + Environment.NewLine);
            copy.Enabled = true;
            status.Text = "设备 A 桌面像素差 · " + delta;
        }

        static string Position(PixelPoint p)
        {
            return string.Format(CultureInfo.InvariantCulture, "({0}, {1})", p.X, p.Y);
        }

        static string Difference(PixelPoint before, PixelPoint after)
        {
            return string.Format(CultureInfo.InvariantCulture, "ΔX {0:+0;-0;0} px，ΔY {1:+0;-0;0} px",
                (long)after.X - before.X, (long)after.Y - before.Y);
        }
    }
}
