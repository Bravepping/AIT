using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

internal sealed class ButtonStatusApp : Form
{
    [DllImport("user32.dll")]
    static extern short GetAsyncKeyState(int key);

    static readonly int[] MouseKeys = { 1, 2, 4, 5, 6 };
    readonly CancellationTokenSource stop = new CancellationTokenSource();
    readonly Label status = new Label { Dock = DockStyle.Top, AutoSize = true };
    UdpClient listener;
    Thread network;
    volatile bool received;

    [STAThread]
    static void Main(string[] args)
    {
        Application.EnableVisualStyles();
        using (var window = new ButtonStatusApp())
        {
            if (args.Length == 2 && args[0] == "--verify-start") window.Shown += delegate
            {
                File.WriteAllText(args[1], "DEVICE_A_STATUS_WINDOW_READY");
                window.Close();
            };
            Application.Run(window);
        }
    }

    ButtonStatusApp()
    {
        Text = "NDI Vision Control · 设备 A 按键确认";
        Font = new Font("Microsoft YaHei UI", 10);
        ClientSize = new Size(620, 222);
        StartPosition = FormStartPosition.CenterScreen;
        var body = new FlowLayoutPanel { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, Padding = new Padding(16) };
        Controls.Add(body);
        body.Controls.Add(new Label { AutoSize = true, Text = "在设备 A 的交互桌面运行。选择设备 A 的局域网地址，然后开始确认。" });
        var address = new ComboBox { Width = 260, DropDownStyle = ComboBoxStyle.DropDownList };
        address.Items.AddRange(Dns.GetHostAddresses(Dns.GetHostName()).Where(ip => ip.AddressFamily == AddressFamily.InterNetwork &&
            !IPAddress.IsLoopback(ip)).Select(ip => (object)ip).ToArray());
        if (address.Items.Count != 0) address.SelectedIndex = 0;
        body.Controls.Add(address);
        var code = new TextBox { ReadOnly = true, Width = 570 };
        var begin = new Button { Text = "开始按键确认", AutoSize = true };
        body.Controls.Add(begin);
        body.Controls.Add(new Label { AutoSize = true, Text = "把下面这一整行配对信息填到设备 B 的「设备 A 按键确认」。" });
        body.Controls.Add(code);
        body.Controls.Add(status);
        var timer = new System.Windows.Forms.Timer { Interval = 150 };
        timer.Tick += delegate { if (received) status.Text = "设备 B 已收到按键读数。保持窗口开启，关闭窗口立即停止确认。"; };
        begin.Click += delegate
        {
            var ip = address.SelectedItem as IPAddress;
            if (ip == null) { status.Text = "请选择连接设备 B 所在局域网的 IPv4 地址。"; return; }
            try { listener = new UdpClient(new IPEndPoint(ip, ButtonConfirmation.Port)); }
            catch (SocketException error) { status.Text = "无法开启按键确认端口，" + error.Message; return; }
            byte[] key = ButtonConfirmation.NewKey();
            code.Text = ip + "/" + BitConverter.ToString(key).Replace("-", "");
            begin.Enabled = address.Enabled = false;
            status.Text = "已开始，等待设备 B 配对。";
            network = new Thread(() =>
            {
                try { ButtonConfirmation.Respond(listener, key, ReadButtons, () => received = true, stop.Token); }
                catch (SocketException) { }
            }) { IsBackground = true };
            network.Start();
            timer.Start();
        };
        FormClosed += delegate
        {
            timer.Dispose();
            stop.Cancel();
            if (network != null) network.Join();
            if (listener != null) listener.Dispose();
            stop.Dispose();
        };
    }

    static byte ReadButtons()
    {
        byte mask = 0;
        for (int i = 0; i < MouseKeys.Length; i++)
            if ((GetAsyncKeyState(MouseKeys[i]) & 0x8000) != 0) mask |= (byte)(1 << i);
        return mask;
    }
}
