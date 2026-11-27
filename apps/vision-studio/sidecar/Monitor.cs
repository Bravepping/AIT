using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using VisionHardwareCheck;

internal static class MonitorHost
{
    static readonly object output = new object();
    static readonly JavaScriptSerializer json = new JavaScriptSerializer();
    static readonly Stopwatch clock = Stopwatch.StartNew();
    static long lastSent = -100;
    static KmBox.Snapshot previous;
    static CancellationTokenSource confirmation;
    static Thread confirmationThread;

    static void Emit(object message)
    {
        lock (output) Console.WriteLine(json.Serialize(message));
    }

    static void Publish(KmBox.Snapshot current)
    {
        long now = clock.ElapsedMilliseconds;
        if (previous != null && now - lastSent < 100 && current.Connected == previous.Connected &&
            current.Monitoring == previous.Monitoring && current.Buttons == previous.Buttons &&
            current.Halted == previous.Halted && current.Busy == previous.Busy && current.Connection == previous.Connection &&
            current.Monitor == previous.Monitor) return;
        previous = current;
        lastSent = now;
        Emit(new { kind = "kmbox", connected = current.Connected, monitoring = current.Monitoring,
            buttons = current.Buttons, reports = current.Reports, age_ms = current.Age, busy = current.Busy,
            report_at_ms = current.Age >= 0 ? KmBox.EpochMs(current.LastReportAt) : 0,
            link_at_ms = current.Connected ? KmBox.EpochMs(current.LinkAt) : 0,
            longest_gap_ms = current.LongestGap, ignored_reports = current.IgnoredReports,
            stationary_reports = current.StationaryReports, stationary_ms = current.StationaryDuration,
            stationary_max_gap_ms = current.StationaryMaxGap,
            connection = current.Connection, monitor = current.Monitor, halted = current.Halted });
    }

    static string Value(Dictionary<string, object> message, string key)
    {
        object value;
        return message.TryGetValue(key, out value) ? Convert.ToString(value) : "";
    }

    static void Main()
    {
        Console.InputEncoding = Console.OutputEncoding = new UTF8Encoding(false);
        using (var device = new KmBox(text => Emit(new { kind = "log", text = text }), Publish))
        {
            string line;
            while ((line = Console.ReadLine()) != null)
            {
                try
                {
                    Dictionary<string, object> message;
                    lock (output) message = json.Deserialize<Dictionary<string, object>>(line);
                    switch (Value(message, "cmd"))
                    {
                        case "exit": StopConfirmation(); return;
                        case "confirm_input":
                            string[] pairing = Value(message, "pairing").Trim().Split('/');
                            IPAddress inputAddress;
                            if (pairing.Length != 2 || !IPAddress.TryParse(pairing[0], out inputAddress) ||
                                inputAddress.AddressFamily != AddressFamily.InterNetwork || inputAddress.Equals(IPAddress.Any) ||
                                inputAddress.GetAddressBytes()[0] >= 224)
                                throw new ArgumentException("配对信息格式无效，请复制设备 A 窗口中的整行");
                            byte[] key = ButtonConfirmation.ParseKey(pairing[1]);
                            StopConfirmation();
                            confirmation = new CancellationTokenSource();
                            var token = confirmation.Token;
                            confirmationThread = new Thread(() =>
                            {
                                bool first = true;
                                try { ButtonConfirmation.Poll(inputAddress, key, (mask, at) =>
                                {
                                    if (first) { Emit(new { kind = "log", text = "设备 A 按键确认已配对" }); first = false; }
                                    Emit(new { kind = "kmbox_confirmation", mask = mask, at_ms = at });
                                }, token); }
                                catch (SocketException) { Emit(new { kind = "kmbox_error", error = "设备 A 确认连接不可用" }); }
                            }) { IsBackground = true };
                            confirmationThread.Start();
                            Emit(new { kind = "log", text = "已请求设备 A 按键确认，等待新鲜读数" });
                            break;
                        case "stop_confirmation": StopConfirmation(); break;
                        case "connect":
                            IPEndPoint endpoint;
                            uint uuid;
                            if (!KmBox.ParseAddress(Value(message, "ip"), Value(message, "port"),
                                Value(message, "uuid"), out endpoint, out uuid))
                                throw new ArgumentException("请检查盒子 IPv4、命令端口和 8 位十六进制 UUID");
#if KMBOX_INTEGRATION_LOOPBACK
                            if (!IPAddress.IsLoopback(endpoint.Address))
                                throw new ArgumentException("集成测试仅允许本机模拟端口");
#endif
                            if (!device.Connect(endpoint, uuid)) throw new InvalidOperationException("KMBox 正忙");
                            break;
                        case "monitor":
                            int port;
                            if (!int.TryParse(Value(message, "port"), out port) || port < 1024 || port > 49151)
                                throw new ArgumentException("监控端口须在 1024–49151");
                            if (!device.StartMonitor(port)) throw new InvalidOperationException("KMBox 正忙");
                            break;
                        case "stop_monitor":
                            if (!device.StopMonitor()) throw new InvalidOperationException("KMBox 正忙");
                            break;
                        case "disconnect":
                            device.Emergency(true);
                            break;
#if KMBOX_AUTO_TRACKING
                        case "auto_enable":
                            Emit(device.EnableAuto() ? new { kind = "auto_available" } : new { kind = "auto_unavailable" });
                            break;
                        case "auto_stop":
                            device.StopAuto();
                            break;
#endif
#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
                        case "relative_move":
                            int dx, dy, holdMask, report;
                            long id, expires, reportAt;
                            if (!int.TryParse(Value(message, "dx"), out dx) || !int.TryParse(Value(message, "dy"), out dy)
                                || !long.TryParse(Value(message, "id"), out id) || !long.TryParse(Value(message, "expires_at_ms"), out expires)
                                || !int.TryParse(Value(message, "hold_mask"), out holdMask) || !int.TryParse(Value(message, "report"), out report)
                                || !long.TryParse(Value(message, "report_at_ms"), out reportAt))
                                throw new ArgumentException("无效的相对位移");
                            long submitted = clock.ElapsedMilliseconds;
                            Action<string> outcome = status => Emit(new { kind = "move_status", id = id, status = status,
                                dx = dx, dy = dy, elapsed_ms = clock.ElapsedMilliseconds - submitted });
                            if (!device.MoveHeld(dx, dy, id, expires, holdMask, report, reportAt, outcome)) outcome("rejected");
                            break;
#endif
                        default:
                            throw new ArgumentException("只读监控进程不支持此命令");
                    }
                }
                catch (Exception error)
                {
                    Emit(new { kind = "kmbox_error", error = error.Message });
                }
            }
        }
        StopConfirmation();
    }

    static void StopConfirmation()
    {
        if (confirmation == null) return;
        confirmation.Cancel();
        confirmationThread.Join();
        confirmation.Dispose();
        confirmation = null;
        confirmationThread = null;
        Emit(new { kind = "kmbox_confirmation", at_ms = 0 });
    }
}
