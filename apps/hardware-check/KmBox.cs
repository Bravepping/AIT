using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;

namespace VisionHardwareCheck
{
    // Wire formats from kvmaibox/kmboxnet, c++_demo/NetConfig/kmboxNet.{h,cpp}.
    // Read the actual UDP reports so timestamps and keys belong to the same packet.
    internal sealed class KmBox : IDisposable
    {
        internal const int ReportTimeoutMs = 1200;
        internal const int HeartbeatMs = 750;
#if !KMBOX_READONLY
        internal const int ManualAxisStep = 10;
        const int ManualQueueTimeoutMs = 150;
        const int MovePreflightTimeoutMs = 120;
        const int MoveAckTimeoutMs = 300;
#endif
        internal const int LeftRight = 3;
        const uint ConnectCommand = 0xaf3c2828;
        const uint MonitorCommand = 0x27388020;
#if !KMBOX_READONLY
        const uint MoveCommand = 0xaede7345;
#endif
        const uint MonitorEnable = 0xaa550000;
        const int HeaderSize = 16;
        const int ReportSize = 20;
        static readonly DateTime UnixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        [DllImport("kernel32.dll", ExactSpelling = true)]
        static extern void GetSystemTimePreciseAsFileTime(out long fileTime);

        static DateTime PreciseUtcNow()
        {
            long fileTime;
            GetSystemTimePreciseAsFileTime(out fileTime);
            return DateTime.FromFileTimeUtc(fileTime);
        }

        readonly object commands = new object();
        readonly AutoResetEvent wake = new AutoResetEvent(false);
        readonly Stopwatch clock = Stopwatch.StartNew();
        readonly Thread worker;
        readonly Action<string> log;
        readonly Action<Snapshot> observe;
        readonly Random random = new Random();
        readonly byte[] report = new byte[2048];
        readonly byte[] reply = new byte[2048];
        Socket command, monitor;
        IPEndPoint endpoint;
        uint uuid, sequence;
        int monitorPort;
        int monitorCleanupPort;
        int buttons = -1, count;
        long lastReport = -1, heartbeatDue;
        DateTime lastReportAt;
        DateTime linkAt;
        int reportCount, longestGap;
        int heartbeatAcksSinceReport;
        int heartbeatAcks;
        StreamWriter wireTrace;
        string wireTracePath;
        int wireTraceRecords;
        long wireTraceDeadline;
        int ignoredReports, stationaryReports, stationaryMaxGap, stationaryDuration;
        long stationarySince = -1;
        bool combo, comboReady;
        Action pending;
        bool busy;
        int stopVersion;
#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
        readonly int[] releasedAtReport = new int[5];
        long monitorEpochMs, lastMoveId;
#endif
#if KMBOX_AUTO_TRACKING
        volatile bool autoAllowed;
#endif
        volatile bool closing, halted, closeRequested;
        string connection = "未连接", monitorStatus = "已关闭";
        volatile Snapshot snapshot = new Snapshot();

        internal KmBox(Action<string> log, Action<Snapshot> observe = null)
        {
            this.log = log;
            this.observe = observe;
            worker = new Thread(Loop) { IsBackground = true, Name = "KMBox UDP" };
            worker.Start();
        }

        internal sealed class Snapshot
        {
            internal bool Connected, Monitoring, Halted, Busy, Combo;
            internal int Buttons = -1, Count, Reports, LongestGap, HeartbeatAcks;
            internal bool TraceActive;
            internal long Age = -1;
            internal int IgnoredReports, StationaryReports, StationaryMaxGap, StationaryDuration;
            internal DateTime LastReportAt, LinkAt;
            internal string Connection = "未连接", Monitor = "已关闭";
            internal string ManualArmBlockReason
            {
                get
                {
                    if (Halted) return "急停或通信故障已锁止";
                    if (!Connected) return "盒子尚未握手";
                    if (!Monitoring) return "本机物理按键监控未开启";
                    return Busy ? "驱动正在等待上一条命令" : null;
                }
            }
            internal string MoveBlockReason
            {
                get
                {
                    string blocked = ManualArmBlockReason;
                    if (blocked != null) return blocked;
                    if (Age < 0) return "未收到物理报文，请在设备 A 轻动鼠标或按键";
                    if (Buttons < 0 || Age > ReportTimeoutMs) return "物理报文已过期，请在设备 A 刷新按键状态";
                    return null;
                }
            }
            internal bool CanMove { get { return MoveBlockReason == null; } }
        }

        internal Snapshot Current { get { return snapshot; } }

        internal static bool ParseAddress(string address, string port, string mac, out IPEndPoint target, out uint id)
        {
            target = null;
            id = 0;
            IPAddress ip;
            ushort number;
            string identifier = mac.Trim().Replace(":", "").Replace("-", "");
            if (!IPAddress.TryParse(address.Trim(), out ip) || ip.AddressFamily != AddressFamily.InterNetwork ||
                ip.Equals(IPAddress.Any) || ip.Equals(IPAddress.Broadcast) || ip.GetAddressBytes()[0] >= 224 ||
                !ushort.TryParse(port.Trim(), out number) || number == 0 ||
                identifier.Length != 8 || !uint.TryParse(identifier, NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out id))
                return false;
            target = new IPEndPoint(ip, number);
            return true;
        }

        internal bool Connect(IPEndPoint target, uint id)
        {
            return Enqueue(delegate
            {
                CloseLink();
                endpoint = target;
                uuid = id;
                sequence = 0;
                connection = "连接中…";
                Publish();
                command = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                command.Connect(target);
                Exchange("连接握手", ConnectCommand, RandomValue(), null, 1500, false);
                connection = "盒子已应答";
                heartbeatDue = clock.ElapsedMilliseconds + HeartbeatMs;
                log("KMBox 握手应答 " + target);
            }, true);
        }

        internal bool StartMonitor(int port)
        {
            return Enqueue(delegate
            {
                if (command == null || monitor != null) return;
                if (port < 1024 || port > 49151) throw new ArgumentOutOfRangeException("port");
                Socket receiver = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
                try
                {
                    receiver.ExclusiveAddressUse = true;
                    IPAddress localIp = ((IPEndPoint)command.LocalEndPoint).Address;
                    receiver.Bind(new IPEndPoint(localIp, port));
                }
                catch (SocketException ex)
                {
                    receiver.Dispose();
                    log("监控端口绑定失败 · " + ((IPEndPoint)command.LocalEndPoint).Address + ":" + port +
                        " · " + ex.SocketErrorCode + " / " + ex.NativeErrorCode + " · " + ex.Message);
                    monitorStatus = "端口绑定失败";
                    return;
                }
                monitor = receiver;
                monitorPort = monitorCleanupPort = port;
                monitorStatus = "等待盒子监控应答";
                ResetReports();
                Exchange("开启监控", MonitorCommand, MonitorEnable | (uint)port, null, 750, false);
                heartbeatDue = clock.ElapsedMilliseconds + HeartbeatMs;
                monitorStatus = "等待物理报文";
                log("监控开启，本机监听 " + receiver.LocalEndPoint + "；请试按或移动物理鼠标");
            }, false);
        }

        internal bool StopMonitor()
        {
            return Enqueue(delegate
            {
                CloseMonitor();
                if (command != null) Exchange("停止监控", MonitorCommand, 0, null, 750, false);
                monitorCleanupPort = 0;
                log("物理监控已关闭");
            }, false);
        }

        internal bool StartWireTrace()
        {
            if (!Current.Monitoring) return false;
            return Enqueue(delegate
            {
                if (monitor == null) { log("KMBox 报文记录未开始 · 监控已关闭"); return; }
                if (wireTrace != null) return;
                try
                {
                    string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                        "local.ndi.vision-control", "logs");
                    Directory.CreateDirectory(folder);
                    wireTracePath = Path.Combine(folder, "kmbox-wire-" + Guid.NewGuid().ToString("N") + ".txt");
                    wireTrace = new StreamWriter(wireTracePath, false, new System.Text.UTF8Encoding(false), 65536);
                    wireTraceRecords = 0;
                    wireTraceDeadline = clock.ElapsedMilliseconds + 30000;
                    Trace(PreciseUtcNow(), "begin threshold_ms=" + ReportTimeoutMs + " previous_report=" + reportCount);
                    if (wireTrace != null) log("KMBox 报文记录已开始（最多 30 秒或 50000 条）");
                }
                catch (Exception ex) { log("KMBox 报文记录无法启动 · " + ex.Message); }
            }, false);
        }

        void Trace(DateTime at, string fields)
        {
            if (wireTrace == null) return;
            if (clock.ElapsedMilliseconds >= wireTraceDeadline || wireTraceRecords >= 50000)
            {
                FinishWireTrace("达到记录上限");
                return;
            }
            try
            {
                wireTrace.WriteLine(at.ToString("O", CultureInfo.InvariantCulture) + " " + fields);
                wireTraceRecords++;
            }
            catch (Exception ex) { FinishWireTrace("写入失败 · " + ex.Message); }
        }

        void FinishWireTrace(string reason)
        {
            if (wireTrace == null) return;
            StreamWriter finished = wireTrace;
            wireTrace = null;
            try
            {
                using (finished)
                    finished.WriteLine(PreciseUtcNow().ToString("O", CultureInfo.InvariantCulture) + " end " + reason);
                log("KMBox 报文记录已保存 · " + wireTracePath + " · " + reason);
            }
            catch (Exception ex) { log("KMBox 报文记录保存失败 · " + ex.Message); }
        }

        internal void Emergency(bool disconnect)
        {
            lock (commands)
            {
                halted = true;
                stopVersion++;
                pending = null;
                if (disconnect) closeRequested = true;
            }
            log(disconnect ? "请求断开，已撤销待处理命令" : "紧急停止，已撤销待处理命令");
            wake.Set();
        }

#if !KMBOX_READONLY
        internal bool Resume()
        {
            lock (commands)
            {
                if (closing || closeRequested || busy || pending != null) return false;
                int version = stopVersion;
                // Reset the latch on the worker before allowing any new movement.
                pending = delegate
                {
                    lock (commands)
                    {
                        if (stopVersion != version || closing) return;
                        comboReady = false;
                        halted = false;
                    }
                    log("已解除急停，须重新按下组合键");
                };
            }
            wake.Set();
            return true;
        }

        internal bool Move(int x, int y)
        {
            if (Math.Abs((long)x) + Math.Abs((long)y) != ManualAxisStep || (x != 0 && y != 0)) return false;
            return MoveRelative(x, y, ManualAxisStep);
        }

        internal bool MoveRelative(int x, int y, int limit)
        {
            return QueueRelative(x, y, limit, clock.ElapsedMilliseconds + ManualQueueTimeoutMs, null, null, 0);
        }

#if KMBOX_CALIBRATION
        internal bool MoveCalibrationStep(int x, int y)
        {
            if (Math.Abs((long)x) + Math.Abs((long)y) != ManualAxisStep || (x != 0 && y != 0)) return false;
            // Explicit single click in the standalone calibration tool. The live preflight is still required.
            return QueueRelative(x, y, ManualAxisStep, clock.ElapsedMilliseconds + ManualQueueTimeoutMs,
                null, null, 0, requireFreshReport: false);
        }
#endif

#if KMBOX_AUTO_TRACKING
        internal bool EnableAuto()
        {
            lock (commands)
            {
                var current = Current;
                if (closing || closeRequested || busy || pending != null || !current.CanMove || current.Buttons != 0)
                    return false;
                autoAllowed = true;
                return true;
            }
        }

        internal void StopAuto()
        {
            lock (commands)
            {
                autoAllowed = false;
                stopVersion++;
            }
            wake.Set();
        }
#endif

#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
        internal bool MoveHeld(int x, int y, long id, long expiresAtMs, int mask, int number, long reportAtMs, Action<string> outcome)
        {
            long utc = EpochMs(PreciseUtcNow());
            if (id <= 0 || mask < 1 || mask > 31 || number <= 0 || x < -24 || x > 24 || y < -24 || y > 24 ||
                expiresAtMs <= utc || expiresAtMs - utc > 45 || reportAtMs > utc || reportAtMs <= 0 ||
                utc - reportAtMs > ReportTimeoutMs) return false;
            return QueueRelative(x, y, 24, clock.ElapsedMilliseconds + (expiresAtMs - utc), () =>
            {
#if KMBOX_AUTO_TRACKING
                if (!autoAllowed || EpochMs(PreciseUtcNow()) - EpochMs(linkAt) > ReportTimeoutMs) return false;
#endif
                if (EpochMs(PreciseUtcNow()) >= expiresAtMs || reportAtMs <= monitorEpochMs || number > reportCount ||
                    reportAtMs > EpochMs(lastReportAt) || (number == reportCount && reportAtMs != EpochMs(lastReportAt)) ||
                    (buttons & mask) != mask || monitor.Poll(0, SelectMode.SelectRead)) return false;
                for (int bit = 0; bit < releasedAtReport.Length; ++bit)
                    if ((mask & (1 << bit)) != 0 && releasedAtReport[bit] > number) return false;
                return true;
            }, outcome, id);
        }
#endif

        bool QueueRelative(int x, int y, int limit, long expiresAt, Func<bool> inputPermits, Action<string> outcome, long id,
            bool requireFreshReport = true)
        {
            if (limit < 1 || limit > 24 || (x == 0 && y == 0)) return false;
            int dx = Math.Max(-limit, Math.Min(limit, x));
            int dy = Math.Max(-limit, Math.Min(limit, y));
            lock (commands)
            {
#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
                if (id != 0 && id <= lastMoveId) return false;
                if (id != 0) lastMoveId = id;
#endif
                var current = Current;
                string unavailable = requireFreshReport ? current.MoveBlockReason : current.ManualArmBlockReason;
                if (unavailable != null || closing || closeRequested || halted || busy || pending != null) return false;
                int version = stopVersion;
                Func<string> blocked = () => {
                    if (version != stopVersion || halted || closing || closeRequested) return "已急停或断开";
                    if (command == null || monitor == null) return "命令连接或物理监控已关闭";
                    if (clock.ElapsedMilliseconds >= expiresAt) return "位移期限已过";
                    if (requireFreshReport && !IsFresh()) return "物理报文已过期";
                    return inputPermits != null && !inputPermits() ? "按住授权已失效" : null;
                };
                Func<bool> canSend = () => blocked() == null;
                pending = delegate
                {
                    try
                    {
                        DrainReports();
                        string reason = blocked();
                        if (reason != null)
                        {
                            if (outcome != null) outcome("cancelled");
                            else log("发包前取消位移：" + reason);
                            return;
                        }
                        bool requirePreflight = true;
#if KMBOX_AUTO_TRACKING
                        requirePreflight = id == 0;
#endif
                        if (requirePreflight)
                        {
                            Exchange("位移预检查", MonitorCommand, MonitorEnable | (uint)monitorPort, null, MovePreflightTimeoutMs, false);
                            heartbeatDue = clock.ElapsedMilliseconds + HeartbeatMs;
                        }
                        DrainReports();
                        byte[] mouse = new byte[56];
                        Write32(mouse, 4, unchecked((uint)dx));
                        Write32(mouse, 8, unchecked((uint)dy));
                        bool confirmed = Exchange("位移应答", MoveCommand, RandomValue(), mouse,
                            id == 0 ? MoveAckTimeoutMs : 100, true, canSend);
                        if (outcome != null) outcome(confirmed ? "acknowledged" : "cancelled");
                        else if (confirmed) log("盒子已应答位移 (" + dx + "," + dy + ")；请在设备 A 观察");
                        else log("发包前取消位移：" + (blocked() ?? "发送许可已改变"));
                    }
                    catch
                    {
                        if (outcome != null) outcome("fault");
                        throw;
                    }
                };
            }
            wake.Set();
            return true;
        }
#endif

        bool Enqueue(Action action, bool resetStop)
        {
            lock (commands)
            {
                if (closing || closeRequested || busy || pending != null || (halted && !resetStop)) return false;
                if (resetStop) halted = false;
                pending = action;
            }
            wake.Set();
            return true;
        }

        void Loop()
        {
            try
            {
                while (!closing)
                {
                    Action next;
                    bool disconnect;
                    lock (commands)
                    {
                        disconnect = closeRequested;
                        closeRequested = false;
                        next = pending;
                        pending = null;
                        busy = disconnect || next != null;
                    }
                    if (busy) Publish();
                    try
                    {
                        if (disconnect) CloseLink();
                        if (next != null && !closing) next();
                        DrainReports();
                        UpdateCombo();
                        if (command != null && clock.ElapsedMilliseconds >= heartbeatDue)
                        {
                            uint option = monitorPort == 0 ? 0 : MonitorEnable | (uint)monitorPort;
                            try { Exchange("监控心跳", MonitorCommand, option, null, 350, false); }
                            catch (TimeoutException ex)
                            {
                                log(ex.Message + "，重试一次心跳。");
                                Exchange("监控心跳重试", MonitorCommand, option, null, 350, false);
                            }
                            heartbeatDue = clock.ElapsedMilliseconds + HeartbeatMs;
                        }
                        if (wireTrace != null && clock.ElapsedMilliseconds >= wireTraceDeadline) FinishWireTrace("达到记录上限");
                    }
                    catch (Exception ex)
                    {
                        if (wireTrace != null) Trace(PreciseUtcNow(), "lockout reason=transport_failure detail=" + ex.Message);
                        log("KMBox 通信故障：" + ex.Message);
                        halted = true;
                        CloseLink();
                        connection = "通信失败，移动已锁止";
                    }
                    lock (commands) busy = false;
                    Publish();
                    wake.WaitOne(8);
                }
            }
            finally
            {
                CloseLink();
                Publish();
            }
        }

        bool IsFresh() { return lastReport >= 0 && clock.ElapsedMilliseconds - lastReport <= ReportTimeoutMs; }

        void DrainReports()
        {
            if (monitor == null) return;
            EndPoint sender = new IPEndPoint(IPAddress.Any, 0);
            for (int i = 0; i < 128 && monitor.Poll(0, SelectMode.SelectRead); ++i)
            {
                int length = monitor.ReceiveFrom(report, ref sender);
                if (length != ReportSize || !((IPEndPoint)sender).Address.Equals(endpoint.Address))
                {
                    ignoredReports++;
                    if (wireTrace != null) Trace(PreciseUtcNow(), "ignored_monitor_datagram size=" + length);
                    continue;
                }
                long now = clock.ElapsedMilliseconds;
                DateTime receivedAt = PreciseUtcNow();
                bool previousExpired = lastReport >= 0 && now - lastReport > ReportTimeoutMs;
                if (previousExpired)
                {
                    UpdateCombo();
                    // Publish the stale edge even when recovery carries an unchanged mask.
                    Publish();
                }
                if (lastReport >= 0) longestGap = (int)Math.Min(int.MaxValue, Math.Max(longestGap, now - lastReport));
                int keys = report[1] & 31;
                bool stationary = keys != 0 && BitConverter.ToInt16(report, 2) == 0 &&
                    BitConverter.ToInt16(report, 4) == 0 && BitConverter.ToInt16(report, 6) == 0;
                if (stationary)
                {
                    if (stationarySince < 0 || keys != buttons || !IsFresh())
                    {
                        stationarySince = now;
                        stationaryReports = stationaryMaxGap = stationaryDuration = 0;
                    }
                    if (stationaryReports != 0) stationaryMaxGap = Math.Max(stationaryMaxGap, (int)(now - lastReport));
                    stationaryReports++;
                    stationaryDuration = (int)Math.Min(int.MaxValue, now - stationarySince);
                }
                else stationarySince = -1;
                if (buttons != keys || !IsFresh())
                    log("物理按键报文 L/R/M/S1/S2=" + string.Join("/", new[] { keys & 1, (keys >> 1) & 1, (keys >> 2) & 1, (keys >> 3) & 1, (keys >> 4) & 1 }) + "，包 #" + (reportCount + 1));
#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
                int released = buttons < 0 ? 0 : buttons & ~keys;
                for (int bit = 0; bit < releasedAtReport.Length; ++bit)
                    if ((released & (1 << bit)) != 0) releasedAtReport[bit] = reportCount + 1;
#endif
                buttons = keys;
                lastReport = now;
                lastReportAt = receivedAt;
                reportCount++;
                heartbeatAcksSinceReport = 0;
                monitorStatus = "已收到物理报文";
                if (wireTrace != null) Trace(receivedAt, "physical local_sequence=" + reportCount + " size=" + length +
                    " mask=" + keys + " expired=false previous_expired=" + previousExpired +
                    " mouse_bytes=" + BitConverter.ToString(report, 0, 8));
                UpdateCombo();
                if (observe != null) Publish();
            }
        }

        void UpdateCombo()
        {
            bool fresh = IsFresh();
#if KMBOX_AUTO_TRACKING
            if (!fresh && autoAllowed)
            {
                StopAuto();
                log("物理按键报文已过期，真实跟随许可已撤销，请重新武装");
            }
#endif
            bool eligible = fresh && !halted && !closing;
            bool both = (buttons & LeftRight) == LeftRight;
            if (!eligible) comboReady = false;
            else if (!both) comboReady = true;
            bool active = eligible && comboReady && both;
            if (active != combo)
            {
                combo = active;
                if (active) log("组合键成立 #" + (++count) + "，只读");
                else log(eligible ? "组合键退出：已观测到至少一键松开" : "组合键退出：监控过期或已停止");
            }
            if (monitor != null && lastReport >= 0 && !fresh && monitorStatus != "报文超时")
            {
                monitorStatus = "报文超时";
#if KMBOX_CALIBRATION
                const string state = "按键状态未知，组合键退出。手动校准仍需逐次启用。";
#elif KMBOX_READONLY
                const string state = "KMBox 原始按键状态未知，盒子侧组合退出。";
#else
                const string state = "按键状态未知，移动锁止。";
#endif
                if (wireTrace != null) Trace(PreciseUtcNow(), "physical_expired local_sequence=" + reportCount +
                    " last_mask=" + buttons + " age_ms=" + (clock.ElapsedMilliseconds - lastReport) + " expired=true reason=no_fresh_physical_report");
                log("物理报文 " + ReportTimeoutMs + " ms 内未更新；" + state + "最近静止持键段 " +
                    stationaryReports + " 包 / " + stationaryDuration + " ms，最大间隔 " + stationaryMaxGap + " ms" +
                    "。上次物理报文 #" + reportCount + "，按键掩码 " + buttons +
                    "；此后监控心跳应答 " + heartbeatAcksSinceReport + " 次");
            }
        }

        uint RandomValue() { return (uint)random.Next(0, 32768); }

        bool Exchange(string stage, uint opcode, uint parameter, byte[] payload, int timeout, bool movement, Func<bool> canSend = null)
        {
            uint packetId = opcode == ConnectCommand ? 0 : ++sequence;
            byte[] packet = new byte[HeaderSize + (payload == null ? 0 : payload.Length)];
            Write32(packet, 0, uuid);
            Write32(packet, 4, parameter);
            Write32(packet, 8, packetId);
            Write32(packet, 12, opcode);
            if (payload != null) Buffer.BlockCopy(payload, 0, packet, HeaderSize, payload.Length);
            if (movement)
            {
                lock (commands)
                {
                    if (halted || closing || canSend == null || !canSend()) return false;
                    SendPacket(stage, packet);
                }
            }
            else SendPacket(stage, packet);

            long deadline = clock.ElapsedMilliseconds + timeout;
            int mismatched = 0;
            try
            {
                do
                {
                    DrainReports();
                    UpdateCombo();
                    Publish();
                    if (!command.Poll(5000, SelectMode.SelectRead)) continue;
                    int size = command.Receive(reply);
                    if (size >= HeaderSize && Read32(reply, 8) == packetId && Read32(reply, 12) == opcode)
                    {
                        linkAt = PreciseUtcNow();
                        if (stage == "监控心跳" || stage == "监控心跳重试")
                        {
                            heartbeatAcks++;
                            if (monitor != null) heartbeatAcksSinceReport++;
                        }
                        if (opcode == MonitorCommand && wireTrace != null) Trace(linkAt, "monitor_ack size=" + size +
                            " sequence=" + packetId + " opcode=0x" + opcode.ToString("X8") + " parameter=0x" +
                            Read32(reply, 4).ToString("X8") + " refreshes_physical=false");
                        return true;
                    }
                    mismatched++;
                } while (clock.ElapsedMilliseconds < deadline);
            }
            catch (SocketException ex)
            {
                throw new InvalidOperationException(stage + " · 等待应答时 UDP 故障 · " + endpoint + " · " +
                    ex.SocketErrorCode + " / " + ex.NativeErrorCode + (movement ? " · 位移已发出，请先观察设备 A 光标" : ""), ex);
            }
            throw new TimeoutException(stage + "超时 · " + endpoint + " · " + timeout + " ms · 命令 0x" +
                opcode.ToString("X8") + " · 序号 " + packetId + " · 不匹配应答 " + mismatched + " 个" +
                (movement ? " · UDP 已发送，请先观察设备 A 光标；位移不会重发" : ""));
        }

        void SendPacket(string stage, byte[] packet)
        {
            try { command.Send(packet); }
            catch (SocketException ex)
            {
                throw new InvalidOperationException(stage + " · UDP 发送失败 · " + endpoint + " · " +
                    ex.SocketErrorCode + " / " + ex.NativeErrorCode + " · " + ex.Message, ex);
            }
        }

        static uint Read32(byte[] bytes, int offset) { return BitConverter.ToUInt32(bytes, offset); }
        internal static long EpochMs(DateTime utc) { return (utc.Ticks - UnixEpoch.Ticks) / TimeSpan.TicksPerMillisecond; }
        static void Write32(byte[] bytes, int offset, uint value)
        {
            Buffer.BlockCopy(BitConverter.GetBytes(value), 0, bytes, offset, 4);
        }

        void ResetReports()
        {
#if KMBOX_AUTO_TRACKING
            autoAllowed = false;
#endif
            lastReport = -1;
            lastReportAt = default(DateTime);
            reportCount = longestGap = 0;
            heartbeatAcks = heartbeatAcksSinceReport = 0;
#if KMBOX_INTEGRATION_LOOPBACK || KMBOX_AUTO_TRACKING
            monitorEpochMs = EpochMs(PreciseUtcNow());
            Array.Clear(releasedAtReport, 0, releasedAtReport.Length);
#endif
            ignoredReports = stationaryReports = stationaryDuration = stationaryMaxGap = 0;
            stationarySince = -1;
            buttons = -1;
            comboReady = false;
            UpdateCombo();
        }

        void CloseMonitor()
        {
            FinishWireTrace("监控已关闭或连接中断");
            if (monitor != null) monitor.Dispose();
            monitor = null;
            monitorPort = 0;
            monitorStatus = "已关闭";
            ResetReports();
        }

        void CloseLink()
        {
            CloseMonitor();
            if (command != null)
            {
                if (monitorCleanupPort != 0)
                {
                    try { Exchange("断开时停止监控", MonitorCommand, 0, null, 250, false); }
                    catch (Exception ex) { log("盒子未确认停止监控：" + ex.Message); }
                }
                command.Dispose();
            }
            command = null;
            linkAt = default(DateTime);
            monitorCleanupPort = 0;
            connection = "未连接";
        }

        void Publish()
        {
            snapshot = new Snapshot
            {
                Connected = command != null && connection == "盒子已应答", Monitoring = monitor != null,
                Halted = halted, Busy = busy, Combo = combo, Buttons = IsFresh() ? buttons : -1, Count = count,
                Reports = reportCount, LongestGap = longestGap, Age = lastReport < 0 ? -1 : clock.ElapsedMilliseconds - lastReport,
                HeartbeatAcks = heartbeatAcks, TraceActive = wireTrace != null,
                LastReportAt = lastReportAt, LinkAt = linkAt, Connection = connection, Monitor = monitorStatus,
                IgnoredReports = ignoredReports, StationaryReports = stationaryReports,
                StationaryDuration = stationaryDuration, StationaryMaxGap = stationaryMaxGap
            };
            if (observe != null) observe(snapshot);
        }

        public void Dispose()
        {
            lock (commands)
            {
                closing = halted = true;
                pending = null;
            }
            wake.Set();
            worker.Join();
            wake.Dispose();
        }
    }
}
