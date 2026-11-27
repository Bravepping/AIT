using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace VisionHardwareCheck
{
    internal static class KmBoxSelfTest
    {
        internal static int Run(string output)
        {
            var checks = new List<string>();
            var deviceLog = new System.Collections.Concurrent.ConcurrentQueue<string>();
            try
            {
                using (var device = new FakeDevice())
                using (var box = new KmBox(deviceLog.Enqueue))
                {
                    int port;
                    using (var reserve = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0)))
                        port = ((IPEndPoint)reserve.Client.LocalEndPoint).Port;
                    if (port > 49151)
                    {
                        for (port = 25000; port < 49151; port++)
                        {
                            try { using (var reserve = new UdpClient(new IPEndPoint(IPAddress.Loopback, port))) break; }
                            catch (SocketException) { }
                        }
                    }

                    Require(box.Connect(device.Address, 0x1234ABCD), "connect queued");
                    Wait(() => box.Current.Connected && !box.Current.Busy, "command handshake");
                    Require(!box.Move(10, 0), "read-only before monitoring");
                    Require(box.StartMonitor(port), "monitor queued");
                    Wait(() => box.Current.Monitoring && !box.Current.Busy && device.MonitorPort == port, "monitor start");
                    Require(box.Current.ManualArmBlockReason == null && !box.Current.CanMove && !box.Move(10, 0),
                        "arming eligibility is independent of reports; move is blocked before a report");
#if KMBOX_CALIBRATION
                    Require(!box.MoveCalibrationStep(11, 0) && !box.MoveCalibrationStep(10, 10), "calibration step is exactly one axis");
                    Require(box.MoveCalibrationStep(10, 0), "explicit calibration can move with no physical report");
                    Wait(() => device.Moves == 1 && !box.Current.Busy, "stationary calibration acknowledged");
                    Require(device.X == 10 && device.Y == 0 && box.Current.Age < 0, "one explicit delta without prior mouse input");
                    device.DelayMonitorAck = true;
                    Require(box.MoveCalibrationStep(0, 10), "calibration preflight can start without a report");
                    Wait(() => device.MonitorAckWaiting, "calibration preflight pending");
                    box.Emergency(false);
                    device.DelayMonitorAck = false;
                    Wait(() => box.Current.Halted && !box.Current.Busy, "preflight emergency");
                    Require(device.Moves == 1 && !box.MoveCalibrationStep(0, 10), "emergency blocks stationary calibration");
                    Require(box.Resume(), "resume stationary calibration");
                    Wait(() => !box.Current.Halted && !box.Current.Busy, "resume complete");
                    checks.Add("SIMULATED PASS — explicit calibration works without physical reports, F8 cancels during preflight");
#endif
                    device.Report(3, 8);
                    Thread.Sleep(100);
                    Require(box.Current.Buttons == -1, "short packet rejected");
                    Report(device, box, 0);
                    checks.Add("SIMULATED PASS — UDP handshake, monitor port, malformed packet rejected");

                    for (int round = 0; round < 20; round++)
                    {
                        int first = round % 2 == 0 ? 1 : 2;
                        Report(device, box, first | 28);
                        Require(!box.Current.Combo && box.Current.Count == round, "one button and side keys");
                        Report(device, box, 31);
                        Require(box.Current.Combo && box.Current.Count == round + 1, "both keys");
                        Report(device, box, 3 ^ first);
                        Require(!box.Current.Combo, "one release");
                        Report(device, box, 0);
                    }
                    checks.Add("SIMULATED PASS — 10 left-first + 10 right-first, side/middle bits, release, counts");
                    foreach (int[] axis in new[] { new[] { -10, 0 }, new[] { 10, 0 }, new[] { 0, -10 }, new[] { 0, 10 } })
                    {
                        int expected = device.Moves + 1;
                        Require(box.Move(axis[0], axis[1]), "valid motion queued");
                        Wait(() => device.Moves == expected && !box.Current.Busy, "motion acknowledged");
                        Require(device.X == axis[0] && device.Y == axis[1], "motion coordinates");
                    }
                    Require(!box.Move(5, 5), "diagonal rejected");
                    checks.Add("SIMULATED PASS — four explicit movement datagrams, no button or wheel bytes");

#if KMBOX_CALIBRATION
                    int bounded = device.Moves + 1;
                    Require(!box.MoveRelative(100, 100, 25), "transport rejects limit above calibration maximum");
                    Require(box.MoveRelative(int.MaxValue, int.MinValue, 12), "bounded manual motion queued");
                    Wait(() => device.Moves == bounded && !box.Current.Busy, "bounded motion acknowledged");
                    Require(device.X == 12 && device.Y == -12, "wire deltas clamped safely");
                    device.DelayMonitorAck = true;
                    Require(box.MoveRelative(12, 0, 12), "emergency test queued");
                    Wait(() => device.MonitorAckWaiting, "preflight is pending");
                    box.Emergency(false);
                    device.DelayMonitorAck = false;
                    Wait(() => box.Current.Halted && !box.Current.Busy, "stop during preflight");
                    Require(device.Moves == bounded, "stop during preflight cancels send");
                    Require(box.Resume(), "resume after cancellation");
                    Wait(() => !box.Current.Halted && !box.Current.Busy, "resume after cancellation completed");
                    checks.Add("SIMULATED PASS — hard clamp, no button/wheel bytes, F8/preflight race drops unsent motion");
#endif

                    Report(device, box, 3);
                    for (int index = 0; index < 3; index++)
                    {
                        Thread.Sleep(30);
                        Report(device, box, 3);
                    }
                    Require(box.Current.StationaryReports == 4 && box.Current.StationaryDuration >= 90 &&
                        box.Current.StationaryMaxGap >= 30 && box.Current.IgnoredReports == 1, "stationary reports diagnostic");
                    Wait(() => box.Current.Buttons == -1 && !box.Current.Combo, "report expiry", 2500);
                    Require(box.Current.Connected && !box.Move(10, 0), "stale report cannot move");
                    Require(box.Current.ManualArmBlockReason == null && !box.Current.CanMove,
                        "linked idle device can prepare a shot while move authorization remains locked");
#if KMBOX_CALIBRATION
                    int stationaryMove = device.Moves + 1;
                    Require(box.MoveCalibrationStep(0, 10), "explicit calibration works after physical reports expire");
                    Wait(() => device.Moves == stationaryMove && !box.Current.Busy, "stale-input calibration acknowledged");
                    Require(device.X == 0 && device.Y == 10 && !box.Current.CanMove, "ordinary movement remains locked");
#endif
                    Report(device, box, 3);
                    Require(!box.Current.Combo, "stale held combo must be re-pressed");
                    Report(device, box, 1);
                    Report(device, box, 3);
                    Require(box.Current.Combo, "combo restored after press");
                    box.Emergency(false);
                    Wait(() => box.Current.Halted && !box.Current.Combo, "emergency");
                    Require(!box.Move(10, 0), "emergency blocks move");
                    Require(box.Resume(), "resume");
                    Wait(() => !box.Current.Halted && !box.Current.Busy, "resume completed");
                    Require(!box.Current.Combo, "resume cannot activate held combo");
                    checks.Add("SIMULATED PASS — stale input, emergency stop, reactivation requires a fresh chord");
                    checks.Add("SIMULATED PASS — stationary press report intervals, continuous ACKs do not keep old input fresh");

                    Require(box.StopMonitor(), "monitor stop queued");
                    Wait(() => !box.Current.Monitoring && !box.Current.Busy && device.MonitorPort == 0, "monitor stopped");
                    Require(box.StartMonitor(port), "rebind monitor");
                    Wait(() => box.Current.Monitoring && !box.Current.Busy && device.MonitorPort == port, "monitor restarted");
                    Report(device, box, 0);
                    int movesBeforeLinkFailure = device.Moves;
                    device.IgnoreCommands = true;
                    Wait(() => !box.Current.Connected && box.Current.Halted && !box.Current.Monitoring, "link failure", 2500);
                    Require(!box.Move(10, 0) && device.Moves == movesBeforeLinkFailure, "link failure locks movement");
                    checks.Add("SIMULATED PASS — stop/rebind local socket, missing heartbeat closes link, movement locked");
#if KMBOX_CALIBRATION
                    for (int attempt = 0; attempt < 2; attempt++)
                    {
                        device.IgnoreCommands = device.DropMonitorAck = device.DropMoveAck = false;
                        Require(box.Connect(device.Address, 0x1234ABCD), "reconnect for timeout test");
                        Wait(() => box.Current.Connected && !box.Current.Busy, "reconnect handshake");
                        Require(box.StartMonitor(port), "open monitor for timeout test");
                        Wait(() => box.Current.Monitoring && !box.Current.Busy && device.MonitorPort == port, "reopen monitor");
                        Report(device, box, 0);
                        int before = device.Moves;
                        device.DropMonitorAck = attempt == 0;
                        device.DropMoveAck = attempt == 1;
                        Require(box.MoveCalibrationStep(10, 0), "queue calibration for timeout test");
                        Wait(() => !box.Current.Connected && box.Current.Halted && !box.Current.Busy, "ack timeout fails closed");
                        string notes = string.Join("\n", deviceLog.ToArray());
                        Require(device.Moves == before + attempt, "preflight blocks send; dropped movement ACK never resends");
                        Require(notes.Contains(attempt == 0 ? "位移预检查超时" : "UDP 已发送，请先观察"), "timeout identifies the failed stage");
                    }
                    checks.Add("SIMULATED PASS — preflight timeout sends nothing; lost movement acknowledgement sends once and locks out");
#endif
#if PHASE2
                    VerifyDryRun();
                    checks.Add("SIMULATED PASS — Dry Run gate requires manual arm, fresh video, inference, KMBox, monitor and both buttons");
#endif
                    checks.Add("REAL DEVICE SKIP for KMBox self-test — physical box and mouse");
                }
                File.WriteAllLines(output, checks);
                return 0;
            }
            catch (Exception ex)
            {
                checks.Add("SIMULATED FAIL — " + ex);
                checks.AddRange(deviceLog);
                File.WriteAllLines(output, checks);
                return 1;
            }
        }

#if PHASE2
        static void VerifyDryRun()
        {
            var km = new KmBox.Snapshot();
            // Exhaustively exercise independent inputs, including a stale cached chord.
            for (int mask = 0; mask < 256; mask++)
            {
                bool armed = (mask & 1) != 0, video = (mask & 2) != 0, inference = (mask & 4) != 0;
                km.Connected = (mask & 8) != 0;
                km.Monitoring = (mask & 16) != 0;
                km.Halted = (mask & 32) == 0;
                km.Buttons = (mask & 64) != 0 ? KmBox.LeftRight : -1;
                km.Combo = (mask & 128) != 0;
                Require(DryRunGate.IsOpen(armed, DryRunGate.BlockReason(km, video, inference), km) == (mask == 255),
                        "Dry Run gate inputs " + mask);
            }
            foreach (int released in new[] { 1, 2 })
            {
                km.Buttons = released;
                Require(!DryRunGate.IsOpen(true, DryRunGate.BlockReason(km, true, true), km), "one-button Dry Run release");
            }
        }
#endif

        static void Report(FakeDevice device, KmBox box, int keys)
        {
            int expected = box.Current.Reports + 1;
            device.Report(keys, 20);
            Wait(() => box.Current.Buttons == keys && box.Current.Reports >= expected, "physical report " + keys);
        }

        static void Wait(Func<bool> predicate, string check, int timeout = 1500)
        {
            var time = Stopwatch.StartNew();
            while (!predicate() && time.ElapsedMilliseconds < timeout) Thread.Sleep(5);
            Require(predicate(), check);
        }

        static void Require(bool condition, string check)
        {
            if (!condition) throw new InvalidOperationException(check);
        }

        sealed class FakeDevice : IDisposable
        {
            readonly UdpClient endpoint = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
            readonly Thread listener;
            internal volatile bool IgnoreCommands;
#if KMBOX_CALIBRATION
            internal volatile bool DelayMonitorAck, MonitorAckWaiting, DropMonitorAck, DropMoveAck;
#endif
            internal volatile int MonitorPort, Moves, X, Y;
            volatile bool disposed;
            internal IPEndPoint Address { get { return (IPEndPoint)endpoint.Client.LocalEndPoint; } }

            internal FakeDevice()
            {
                listener = new Thread(Listen) { IsBackground = true };
                listener.Start();
            }

            internal void Report(int keys, int length)
            {
                byte[] frame = new byte[length];
                frame[0] = 1;
                frame[1] = (byte)keys;
                endpoint.Send(frame, frame.Length, new IPEndPoint(IPAddress.Loopback, MonitorPort));
            }

            void Listen()
            {
                while (!disposed)
                {
                    IPEndPoint sender = null;
                    byte[] packet;
                    try
                    {
                        if (!endpoint.Client.Poll(10000, SelectMode.SelectRead)) continue;
                        packet = endpoint.Receive(ref sender);
                    }
                    catch (SocketException) { continue; }
                    if (IgnoreCommands || packet.Length < 16 || BitConverter.ToUInt32(packet, 0) != 0x1234ABCD) continue;
                    uint opcode = BitConverter.ToUInt32(packet, 12);
                    switch (opcode)
                    {
                        case 0xaf3c2828:
                            if (packet.Length != 16 || BitConverter.ToUInt32(packet, 8) != 0) continue;
                            break;
                        case 0x27388020:
                            uint flags = BitConverter.ToUInt32(packet, 4);
                            if (flags != 0 && (flags >> 16) != 0xaa55) continue;
                            MonitorPort = (int)(flags & 65535);
#if KMBOX_CALIBRATION
                            if (DropMonitorAck) continue;
                            if (DelayMonitorAck)
                            {
                                MonitorAckWaiting = true;
                                var deadline = Stopwatch.StartNew();
                                while (DelayMonitorAck && !disposed && deadline.ElapsedMilliseconds < 130) Thread.Sleep(1);
                                MonitorAckWaiting = false;
                            }
#endif
                            break;
                        case 0xaede7345:
                            if (packet.Length != 72 || BitConverter.ToInt32(packet, 16) != 0 ||
                                BitConverter.ToInt32(packet, 28) != 0) continue;
                            X = BitConverter.ToInt32(packet, 20);
                            Y = BitConverter.ToInt32(packet, 24);
                            Moves++;
#if KMBOX_CALIBRATION
                            if (DropMoveAck) continue;
#endif
                            break;
                        default:
                            continue;
                    }
                    endpoint.Send(packet, 16, sender);
                }
            }

            public void Dispose()
            {
                disposed = true;
                listener.Join();
                endpoint.Close();
            }
        }
    }
}
