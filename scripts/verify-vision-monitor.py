"""Exercise the read-only and integration builds against a loopback KMBox fixture."""

import json
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path


def local_monitor_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for port in range(24100, 24200):
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("No test monitor port is available")


def verify(executable, output, controller=None, auto_tracking=False, hold_diagnostics=False):
    events = queue.Queue()
    commands = []
    moves = queue.Queue()
    delay_preflight = threading.Event()
    drop_move_ack = threading.Event()
    drop_monitor_ack = threading.Event()
    preflights = queue.Queue()
    stopping = threading.Event()
    hold_result = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as fake:
        fake.bind(("127.0.0.1", 0))
        fake.settimeout(0.05)

        def ack():
            while not stopping.is_set():
                try:
                    packet, source = fake.recvfrom(2048)
                except TimeoutError:
                    continue
                if len(packet) >= 16:
                    opcode, = struct.unpack_from("<I", packet, 12)
                    commands.append(opcode)
                    if opcode == 0xaede7345:
                        moves.put(packet[16:])
                        if drop_move_ack.is_set():
                            continue
                    if opcode == 0x27388020 and delay_preflight.is_set():
                        delay_preflight.clear()
                        preflights.put((packet[:16], source))
                        continue
                    if opcode == 0x27388020 and drop_monitor_ack.is_set():
                        continue
                    fake.sendto(packet[:16], source)

        listener = threading.Thread(target=ack, daemon=True)
        listener.start()
        try:
            with subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW) as child:
                def read():
                    for line in child.stdout:
                        events.put(json.loads(line))

                threading.Thread(target=read, daemon=True).start()

                def send(command, **fields):
                    child.stdin.write(json.dumps({"cmd": command, **fields}) + "\n")
                    child.stdin.flush()

                def wait(predicate):
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline:
                        event = events.get(timeout=max(0.001, deadline - time.monotonic()))
                        if predicate(event):
                            return event
                    raise AssertionError("Missing expected monitor status")

                def keys(mask, *, idle=False):
                    fake.sendto(bytes((1, mask)) + bytes(18), ("127.0.0.1", port))
                    return wait(lambda item: item.get("buttons") == mask and (not idle or not item.get("busy")))

                def long_holds():
                    send("trace_wire")
                    wait(lambda item: item.get("wire_trace_active"))
                    periods = []
                    for mask in (1, 2, 3):
                        report = keys(mask)
                        began = time.monotonic()
                        first = report
                        while time.monotonic() - began < 5:
                            time.sleep(0.54)
                            fake.sendto(bytes((1, mask)) + bytes(18), ("127.0.0.1", port))
                            updated = wait(lambda item: item.get("kind") == "kmbox" and
                                           (item["reports"] > report["reports"] or item["buttons"] == -1))
                            assert updated["buttons"] == mask and updated["report_at_ms"] > report["report_at_ms"], updated
                            assert updated["combo"] == (mask == 3), updated
                            report = updated
                        periods.append({"mask": mask, "seconds": round(time.monotonic() - began, 2),
                                        "physical_reports": report["reports"] - first["reports"] + 1})
                        released = keys(2 if mask == 3 else 0)
                        assert not released["combo"], released
                        keys(0)

                    assert keys(3)["combo"]
                    assert not keys(1)["combo"]
                    held = keys(3)
                    ack = struct.pack("<IIII", 0x1234ABCD, 0xAA550000 | port, 17, 0x27388020)
                    fake.sendto(ack, ("127.0.0.1", port))
                    expired = wait(lambda item: item.get("age_ms", -1) > 1200 and item.get("buttons") == -1)
                    assert expired["reports"] == held["reports"] and expired["report_at_ms"] == held["report_at_ms"], expired
                    assert expired["heartbeat_acks"] > held["heartbeat_acks"] and expired["ignored_reports"] > 0, expired
                    assert not expired["combo"]
                    assert not keys(3)["combo"]  # A late same-mask frame cannot restart the combo.
                    keys(0)
                    send("auto_enable")
                    wait(lambda item: item.get("kind") == "kmbox_error")
                    send("relative_move", dx=10, dy=0)
                    wait(lambda item: item.get("kind") == "kmbox_error")
                    drop_monitor_ack.set()
                    saved = wait(lambda item: item.get("text", "").startswith("KMBox 报文记录已保存"))
                    path = Path(saved["text"].split(" · ")[1])
                    trace = path.read_text(encoding="utf-8")
                    assert "local_sequence=" in trace and "mask=3" in trace and "mouse_bytes=" in trace
                    assert "monitor_ack size=16" in trace and "refreshes_physical=false" in trace
                    assert "ignored_monitor_datagram size=16" in trace and "reason=no_fresh_physical_report" in trace
                    assert "lockout reason=transport_failure" in trace
                    fault = wait(lambda item: item.get("kind") == "kmbox" and not item["connected"] and item["halted"])
                    assert not fault["combo"] and fault["buttons"] == -1
                    drop_monitor_ack.clear()
                    send("connect", ip="127.0.0.1", port=fake.getsockname()[1], uuid="1234ABCD")
                    wait(lambda item: item.get("connected"))
                    send("monitor", port=port)
                    wait(lambda item: item.get("monitoring"))
                    assert not keys(3)["combo"]
                    hold_result.update({"periodic_full_state_holds": periods, "release_each_key": True,
                                        "heartbeat_only_state_expires": True, "expiry_age_ms": expired["age_ms"],
                                        "lost_release_lockout": True, "network_lockout": True,
                                        "combined_hold_not_reused_after_reconnect": True,
                                        "udp_trace_verified": True, "automatic_movement_disabled": True})

                def status(command_id, expected):
                    event = wait(lambda item: item.get("kind") == "move_status" and item["id"] == command_id)
                    assert event["status"] == expected, event

                def auto_intent(command_id, held, *, lifetime=45):
                    send("relative_move", dx=1, dy=-1, id=command_id,
                         expires_at_ms=time.time_ns() // 1_000_000 + lifetime, hold_mask=3,
                         report=held["reports"], report_at_ms=held["report_at_ms"])

                try:
                    send("connect", ip="127.0.0.1", port=fake.getsockname()[1], uuid="1234ABCD")
                    wait(lambda item: item.get("connected"))
                    port = local_monitor_port()
                    send("monitor", port=port)
                    wait(lambda item: item.get("monitoring"))
                    for buttons in (1, 2, 4, 8, 16, 3, 2, 0):
                        keys(buttons)
                    if hold_diagnostics:
                        long_holds()
                    if auto_tracking:
                        held = keys(3, idle=True)
                        auto_intent(1, held)
                        status(1, "cancelled")
                        keys(0, idle=True)
                        send("auto_enable")
                        wait(lambda item: item.get("kind") == "auto_available")
                        held = keys(3, idle=True)
                        auto_intent(2, held)
                        payload = moves.get(timeout=2)
                        status(2, "acknowledged")
                        expected = bytearray(56)
                        struct.pack_into("<ii", expected, 4, 1, -1)
                        assert payload == expected, payload
                        auto_intent(2, held)
                        status(2, "rejected")
                        auto_intent(3, held, lifetime=-1)
                        status(3, "rejected")
                        keys(2)
                        held_again = keys(3, idle=True)
                        auto_intent(4, held)
                        status(4, "cancelled")
                        wait(lambda item: item.get("buttons") == -1 and item.get("age_ms", 0) > 1200
                             and item.get("connected") and item.get("monitoring"))
                        auto_intent(5, held_again)
                        status(5, "rejected")
                        keys(0, idle=True)
                        held_again = keys(3, idle=True)
                        auto_intent(6, held_again)
                        status(6, "cancelled")
                        send("auto_stop")
                        auto_intent(7, held_again)
                        status(7, "cancelled")
                        keys(0, idle=True)
                        send("auto_enable")
                        wait(lambda item: item.get("kind") == "auto_available")
                        held = keys(3, idle=True)
                        drop_move_ack.set()
                        auto_intent(8, held)
                        assert moves.get(timeout=2) == expected
                        status(8, "fault")
                        wait(lambda item: item.get("kind") == "kmbox" and item["connected"] is False)
                    elif controller:
                        held = keys(3, idle=True)
                        command = subprocess.check_output([str(controller), "--verify-kmbox-commands",
                                                           str(held["reports"]), str(held["report_at_ms"])],
                                                          creationflags=subprocess.CREATE_NO_WINDOW,
                                                          encoding="utf-8", timeout=5)
                        request = json.loads(command)
                        assert (request["cmd"], request["dx"], request["dy"], request["hold_mask"], request["report"]) == (
                            "relative_move", -24, 24, 3, held["reports"]), command
                        child.stdin.write(command)
                        child.stdin.flush()
                        payload = moves.get(timeout=2)
                        status(1, "acknowledged")
                        expected = bytearray(56)
                        struct.pack_into("<ii", expected, 4, -24, 24)
                        assert payload == expected, payload
                        request.pop("cmd")

                        def intent(command_id, held, *, lifetime=45, mask=3):
                            fields = {**request, "id": command_id, "expires_at_ms": time.time_ns() // 1_000_000 + lifetime,
                                      "hold_mask": mask, "report": held["reports"], "report_at_ms": held["report_at_ms"]}
                            send("relative_move", **fields)

                        intent(1, held)
                        status(1, "rejected")
                        intent(2, held, lifetime=-1)
                        status(2, "rejected")
                        intent(3, held, mask=0)
                        status(3, "rejected")
                        keys(1)
                        pressed_again = keys(3, idle=True)
                        intent(4, held)
                        status(4, "cancelled")

                        delay_preflight.set()
                        intent(5, pressed_again)
                        response, destination = preflights.get(timeout=1)
                        keys(2)
                        keys(3)
                        fake.sendto(response, destination)
                        status(5, "cancelled")

                        keys(1)
                        fresh_hold = keys(3, idle=True)
                        delay_preflight.set()
                        intent(6, fresh_hold)
                        response, destination = preflights.get(timeout=1)
                        time.sleep(0.06)
                        fake.sendto(response, destination)
                        status(6, "cancelled")
                    if not auto_tracking and not hold_diagnostics:
                        wait(lambda item: item.get("buttons") == -1 and item.get("age_ms", 0) >= 1200)
                        if controller:
                            intent(7, fresh_hold)
                            status(7, "rejected")
                        else:
                            send("relative_move", dx=10, dy=0)
                            wait(lambda item: item["kind"] == "kmbox_error")
                        send("stop_monitor")
                        wait(lambda item: item.get("monitoring") is False)
                finally:
                    send("exit")
                    child.wait(timeout=5)
        finally:
            stopping.set()
            listener.join()
    expected_commands = {0xaf3c2828, 0x27388020}
    if auto_tracking or controller:
        expected_commands.add(0xaede7345)
        assert commands.count(0xaede7345) == (2 if auto_tracking else 1)
    assert set(commands) == expected_commands, commands
    result = {"status": "PASS", "input": "simulated UDP device", "physical_keys": 5,
              "stale_report_locks": True, "move_rejected": True,
              "relative_move_transport": "loopback" if controller or auto_tracking else "locked in release",
              "opcodes_observed": [f"{value:#010x}" for value in sorted(set(commands))]}
    if hold_diagnostics:
        result.update(hold_result)
    else:
        result.update({"arming_stop_and_missing_ack": auto_tracking,
                       "fresh_reports_after_stale_require_rearm": auto_tracking,
                       "terminal_status_duplicate_expiry_and_release": bool(controller) or auto_tracking})
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    auto_tracking = len(sys.argv) > 3 and sys.argv[3] == "--auto-tracking"
    hold_diagnostics = len(sys.argv) > 3 and sys.argv[3] == "--hold-diagnostics"
    verify(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(),
           Path(sys.argv[3]).resolve() if len(sys.argv) > 3 and not (auto_tracking or hold_diagnostics) else None,
           auto_tracking, hold_diagnostics)
