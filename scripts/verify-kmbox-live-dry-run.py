"""Feed the production Dry Run controller real NDI and a read-only KMBox sidecar."""

import argparse
import contextlib
import json
import queue
import subprocess
import threading
import time
from pathlib import Path


def send(child, **message):
    child.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    child.stdin.flush()


def read(name, child, events):
    for line in child.stdout:
        events.put((name, json.loads(line)))
    events.put((name, None))


def stop(child):
    child.stdin.close()
    try:
        child.wait(timeout=4)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=4)


def verify(args):
    runtime = args.runtime.resolve()
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    instruction = run / "operator.txt"
    status_path = run / "status.json"
    inbox = queue.Queue(maxsize=4096)
    status = {"phase": "connecting", "model": None, "stream": None, "monitoring": False,
              "frames": 0, "frames_with_targets": 0, "controller": None}

    with contextlib.ExitStack() as resources:
        output = resources.enter_context((run / "live-dry-run.jsonl").open("w", encoding="utf-8"))

        def record(channel, event):
            output.write(json.dumps({"channel": channel, "observed_at_ms": time.time_ns() // 1_000_000,
                                     **event}, ensure_ascii=False) + "\n")
            output.flush()

        def spawn(name, argv):
            child = subprocess.Popen(argv, cwd=runtime, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
            resources.callback(stop, child)
            threading.Thread(target=read, args=(name, child, inbox), daemon=True).start()
            return child

        worker = spawn("ndi", [str(runtime / "vision-runtime/python.exe"), "-B", "-X", "utf8", "-u",
                               str(runtime / "vision-runtime/worker.py"), "--desktop"])
        monitor = spawn("kmbox", [str(args.monitor.resolve())])
        control = spawn("control", [str(args.executable.resolve()), "--verify-physical-dry-run"])
        send(monitor, cmd="connect", ip=args.ip, port=args.port, uuid=args.uuid)
        monitor_requested = False
        trace_started = 0.0
        prior_km = None
        saw_instruction = ""
        last_status = 0.0
        started = time.monotonic()
        try:
            while time.monotonic() - started < args.seconds:
                now = time.monotonic()
                if now - last_status > 0.25:
                    status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
                    last_status = now
                    if instruction.is_file():
                        action = instruction.read_text(encoding="utf-8-sig").strip()
                        if action != saw_instruction:
                            saw_instruction = action
                            record("operator", {"event": action})
                            if action == "stop":
                                break
                            command, _, name = action.partition(" ")
                            if command in ("arm", "arm-on-left"):
                                status["phase"] = name
                                send(control, kind="operator_arm", wait_for_mask=1 if command == "arm-on-left" else 0)
                try:
                    channel, event = inbox.get(timeout=0.05)
                except queue.Empty:
                    send(control, kind="clock")
                    continue
                if event is None:
                    raise RuntimeError(f"{channel} exited during measurement")
                kind = event.get("kind")
                if channel == "ndi":
                    if kind == "ready":
                        send(worker, cmd="discover")
                    elif kind == "sources":
                        name, url = next(pair for pair in event["sources"] if args.source in pair[0])
                        status["stream"] = name
                        send(worker, cmd="model", model=1, provider="TensorRT", path=str(runtime / "1013v26s_640.onnx"))
                        send(worker, cmd="stream", source=1, name=name, url=url)
                    elif kind == "model":
                        if event.get("error") or event.get("provider") != "TensorRT":
                            raise RuntimeError(f"TensorRT failed to load: {event.get('error')}")
                        status["model"] = event.get("backend")
                        record(channel, event)
                    elif kind == "detections":
                        status["frames"] += 1
                        status["frames_with_targets"] += bool(event["boxes"])
                    elif kind == "stream_state":
                        record(channel, event)
                    if kind in ("model", "stream_state", "ndi_health", "detections"):
                        send(control, **event)
                elif channel == "kmbox":
                    if kind == "kmbox":
                        status["monitoring"] = event["monitoring"]
                        if event["connected"] and not monitor_requested:
                            monitor_requested = True
                            send(monitor, cmd="monitor", port=args.monitor_port)
                        if event["monitoring"] and not event.get("wire_trace_active") and now - trace_started > 2:
                            trace_started = now
                            send(monitor, cmd="trace_wire")
                        identity = (event["reports"], event["buttons"], event["connected"], event["monitoring"])
                        if identity != prior_km:
                            record(channel, event)
                            prior_km = identity
                        send(control, **event)
                    elif kind in ("log", "kmbox_error"):
                        record(channel, event)
                else:
                    record(channel, event)
                    status["controller"] = {key: event.get(key) for key in ("state", "reason", "event", "observation", "candidate")}
        finally:
            status["ended"] = True
            status_path.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
            send(monitor, cmd="exit")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    channel, event = inbox.get(timeout=0.05)
                except queue.Empty:
                    continue
                if channel == "kmbox" and event is None:
                    break
                if channel == "kmbox" and event.get("kind") == "log":
                    record(channel, event)
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("monitor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--monitor-port", type=int, required=True)
    parser.add_argument("--seconds", type=int, default=180)
    verify(parser.parse_args())
