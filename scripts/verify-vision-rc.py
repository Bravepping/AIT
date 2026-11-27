"""Exercise a live NDI feed and the shipping Dry Run controller for three minutes."""

import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def events(process, messages):
    try:
        for line in process.stdout:
            messages.put(json.loads(line))
    finally:
        messages.put(None)


def write(process, message):
    process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
    process.stdin.flush()


def summarize(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return {"samples": len(samples), "median": round(statistics.median(ordered), 3),
            "min": round(ordered[0], 3), "max": round(ordered[-1], 3)}


def verify(app, controller_exe, output, source):
    runtime = app / "vision-runtime"
    options = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
                   creationflags=subprocess.CREATE_NO_WINDOW)
    with subprocess.Popen([str(runtime / "python.exe"), "-B", "-X", "utf8", "-u",
                           str(runtime / "worker.py"), "--desktop"], cwd=app, **options) as worker:
        mailbox = queue.Queue(maxsize=1024)
        threading.Thread(target=events, args=(worker, mailbox), daemon=True).start()

        def next_event(kind, seconds):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                event = mailbox.get(timeout=deadline-time.monotonic())
                if event is None:
                    raise RuntimeError("Inference process exited")
                if event.get("error") and event["kind"] in ("model", "stream"):
                    raise RuntimeError(event["error"])
                if event["kind"] == kind:
                    return event
            raise TimeoutError(kind)

        try:
            next_event("ready", 20)
            write(worker, {"cmd": "discover"})
            discovered = next_event("sources", 15)
            name, url = next(pair for pair in discovered["sources"] if pair[0] == source)
            write(worker, {"cmd": "model", "model": 1, "provider": "TensorRT",
                           "path": str(app / "1013v26s_640.onnx")})
            model = next_event("model", 25)
            if model.get("provider") != "TensorRT":
                raise RuntimeError("TensorRT backend did not load")
            write(worker, {"cmd": "stream", "source": 1, "name": name, "url": url})
            stream = next_event("stream_state", 20)
            with subprocess.Popen([str(controller_exe), "--verify-control-stream"], **options) as controller:
                reports = queue.Queue()
                threading.Thread(target=events, args=(controller, reports), daemon=True).start()
                write(controller, model)
                write(controller, stream)
                begin = time.monotonic()
                until = begin + 180
                preview_phase = 0
                counts = {"frames": 0, "frames_with_boxes": 0, "preview_frames": 0}
                statistics_by_phase = {phase: {key: [] for key in ("ndi_fps", "inference_fps", "frame_age_ms")}
                                       for phase in ("off", "on")}
                losses = []
                last_progress = -30
                while time.monotonic() < until:
                    elapsed = time.monotonic() - begin
                    if elapsed >= 60 and preview_phase == 0:
                        write(worker, {"cmd": "preview", "fps": 5, "preview": 1})
                        preview_phase = 1
                    if elapsed >= 90 and preview_phase == 1:
                        write(worker, {"cmd": "preview", "fps": 0, "preview": 2})
                        preview_phase = 2
                    try:
                        event = mailbox.get(timeout=min(1, max(0.001, until-time.monotonic())))
                    except queue.Empty:
                        continue
                    if event is None:
                        raise RuntimeError("Inference process exited during stability test")
                    if event.get("error") and event["kind"] in ("model", "stream"):
                        raise RuntimeError(event["error"])
                    phase = statistics_by_phase["on" if preview_phase == 1 else "off"]
                    kind = event["kind"]
                    if kind in ("detections", "ndi_health", "stream_state", "model"):
                        write(controller, event)
                    if kind == "detections":
                        counts["frames"] += 1
                        counts["frames_with_boxes"] += bool(event["boxes"])
                    elif kind == "ndi_health":
                        phase["frame_age_ms"].append(time.time_ns() // 1_000_000 - event["frame_at"])
                    elif kind == "stream":
                        phase["ndi_fps"].append(event["fps"])
                    elif kind == "inference" and event.get("inference_fps") is not None:
                        phase["inference_fps"].append(event["inference_fps"])
                    elif kind == "stream_state" and event["state"] == "lost":
                        losses.append(round(elapsed, 3))
                    elif kind == "preview":
                        counts["preview_frames"] += 1
                    if elapsed >= last_progress + 30:
                        print(f"{elapsed:.0f}s frames={counts['frames']}, with boxes={counts['frames_with_boxes']}", flush=True)
                        last_progress = elapsed
                controller.stdin.close()
                if controller.wait(timeout=5):
                    raise RuntimeError("Dry Run controller exited with error")
                controller_events = list(iter(lambda: reports.get(timeout=2), None))
                summary = next(item for item in controller_events if item.get("event") == "controller_summary")
                report = {"source": name, "seconds": round(time.monotonic()-begin, 1), "backend": model.get("backend"),
                          **counts, "stream_loss_seconds": losses, "controller": summary,
                          "controller_faults": [event for event in controller_events if event["event"] == "controller_fault"],
                          "statistics": {phase: {key: summarize(values) for key, values in sample.items()}
                                         for phase, sample in statistics_by_phase.items()}}
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            if worker.poll() is None:
                write(worker, {"cmd": "exit"})
                worker.wait(timeout=7)


if __name__ == "__main__":
    app, controller, output = (Path(item).resolve() for item in sys.argv[1:4])
    verify(app, controller, output, sys.argv[4])
