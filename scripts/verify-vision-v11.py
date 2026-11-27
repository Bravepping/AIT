"""Record a preview-off/on stability run against an explicitly selected NDI source."""

import argparse
import ctypes as c
import os
import json
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import ExitStack
from ctypes import wintypes as w
from pathlib import Path


class ProcessMemory(c.Structure):
    _fields_ = [("size", w.DWORD), ("faults", w.DWORD),
                *((field, c.c_size_t) for field in ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
                                                  "nonpaged", "pagefile", "peak_pagefile", "private"))]


def monitor_resources(pid, samples, done):
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
    kernel.OpenProcess.restype = w.HANDLE
    kernel.GetProcessTimes.argtypes = (w.HANDLE, *([c.POINTER(w.FILETIME)] * 4))
    kernel.GetSystemTimes.argtypes = (*([c.POINTER(w.FILETIME)] * 3),)
    kernel.K32GetProcessMemoryInfo.argtypes = (w.HANDLE, c.POINTER(ProcessMemory), w.DWORD)
    kernel.CloseHandle.argtypes = (w.HANDLE,)
    handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
    previous = None
    previous_host = None

    def ticks(value):
        return (value.dwHighDateTime << 32) + value.dwLowDateTime

    try:
        while handle and not done.is_set():
            now = time.monotonic()
            memory = ProcessMemory()
            memory.size = c.sizeof(memory)
            metrics = {}
            if kernel.K32GetProcessMemoryInfo(handle, c.byref(memory), memory.size):
                metrics.update(worker_rss_mib=round(memory.rss / 2**20, 2), worker_private_mib=round(memory.private / 2**20, 2))
            times = [w.FILETIME() for _ in range(4)]
            if kernel.GetProcessTimes(handle, *(c.byref(value) for value in times)):
                cpu = sum(ticks(value) for value in times[2:]) / 1e7
                if previous:
                    metrics["worker_cpu_pct"] = round(100 * (cpu - previous[1]) / ((now - previous[0]) * (os.cpu_count() or 1)), 2)
                previous = (now, cpu)
            system = [w.FILETIME() for _ in range(3)]
            if kernel.GetSystemTimes(*(c.byref(value) for value in system)):
                idle, kern, user = map(ticks, system)
                host = (idle, kern + user)
                if previous_host and host[1] > previous_host[1]:
                    metrics["host_cpu_pct"] = round(100 * (1 - (host[0] - previous_host[0]) / (host[1] - previous_host[1])), 2)
                previous_host = host
            try:
                data = subprocess.check_output(
                    ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], creationflags=subprocess.CREATE_NO_WINDOW,
                    stderr=subprocess.DEVNULL, text=True, timeout=3)
                gpu, used, total = [int(item) for item in data.split(",")]
                metrics.update(gpu_pct=gpu, gpu_used_mib=used, gpu_total_mib=total)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            samples.put({"at": now, **metrics})
            done.wait(10)
    finally:
        if handle:
            kernel.CloseHandle(handle)


def percentile_histogram(histogram, fraction):
    rank = max(1, -int(-(sum(histogram.values()) * fraction) // 1))
    for value, count in sorted(histogram.items()):
        rank -= count
        if rank <= 0:
            return value
    return None


class ControlProbe:
    def __init__(self, path):
        self.child = subprocess.Popen([str(path), "--verify-control-stream"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      creationflags=subprocess.CREATE_NO_WINDOW, encoding="utf-8")
        self.samples = queue.Queue()
        self.lock = threading.Lock()
        self.latest = None
        self.failed = None
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        try:
            for line in self.child.stdout:
                self.samples.put(json.loads(line))
        except (OSError, ValueError) as error:
            self.failed = error

    def feed(self, line):
        with self.lock:
            if not self.child.stdin.closed:
                try:
                    self.child.stdin.write(line)
                    self.child.stdin.flush()
                except OSError as error:
                    self.failed = error

    def drain(self, log):
        while not self.samples.empty():
            sample = self.samples.get_nowait()
            kind = sample.pop("event")
            if kind == "controller_sample":
                self.latest = sample
            else:
                log(kind, **sample)

    def stop(self):
        with self.lock:
            if not self.child.stdin.closed:
                self.child.stdin.close()
        try:
            code = self.child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait()
            raise
        self.reader.join(timeout=2)
        if code or self.failed:
            raise RuntimeError(f"Dry Run probe stopped, exit={code}, error={self.failed}")


def verify(root, source, output, duration, preview_cycles=False, controller=None):
    runtime = root / "vision-runtime"
    events = queue.Queue()
    with ExitStack() as cleanup:
        child = cleanup.enter_context(subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-u", str(runtime / "worker.py"), "--desktop"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8"))
        control = ControlProbe(controller) if controller else None
        if control:
            cleanup.callback(control.stop)
        def receive():
            try:
                for line in child.stdout:
                    event = json.loads(line)
                    if event["kind"] == "detections":
                        event["ipc_ms"] = time.time_ns() // 1_000_000 - event["frame_at"]
                    if event["kind"] != "preview":
                        events.put(event)
                    if control and event["kind"] in ("detections", "ndi_health", "model", "stream_state", "stream"):
                        control.feed(line)
            finally:
                events.put({"kind": "exit"})

        threading.Thread(target=receive, daemon=True).start()
        resource_samples = queue.Queue()
        resource_done = threading.Event()
        resource_thread = threading.Thread(target=monitor_resources, args=(child.pid, resource_samples, resource_done), daemon=True)
        resource_thread.start()

        def send(cmd, **fields):
            child.stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
            child.stdin.flush()

        def wait(kind, predicate=lambda _: True, timeout=15):
            deadline = time.monotonic() + timeout
            while True:
                event = events.get(timeout=max(0, deadline - time.monotonic()))
                if event["kind"] == "exit":
                    raise RuntimeError("worker exited")
                if event["kind"] == kind and predicate(event):
                    return event

        def load(revision, provider):
            send("model", model=revision, path=str(root / "1013v26s_640.onnx"), provider=provider)
            result = wait("model", lambda item: item["model"] == revision, timeout=360)
            if result.get("provider") != provider:
                raise RuntimeError(result)
            return result["backend"]

        with output.open("w", encoding="utf-8") as report:
            def log(kind, **values):
                report.write(json.dumps({"event": kind, **values}, ensure_ascii=False) + "\n")
                report.flush()

            try:
                wait("ready")
                send("discover")
                found = wait("sources")
                matching = [item for item in found["sources"] if item[0] == source]
                if len(matching) != 1:
                    raise RuntimeError(f"Requested NDI source missing: {found}")
                name, url = matching[0]
                log("model", provider="TensorRT", backend=load(1, "TensorRT"))
                send("stream", source=1, name=name, url=url)
                wait("stream_state", lambda item: item["state"] == "live")
                detection = wait("detections")
                started = time.monotonic()
                log("start", source=name, duration=duration, utc_ms=time.time_ns() // 1_000_000)
                stages = [(0, 0)]
                if preview_cycles:
                    stages += [(index * duration / 7, fps) for index, fps in enumerate((30, 0, 60, 0, 15, 0), 1)]
                next_stage = 1
                preview_id, preview_rate = 0, 0
                acknowledged = True
                preview_frames = 0
                last_stream = latest_stats = gpu = None
                latest_health = None
                last_detection = started
                next_log = started
                detected_frames = int(bool(detection["boxes"]))
                count = 1
                ipc_histogram = Counter()
                dips = {"input": None, "inference": None}
                last_sample = 0.0
                dropouts = 0
                while time.monotonic() - started < duration:
                    now = time.monotonic()
                    if next_stage < len(stages) and now - started >= stages[next_stage][0]:
                        preview_id = next_stage
                        preview_rate = stages[next_stage][1]
                        acknowledged = False
                        send("preview", fps=preview_rate, preview=preview_id)
                        next_stage += 1
                    event = events.get(timeout=8)
                    if control:
                        control.drain(log)
                    kind = event["kind"]
                    if kind == "exit" or event.get("error"):
                        raise RuntimeError(event)
                    if kind == "stream_state":
                        log("stream_state", second=round(now - started, 3), state=event["state"], gap_ms=event["gap_ms"])
                        dropouts += event["state"] == "lost"
                    if kind == "detections":
                        last_detection = now
                        detected_frames += int(bool(event["boxes"]))
                        count += 1
                        if event["ipc_ms"] >= 0:
                            ipc_histogram[event["ipc_ms"]] += 1
                    elif kind == "preview_state" and event["preview"] == preview_id:
                        if event["fps"] != preview_rate:
                            raise RuntimeError(event)
                        acknowledged = True
                        preview_frames = event["frames"]
                        log("preview", second=round(now - started, 2), fps=preview_rate, frames=preview_frames)
                    elif kind == "inference":
                        latest_stats = event
                        if now - started > 3:
                            note_dip(log, dips, "inference", now - started, event.get("inference_fps"), 180,
                                     latest_health, last_stream, latest_stats, gpu, control.latest if control else None)
                    elif kind == "gpu":
                        gpu = event
                    elif kind == "stream":
                        last_stream = event
                        if now - started > 3:
                            note_dip(log, dips, "input", now - started, event.get("fps"), 216,
                                     latest_health, last_stream, latest_stats, gpu, control.latest if control else None)
                        if acknowledged and not preview_rate and event["preview_frames"] != preview_frames:
                            raise RuntimeError("Preview counter advanced while off")
                    elif kind == "ndi_health":
                        latest_health = event
                    if now - last_detection > 4:
                        raise RuntimeError("No fresh inference for four seconds")
                    if now >= next_log:
                        log("sample", second=round(now - started, 2), preview=preview_rate, stream=last_stream,
                            inference=latest_stats, gpu=gpu, health=latest_health, detected_frames=detected_frames,
                            controller=control.latest if control else None)
                        next_log = now + 10
                    while not resource_samples.empty():
                        sample = resource_samples.get_nowait()
                        at = sample.pop("at") - started
                        if at >= 0:
                            log("resources", second=round(at, 2), **sample)
                    last_sample = now - started
                for metric, start in dips.items():
                    if start is not None:
                        log("fps_dip_open_at_finish", metric=metric, start=round(start, 3), observed_seconds=round(last_sample - start, 3))
                if control:
                    control.stop()
                    control.drain(log)
                samples = sum(ipc_histogram.values())
                log("continuous_pass", seconds=round(time.monotonic() - started, 2), inference_frames=count,
                    detected_frames=detected_frames, preview_cycles=next_stage - 1, recovered_short_dropouts=dropouts,
                    whole_run_receiver_to_ipc={"samples": samples,
                        "mean_ms": round(sum(key * val for key, val in ipc_histogram.items()) / samples, 2) if samples else None,
                        "p95_ms": percentile_histogram(ipc_histogram, .95), "p99_ms": percentile_histogram(ipc_histogram, .99)})
                send("stream", source=2, name="", url="")
                log("reload", provider="CUDA", backend=load(2, "CUDA"))
                log("reload", provider="TensorRT", backend=load(3, "TensorRT"))
                send("stream", source=3, name=name, url=url)
                wait("stream_state", lambda item: item["source"] == 3 and item["state"] == "live")
                wait("detections", lambda item: item["model"] == 3 and item["source"] == 3)
                log("reconnect_pass", method="receiver disconnect and reconnect")
                send("stream", source=4, name="", url="")
                send("stream", source=5, name=name, url=url)
                wait("detections", lambda item: item["model"] == 3 and item["source"] == 5)
                log("reconnect_pass", method="second receiver disconnect and reconnect")
            except Exception as error:
                log("failure", error=str(error))
                raise
            finally:
                resource_done.set()
                resource_thread.join(timeout=4)
                if child.poll() is None:
                    send("exit")
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        child.kill()


def note_dip(log, dips, metric, now, fps, threshold, health, stream, inference, gpu, controller):
    if fps is None:
        return
    start = dips[metric]
    context = {"stream": stream, "inference": inference, "gpu": gpu, "health": health, "controller": controller}
    if health and health.get("frame_at"):
        context["capture_age_ms"] = time.time_ns() // 1_000_000 - health["frame_at"]
    if fps < threshold and start is None:
        dips[metric] = now
        log("fps_dip_start", metric=metric, second=round(now, 3), fps=fps, threshold=threshold, **context)
    elif fps >= threshold and start is not None:
        dips[metric] = None
        log("fps_dip_recovered", metric=metric, second=round(now, 3), fps=fps,
            observed_seconds=round(now - start, 3), **context)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("root", type=Path)
    args.add_argument("source")
    args.add_argument("output", type=Path)
    args.add_argument("--seconds", type=int, default=180, help="Continuous test duration in seconds (default: 180).")
    args.add_argument("--preview-cycles", action="store_true", help="Exercise the independent preview setting within the test window.")
    args.add_argument("--controller", type=Path, help="V1.3 executable with its stdin-only Dry Run test entry.")
    options = args.parse_args()
    if options.seconds <= 0:
        args.error("--seconds must be positive")
    execution_state = c.windll.kernel32.SetThreadExecutionState
    execution_state.argtypes = (w.DWORD,)
    execution_state.restype = w.DWORD
    execution_state(0x80000001)  # Keep the system awake only while this Windows test runs.
    try:
        verify(options.root.resolve(), options.source, options.output.resolve(), options.seconds, options.preview_cycles,
               options.controller.resolve() if options.controller else None)
    finally:
        execution_state(0x80000000)
