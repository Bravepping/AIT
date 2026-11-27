"""Actual ORT CPU and NDI SDK exercised with synthetic local frames."""

import base64
import ctypes as c
import io
import itertools
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ndi_capture import NdiRuntime, Video, bind, copy_rgb
from vision_model import decode, letterbox


class SendConfig(c.Structure):
    _fields_ = [("name", c.c_char_p), ("groups", c.c_char_p), ("clock_video", c.c_bool), ("clock_audio", c.c_bool)]


class LocalSender:
    def __init__(self, frame=None, rate=30, paced=False):
        self.frame = frame
        self.rate = rate
        self.paced = paced

    def __enter__(self):
        self.ndi = NdiRuntime()
        lib = self.ndi.library
        self.name = f"VisionHardwareCheck Test {os.getpid()}"
        create = bind(lib, "send_create", c.c_void_p, c.POINTER(SendConfig))
        self.free = bind(lib, "send_destroy", None, c.c_void_p)
        self.send = bind(lib, "send_send_video_v2", None, c.c_void_p, c.POINTER(Video))
        self.handle = create(c.byref(SendConfig(self.name.encode(), None, False, False)))
        assert self.handle, "NDI sender creation"
        self.stopped = threading.Event()
        self.paused = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        height, width = self.frame.shape[:2] if self.frame is not None else (360, 640)
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        if self.frame is None:
            image[:, :, 0] = 64
        else:
            image[:, :, :3] = self.frame
        video = Video(width, height, int.from_bytes(b"RGBX", "little"), self.rate, 1, width / height,
                      1, (1 << 63) - 1, image.ctypes.data, width * 4, None, 0)
        deadline = time.perf_counter()
        for number in itertools.count():
            if self.stopped.is_set():
                return
            if not self.paused.is_set():
                if self.frame is None:
                    image[90:170, 140:250, 1] = number % 250
                self.send(self.handle, c.byref(video))
            if self.paced:
                now = time.perf_counter()
                deadline = max(now, deadline + 1 / self.rate)
                time.sleep(deadline - now)
            elif self.stopped.wait(1 / self.rate):
                return

    def halt(self):
        self.stopped.set()
        self.thread.join()
        if self.handle:
            self.free(self.handle)
            self.handle = None

    def __exit__(self, *_):
        self.halt()
        self.ndi.shutdown()


class WorkerProcess:
    def __init__(self, runtime):
        self.child = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-u", str(runtime / "worker.py")],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.messages = queue.Queue()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.child.stdout:
            self.messages.put(json.loads(line))

    def send(self, cmd, **params):
        self.child.stdin.write(json.dumps({"cmd": cmd, **params}) + "\n")
        self.child.stdin.flush()

    def wait(self, kind, predicate=lambda _: True, limit=12, prohibit_preview=False):
        until = time.monotonic() + limit
        while time.monotonic() < until:
            remaining = until - time.monotonic()
            try:
                message = self.messages.get(timeout=max(0, remaining))
            except queue.Empty:
                break
            if prohibit_preview:
                assert message["kind"] != "preview", "preview transmitted while off"
            if message["kind"] == kind and predicate(message):
                return message
        raise AssertionError("timed out waiting for " + kind)

    def __enter__(self):
        ready = self.wait("ready")
        assert ready["preview_fps"] == ready["preview_frames"] == 0, ready
        return self

    def __exit__(self, *_):
        if self.child.poll() is None:
            self.send("exit")
            self.child.stdin.close()
            try:
                self.child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
                raise AssertionError("worker failed to stop")


def verify_sample(worker, sample, model, path, provider):
    worker.send("sample", sample=sample, model=model, path=str(path))
    result = worker.wait("sample", lambda item: item["sample"] == sample, prohibit_preview=True)
    assert result["model"] == model and result.get("provider") == provider, result.keys()
    assert result["filename"] == path.name and (result["width"], result["height"]) == (640, 360)
    assert result["threshold"] == 0.55 and result["elapsed"] > 0 and result["count"] >= 0
    with Image.open(io.BytesIO(base64.b64decode(result["image"]))) as jpeg:
        assert jpeg.format == "JPEG" and jpeg.size == (640, 360)
        jpeg.load()
    print(f"LOCAL {provider} PASS — image file → model → JPEG result, {result['elapsed']} ms, "
          f"detections {result['count']}", flush=True)


def verify_trt_recovery(model, runtime, output):
    summaries = []
    with WorkerProcess(runtime) as worker, LocalSender() as sender:
        worker.send("model", model=1, path=str(model), provider="TensorRT")
        ready = worker.wait("model", limit=360)
        assert ready.get("provider") == "TensorRT", ready
        selected = None
        for _ in range(3):
            worker.send("discover")
            seen = worker.wait("sources")
            selected = next((item for item in seen["sources"] if sender.name in item[0]), None)
            if selected:
                break
        assert selected, seen
        worker.send("stream", source=1, name=selected[0], url=selected[1])
        worker.wait("stream_state", lambda item: item["state"] == "live", prohibit_preview=True)
        for revision, rate in enumerate((5, 10, 15, 30, 60, 0), 1):
            worker.send("preview", fps=rate, preview=revision)
            state = worker.wait("preview_state", lambda item: item["preview"] == revision)
            assert state["fps"] == rate, state
            if rate:
                worker.wait("preview", lambda item: item["preview"] == revision)
            else:
                status = worker.wait("stream", prohibit_preview=True)
                assert status["preview_frames"] == state["frames"], status
            summaries.append({"fps": rate, "acknowledged": True, "preview_frames": state["frames"]})
        before = worker.wait("inference", lambda item: item.get("warm_samples", 0) > 2, prohibit_preview=True)
        sender.paused.set()
        lost = worker.wait("stream_state", lambda item: item["state"] == "lost", prohibit_preview=True)
        assert lost["frame_at"] == 0 and lost["gap_ms"] >= 750, lost
        sender.paused.clear()
        resumed = worker.wait("stream_state", lambda item: item["state"] == "resumed", prohibit_preview=True)
        recovered = worker.wait("inference", lambda item: item["frame_at"] >= resumed["frame_at"], prohibit_preview=True)
        assert recovered["runs"] > before["runs"], recovered
    output.write_text(json.dumps({"status": "PASS", "source": "synthetic local NDI",
                                  "provider": "TensorRT", "backend": ready["backend"], "preview_rates": summaries,
                                  "loss_ms": lost["gap_ms"], "fresh_frame_only": True}, indent=2), encoding="utf-8")


def compare_ndi(model, runtime, image_path, output):
    with Image.open(image_path) as image:
        frame = np.asarray(ImageOps.pad(image.convert("RGB"), (640, 640), method=Image.Resampling.BILINEAR,
                                        color=(114, 114, 114)))
    summaries = []
    with WorkerProcess(runtime) as worker, LocalSender(frame, rate=120) as sender:
        worker.send("discover")
        seen = worker.wait("sources")
        selected = next((item for item in seen["sources"] if sender.name in item[0]), None)
        assert selected, seen
        worker.send("stream", source=1, name=selected[0], url=selected[1])
        worker.wait("stream_state", lambda item: item["state"] == "live", prohibit_preview=True)
        for index, provider in enumerate(("CUDA", "TensorRT", "TensorRT for RTX"), 1):
            worker.send("model", model=index, path=str(model), provider=provider)
            ready = worker.wait("model", lambda item: item["model"] == index, limit=900, prohibit_preview=True)
            assert ready.get("provider") == provider, ready
            measured = worker.wait("inference", lambda item: item["model"] == index and item.get("warm_samples", 0) >= 160,
                                   limit=40, prohibit_preview=True)
            stats = worker.wait("stream", lambda item: item.get("fps", 0) > 0, prohibit_preview=True)
            assert stats["preview_frames"] == stats["preview_fps"] == 0, stats
            measured.update(provider=provider, backend=ready["backend"], ndi_fps=stats["fps"], ndi_dropped=stats["dropped"])
            summaries.append(measured)
            print(f'LOCAL NDI LOOPBACK PASS — {provider}: model {measured["model_avg"]}/{measured["model_p95"]} ms, '
                  f'receiver-to-offset {measured["receiver_avg"]}/{measured["receiver_p95"]} ms, '
                  f'{measured["inference_fps"]} FPS, preview frames 0', flush=True)
    output.write_text(json.dumps({"source": "local NDI SDK loopback, fixed image", "frame": list(frame.shape[:2][::-1]),
                                  "nominal_fps": 120, "preview_frames": 0, "results": summaries}, indent=2), encoding="utf-8")


def run(model, runtime):
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    tensor, transform = letterbox(rgb)
    assert tensor.shape == (1, 3, 640, 640)
    assert np.isclose(tensor[0, 0, 0, 0], 114 / 255) and tensor[0, 0, 140, 0] == 0
    input_rgb = np.random.default_rng(42).integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
    expected = np.ascontiguousarray(input_rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
    native, native_transform = letterbox(input_rgb)
    assert np.array_equal(expected, native) and native_transform == (640, 640, 1.0, 1.0, 0, 0)
    packed = np.arange(32, dtype=np.uint8).reshape(2, 16)
    video = Video(3, 2, int.from_bytes(b"RGBX", "little"), 30, 1, 1.5, 1, 0, packed.ctypes.data, 16, None, 0)
    expected_rgb = packed[:, :12].reshape(2, 3, 4)[:, :, :3].copy()
    owned_rgb = copy_rgb(video)
    packed.fill(0)
    assert np.array_equal(owned_rgb, expected_rgb)
    assert np.array_equal(letterbox(owned_rgb)[0], letterbox(expected_rgb)[0])
    target = [50, 60, 160, 180]
    boxes = decode(np.array([[50, 200, 160, 320, 0.9, 1], [0, 0, 1, 1, 0.2, 4]], dtype=np.float32), 0.55, transform)
    assert len(boxes) == 1 and np.allclose(boxes[0, :4], target)
    print("SIMULATED PASS — letterbox / RGB normalization / confidence and inverse box mapping", flush=True)
    with tempfile.TemporaryDirectory(prefix="vision-sample-", dir=runtime.parent.parent) as directory, WorkerProcess(runtime) as worker:
        sample_image = Path(directory) / "本地样例.png"
        Image.fromarray(rgb).save(sample_image)
        broken_image = Path(directory) / "broken.png"
        broken_image.write_bytes(b"not an image")
        worker.send("model", model=1, path=str(model), provider="CPU")
        engine = worker.wait("model", lambda item: item["model"] == 1, limit=20)
        assert engine.get("provider") == "CPU" and engine["elapsed"] > 0 and engine["count"] >= 0, engine
        print(f"LOCAL CPU PASS — CPUExecutionProvider; {engine['metadata']}; self-test {engine['elapsed']} ms, "
              f"detections {engine['count']}", flush=True)
        worker.send("sample", sample=1, model=1, path=str(broken_image))
        broken = worker.wait("sample", lambda item: item["sample"] == 1, prohibit_preview=True)
        assert broken.get("error") and "image" not in broken, broken
        verify_sample(worker, 2, 1, sample_image, "CPU")
        with LocalSender() as sender:
            selected = None
            for _ in range(3):
                worker.send("discover")
                seen = worker.wait("sources")
                selected = next((item for item in seen["sources"] if sender.name in item[0]), None)
                if selected:
                    break
            assert selected, seen
            worker.send("stream", source=1, name=selected[0], url=selected[1])
            live = worker.wait("stream_state", lambda item: item["state"] == "live")
            assert live["frame_at"] > 0, live
            status = worker.wait("stream", lambda item: item.get("width") == 640 and item.get("height") == 360 and item["fps"] > 0,
                                 prohibit_preview=True)
            assert status["dropped"] >= 0 and status["preview_fps"] == status["preview_frames"] == 0
            first = worker.wait("inference", lambda item: item["runs"] >= 3, prohibit_preview=True)
            worker.send("cuda_check")
            diagnostic = worker.wait("cuda_check", prohibit_preview=True)
            assert diagnostic.get("status") and diagnostic.get("details"), diagnostic
            print("LOCAL DIAGNOSTICS PASS — CUDA environment report returned while CPU inference remained active: "
                  + diagnostic["status"], flush=True)
            after = worker.wait("inference", lambda item: item["runs"] >= first["runs"] + 15, prohibit_preview=True)
            assert after["count"] >= 0 and after["average"] > 0
            print(f"SIMULATED PASS — real NDI SDK local discovery/receive, 640×360, SDK drops {status['dropped']}; "
                  f"preview OFF inference runs {first['runs']}→{after['runs']}, CPU mean {after['average']} ms, detections {after['count']}", flush=True)
            worker.send("preview", fps=10, preview=1)
            enabled = worker.wait("preview_state", lambda item: item["preview"] == 1)
            assert enabled["fps"] == 10 and enabled["frames"] == 0, enabled
            image = worker.wait("preview", lambda item: item["preview"] == 1)
            with Image.open(io.BytesIO(base64.b64decode(image["image"]))) as jpeg:
                assert jpeg.size == (640, 360)
            worker.send("preview", fps=0, preview=2)
            disabled = worker.wait("preview_state", lambda item: item["preview"] == 2)
            assert disabled["fps"] == 0 and disabled["frames"] > 0, disabled
            inference_off = worker.wait("inference", lambda item: item["runs"] > after["runs"], prohibit_preview=True)
            for _ in range(2):
                heartbeat = worker.wait("stream", prohibit_preview=True)
                assert heartbeat["preview_fps"] == 0 and heartbeat["preview_frames"] == disabled["frames"], heartbeat
            print("SIMULATED PASS — preview JPEG, off acknowledgement and constant preview frame counter while inference continues", flush=True)
            sender.paused.set()
            lost = worker.wait("stream_state", lambda item: item["state"] == "lost", prohibit_preview=True)
            assert lost["gap_ms"] >= 750 and lost["frame_at"] == 0, lost
            stopped = worker.wait("stream", lambda item: item["frame_at"] == 0, prohibit_preview=True)
            assert stopped["preview_frames"] == disabled["frames"], stopped
            sender.paused.clear()
            resumed = worker.wait("stream_state", lambda item: item["state"] == "resumed", prohibit_preview=True)
            recovered = worker.wait("inference", lambda item: item["frame_at"] >= resumed["frame_at"], prohibit_preview=True)
            assert recovered["runs"] > inference_off["runs"], recovered
            print(f"SIMULATED PASS — source silence signaled after {lost['gap_ms']} ms, old frame invalidated, fresh inference recovered", flush=True)
            worker.send("stream", source=2, name="", url="")
        worker.send("model", model=2, path=str(model), provider="CUDA")
        gpu = worker.wait("model", lambda item: item["model"] == 2, limit=30)
        if gpu.get("provider") == "CUDA":
            print("LOCAL GPU PASS — actual CUDA node execution confirmed by ORT profiling", flush=True)
            verify_sample(worker, 3, 2, sample_image, "CUDA")
        else:
            assert "error" in gpu and "provider" not in gpu, gpu
            print("RTX 3060 SKIP — local CUDA request failed explicitly, no CPU fallback reported: " + gpu["error"], flush=True)
    print("REAL DEVICE SKIP for this build — device A NDI stream, KMBox input/movement, double-machine Dry Run", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--compare-ndi":
        compare_ndi(*(Path(argument).resolve() for argument in sys.argv[2:]))
    elif sys.argv[1] == "--verify-trt-recovery":
        verify_trt_recovery(*(Path(argument).resolve() for argument in sys.argv[2:]))
    else:
        run(*(Path(argument).resolve() for argument in sys.argv[1:]))
