"""Local stdio worker for NDI and ONNX. No mouse or network command output."""

import base64
import io
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from cuda_check import inspect_cuda
from ndi_capture import NdiRuntime
from vision_model import ENGINE_CHOICES, FAST_RGBX, VisionModel, letterbox


CONFIDENCE_THRESHOLD = 0.55
WARMUP_FRAMES = 20
FRAME_MAX_AGE_MS = 250
MIN_NEWER_FRAME_MS = 3


class VisionWorker:
    def __init__(self, emit, desktop=False):
        self.emit = emit
        self.desktop = desktop
        self.changed = threading.Condition()
        self.closed = False
        self.source_id = 0
        self.frame_sequence = 0
        self.first_live_sequence = 1
        self.frame = None
        self.prepared = None
        self.input_buffers = [np.empty((1, 3, 640, 640), dtype=np.float32) for _ in range(2)]
        self.inference_buffer = 0
        self.model_online = None
        self.result = None
        self.last_preparation = 0
        self.replaced = 0
        self.model_id = 0
        self.model_request = None
        self.sample_request = None
        self.preview_fps = 0
        self.preview_id = 0
        self.preview_frames = 0
        self.preview_output = threading.Lock()
        self.runtime = None
        self.runtime_lock = threading.Lock()
        self.discover_thread = None
        self.cuda_thread = None
        self.receiver = None
        self.receiver_stop = None
        self.health_stop = threading.Event()
        self.preprocess = threading.Thread(target=self.prepare_frames, name="Preprocess", daemon=True)
        self.inference = threading.Thread(target=self.run_inference, name="ONNX", daemon=True)
        self.preview = threading.Thread(target=self.run_preview, name="Preview", daemon=True)
        self.inference.start()
        self.preprocess.start()
        self.preview.start()
        if desktop:
            threading.Thread(target=self.report_video_health, name="NDI health", daemon=True).start()
            threading.Thread(target=self.report_vram, name="GPU telemetry", daemon=True).start()

    def ndi(self):
        with self.runtime_lock:
            if self.runtime is None:
                self.runtime = NdiRuntime()
            return self.runtime

    def discover(self):
        if self.discover_thread is not None and self.discover_thread.is_alive():
            return

        def run():
            try:
                self.emit("sources", sources=self.ndi().discover())
            except Exception as ex:
                self.emit("sources", sources=[], error=str(ex))

        self.discover_thread = threading.Thread(target=run, name="NDI discovery", daemon=True)
        self.discover_thread.start()

    def check_cuda(self):
        if self.cuda_thread is not None and self.cuda_thread.is_alive():
            return

        def run():
            try:
                report = inspect_cuda()
            except Exception as ex:
                report = {"status": "自检失败", "details": [str(ex)]}
            with self.changed:
                closed = self.closed
            if not closed:
                self.emit("cuda_check", **report)

        self.cuda_thread = threading.Thread(target=run, name="CUDA check", daemon=True)
        self.cuda_thread.start()

    def stream(self, source_id, name, url):
        with self.changed:
            self.source_id = source_id
            self.frame = self.result = self.prepared = None
            self.first_live_sequence = self.frame_sequence + 1
            self.replaced = 0
            self.changed.notify_all()
        if self.receiver is not None:
            self.receiver_stop.set()
            self.receiver.join()
            self.receiver = None
        if not name:
            return
        stop = threading.Event()

        def frame(rgb, received):
            with self.changed:
                if self.source_id != source_id:
                    return
                if self.frame is not None and self.model_online is not None and self.frame[0] > self.last_preparation:
                    self.replaced += 1
                self.frame_sequence += 1
                self.frame = self.frame_sequence, source_id, rgb, received, time.perf_counter_ns()
                self.changed.notify_all()

        def status(width, height, nominal, fps, dropped, capture_wait_ms, copy_ms):
            with self.preview_output:
                with self.changed:
                    if self.source_id != source_id:
                        return
                    last = self.frame[3] if self.frame is not None else 0
                    replaced = self.replaced
                    fps_limit, preview_frames = self.preview_fps, self.preview_frames
                self.emit("stream", source=source_id, width=width, height=height, nominal=round(nominal, 2),
                          fps=round(fps, 2), dropped=dropped, replaced=replaced, frame_at=last,
                          preview_fps=fps_limit, preview_frames=preview_frames,
                          ndi_capture_wait_avg=round(capture_wait_ms, 3) if capture_wait_ms is not None else None,
                          ndi_copy_avg=round(copy_ms, 3) if copy_ms is not None else None)

        def state(value, frame_at, gap_ms):
            with self.changed:
                if self.source_id != source_id:
                    return
                if value == "lost":
                    self.frame = self.result = self.prepared = None
                    self.first_live_sequence = self.frame_sequence + 1
                    self.changed.notify_all()
            self.emit("stream_state", source=source_id, state=value, frame_at=frame_at, gap_ms=gap_ms)

        def run():
            try:
                self.ndi().receive(name, url, stop, frame, status, state)
            except Exception as ex:
                with self.changed:
                    if self.source_id != source_id:
                        return
                    self.frame = self.result = self.prepared = None
                    self.changed.notify_all()
                self.emit("stream", source=source_id, frame_at=0, error=str(ex))

        self.receiver_stop = stop
        self.receiver = threading.Thread(target=run, name="NDI capture", daemon=True)
        self.receiver.start()

    def report_video_health(self):
        while not self.health_stop.wait(0.04):
            with self.changed:
                frame = self.frame
                if frame is None:
                    continue
                sequence, source, _, frame_at, _ = frame
                raw_pending = int(sequence > self.last_preparation)
                prepared_pending = int(self.prepared is not None)
            self.emit("ndi_health", source=source, frame_at=frame_at, sequence=sequence,
                      raw_pending=raw_pending, prepared_pending=prepared_pending)

    def load(self, model_id, path, provider):
        with self.changed:
            self.model_id = model_id
            self.model_request = (path, provider) if path else None
            self.model_online = None
            self.last_preparation = 0
            self.sample_request = None
            self.result = self.prepared = None
            self.replaced = 0
            self.changed.notify_all()

    def sample(self, sample_id, model_id, path):
        with self.changed:
            if self.model_id == model_id:
                self.sample_request = (sample_id, path)
                self.changed.notify_all()

    def prepare_frames(self):
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.closed or (
                    self.model_online is not None and self.frame is not None
                    and self.prepared is None
                    and self.frame[0] > self.last_preparation))
                if self.closed:
                    return
                frame, model_id = self.frame, self.model_online
                self.last_preparation = frame[0]
                # The consumer can only acquire a buffer after we publish it.
                # Use the other buffer even if a stream switch cleared the ready slot.
                slot = 1 - self.inference_buffer
            sequence, source, _, frame_at, _ = frame
            if time.time_ns() // 1_000_000 - frame_at > FRAME_MAX_AGE_MS:
                continue
            batch = self.prepare_frame(frame, slot)
            with self.changed:
                if (self.model_online == model_id and self.source_id == source
                        and self.frame is not None and sequence >= self.first_live_sequence):
                    self.prepared = batch
                    self.changed.notify_all()

    def prepare_frame(self, frame, slot):
        age_ms = max(0, time.time_ns() / 1_000_000 - frame[3])
        started_ns = time.perf_counter_ns()
        try:
            prepared = letterbox(frame[2], output=self.input_buffers[slot])
        except Exception as error:
            prepared = error
        ready_ns = time.perf_counter_ns()
        return frame, prepared, ready_ns, (age_ms, (ready_ns - started_ns) / 1e6,
                                          (started_ns - frame[4]) / 1e6), slot

    def report_vram(self):
        executable = shutil.which("nvidia-smi")
        while executable and not self.closed:
            try:
                data = subprocess.check_output(
                    [executable, "-i", "0", "--query-gpu=memory.used,memory.total",
                     "--format=csv,noheader,nounits"], timeout=3, text=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stderr=subprocess.DEVNULL)
                used, total = map(int, data.strip().split(","))
                self.emit("gpu", used_mib=used, total_mib=total)
            except (OSError, ValueError, subprocess.SubprocessError):
                self.emit("gpu", used_mib=None, total_mib=None)
            with self.changed:
                self.changed.wait_for(lambda: self.closed, timeout=3)

    def run_inference(self):
        current_id = -1
        model = None
        last_source = -1
        total_ms = total_runs = 0
        reselected = 0
        next_update = 0
        timings = deque(maxlen=240)
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.closed or self.model_id != current_id or self.sample_request is not None or (
                    model is not None and self.prepared is not None))
                if self.closed:
                    if model is not None:
                        model.close()
                    return
                model_id = self.model_id
                request = self.model_request
                sample = self.sample_request if model_id == current_id else None
                if sample is not None:
                    self.sample_request = None
                batch = None
                fresh = None
                if model_id == current_id and sample is None and self.prepared is not None:
                    batch, self.prepared = self.prepared, None
                    self.inference_buffer = batch[4]
                    newest = self.frame
                    if (FAST_RGBX and model.provider == "TensorRT" and newest is not None
                            and newest[0] > batch[0][0] and newest[2].shape == (640, 640, 3)
                            and newest[3] - batch[0][3] >= MIN_NEWER_FRAME_MS
                            and time.time_ns() // 1_000_000 - newest[3] <= FRAME_MAX_AGE_MS):
                        # Reserve the newer raw frame so the producer cannot prepare it a second time.
                        fresh = newest
                        self.last_preparation = newest[0]
                    self.changed.notify_all()
            if model_id != current_id:
                current_id = model_id
                if model is not None:
                    model.close()
                model = None
                total_runs = total_ms = next_update = 0
                reselected = 0
                timings.clear()
                if request is None:
                    continue
                requested = request[1]
                self.emit("model_loading", model=model_id, status=f"正在初始化 {requested}，首次构建引擎可能需要数分钟")
                try:
                    model = VisionModel(*request)
                    self.emit("model", model=model_id, provider=model.provider, metadata=model.description,
                              elapsed=round(model.test_ms, 2), count=model.test_count,
                              requested=requested, backend=model.backend_details)
                    with self.changed:
                        if self.model_id == model_id:
                            self.model_online = model_id
                            self.changed.notify_all()
                except Exception as ex:
                    model = None
                    self.emit("model", model=model_id, requested=requested, error=str(ex), phase="load")
                continue
            if sample is not None:
                self.infer_sample(model, model_id, *sample)
                continue
            frame, prepared, ready_ns, prep_timing, _ = self.prepare_frame(fresh, batch[4]) if fresh is not None else batch
            sequence, source, rgb, frame_at, _ = frame
            if source != last_source:
                total_runs = total_ms = next_update = 0
                reselected = 0
                timings.clear()
                last_source = source
            if time.time_ns() // 1_000_000 - frame_at > FRAME_MAX_AGE_MS:
                continue
            try:
                if isinstance(prepared, Exception):
                    raise prepared
                prepared_wait = (time.perf_counter_ns() - ready_ns) / 1e6
                boxes, elapsed = model.infer_prepared(prepared, CONFIDENCE_THRESHOLD)
                total_runs += 1
                reselected += int(fresh is not None)
                total_ms += elapsed
                height, width = rgb.shape[:2]
                offset = nearest_offset(boxes, width, height)
                completed = time.monotonic()
                receiver_ms = max(0, time.time_ns() // 1_000_000 - frame_at)
                emit_result = False
                with self.changed:
                    if (self.source_id != source or self.model_id != model_id or self.frame is None
                            or sequence < self.first_live_sequence):
                        continue
                    self.result = (sequence, source, model_id, rgb, frame_at, boxes)
                    if total_runs > WARMUP_FRAMES:
                        timings.append((elapsed, receiver_ms, completed, *prep_timing, prepared_wait))
                    ready = time.monotonic()
                    if ready >= next_update:
                        next_update = ready + 0.08
                        emit_result = True
                if self.desktop and receiver_ms <= FRAME_MAX_AGE_MS:
                    self.emit("detections", model=model_id, source=source, frame_at=frame_at, sequence=sequence,
                              width=width, height=height, boxes=boxes.tolist())
                if emit_result:
                    self.emit("inference", model=model_id, source=source, frame_at=frame_at,
                              elapsed=round(elapsed, 2), average=round(total_ms / total_runs, 2),
                              count=len(boxes), runs=total_runs, offset=offset, receiver_ms=receiver_ms,
                              reselected=reselected,
                              **stream_timings(timings))
            except Exception as ex:
                model.close()
                model = None
                with self.changed:
                    if self.model_id == model_id:
                        self.model_online = None
                        self.result = self.prepared = None
                self.emit("model", model=model_id, requested=requested, error=str(ex), phase="inference")

    def infer_sample(self, model, model_id, sample_id, path):
        try:
            if model is None:
                raise RuntimeError("请先加载模型")
            with Image.open(path, formats=("PNG", "JPEG", "BMP")) as source:
                if source.width * source.height > 16_000_000:
                    raise ValueError("图片超过 1600 万像素，请裁剪后重试")
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
            boxes, elapsed = model.infer(rgb, CONFIDENCE_THRESHOLD)
            height, width = rgb.shape[:2]
            fields = dict(provider=model.provider, filename=Path(path).name, width=width, height=height,
                          count=len(boxes), elapsed=round(elapsed, 2), threshold=CONFIDENCE_THRESHOLD,
                          max_score=round(float(boxes[:, 4].max()), 3) if len(boxes) else None,
                          image=render_preview(rgb, boxes, bounds=(960, 640), draw_center=False))
        except Exception as ex:
            fields = {"error": str(ex)}
        self.emit("sample", sample=sample_id, model=model_id, **fields)

    def set_preview(self, fps, preview_id, error=None):
        with self.preview_output:
            with self.changed:
                if error is not None and preview_id != self.preview_id:
                    return
                self.preview_fps = fps if fps in (5, 10, 15, 30, 60) else 0
                self.preview_id = preview_id
                count = self.preview_frames
                self.changed.notify_all()
            self.emit("preview_state", fps=self.preview_fps, preview=preview_id, frames=count, error=error)

    def run_preview(self):
        last_key = None
        interval = 0
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.closed or self.preview_fps and self.frame is not None, timeout=0.25)
                if self.closed:
                    return
                fps, preview_id, frame, result = self.preview_fps, self.preview_id, self.frame, self.result
            if not fps or frame is None:
                continue
            now = time.monotonic()
            if now < interval:
                time.sleep(interval - now)
                continue
            sequence, source, rgb, frame_at, _ = frame
            annotated = result is not None and result[1] == source and frame_at - result[4] <= 250
            if annotated:
                sequence, source, model, rgb, frame_at, boxes = result
            else:
                model, boxes = None, []
            key = preview_id, sequence, annotated
            if key == last_key or time.time_ns() // 1_000_000 - frame_at > 1000:
                time.sleep(0.02)
                continue
            try:
                image = render_preview(rgb, boxes)
                with self.preview_output:
                    with self.changed:
                        if (self.preview_id != preview_id or not self.preview_fps or self.source_id != source
                                or self.frame is None or sequence < self.first_live_sequence):
                            continue
                        if model is not None and self.model_id != model:
                            continue
                        self.preview_frames += 1
                    self.emit("preview", source=source, preview=preview_id, frame_at=frame_at, image=image)
                last_key, interval = key, now + 1 / fps
            except Exception as ex:
                self.set_preview(0, preview_id, str(ex))

    def close(self):
        self.health_stop.set()
        if self.receiver is not None:
            self.receiver_stop.set()
            self.receiver.join()
        if self.discover_thread is not None:
            self.discover_thread.join()
        with self.changed:
            self.closed = True
            self.changed.notify_all()
        self.inference.join(timeout=1)
        self.preprocess.join(timeout=1)
        self.preview.join(timeout=1)
        if self.runtime is not None:
            self.runtime.shutdown()


def nearest_offset(boxes, width, height):
    if not len(boxes):
        return None
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    centers[:, 0] -= width / 2
    centers[:, 1] -= height / 2
    index = (centers ** 2).sum(axis=1).argmin()
    return [float(round(float(value), 1)) for value in centers[index]]

def stream_timings(timings):
    count = len(timings)
    if count < 2:
        return {"warm_samples": count}
    model, receiver, times, prep_age, preprocessing, raw_wait, prepared_wait = zip(*timings)
    model, receiver = sorted(model), sorted(receiver)
    p95 = math.ceil(count * 0.95) - 1
    p99 = math.ceil(count * 0.99) - 1
    return {"warm_samples": count, "model_avg": round(sum(model) / count, 2), "model_p95": round(model[p95], 2),
            "receiver_avg": round(sum(receiver) / count, 2), "receiver_p95": receiver[p95],
            "model_p99": round(model[p99], 2), "receiver_p99": receiver[p99],
            "before_preprocess_avg": round(sum(prep_age) / count, 2),
            "preprocess_avg": round(sum(preprocessing) / count, 2),
            "raw_wait_avg": round(sum(raw_wait) / count, 3),
            "prepared_wait_avg": round(sum(prepared_wait) / count, 2),
            "inference_fps": round((count - 1) / max(times[-1] - times[0], 1e-6), 2)}


def render_preview(rgb, boxes, bounds=(640, 360), draw_center=True):
    image = Image.fromarray(rgb)
    width, height = image.size
    image.thumbnail(bounds)
    sx, sy = image.width / width, image.height / height
    draw = ImageDraw.Draw(image)
    for left, top, right, bottom, score, category in boxes:
        rect = (left * sx, top * sy, right * sx, bottom * sy)
        draw.rectangle(rect, outline="#48eaa0", width=2)
        draw.text(rect[:2], f"{int(category)} {score:.2f}", fill="#48eaa0")
    if draw_center:
        x, y = image.width / 2, image.height / 2
        draw.line((x - 5, y, x + 5, y), fill="#ffffff")
        draw.line((x, y - 5, x, y + 5), fill="#ffffff")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=75)
    return base64.b64encode(output.getvalue()).decode("ascii")


def main():
    output = threading.Lock()

    def emit(kind, **fields):
        line = json.dumps({"kind": kind, **fields}, ensure_ascii=True, separators=(",", ":"))
        with output:
            print(line, flush=True)

    worker = VisionWorker(emit, desktop="--desktop" in sys.argv[1:])
    try:
        emit("ready", preview_fps=0, preview_frames=0, providers=ENGINE_CHOICES)
        for line in sys.stdin:
            try:
                message = json.loads(line)
                command = message["cmd"]
                if command == "exit":
                    break
                if command == "discover":
                    worker.discover()
                elif command == "stream":
                    worker.stream(message["source"], message["name"], message["url"])
                elif command == "model":
                    worker.load(message["model"], message["path"], message["provider"])
                elif command == "sample":
                    worker.sample(message["sample"], message["model"], message["path"])
                elif command == "preview":
                    worker.set_preview(message["fps"], message["preview"])
                elif command == "cuda_check":
                    worker.check_cuda()
            except Exception as ex:
                emit("log", text=str(ex))
    finally:
        worker.close()


if __name__ == "__main__":
    main()
