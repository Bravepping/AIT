"""Compare warmed inference backends in separate processes using identical image files."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time


ENGINES = ("Auto", "CUDA", "TensorRT", "TensorRT for RTX", "CPU")
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp"}
RESULT_PREFIX = "VISION_BENCHMARK_RESULT "
GPU_QUERY = "index,name,driver_version,utilization.gpu,memory.used,memory.total"
MIN_IOU = 0.5
MAX_CONFIDENCE_DELTA = 0.10


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def frame_paths(inputs):
    paths = []
    for item in inputs:
        path = Path(item).resolve(strict=True)
        if path.is_dir():
            paths.extend(sorted(child for child in path.iterdir()
                                if child.is_file() and child.suffix.lower() in IMAGE_TYPES))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("未找到 PNG、JPEG 或 BMP 测试图片")
    return paths


def timing(values):
    ranked = sorted(values)
    return {
        "average_ms": round(statistics.mean(ranked), 3),
        "p95_ms": round(ranked[math.ceil(len(ranked) * 0.95) - 1], 3),
    }


def measure(config):
    sys.path.insert(0, config["runtime"])
    import numpy as np
    from PIL import Image, ImageOps
    from vision_model import VisionModel
    from worker import CONFIDENCE_THRESHOLD, nearest_offset

    frames = []
    for entry in config["inputs"]:
        path = Path(entry["path"])
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != entry["sha256"]:
            raise ValueError(f"测试文件在测试期间发生变化：{path}")
        with Image.open(io.BytesIO(encoded), formats=("PNG", "JPEG", "BMP")) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError(f"图片超过 1600 万像素：{path}")
            frames.append(np.asarray(ImageOps.exif_transpose(image).convert("RGB")))

    started = time.perf_counter()
    model = VisionModel(config["model"], config["engine"])
    load_ms = (time.perf_counter() - started) * 1000
    if config["engine"] != "Auto" and model.provider != config["engine"]:
        raise RuntimeError(f"请求 {config['engine']}，实际执行 {model.provider}")

    def infer(frame):
        boxes, model_ms = model.infer(frame, CONFIDENCE_THRESHOLD)
        height, width = frame.shape[:2]
        return boxes, model_ms, nearest_offset(boxes, width, height)

    for index in range(config["warmup"]):
        infer(frames[index % len(frames)])

    model_times, pipeline_times = [], []
    measured_start_ns = time.time_ns()
    cpu_start = time.process_time_ns()
    wall_start = time.perf_counter_ns()
    for index in range(config["runs"]):
        frame = frames[index % len(frames)]
        start = time.perf_counter_ns()
        _, model_ms, _ = infer(frame)
        pipeline_times.append((time.perf_counter_ns() - start) / 1e6)
        model_times.append(model_ms)
    wall_duration = time.perf_counter_ns() - wall_start
    cpu_duration = time.process_time_ns() - cpu_start
    measured_end_ns = time.time_ns()

    detections = []
    for frame in frames:
        boxes, _, offset = infer(frame)
        detections.append({"boxes": boxes.tolist(), "offset_px": offset})
    logical_cpus = os.cpu_count()
    return {
        "status": "ok",
        "actual_engine": model.provider,
        "backend_details": getattr(model, "backend_details", {}),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "load_and_self_test_ms": round(load_ms, 3),
        "model": timing(model_times),
        "preprocess_infer_decode_offset": timing(pipeline_times),
        "inference_fps": round(config["runs"] * 1e9 / wall_duration, 2),
        "process_cpu_one_logical_core_percent": round(cpu_duration * 100 / wall_duration, 2),
        "process_cpu_host_capacity_percent": (
            round(cpu_duration * 100 / (wall_duration * logical_cpus), 2) if logical_cpus else None
        ),
        "measured_start_ns": measured_start_ns,
        "measured_end_ns": measured_end_ns,
        "detections": detections,
    }


class GpuSamples:
    """Collect device-wide NVSMI samples while the benchmark child runs."""

    def __init__(self):
        self.process = None
        self.thread = None
        self.samples = []
        self.message = None

    def __enter__(self):
        executable = shutil.which("nvidia-smi")
        if not executable:
            self.message = "nvidia-smi unavailable"
            return self
        try:
            self.process = subprocess.Popen(
                [executable, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits", "--loop-ms=100"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            )
            self.thread = threading.Thread(target=self.read, daemon=True)
            self.thread.start()
        except OSError as failure:
            self.message = str(failure)
        return self

    def read(self):
        for row in csv.reader(self.process.stdout, skipinitialspace=True):
            if len(row) == 6 and row[0].isdigit():
                self.samples.append((time.time_ns(), row))

    def __exit__(self, *_):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait()
            self.thread.join()
            self.process.stdout.close()

    def summarize(self, start, end):
        devices = {}
        for received_at, row in self.samples:
            if start <= received_at <= end:
                devices.setdefault(row[0], []).append(row)
        summaries = []
        for index, rows in sorted(devices.items()):
            utilization = numbers(rows, 3)
            memory = numbers(rows, 4)
            total = numbers(rows, 5)
            summaries.append({
                "gpu_index": int(index),
                "name": rows[0][1],
                "driver": rows[0][2],
                "samples": len(rows),
                "utilization_average_percent": round(statistics.mean(utilization), 2) if utilization else None,
                "memory_used_average_mib": round(statistics.mean(memory), 2) if memory else None,
                "memory_used_peak_mib": max(memory) if memory else None,
                "memory_total_mib": max(total) if total else None,
            })
        return {
            "scope": "device-wide; includes other processes",
            "sample_interval_ms": 100,
            "devices": summaries,
            "unavailable": None if summaries else (self.message or "no samples during the measured window"),
        }


def numbers(rows, index):
    values = []
    for row in rows:
        try:
            values.append(float(row[index]))
        except ValueError:
            pass
    return values


def run_engine(config, timeout):
    engine = config["engine"]
    command = [sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()), "--worker"]
    stdout, stderr = "", ""
    try:
        with GpuSamples() as gpu:
            result = subprocess.run(
                command, input=json.dumps(config), encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            )
        stdout, stderr = result.stdout, result.stderr
        payload = next((line.removeprefix(RESULT_PREFIX) for line in reversed(stdout.splitlines())
                        if line.startswith(RESULT_PREFIX)), None)
        if result.returncode != 0 or payload is None:
            raise RuntimeError(f"引擎进程退出码 {result.returncode}，未返回检测数据")
        entry = json.loads(payload)
        if entry["status"] == "ok":
            entry["gpu_device_usage"] = gpu.summarize(entry.pop("measured_start_ns"), entry.pop("measured_end_ns"))
    except subprocess.TimeoutExpired:
        entry = {"status": "error", "error": f"超时（{timeout} 秒），请检查引擎构建和驱动日志"}
    except (OSError, RuntimeError, ValueError) as failure:
        entry = {"status": "error", "error": str(failure)}
    if entry["status"] == "error" and stderr.strip():
        entry["diagnostic"] = stderr.strip()[-3000:]
    return {"requested_engine": engine, **entry}


def iou(first, second):
    width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = width * height
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - overlap)
    return overlap / union if union > 0 else 0


def compare_frame(reference, candidate):
    expected, actual = reference["boxes"], candidate["boxes"]
    pairs = sorted(
        ((iou(left, right), index, other) for index, left in enumerate(expected)
         for other, right in enumerate(actual) if left[5] == right[5]), reverse=True,
    )
    matches, unmatched_expected, unmatched_actual = [], set(range(len(expected))), set(range(len(actual)))
    for overlap, index, other in pairs:
        if overlap < MIN_IOU:
            break
        if index not in unmatched_expected or other not in unmatched_actual:
            continue
        unmatched_expected.remove(index)
        unmatched_actual.remove(other)
        matches.append({
            "reference_box": index,
            "candidate_box": other,
            "iou": round(overlap, 5),
            "confidence_delta": abs(expected[index][4] - actual[other][4]),
        })
    shift = None
    if reference["offset_px"] is not None and candidate["offset_px"] is not None:
        shift = round(math.dist(reference["offset_px"], candidate["offset_px"]), 3)
    consistent = not unmatched_expected and not unmatched_actual and all(
        match["confidence_delta"] <= MAX_CONFIDENCE_DELTA for match in matches
    )
    return {
        "status": "pass" if consistent else "mismatch",
        "matches": matches,
        "missing_reference_boxes": sorted(unmatched_expected),
        "extra_candidate_boxes": sorted(unmatched_actual),
        "offset_delta_px": shift,
    }


def compare_results(results, reference):
    baseline = next((entry for entry in results
                     if entry["requested_engine"] == reference and entry["status"] == "ok"), None)
    comparisons = []
    if baseline is not None:
        for entry in results:
            if entry is baseline or entry["status"] != "ok":
                continue
            frames = [compare_frame(want, got) for want, got in zip(baseline["detections"], entry["detections"], strict=True)]
            comparisons.append({
                "engine": entry["requested_engine"],
                "status": "pass" if all(frame["status"] == "pass" for frame in frames) else "mismatch",
                "frames": frames,
            })
    return {
        "reference": reference,
        "status": "available" if baseline is not None else "reference engine unavailable",
        "minimum_iou": MIN_IOU,
        "maximum_confidence_delta": MAX_CONFIDENCE_DELTA,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", required=True, help="Image files or directories with extracted frames.")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=["CPU", "CUDA", "TensorRT", "TensorRT for RTX"])
    parser.add_argument("--reference", choices=ENGINES, default="CUDA")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=900, help="Timeout in seconds for each engine, including engine build.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the JSON report.")
    args = parser.parse_args()
    if min(args.warmup, args.runs, args.timeout) < 1:
        parser.error("--warmup, --runs and --timeout must be positive")
    try:
        frames = frame_paths(args.frames)
        model_path = args.model.resolve(strict=True)
        config = {
            "runtime": str(args.runtime.resolve(strict=True)),
            "model": str(model_path),
            "inputs": [{"path": str(path), "sha256": fingerprint(path)} for path in frames],
            "warmup": args.warmup,
            "runs": args.runs,
        }
        model_sha = fingerprint(model_path)
    except (OSError, ValueError) as failure:
        parser.error(str(failure))
    if args.runs < len(frames):
        parser.error("--runs must cover every input image at least once")

    results = []
    for engine in dict.fromkeys(args.engines):
        if fingerprint(model_path) != model_sha:
            parser.error("ONNX model changed during the benchmark")
        print(f"Testing {engine} ({args.warmup} warmup, {args.runs} measured)", flush=True)
        entry = run_engine({**config, "engine": engine}, args.timeout)
        results.append(entry)
        if entry["status"] == "ok":
            measured = entry["preprocess_infer_decode_offset"]
            print(f"  {entry['actual_engine']}: {measured['average_ms']} ms avg, "
                  f"{measured['p95_ms']} ms P95, {entry['inference_fps']} FPS", flush=True)
        else:
            print(f"  Error: {entry['error']}", flush=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "logical_processors": os.cpu_count(),
        "input_mode": "fixed images; no NDI capture or preview",
        "model": {"name": model_path.name, "sha256": model_sha},
        "frames": [{"name": path.name, "sha256": entry["sha256"]}
                   for path, entry in zip(frames, config["inputs"], strict=True)],
        "warmup_iterations": args.warmup,
        "measured_iterations": args.runs,
        "p95_method": "nearest rank",
        "results": results,
        "detection_parity": compare_results(results, args.reference),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output}", flush=True)
    return 0 if any(entry["status"] == "ok" for entry in results) else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        try:
            outcome = measure(json.load(sys.stdin))
        except Exception as failure:
            outcome = {"status": "error", "error": str(failure)}
        print(RESULT_PREFIX + json.dumps(outcome, allow_nan=False), flush=True)
    else:
        sys.exit(main())
