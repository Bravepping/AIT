"""Compare 240 FPS nominal NDI replay across two versions of the vision worker."""

import argparse
from contextlib import nullcontext
import json
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "vision-phase2"))
from test_phase2 import LocalSender


def quantile(values, fraction):
    ranked = sorted(values)
    return ranked[max(0, int(np.ceil(len(ranked) * fraction)) - 1)]


def test(runtime, model, source_name, preview, seconds):
    events = queue.Queue()
    with subprocess.Popen([str(runtime / "python.exe"), "-B", "-X", "utf8", "-u",
                           str(runtime / "worker.py"), "--desktop"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8") as worker:
        def read():
            for line in worker.stdout:
                event = json.loads(line)
                if event["kind"] != "preview":
                    events.put((time.perf_counter(), time.time_ns() / 1e6, event))
            events.put((time.perf_counter(), 0, {"kind": "exit"}))

        def send(cmd, **fields):
            worker.stdin.write(json.dumps({"cmd": cmd, **fields}) + "\n")
            worker.stdin.flush()

        def wait(kind, seconds=10):
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                _, _, event = events.get(timeout=max(0.01, deadline - time.perf_counter()))
                if event["kind"] == kind:
                    return event
                if event["kind"] == "exit":
                    raise RuntimeError("Worker exited")
            raise TimeoutError(kind)

        threading.Thread(target=read, daemon=True).start()
        try:
            wait("ready")
            send("model", path=str(model), model=1, provider="TensorRT")
            loaded = wait("model", seconds=360)
            assert loaded.get("provider") == "TensorRT", loaded
            selected = None
            for _ in range(3):
                send("discover")
                sources = wait("sources")
                matches = [item for item in sources["sources"] if item[0] == source_name or
                           (source_name.startswith("VisionHardwareCheck Test ") and source_name in item[0])]
                selected = matches[0] if len(matches) == 1 else None
                if selected:
                    break
            assert selected, sources
            send("stream", source=1, name=selected[0], url=selected[1])
            wait("stream_state")
            send("preview", fps=preview, preview=1)
            assert wait("preview_state")["fps"] == preview
            start = time.perf_counter() + 3
            deadline = start + seconds
            latencies, inf, stream = [], [], []
            detections = None
            while True:
                clock, wall, event = events.get(timeout=8)
                kind = event["kind"]
                if kind == "exit" or event.get("error") or event.get("state") == "lost":
                    raise RuntimeError(event)
                if clock < start:
                    continue
                if clock >= deadline:
                    break
                if kind == "detections":
                    latencies.append(wall - event["frame_at"])
                    detections = event["boxes"]
                elif kind == "inference" and event.get("warm_samples", 0) > 1:
                    inf.append(event)
                elif kind == "stream":
                    stream.append(event)
            assert latencies and inf and stream and detections is not None
            median = lambda series, key: round(statistics.median(item[key] for item in series if key in item), 2)
            result = {"input_fps": median(stream, "fps"), "inference_fps": round(len(latencies) / seconds, 2),
                      "preview_fps": round((stream[-1]["preview_frames"] - stream[0]["preview_frames"]) /
                                           max(1, len(stream) - 1), 2),
                      "receiver_avg": median(inf, "receiver_avg"), "receiver_p95": median(inf, "receiver_p95"),
                      "receiver_p99": median(inf, "receiver_p99"), "model_avg": median(inf, "model_avg"),
                      "receiver_to_ipc_avg": round(statistics.mean(latencies), 2),
                      "receiver_to_ipc_p95": round(quantile(latencies, 0.95), 2), "boxes": detections}
            for key in ("before_preprocess_avg", "preprocess_avg", "prepared_wait_avg"):
                if key in inf[-1]:
                    result[key] = median(inf, key)
            if "reselected" in inf[-1] and inf[-1]["runs"] > inf[0]["runs"]:
                result["reselected_percent"] = round(
                    100 * (inf[-1]["reselected"] - inf[0]["reselected"]) / (inf[-1]["runs"] - inf[0]["runs"]), 2)
            return result
        finally:
            if worker.poll() is None:
                send("exit")
                try:
                    worker.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "model", "image", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--source", help="Exact existing NDI source name. Skip fixed-frame replay.")
    parser.add_argument("--preview", type=int, choices=(0, 60), nargs="+", default=[0, 60])
    config = parser.parse_args()
    sender = nullcontext()
    if config.source is None:
        with Image.open(config.image) as image:
            frame = np.asarray(ImageOps.pad(image.convert("RGB"), (640, 640), method=Image.Resampling.BILINEAR,
                                            color=(114, 114, 114)))
        sender = LocalSender(frame, rate=240, paced=True)
    paths = {"baseline": config.baseline.resolve(), "candidate": config.candidate.resolve()}
    results = []
    with sender as local:
        source_name = config.source if config.source is not None else local.name
        for preview in config.preview:
            for repeat in range(config.repeats):
                order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
                for name in order:
                    result = test(paths[name], config.model.resolve(), source_name, preview, config.seconds)
                    result.update(variant=name, preview=preview, repeat=repeat + 1)
                    results.append(result)
                    print(json.dumps({key: value for key, value in result.items() if key != "boxes"}), flush=True)
    parity = None
    if config.source is None:
        reference = np.array(results[0]["boxes"])
        parity = bool(reference.size and all(np.array(item["boxes"]).shape == reference.shape and
                      np.allclose(np.array(item["boxes"]), reference, rtol=0, atol=1e-4) for item in results))
    source = f"real NDI: {config.source}" if config.source else "local NDI SDK fixed-frame replay"
    report = {"source": source, "gui_rendering": False,
              "box_parity": parity, "seconds_per_run": config.seconds, "runs": results}
    summary = []
    for preview in config.preview:
        for name in paths:
            group = [item for item in results if item["variant"] == name and item["preview"] == preview]
            metrics = {key: round(statistics.mean(item[key] for item in group), 2) for key in (
                "input_fps", "inference_fps", "receiver_avg", "receiver_p95", "receiver_p99",
                "model_avg", "receiver_to_ipc_avg", "receiver_to_ipc_p95", "before_preprocess_avg",
                "preprocess_avg", "prepared_wait_avg") if all(key in item for item in group)}
            summary.append({"variant": name, "preview": preview, **metrics})
    report["summary_averaged_across_runs"] = summary
    config.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if parity is False:
        raise AssertionError("Detection boxes changed")


if __name__ == "__main__":
    main()
