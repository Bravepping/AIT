"""Pause and resume only a generated local NDI source with the controller armed."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "vision-phase2"))
from test_phase2 import LocalSender
from ndi_capture import NdiRuntime


def verify(application, image_path, marker, output):
    with Image.open(image_path) as image:
        frame = np.asarray(ImageOps.pad(image.convert("RGB"), (640, 640),
                                        method=Image.Resampling.BILINEAR, color=(114, 114, 114)))
    with LocalSender(frame=frame, rate=60, paced=True) as sender:
        runtime = NdiRuntime()
        try:
            source = next(name for name, _ in runtime.discover() if sender.name in name)
        finally:
            runtime.shutdown()
        with subprocess.Popen([str(application / "VisionStudio.exe"), "--verify-ndi-watchdog",
                               str(marker), source, str(output)], creationflags=subprocess.CREATE_NO_WINDOW) as test:
            try:
                deadline = time.monotonic() + 40
                while not marker.is_file() and test.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not marker.is_file() or marker.read_text() != "DRY_RUN_RUNNING":
                    raise RuntimeError("Controller did not produce a Dry Run candidate for the test frame")
                sender.paused.set()
                deadline = time.monotonic() + 5
                while marker.read_text() != "NDI_LOST" and test.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                if marker.read_text() != "NDI_LOST":
                    raise RuntimeError("Controller did not report the SDK dropout")
                sender.paused.clear()
                if test.wait(timeout=8) != 0:
                    raise RuntimeError("Watchdog test failed")
            except RuntimeError as error:
                details = application / "VisionStudio.test-error.txt"
                if details.is_file():
                    raise RuntimeError(details.read_text(encoding="utf-8")) from error
                raise
            finally:
                if test.poll() is None:
                    test.kill()
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["queued_candidate_cleared"]
        assert report["rearm_required"] and report["resumed_after_manual_rearm"]
        assert 0 < report["observed_frame_age_at_lock_ms"] < 220 < report["sdk_gap_ms"], report
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    verify(*(Path(argument).resolve() for argument in sys.argv[1:]))
