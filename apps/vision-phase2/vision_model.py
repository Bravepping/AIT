import contextlib
import ctypes
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cuda_check import cuda_dll_folders, cuda_driver


ENGINE_CHOICES = ("Auto", "CUDA", "TensorRT", "TensorRT for RTX", "CPU")


def load_fast_pixels():
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name("VisionFastPixels.dll").resolve()))
        convert = library.vision_rgbx640_to_chw
    except (OSError, AttributeError):
        return None
    convert.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    convert.restype = None
    return convert


FAST_RGBX = load_fast_pixels()


class VisionModel:
    def __init__(self, path, provider):
        self.native = None
        self.session = None
        self.backend_details = {}
        if provider == "Auto":
            failures = []
            for engine in ("TensorRT", "CUDA", "CPU"):
                try:
                    self._load(path, engine)
                except Exception as error:
                    self.close()
                    failures.append({"provider": engine, "error": str(error)})
                else:
                    self.backend_details["previous_attempts"] = failures
                    return
            raise RuntimeError("Auto 无可用引擎：" + "；".join(f"{item['provider']} {item['error']}" for item in failures))
        try:
            self._load(path, provider)
        except Exception:
            self.close()
            raise

    def _load(self, path, provider):
        self.provider = provider
        if provider in ("TensorRT", "TensorRT for RTX"):
            from trt_engine import TensorRTEngine

            self.native = TensorRTEngine(path, provider)
            self.input_name, self.output_name = self.native.input_name, self.native.output_name
            self.description = f"{self.input_name} [1, 3, 640, 640] → {self.output_name} [1, 300, 6]"
            self.backend_details = self.native.details
            self._self_test()
            self.native.save_runtime_cache()
            return

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.backend_details = {"version": ort.__version__, "precision": "FP32"}
        if provider == "CPU":
            self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
            self.session.disable_fallback()
            if self.session.get_providers() != ["CPUExecutionProvider"]:
                raise RuntimeError("ONNX Runtime 未选中 CPU")
            self._read_metadata()
            self._self_test()
        elif provider == "CUDA":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("本 ONNX Runtime 未提供 CUDA Execution Provider")
            cuda_driver()
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            with tempfile.TemporaryDirectory(prefix="vision-ort-", ignore_cleanup_errors=True) as directory:
                options.enable_profiling = True
                options.log_severity_level = 3
                options.profile_file_prefix = str(Path(directory) / "providers")
                try:
                    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as libraries:
                        preload_cuda(ort, libraries)
                        self.session = ort.InferenceSession(path, sess_options=options, providers=["CUDAExecutionProvider"])
                except Exception as ex:
                    raise RuntimeError("CUDA 初始化失败；确认 CUDA 12.8+、cuDNN 9 和 NVIDIA 驱动。"
                                       "如果存在 CPU 节点，严格 CUDA 模式会拒绝该模型。详情：" + str(ex)) from ex
                self.session.disable_fallback()
                if self.session.get_providers()[0] != "CUDAExecutionProvider":
                    raise RuntimeError("CUDA 初始化失败，已拒绝 CPU 回退。请检查 NVIDIA 驱动、CUDA 12.8+ 和 cuDNN 9。")
                self._read_metadata()
                try:
                    self._self_test()
                finally:
                    profile = self.session.end_profiling()
                events = json.loads(Path(profile).read_text(encoding="utf-8"))
                providers = {
                    event.get("args", {}).get("provider")
                    for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider")
                }
                if providers != {"CUDAExecutionProvider"}:
                    raise RuntimeError("CUDA 运行验证未通过，执行节点 provider：" + ", ".join(sorted(providers)))
        else:
            raise ValueError("未知推理引擎")

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
        self.session = None

    def _self_test(self):
        boxes, self.test_ms = self.infer(np.zeros((640, 640, 3), dtype=np.uint8), 0.55)
        self.test_count = len(boxes)

    def _read_metadata(self):
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("需要单输入、单输出的 YOLO26s 模型")
        source, result = inputs[0], outputs[0]
        if (source.type, list(source.shape)) != ("tensor(float)", [1, 3, 640, 640]):
            raise ValueError("不支持的输入：" + str((source.name, source.type, source.shape)))
        if (result.type, list(result.shape)) != ("tensor(float)", [1, 300, 6]):
            raise ValueError("不支持的输出：" + str((result.name, result.type, result.shape)))
        self.input_name, self.output_name = source.name, result.name
        self.description = f"{source.name} {source.shape} → {result.name} {result.shape}"

    def infer(self, rgb, threshold):
        return self.infer_prepared(letterbox(rgb), threshold)

    def infer_prepared(self, prepared, threshold):
        tensor, transform = prepared
        started = time.perf_counter()
        if self.native is None:
            rows, = self.session.run([self.output_name], {self.input_name: tensor})
        else:
            rows = self.native.run(tensor)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if rows.shape != (1, 300, 6) or not np.isfinite(rows).all():
            raise ValueError("模型输出尺寸或数值异常")
        detections = decode(rows[0], threshold, transform)
        return detections, elapsed_ms


def preload_cuda(ort, libraries):
    cuda, cudnn = cuda_dll_folders()
    for path in dict.fromkeys(path for path in (cuda, cudnn) if path):
        libraries.enter_context(os.add_dll_directory(str(path.resolve())))
    if cuda:
        ort.preload_dlls(cuda=True, cudnn=False, msvc=False, directory=str(cuda.resolve()))
    if cudnn:
        ort.preload_dlls(cuda=False, cudnn=True, msvc=False, directory=str(cudnn.resolve()))
    ort.preload_dlls(cuda=cuda is None, cudnn=cudnn is None)


def letterbox(rgb, *, output=None):
    height, width = rgb.shape[:2]
    tensor = output if output is not None else np.empty((1, 3, 640, 640), dtype=np.float32)
    if (FAST_RGBX is not None and rgb.shape == (640, 640, 3) and rgb.dtype == np.uint8
            and rgb.strides == (2560, 4, 1)
            and isinstance(rgb.base, np.ndarray) and rgb.base.flags.owndata and rgb.base.flags.c_contiguous
            and rgb.base.nbytes >= 640 * 640 * 4
            and rgb.ctypes.data == rgb.base.ctypes.data):
        FAST_RGBX(rgb.ctypes.data, tensor.ctypes.data)
        return tensor, (640, 640, 1.0, 1.0, 0, 0)
    scale = min(640 / width, 640 / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (640 - w) // 2, (640 - h) // 2
    resized = rgb if (w, h) == (width, height) else np.asarray(
        Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR))
    if (w, h) == (640, 640):
        padded = resized
    else:
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[top:top + h, left:left + w] = resized
    np.divide(padded.transpose(2, 0, 1), np.float32(255), dtype=np.float32, out=tensor[0])
    return tensor, (width, height, w / width, h / height, left, top)


def decode(rows, threshold, transform):
    width, height, scale_x, scale_y, left, top = transform
    boxes = rows[rows[:, 4] >= threshold].copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / scale_x, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / scale_y, 0, height)
    return boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
