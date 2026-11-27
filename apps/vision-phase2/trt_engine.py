"""Native TensorRT execution with CUDA buffers and a per-device engine cache."""

import ctypes as c
import hashlib
import importlib
import json
import os
import sys
import tempfile
from contextlib import ExitStack, suppress
from pathlib import Path

import numpy as np

from cuda_check import cuda_dll_folders, cuda_driver


PROVIDERS = {"TensorRT": "tensorrt_bindings", "TensorRT for RTX": "tensorrt_rtx_bindings"}
CACHE_VERSION = 1


def cache_identity():
    driver = cuda_driver()
    driver.cuInit.argtypes = [c.c_uint]
    driver.cuDeviceGet.argtypes = [c.POINTER(c.c_int), c.c_int]
    driver.cuDeviceGetUuid.argtypes = [c.c_void_p, c.c_int]
    driver.cuDeviceGetAttribute.argtypes = [c.POINTER(c.c_int), c.c_int, c.c_int]
    driver.cuDriverGetVersion.argtypes = [c.POINTER(c.c_int)]
    device, version = c.c_int(), c.c_int()
    uuid = (c.c_ubyte * 16)()
    major, minor = c.c_int(), c.c_int()
    for function, args in (
        (driver.cuInit, (0,)),
        (driver.cuDeviceGet, (c.byref(device), 0)),
        (driver.cuDeviceGetUuid, (uuid, device)),
        (driver.cuDeviceGetAttribute, (c.byref(major), 75, device)),
        (driver.cuDeviceGetAttribute, (c.byref(minor), 76, device)),
        (driver.cuDriverGetVersion, (c.byref(version),)),
    ):
        result = function(*args)
        if result != 0:
            raise RuntimeError(f"{function.__name__} 失败 (CUDA Driver {result})")
    return {"gpu": bytes(uuid).hex(), "compute_capability": f"{major.value}.{minor.value}", "driver_api": version.value}


class CudaBuffers:
    def __init__(self, lib, lifetime):
        self.lib = lib
        self.lifetime = lifetime
        self.functions = {}
        self.error_string = lib.cudaGetErrorString
        self.error_string.argtypes = [c.c_int]
        self.error_string.restype = c.c_char_p
        self.call("cudaSetDevice", (c.c_int,), 0)
        stream = c.c_void_p()
        self.call("cudaStreamCreateWithFlags", (c.POINTER(c.c_void_p), c.c_uint), c.byref(stream), 1)
        self.stream = stream
        lifetime.callback(self.call, "cudaStreamDestroy", (c.c_void_p,), stream)

    def call(self, name, types, *args):
        function = self.functions.get(name)
        if function is None:
            function = getattr(self.lib, name)
            function.argtypes = types
            function.restype = c.c_int
            self.functions[name] = function
        result = function(*args)
        if result != 0:
            reason = (self.error_string(result) or b"unknown").decode("ascii", "replace")
            raise RuntimeError(f"{name} 失败 ({result}): {reason}")

    def allocate(self, shape):
        size = int(np.prod(shape)) * c.sizeof(c.c_float)
        gpu, host = c.c_void_p(), c.c_void_p()
        signature = (c.POINTER(c.c_void_p), c.c_size_t)
        self.call("cudaMalloc", signature, c.byref(gpu), size)
        self.lifetime.callback(self.call, "cudaFree", (c.c_void_p,), gpu)
        self.call("cudaMallocHost", signature, c.byref(host), size)
        self.lifetime.callback(self.call, "cudaFreeHost", (c.c_void_p,), host)
        array = np.ctypeslib.as_array((c.c_float * (size // c.sizeof(c.c_float))).from_address(host.value)).reshape(shape)
        return gpu, host, size, array

    def copy(self, target, source, size, direction):
        self.call("cudaMemcpyAsync", (c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p),
                  target, source, size, direction, self.stream)

    def sync(self):
        self.call("cudaStreamSynchronize", (c.c_void_p,), self.stream)


def cache_path(path, provider, version, identity):
    with Path(path).open("rb") as model:
        model_hash = hashlib.file_digest(model, "sha256").hexdigest()
    key = json.dumps([CACHE_VERSION, provider, version, identity, model_hash], sort_keys=True).encode("utf-8")
    folder = os.environ.get("VISION_TRT_CACHE")
    if not folder:
        folder = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "VisionHardwareCheck" / "engine-cache"
    return Path(folder) / (hashlib.sha256(key).hexdigest() + ".engine")


def save_cache(path, data):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        os.replace(temporary, path)
        return "built"
    except OSError as error:
        return f"write_failed: {error}"
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


class TensorRTEngine:
    def __init__(self, path, provider):
        self.lifetime = ExitStack()
        self.context = self.engine = self.runtime = None
        self.runtime_config = self.runtime_cache = None
        try:
            self._load(path, provider)
        except Exception:
            with suppress(RuntimeError):
                self.close()
            raise

    def _load(self, path, provider):
        self.fp16 = provider == "TensorRT"
        identity = cache_identity()
        cuda, _ = cuda_dll_folders()
        if cuda is None:
            raise RuntimeError(f"{provider} 需要 CUDA 12 运行库。请运行 Start-Vision-CUDA.cmd")
        self.lifetime.enter_context(os.add_dll_directory(str(cuda)))
        runtime = Path(__file__).resolve().parent
        native_packages = runtime / "trt-packages"
        if native_packages.is_dir() and str(native_packages) not in sys.path:
            sys.path.insert(0, str(native_packages))
        site = native_packages if native_packages.is_dir() else runtime / "site-packages"
        libraries = site / ("tensorrt_libs" if self.fp16 else "tensorrt_rtx_libs")
        if libraries.is_dir():
            self.lifetime.enter_context(os.add_dll_directory(str(libraries)))
        try:
            trt = importlib.import_module(PROVIDERS[provider])
        except (ImportError, OSError) as error:
            raise RuntimeError(f"{provider} 运行库无法加载。请安装本包对应的 NVIDIA 运行组件。详情：{error}") from error
        self.cuda = CudaBuffers(c.WinDLL(str(cuda / "cudart64_12.dll")), self.lifetime)
        version = trt.__version__
        self.details = {"provider": provider, "version": version, "precision": "FP16 mixed" if self.fp16 else "FP32 original ONNX",
                        **identity}

        errors = []

        class Logger(trt.ILogger):
            def __init__(self):
                trt.ILogger.__init__(self)

            def log(self, severity, message):
                if severity <= trt.ILogger.Severity.ERROR:
                    errors.append(message)

        self.logger = Logger()
        self.runtime = trt.Runtime(self.logger)
        cache = cache_path(path, provider, version, identity)
        try:
            self.engine = self.runtime.deserialize_cuda_engine(cache.read_bytes())
        except (OSError, RuntimeError):
            pass
        needs_build = self.engine is None
        if needs_build:
            errors.clear()
            serialized = self._build(trt, path, errors)
            self.engine = self.runtime.deserialize_cuda_engine(serialized)
            if self.engine is None:
                raise RuntimeError("TensorRT 引擎反序列化失败：" + "；".join(errors[-4:]))

        if self.engine.num_io_tensors != 2:
            raise ValueError("TensorRT 需要恰好一个图像输入和一个检测输出")
        names = [self.engine.get_tensor_name(i) for i in range(2)]
        self.input_name, self.output_name = sorted(names, key=lambda name: self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT)
        for name, expected in ((self.input_name, (1, 3, 640, 640)), (self.output_name, (1, 300, 6))):
            shape = tuple(self.engine.get_tensor_shape(name))
            if (shape != expected or self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError(f"不支持的 TensorRT 张量 {name} {shape}；需要线性 FP32 {expected}")
        if self.engine.get_tensor_mode(self.output_name) != trt.TensorIOMode.OUTPUT:
            raise ValueError("TensorRT 输出张量类型异常")

        if self.fp16:
            self.context = self.engine.create_execution_context()
        else:
            self.jit_path = cache.with_suffix(".jit")
            self.runtime_config = self.engine.create_runtime_config()
            self.runtime_config.set_execution_context_allocation_strategy(trt.ExecutionContextAllocationStrategy.STATIC)
            self.runtime_cache = self.runtime_config.create_runtime_cache()
            self.details["jit_cache"] = "miss"
            try:
                if self.runtime_cache.deserialize(self.jit_path.read_bytes()):
                    self.details["jit_cache"] = "hit"
            except (OSError, RuntimeError):
                pass
            if not self.runtime_config.set_runtime_cache(self.runtime_cache):
                raise RuntimeError("TensorRT for RTX 无法设置运行时缓存")
            self.context = self.engine.create_execution_context(self.runtime_config)
        if self.context is None:
            raise RuntimeError("TensorRT 执行上下文创建失败：" + "；".join(errors[-4:]))
        self.input = self.cuda.allocate((1, 3, 640, 640))
        self.output = self.cuda.allocate((1, 300, 6))
        for name, (gpu, _, _, _) in ((self.input_name, self.input), (self.output_name, self.output)):
            if not self.context.set_tensor_address(name, gpu.value):
                raise RuntimeError(f"TensorRT 无法绑定 {name}")

        self.details["engine_layers"] = self.engine.num_layers
        self.details["cache"] = save_cache(cache, serialized) if needs_build else "hit"
        self.details["cache_file"] = str(cache)
        if self.fp16:
            inspector = self.engine.create_engine_inspector()
            layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON)).get("Layers", [])
            fp16_layers = sum(any(any(term in str(tensor.get("Format/Datatype", "")).lower()
                                      for term in ("fp16", "half"))
                                  for tensor in layer.get("Inputs", []) + layer.get("Outputs", []))
                              for layer in layers if isinstance(layer, dict))
            if not fp16_layers:
                formats = sorted({tensor.get("Format/Datatype", "") for layer in layers if isinstance(layer, dict)
                                  for tensor in layer.get("Inputs", []) + layer.get("Outputs", [])})
                raise RuntimeError(f"TensorRT 引擎检查未发现 FP16 张量；检测到的数据格式：{formats}")
            self.details["fp16_layers"] = fp16_layers

    def _build(self, trt, path, errors):
        builder = trt.Builder(self.logger)
        flags = 0 if self.fp16 else 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(path)):
            detail = "；".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("ONNX 算子解析失败：" + detail)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
        if self.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT 引擎构建失败：" + "；".join(errors[-4:]))
        return bytes(serialized)

    def run(self, tensor):
        input_gpu, input_host, input_size, staging = self.input
        output_gpu, output_host, output_size, result = self.output
        np.copyto(staging, tensor)
        try:
            self.cuda.copy(input_gpu, input_host, input_size, 1)
            if not self.context.execute_async_v3(self.cuda.stream.value):
                raise RuntimeError("TensorRT execute_async_v3 执行失败")
            self.cuda.copy(output_host, output_gpu, output_size, 2)
        finally:
            self.cuda.sync()
        return result

    def save_runtime_cache(self):
        if self.runtime_cache is not None and self.details["jit_cache"] != "hit":
            blob = self.runtime_cache.serialize()
            data = bytes(blob) if blob is not None else b""
            self.details["jit_cache"] = save_cache(self.jit_path, data) if data else "empty"

    def close(self):
        self.context = None
        self.runtime_config = self.runtime_cache = None
        self.engine = self.runtime = None
        with suppress(RuntimeError):
            self.lifetime.close()
