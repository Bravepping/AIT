"""Check the local NVIDIA driver and the DLLs required by the bundled ORT provider."""

import ctypes as c
import hashlib
import os
from contextlib import ExitStack
from pathlib import Path


CUDA_DLLS = ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll")
CUDNN_DLL = "cudnn64_9.dll"


def cuda_dll_folders():
    locations = [os.environ.get("CUDA_PATH"), os.environ.get("CUDNN_PATH")]
    explicit = [path for value in locations if value for path in (Path(value) / "bin", Path(value))]
    runtime = Path(__file__).resolve().parent
    manifest = runtime.parent / "cuda-packages.json"
    try:
        name = "cuda12-" + hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
        bundled = [runtime / "cuda-libraries" / name / "bin"]
    except OSError:
        bundled = []
    paths = [*explicit, *bundled, *(Path(value.strip('"')) for value in os.get_exec_path() if value)]
    preferred = [path for path in dict.fromkeys(paths) if path.is_absolute() and path.is_dir()]
    cuda = next((path for path in preferred if (path / CUDA_DLLS[0]).is_file()), None)
    cudnn = next((path for path in preferred if (path / CUDNN_DLL).is_file()), None)
    return cuda, cudnn


def cuda_driver():
    try:
        return c.WinDLL("nvcuda.dll", winmode=0x800)
    except OSError as ex:
        raise RuntimeError("系统无法加载 NVIDIA 驱动 nvcuda.dll：" + str(ex)) from ex


def driver_devices(driver):
    error_name = driver.cuGetErrorName
    error_name.argtypes = [c.c_int, c.POINTER(c.c_char_p)]

    def call(name, signature, *args):
        function = getattr(driver, name)
        function.argtypes = signature
        result = function(*args)
        if result != 0:
            error = c.c_char_p()
            error_name(result, c.byref(error))
            label = (error.value or b"CUDA_ERROR").decode("ascii", "replace")
            raise RuntimeError(f"{name} 返回 {label} ({result})")

    int_ptr = c.POINTER(c.c_int)
    call("cuInit", [c.c_uint], 0)
    version, count = c.c_int(), c.c_int()
    call("cuDriverGetVersion", [int_ptr], c.byref(version))
    call("cuDeviceGetCount", [int_ptr], c.byref(count))
    devices = []
    for index in range(count.value):
        device = c.c_int()
        name = c.create_string_buffer(256)
        call("cuDeviceGet", [int_ptr, c.c_int], c.byref(device), index)
        call("cuDeviceGetName", [c.POINTER(c.c_char), c.c_int, c.c_int], name, len(name), device)
        devices.append(name.value.decode("utf-8", "replace"))
    return version.value, devices


def inspect_cuda():
    import onnxruntime as ort

    details = [f"ONNX Runtime {ort.__version__}，可用 Provider：{', '.join(ort.get_available_providers())}"]
    cuda, cudnn = cuda_dll_folders()
    details.append("CUDA 12 路径：" + (str(cuda) if cuda else "未在 CUDA_PATH / PATH 找到"))
    details.append("cuDNN 9 路径：" + (str(cudnn) if cudnn else "未在 CUDNN_PATH / CUDA_PATH / PATH 找到"))
    try:
        version, devices = driver_devices(cuda_driver())
    except RuntimeError as ex:
        details.append(str(ex))
        return {"status": "NVIDIA 驱动不可用", "details": details}
    details.append(f"驱动支持的 CUDA API {version // 1000}.{version % 1000 // 10}，"
                   "设备：" + (", ".join(devices) if devices else "无可用 CUDA 设备"))
    if not devices:
        return {"status": "未发现 CUDA 设备", "details": details}
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return {"status": "CUDA Provider 不可用", "details": details}

    capi = Path(ort.__file__).parent / "capi"
    provider_dlls = ("onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll")
    failures = []
    with ExitStack() as libraries:
        for folder in dict.fromkeys(path for path in (capi, cuda, cudnn) if path and path.is_dir()):
            libraries.enter_context(os.add_dll_directory(str(folder)))
        dlls = [(cuda, dll) for dll in CUDA_DLLS] + [(cudnn, CUDNN_DLL)] + [(capi, dll) for dll in provider_dlls]
        for folder, dll in dlls:
            try:
                c.WinDLL(str(folder / dll) if folder else dll)
            except OSError as ex:
                failures.append(f"{dll} 加载失败：{ex}")
    details.extend(failures)
    if failures:
        return {"status": "CUDA 依赖加载失败", "details": details}
    return {"status": "环境自检通过，待模型实际执行", "details": details}
