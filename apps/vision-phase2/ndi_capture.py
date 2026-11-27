"""Bindings for the NDI 6 Windows runtime. Native frame buffers never escape capture."""

import ctypes as c
import os
import time
from pathlib import Path

import numpy as np

NO_VIDEO_SECONDS = 0.75
RGB_FORMATS = (int.from_bytes(b"RGBX", "little"), int.from_bytes(b"RGBA", "little"))


class Source(c.Structure):
    _fields_ = [("name", c.c_char_p), ("url", c.c_char_p)]


class FindConfig(c.Structure):
    _fields_ = [("show_local", c.c_bool), ("groups", c.c_char_p), ("extra_ips", c.c_char_p)]


class ReceiveConfig(c.Structure):
    _fields_ = [
        ("source", Source), ("color", c.c_int), ("bandwidth", c.c_int),
        ("allow_fields", c.c_bool), ("name", c.c_char_p),
    ]


class Video(c.Structure):
    _fields_ = [
        ("width", c.c_int), ("height", c.c_int), ("fourcc", c.c_uint32),
        ("rate_n", c.c_int), ("rate_d", c.c_int), ("aspect", c.c_float),
        ("format", c.c_int), ("timecode", c.c_int64), ("data", c.c_void_p),
        ("stride", c.c_int), ("metadata", c.c_char_p), ("timestamp", c.c_int64),
    ]


class Performance(c.Structure):
    _fields_ = [("video", c.c_int64), ("audio", c.c_int64), ("metadata", c.c_int64)]


def bind(library, name, result, *arguments):
    function = getattr(library, "NDIlib_" + name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


class NdiRuntime:
    def __init__(self):
        dll = "Processing.NDI.Lib.x64.dll"
        roots = [
            os.environ.get("NDI_RUNTIME_DIR_V6"),
            str(Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NDI" / "NDI 6 Runtime" / "v6"),
        ]
        path = next((Path(root) / dll for root in roots if root and (Path(root) / dll).is_file()), None)
        if path is None:
            raise FileNotFoundError("设备 B 需要安装 NDI 6 Runtime（https://ndi.link/NDIRedistV6）")
        self.library = c.CDLL(str(path))
        lib = self.library
        self.shutdown = bind(lib, "destroy", None)
        self.find_create = bind(lib, "find_create_v2", c.c_void_p, c.POINTER(FindConfig))
        self.find_wait = bind(lib, "find_wait_for_sources", c.c_bool, c.c_void_p, c.c_uint32)
        self.find_sources = bind(lib, "find_get_current_sources", c.POINTER(Source), c.c_void_p, c.POINTER(c.c_uint32))
        self.find_destroy = bind(lib, "find_destroy", None, c.c_void_p)
        self.recv_create = bind(lib, "recv_create_v3", c.c_void_p, c.POINTER(ReceiveConfig))
        self.recv_capture = bind(lib, "recv_capture_v3", c.c_int, c.c_void_p, c.POINTER(Video), c.c_void_p, c.c_void_p, c.c_uint32)
        self.recv_free = bind(lib, "recv_free_video_v2", None, c.c_void_p, c.POINTER(Video))
        self.recv_performance = bind(lib, "recv_get_performance", None, c.c_void_p, c.POINTER(Performance), c.POINTER(Performance))
        self.recv_destroy = bind(lib, "recv_destroy", None, c.c_void_p)
        if not bind(lib, "initialize", c.c_bool)():
            raise RuntimeError("NDI 6 Runtime 初始化失败")

    def discover(self, duration=1.6):
        handle = self.find_create(c.byref(FindConfig(True, None, None)))
        if not handle:
            raise RuntimeError("无法创建 NDI 源发现器")
        try:
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                self.find_wait(handle, 100)
            count = c.c_uint32()
            sources = self.find_sources(handle, c.byref(count))
            return [(sources[i].name.decode("utf-8"), (sources[i].url or b"").decode("utf-8")) for i in range(count.value)]
        finally:
            self.find_destroy(handle)

    def receive(self, name, url, stop, on_frame, on_status, on_state):
        config = ReceiveConfig(
            Source(name.encode("utf-8"), url.encode("utf-8") if url else None),
            2, 100, False, b"VisionHardwareCheck",
        )
        handle = self.recv_create(c.byref(config))
        if not handle:
            raise RuntimeError("无法创建 NDI 接收器")
        last_status = time.monotonic()
        last_frame = None
        stream_state = "waiting"
        frames = 0
        capture_wait_ns = copy_ns = 0
        total, dropped = Performance(), Performance()
        width = height = 0
        nominal = 0
        video = Video()
        try:
            while not stop.is_set():
                waiting_since = time.perf_counter_ns()
                kind = self.recv_capture(handle, c.byref(video), None, None, 80)
                if kind == 4:
                    raise RuntimeError("NDI 接收器返回错误")
                if kind == 1:
                    captured_at = time.perf_counter_ns()
                    capture_wait_ns += captured_at - waiting_since
                    received_at = time.time_ns() // 1_000_000
                    try:
                        rgb = copy_rgb(video)
                        width, height = video.width, video.height
                        nominal = video.rate_n / video.rate_d if video.rate_d else 0
                    finally:
                        self.recv_free(handle, c.byref(video))
                    copy_ns += time.perf_counter_ns() - captured_at
                    last_frame = time.monotonic()
                    if stream_state != "live":
                        on_state("resumed" if stream_state == "lost" else "live", received_at, 0)
                        stream_state = "live"
                    on_frame(rgb, received_at)
                    frames += 1
                now = time.monotonic()
                if stream_state == "live" and now - last_frame >= NO_VIDEO_SECONDS:
                    stream_state = "lost"
                    on_state("lost", 0, round((now - last_frame) * 1000))
                if now - last_status >= 1:
                    self.recv_performance(handle, c.byref(total), c.byref(dropped))
                    on_status(width, height, nominal, frames / (now - last_status), int(dropped.video),
                              capture_wait_ns / frames / 1e6 if frames else None,
                              copy_ns / frames / 1e6 if frames else None)
                    last_status, frames = now, 0
                    capture_wait_ns = copy_ns = 0
        finally:
            self.recv_destroy(handle)


def copy_rgb(video):
    width, height, pitch = video.width, video.height, video.stride
    if video.fourcc not in RGB_FORMATS or not video.data or not 0 < width <= 8192 or not 0 < height <= 8192:
        raise ValueError("NDI 视频帧格式或尺寸无效")
    if not width * 4 <= pitch <= width * 4 + 65536 or height * pitch > 512 * 1024 * 1024:
        raise ValueError("NDI 视频帧 stride 无效")
    raw = np.ctypeslib.as_array((c.c_uint8 * (height * pitch)).from_address(video.data))
    # Own the SDK frame before freeing it. Copy packed pixels first so we do not
    # extract interleaved RGB bytes in the capture thread.
    owned = raw.reshape(height, pitch)[:, :width * 4].copy()
    return owned.reshape(height, width, 4)[:, :, :3]
