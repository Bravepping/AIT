"""Install the pinned NVIDIA runtime alongside the portable HardwareCheck app."""

import argparse
import hashlib
import json
import msvcrt
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath


CHUNK = 1024 * 1024
PRIMARY_DLLS = {
    "cudart64_12.dll", "cublaslt64_12.dll", "cublas64_12.dll",
    "cufft64_11.dll", "cudnn64_9.dll", "nvjitlink_120_0.dll",
}


def verified(path, package):
    if not path.is_file() or path.stat().st_size != package["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == package["sha256"]


def get_wheel(cache, package):
    path = cache / package["filename"]
    if verified(path, package):
        print(f"已校验缓存 {package['name']} {package['version']}", flush=True)
        return path

    partial = path.with_suffix(".part")
    whole = package["size"]
    print(f"下载 {package['name']} {package['version']} ({whole / 1e6:.1f} MB)", flush=True)
    digest = hashlib.sha256()
    total = 0
    next_report = 64 * CHUNK
    with urllib.request.urlopen(package["url"], timeout=45) as response, partial.open("wb") as stream:
        for block in iter(lambda: response.read(CHUNK), b""):
            total += len(block)
            if total > whole:
                raise ValueError(f"{path.name} 大小异常")
            stream.write(block)
            digest.update(block)
            if total >= next_report:
                print(f"  {total / 1e6:.0f}/{whole / 1e6:.0f} MB", flush=True)
                next_report += 64 * CHUNK
    if total != whole or digest.hexdigest() != package["sha256"]:
        raise ValueError(f"{path.name} SHA256 或大小校验失败")
    partial.replace(path)
    return path


def extract_runtime(wheel, staging, files, package_name):
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename)
            if entry.is_dir():
                continue
            if name.suffix.lower() == ".dll" and name.parts[0] == "nvidia" and name.parent.name == "bin":
                basename = name.name.lower()
                if basename in files:
                    raise ValueError(f"运行库重名：{name.name}")
                target = staging / "bin" / name.name
                files[basename] = entry.file_size
            elif "license" in name.name.lower() or name.name.lower().startswith("eula"):
                target = staging / "licenses" / package_name / name.name
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                continue
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, CHUNK)


def ready(directory):
    try:
        files = json.loads((directory / "installed.json").read_text(encoding="utf-8"))
        return PRIMARY_DLLS <= files.keys() and all(
            (directory / "bin" / name).stat().st_size == size for name, size in files.items()
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def install(application, lockfile, test_cache=None):
    package_data = lockfile.read_bytes()
    packages = json.loads(package_data)
    runtime = application / "vision-runtime" / "cuda-libraries"
    runtime.mkdir(parents=True, exist_ok=True)
    directory = runtime / ("cuda12-" + hashlib.sha256(package_data).hexdigest()[:12])
    cache = test_cache or runtime / "downloads"
    cache.mkdir(parents=True, exist_ok=True)

    with (runtime / "setup.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as ex:
            raise RuntimeError("另一个 CUDA 安装窗口正在运行") from ex
        if ready(directory):
            print("已找到本程序的 CUDA 运行库。", flush=True)
            return directory / "bin"
        with tempfile.TemporaryDirectory(dir=runtime, prefix="staging-") as temp:
            staging = Path(temp) / "runtime"
            (staging / "bin").mkdir(parents=True)
            files = {}
            for package in packages:
                wheel = get_wheel(cache, package)
                extract_runtime(wheel, staging, files, package["name"])
            if not PRIMARY_DLLS <= files.keys():
                raise ValueError("解压后缺少运行库：" + ", ".join(sorted(PRIMARY_DLLS - files.keys())))
            (staging / "installed.json").write_text(json.dumps(files, sort_keys=True), encoding="utf-8")
            if directory.exists():
                directory.rename(runtime / (directory.name + "-backup-" + uuid.uuid4().hex[:8]))
            staging.rename(directory)
        if test_cache is None:
            for package in packages:
                try:
                    (cache / package["filename"]).unlink()
                except OSError:
                    pass
        return directory / "bin"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--wheel-cache", type=Path)
    args = parser.parse_args()
    application = Path(__file__).resolve().parent
    executable = application / "VisionHardwareCheck.exe"
    if not executable.is_file():
        executable = application / "VisionStudio.exe"
    if not executable.is_file() or not (application / "vision-runtime" / "python.exe").is_file():
        raise RuntimeError("请将补丁解压到 VisionStudio.exe 或 NDI / ONNX 版 VisionHardwareCheck.exe 同一目录")
    folder = install(application, application / "cuda-packages.json", args.wheel_cache)
    print(f"CUDA 运行库已就绪：{folder}", flush=True)
    if args.stage_only:
        return
    env = os.environ.copy()
    env["CUDA_PATH"] = env["CUDNN_PATH"] = str(folder)
    env["PATH"] = str(folder) + os.pathsep + env.get("PATH", "")
    subprocess.Popen([str(executable)], cwd=application, env=env)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as failure:
        print(f"CUDA 补丁失败：{failure}", file=sys.stderr, flush=True)
        sys.exit(1)
