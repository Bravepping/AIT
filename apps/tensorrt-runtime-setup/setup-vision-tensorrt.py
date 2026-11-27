"""Install pinned NVIDIA TensorRT wheels into the portable app's runtime."""

import argparse
import hashlib
import json
import msvcrt
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid
import zipfile


CHUNK = 2 * 1024 * 1024
PROGRESS_STEP = 64 * 1024 * 1024
REQUIRED_FILES = {
    "tensorrt_bindings/tensorrt.cp313-win_amd64.pyd",
    "tensorrt_libs/nvinfer_10.dll",
    "tensorrt_libs/nvonnxparser_10.dll",
    "tensorrt_rtx_bindings/tensorrt_rtx.cp313-win_amd64.pyd",
    "tensorrt_rtx_libs/tensorrt_rtx_1_7.dll",
    "tensorrt_rtx_libs/tensorrt_onnxparser_rtx_1_7.dll",
}


def wheel_matches(path, package):
    if not path.is_file() or path.stat().st_size != package["size"]:
        return False
    with path.open("rb") as archive:
        return hashlib.file_digest(archive, "sha256").hexdigest() == package["sha256"]


def get_wheel(cache, package, offline):
    path = cache / package["filename"]
    if wheel_matches(path, package):
        print(f"已校验 {path.name}", flush=True)
        return path
    if offline:
        raise ValueError(f"离线目录缺少完整且校验通过的文件：{path}")
    address = urlsplit(package["url"])
    if address.scheme != "https" or address.hostname not in {"files.pythonhosted.org", "pypi.nvidia.com"}:
        raise ValueError(f"下载地址异常：{package['url']}")

    partial = path.with_suffix(".part")
    whole = package["size"]
    received = 0
    digest = hashlib.sha256()
    next_progress = PROGRESS_STEP
    print(f"下载 {path.name}（{whole / 1e6:.1f} MB）", flush=True)
    try:
        with urlopen(package["url"], timeout=60) as response, partial.open("wb") as output:
            for block in iter(lambda: response.read(CHUNK), b""):
                received += len(block)
                if received > whole:
                    raise ValueError(f"{path.name} 大小超过清单")
                output.write(block)
                digest.update(block)
                if received >= next_progress:
                    print(f"  已下载 {received / 1e6:.0f}/{whole / 1e6:.0f} MB", flush=True)
                    next_progress += PROGRESS_STEP
        if received != whole or digest.hexdigest() != package["sha256"]:
            raise ValueError(f"{path.name} 的大小或 SHA256 校验失败")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def relative_name(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"无效的运行库相对路径：{value}")
    return path


def extract(wheel, staging, files):
    print(f"解压 {wheel.name}", flush=True)
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = relative_name(entry.filename)
            target = staging.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, CHUNK)
            files[name.as_posix()] = entry.file_size


def ready(directory, manifest_hash):
    try:
        installed = json.loads((directory / "installed.json").read_text(encoding="utf-8"))
        files = installed["files"]
        return (
            installed["packages_sha256"] == manifest_hash
            and REQUIRED_FILES <= files.keys()
            and all(directory.joinpath(*relative_name(name).parts).stat().st_size == size
                    for name, size in files.items())
        )
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        return False


def cleanup(runtime, directory):
    if directory.exists() and directory.resolve().parent == runtime:
        try:
            shutil.rmtree(directory)
        except OSError as error:
            print(f"保留临时文件夹 {directory}：{error}", file=sys.stderr, flush=True)


def install(runtime, manifest, wheel_cache=None, repair=False):
    package_data = manifest.read_bytes()
    packages = json.loads(package_data)
    manifest_hash = hashlib.sha256(package_data).hexdigest()
    runtime = runtime.resolve(strict=True)
    destination = runtime / "trt-packages"
    cache = wheel_cache.resolve(strict=True) if wheel_cache else runtime / "trt-downloads"
    with (runtime / ".tensorrt-install.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("另一个 TensorRT 安装程序正在运行") from error
        if not repair and ready(destination, manifest_hash):
            return destination, True

        if wheel_cache is None:
            cache.mkdir(exist_ok=True)
        wheels = [get_wheel(cache, entry, offline=wheel_cache is not None) for entry in packages]
        suffix = uuid.uuid4().hex
        staging = runtime / f".trt-staging-{suffix}"
        backup = runtime / f".trt-backup-{suffix}"
        staging.mkdir()
        files = {}
        try:
            for wheel in wheels:
                extract(wheel, staging, files)
            if not REQUIRED_FILES <= files.keys():
                missing = sorted(REQUIRED_FILES - files.keys())
                raise ValueError("安装包缺少文件：" + ", ".join(missing))
            metadata = {"packages_sha256": manifest_hash, "files": files}
            (staging / "installed.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            if destination.exists():
                destination.rename(backup)
            try:
                staging.rename(destination)
            except OSError:
                if backup.exists() and not destination.exists():
                    backup.rename(destination)
                raise
        finally:
            cleanup(runtime, staging)
        cleanup(runtime, backup)
        if wheel_cache is None:
            for wheel in wheels:
                try:
                    wheel.unlink()
                except OSError as error:
                    print(f"保留下载缓存 {wheel.name}：{error}", file=sys.stderr, flush=True)
        return destination, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-cache", type=Path, help="Use the four verified wheel files in this directory offline.")
    parser.add_argument("--repair", action="store_true", help="Replace the installed TensorRT runtime.")
    args = parser.parse_args()
    application = Path(__file__).resolve().parent
    runtime = application / "vision-runtime"
    executable = application / "VisionHardwareCheck.exe"
    if not executable.is_file():
        executable = application / "VisionStudio.exe"
    bundled_python = runtime / "python.exe"
    if not executable.is_file() or not bundled_python.is_file() or not Path(sys.executable).samefile(bundled_python):
        raise RuntimeError("请在含有 VisionStudio.exe 或 VisionHardwareCheck.exe 的程序目录运行 Install-TensorRT.cmd")
    if sys.version_info[:2] != (3, 13) or platform.machine().upper() != "AMD64":
        raise RuntimeError("此 TensorRT 安装包需要 Windows x64 的 Python 3.13 版 VisionHardwareCheck")
    installed, reused = install(runtime, application / "tensorrt-packages.json", args.wheel_cache, args.repair)
    status = "已校验现有 TensorRT 运行库" if reused else "TensorRT 与 TensorRT for RTX 运行库已安装"
    print(f"{status}：{installed}", flush=True)
    print(f"CUDA 运行库就绪后，启动 {executable.name} 并选择所需引擎。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as failure:
        print(f"TensorRT 安装失败：{failure}", file=sys.stderr, flush=True)
        sys.exit(1)
