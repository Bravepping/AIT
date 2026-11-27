import shutil
import struct
import sys
import zipfile
from pathlib import Path


def stage(downloads, target):
    if struct.calcsize("P") != 8 or sys.version_info[:2] != (3, 14):
        raise SystemExit("Build requires Windows x64 CPython 3.14")
    base = Path(sys.base_prefix)
    target.mkdir(parents=True, exist_ok=True)
    embed = sorted(downloads.glob("python-3.14.*-embed-amd64.zip"))
    if embed:
        with zipfile.ZipFile(embed[-1]) as archive:
            archive.extractall(target)
    else:
        for name in ["python.exe", "python314.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"]:
            shutil.copy2(base / name, target / name)
        dlls = target / "DLLs"
        dlls.mkdir(exist_ok=True)
        for binary in (base / "DLLs").iterdir():
            if binary.suffix in (".dll", ".pyd") and not binary.name.startswith(("_test", "_tkinter", "tk", "tcl")):
                shutil.copy2(binary, dlls / binary.name)
        stdlib = base / "Lib"
        excluded = {"site-packages", "__pycache__", "test", "idlelib", "tkinter", "ensurepip", "turtledemo"}
        with zipfile.ZipFile(target / "python314.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for source in stdlib.rglob("*"):
                relative = source.relative_to(stdlib)
                if source.suffix in {".py", ".txt", ".pem"} and not excluded.intersection(relative.parts):
                    archive.write(source, relative.as_posix())

    (target / "python314._pth").write_text("python314.zip\n.\nDLLs\nsite-packages\n", encoding="ascii")
    site = target / "site-packages"
    site.mkdir(exist_ok=True)
    wheels = ["numpy-2.5.0-", "pillow-13.0.0-", "onnxruntime_gpu-1.24.4-", "packaging-26.3-", "flatbuffers-25.12.19-"]
    for prefix in wheels:
        wheel, = list(downloads.glob(prefix + "*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)


if __name__ == "__main__":
    stage(*(Path(argument) for argument in sys.argv[1:]))
