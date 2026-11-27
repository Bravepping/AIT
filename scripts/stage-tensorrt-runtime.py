"""Assemble a portable Windows x64 runtime from the pinned, downloaded wheels."""

import argparse
import hashlib
import zipfile
from pathlib import Path


BASE_WHEELS = (
    "numpy-2.5.0-cp313-cp313-win_amd64",
    "pillow-13.0.0-cp313-cp313-win_amd64",
    "onnxruntime_gpu-1.24.4-cp313-cp313-win_amd64",
    "packaging-26.3-py3-none-any",
    "flatbuffers-25.12.19-py2.py3-none-any",
)


def extract(archive, destination, digest=None):
    if digest:
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"SHA256 mismatch: {archive.name}")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


def stage(downloads, target):
    target.mkdir(parents=True)
    extract(downloads / "python-3.13.9-embed-amd64.zip", target,
            "91d828c2da3a029b41699e918674a0cb379c02cf20dab9c501306885f837402a")
    (target / "python313._pth").write_text("python313.zip\n.\nsite-packages\n", encoding="ascii")
    site = target / "site-packages"
    for wheel in BASE_WHEELS:
        extract(downloads / (wheel + ".whl"), site)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("downloads", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    stage(args.downloads.resolve(), args.target.resolve())
