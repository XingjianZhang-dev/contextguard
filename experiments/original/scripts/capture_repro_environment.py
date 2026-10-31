#!/usr/bin/env python3
"""Capture a machine-readable environment record for a clean reproduction run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import site
import subprocess
import sys
from pathlib import Path


PACKAGES = (
    "accelerate",
    "chronos-forecasting",
    "datasets",
    "einops",
    "huggingface-hub",
    "numpy",
    "pandas",
    "pypots",
    "safetensors",
    "scikit-learn",
    "scipy",
    "sktime",
    "tensorboard",
    "timesfm",
    "torch",
    "transformers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-freeze", type=Path, required=True)
    parser.add_argument("--hash-file", action="append", type=Path, default=[])
    args = parser.parse_args()

    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    import torch

    freeze = command_output([sys.executable, "-m", "pip", "freeze"])
    hashes = {}
    for path in args.hash_file:
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path)] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    payload = {
        "cuda": {
            "available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "torch_cuda_version": torch.version.cuda,
        },
        "executable": sys.executable,
        "file_hashes": hashes,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "package_versions": versions,
        "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
        "pip_freeze_sha256": hashlib.sha256((freeze + "\n").encode("utf-8")).hexdigest(),
        "platform": platform.platform(),
        "prefixes": {"base_prefix": sys.base_prefix, "prefix": sys.prefix},
        "python": sys.version,
        "site_packages": site.getsitepackages(),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_json.replace(args.output_json)

    args.output_freeze.parent.mkdir(parents=True, exist_ok=True)
    temporary_freeze = args.output_freeze.with_suffix(args.output_freeze.suffix + ".tmp")
    temporary_freeze.write_text(freeze + "\n", encoding="utf-8")
    temporary_freeze.replace(args.output_freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
