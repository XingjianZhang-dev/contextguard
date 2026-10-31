#!/usr/bin/env python3
"""Train and audit one leakage-free HELIX imputer per GIFT configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def valid_manifest(output: Path) -> dict[str, object] | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = Path(str(manifest["artifact"]))
    if not artifact.is_absolute():
        artifact = output / artifact
    if not artifact.exists() or sha256(artifact) != manifest.get("artifact_sha256"):
        return None
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    shared = config["shared"]
    records: list[dict[str, object]] = []
    failed = False
    summary_path = Path(next(iter(config["datasets"].values()))["output"]).parent / (
        config["experiment"] + "_summary.json"
    )
    for name, dataset in config["datasets"].items():
        output = Path(dataset["output"])
        manifest = valid_manifest(output) if args.resume else None
        if manifest is not None:
            records.append({"dataset": name, "status": "complete", **manifest})
            print(f"SKIP {name}", flush=True)
            continue
        command = [
            config["python"],
            "-u",
            config["trainer"],
            "--dataset-path", dataset["path"],
            "--prediction-length", str(dataset["prediction_length"]),
            "--output", str(output),
        ]
        for key, value in shared.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])
        output.mkdir(parents=True, exist_ok=True)
        print(f"START {name}", flush=True)
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        (output / "matrix_stdout.log").write_text(process.stdout, encoding="utf-8")
        (output / "matrix_stderr.log").write_text(process.stderr, encoding="utf-8")
        manifest = valid_manifest(output)
        if process.returncode == 0 and manifest is not None:
            records.append({"dataset": name, "status": "complete", **manifest})
            print(
                f"COMPLETE {name} val_mse={manifest['val_mse']:.6f} "
                f"val_mae={manifest['val_mae']:.6f}",
                flush=True,
            )
        else:
            failed = True
            failure = {
                "dataset": name,
                "return_code": process.returncode,
                "status": "failed",
                "stderr_tail": process.stderr[-4000:],
            }
            write_json(output / "matrix_failure.json", failure)
            records.append(failure)
            print(f"FAILED {name} return_code={process.returncode}", flush=True)
        write_json(summary_path, {"experiment": config["experiment"], "runs": records})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
