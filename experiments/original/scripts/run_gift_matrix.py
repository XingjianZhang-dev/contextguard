#!/usr/bin/env python3
"""Run a resumable matrix of official-split GIFT-Eval fault experiments."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def identifier(coordinates: dict[str, object]) -> str:
    readable = "_".join(
        str(coordinates[key])
        for key in ("dataset", "model", "corruption", "severity", "repair", "corruption_seed")
    )
    digest = hashlib.sha256(
        json.dumps(coordinates, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return f"{readable}_{digest}".replace("/", "-")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if "extends" in config:
        base_path = (args.config.parent / config["extends"]).resolve()
        base = json.loads(base_path.read_text(encoding="utf-8"))
        overrides = {key: value for key, value in config.items() if key != "extends"}
        matrix_overrides = overrides.pop("matrix", {})
        base.update(overrides)
        base["matrix"].update(matrix_overrides)
        config = base
    axes = list(config["matrix"])
    combinations = itertools.product(*(config["matrix"][axis] for axis in axes))
    output_root = Path(config["output_root"])
    evaluator = Path(config["evaluator"])
    python = config["python"]
    runs: list[dict[str, object]] = []
    failed = False

    for values in combinations:
        coordinates = dict(zip(axes, values, strict=True))
        run_id = identifier(coordinates)
        run_dir = output_root / run_id
        manifest_path = run_dir / "manifest.json"
        if args.resume and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs.append({"run_id": run_id, "status": "complete", **manifest})
            print(f"SKIP {run_id}", flush=True)
            continue

        dataset = config["datasets"][coordinates["dataset"]]
        model = config["models"][coordinates["model"]]
        command = [
            python,
            "-u",
            str(evaluator),
            "--dataset-path",
            dataset["path"],
            "--model",
            model["name"],
            "--model-path",
            model["path"],
            "--split",
            config["split"],
            "--context-length",
            str(config["context_length"]),
            "--prediction-length",
            str(dataset["prediction_length"]),
            "--max-windows",
            str(config["max_windows"]),
            "--batch-size",
            str(config["batch_size"]),
            "--corruption",
            str(coordinates["corruption"]),
            "--severity",
            str(coordinates["severity"]),
            "--corruption-seed",
            str(coordinates["corruption_seed"]),
            "--repair",
            str(coordinates["repair"]),
            "--output",
            str(run_dir),
        ]
        for axis in config.get("pass_through_axes", []):
            command.extend(
                [f"--{axis.replace('_', '-')}", str(coordinates[axis])]
            )
        if coordinates["repair"] == "helix":
            command.extend(["--imputer-path", str(dataset["imputer_path"])])
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"START {run_id}", flush=True)
        process = subprocess.run(command, text=True, capture_output=True, check=False)
        (run_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
        if process.returncode == 0 and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs.append({"run_id": run_id, "status": "complete", **manifest})
            print(f"COMPLETE {run_id} {manifest['metrics']}", flush=True)
        else:
            failed = True
            failure = {
                "run_id": run_id,
                "status": "failed",
                "coordinates": coordinates,
                "return_code": process.returncode,
                "stderr_tail": process.stderr[-4000:],
            }
            write_json(run_dir / "failure.json", failure)
            runs.append(failure)
            print(f"FAILED {run_id} return_code={process.returncode}", flush=True)
        write_json(
            output_root / "summary.json",
            {"experiment": config["experiment"], "runs": runs},
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
