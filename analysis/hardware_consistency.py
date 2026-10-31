#!/usr/bin/env python3
"""Agreement of the revision runs with the original records (Supplementary Section S12).

The 320 topology and ContextGuard runs of the primary design were rerun on a different GPU architecture
with the original pinned software. This script compares every reported metric run by run and checks
that dataset fingerprints and model hashes are identical. Output: ``results/summaries/hardware_consistency.json``.
"""

from __future__ import annotations

import json

from common import ORIGINAL, RESULTS, runs, write_json

METRICS = ("mse", "mae", "quantile_loss", "coverage80", "width80")


def main() -> None:
    original = {}
    for path in (ORIGINAL / "gift_sensor_v4_formal_test").glob("*/manifest.json"):
        j = json.loads(path.read_text())
        a = j["args"]
        original[(a["dataset_path"], a["model"], a["severity"], a["corruption_seed"], a["repair"])] = j
    worst = {m: 0.0 for m in METRICS}
    compared, identity = 0, True
    for key, r in runs("primary").items():
        if key[4] not in ("topology", "contextguard_adaptive"):
            continue
        a = r.manifest["args"]
        ref = original[(a["dataset_path"], a["model"], a["severity"], a["corruption_seed"], a["repair"])]
        identity &= ref["dataset_fingerprint"] == r.manifest["dataset_fingerprint"]
        identity &= ref["model_sha256"] == r.manifest["model_sha256"]
        for m in METRICS:
            x, y = r.metrics[m], ref["metrics"][m]
            worst[m] = max(worst[m], abs(x - y) / max(abs(y), 1e-12))
        compared += 1
    out = {
        "runs_compared": compared,
        "max_relative_difference": worst,
        "fingerprints_and_hashes_identical": bool(identity),
    }
    write_json(out, RESULTS / "summaries" / "hardware_consistency.json")
    print(
        f"{compared} runs; identical fingerprints and model hashes: {identity}; worst relative difference "
        + ", ".join(f"{m} {v:.1e}" for m, v in worst.items())
    )


if __name__ == "__main__":
    main()
