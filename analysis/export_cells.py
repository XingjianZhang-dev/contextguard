#!/usr/bin/env python3
"""Export the per-run metrics of every revision run set to ``results/cells/<set>.csv``."""

from __future__ import annotations

import csv

from common import RESULTS, runs

SETS = ("primary", "primary_clean", "heldout", "heldout_clean", "tirex", "tirex_clean", "native")
FIELDS = ("mse", "mae", "quantile_loss", "coverage80", "width80", "raw_mse")


def main() -> None:
    out_dir = RESULTS / "cells"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        rows = []
        for r in runs(name).values():
            a, m = r.manifest["args"], r.manifest
            rows.append(
                {
                    "dataset": a["dataset_key"],
                    "model": a["model"],
                    "corruption": a["corruption"],
                    "severity": a["severity"],
                    "seed": a["corruption_seed"],
                    "method": a["repair"],
                    **{k: m["metrics"][k] for k in FIELDS},
                    "windows": m["windows_evaluated"],
                    "dataset_fingerprint": m["dataset_fingerprint"],
                    "model_sha256": m["model_sha256"],
                }
            )
        rows.sort(key=lambda r: (r["dataset"], r["model"], r["severity"], r["seed"], r["method"]))
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"results/cells/{name}.csv  {len(rows)} runs")


if __name__ == "__main__":
    main()
