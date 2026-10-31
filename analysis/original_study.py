#!/usr/bin/env python3
"""Original-study summaries used by the revised paper (Sections 5.1-5.4, Fig. 5A-B, Table S-Holm).

Runs the audit programs of the original study (``experiments/original/scripts``) on the original run
records and writes their reports to ``results/summaries/original/``, together with the two tables plotted
in Fig. 5A-B:

    severity_and_boundary.csv   channel dropout of 10/30/50% of the channels and 50% random missingness
    context_sensitivity.csv     context lengths 256, 512 (frozen) and 1024
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys

from common import LOSSES, ORIGINAL, RESULTS, ROOT

SCRIPTS = ROOT / "experiments" / "original" / "scripts"
OUT = RESULTS / "summaries" / "original"


def rel(path) -> str:
    """Repository-relative path when possible, so that the reports carry no machine-specific paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def audit(script: str, *args: str, output: str) -> dict:
    path = OUT / output
    subprocess.run([sys.executable, rel(SCRIPTS / script), *args, "--output", rel(path)], check=True, cwd=ROOT)
    return json.loads(path.read_text())


def rows_for(summary: dict, **fields) -> list[dict]:
    rows = []
    for loss in LOSSES:
        m = summary["metrics"][loss]
        lo, hi = m["dataset_cluster_bootstrap_ci95_percent"]
        rows.append(
            {
                **fields,
                "metric": loss,
                "paired_cells": summary["paired_cells"],
                "geometric_relative_change_percent": m["geometric_relative_change_percent"],
                "cluster_ci_low": lo,
                "cluster_ci_high": hi,
                "wins": m["wins"],
                "losses": m["losses"],
                "ties": m["ties"],
                "wilcoxon_holm_p": m["wilcoxon_less_holm_p"],
            }
        )
    return rows


def write_csv(rows: list[dict], name: str) -> None:
    with open(RESULTS / "summaries" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = lambda name: rel(ORIGINAL / name / "summary.json")  # noqa: E731
    formal = audit("audit_formal_pairs.py", summary("gift_sensor_v4_formal_test"), output="formal.json")
    severity2 = audit("audit_formal_pairs.py", summary("gift_sensor_v4_severity2_test"), output="severity2.json")
    random = audit("audit_formal_pairs.py", summary("gift_random_v4_formal_test"), output="random_missing.json")
    helix = audit(
        "audit_helix_baseline.py",
        summary("gift_sensor_v4_formal_test"),
        summary("gift_helix_formal_test"),
        output="helix.json",
    )
    context = audit(
        "audit_context_sensitivity.py",
        summary("gift_sensor_v4_context_sensitivity_test"),
        summary("gift_sensor_v4_formal_test"),
        output="context_sensitivity.json",
    )

    rows = rows_for(formal["by_severity"]["1"], corruption="sensor_dropout", severity=1, missing_rate_percent=10)
    rows += rows_for(severity2["overall"], corruption="sensor_dropout", severity=2, missing_rate_percent=30)
    rows += rows_for(formal["by_severity"]["3"], corruption="sensor_dropout", severity=3, missing_rate_percent=50)
    rows += rows_for(random["overall"], corruption="random_missing", severity=3, missing_rate_percent=50)
    write_csv(rows, "severity_and_boundary.csv")
    rows = []
    for length in sorted(context["settings"], key=int):
        rows += rows_for(
            context["settings"][length]["comparison"]["overall"],
            context_length=int(length),
            is_default=int(length) == context["frozen_default"],
        )
    write_csv(rows, "context_sensitivity.csv")

    for label, s in (
        ("ContextGuard vs topology", formal["overall"]),
        ("HELIX vs topology", helix["comparisons"]["helix_vs_topology"]["overall"]),
        ("ContextGuard vs HELIX", helix["comparisons"]["contextguard_vs_helix"]["overall"]),
    ):
        print(
            f"{label:26s} "
            + " | ".join(f"{loss} {s['metrics'][loss]['geometric_relative_change_percent']:+.2f}%" for loss in LOSSES)
        )


if __name__ == "__main__":
    main()
