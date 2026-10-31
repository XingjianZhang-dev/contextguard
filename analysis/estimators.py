#!/usr/bin/env python3
"""Classical cross-channel estimators and the calibration gate (Section 5.5, Table 4, Fig. 3).

Paired comparisons on the 160 primary cells (Chronos-2, TimesFM 2.5) and on the 80 TiRex cells of the
primary configurations. Output: ``results/summaries/estimators.json``.
"""

from __future__ import annotations

from common import RESULTS, compare, effect, metrics, write_json

VS_TOPOLOGY = (
    "contextguard_adaptive",
    "contextguard",
    "contextguard_crossfit",
    "ridge_ungated",
    "ridge_gated",
    "lw_ungated",
    "lw_gated",
)
PAIRS = (
    ("contextguard_adaptive", "contextguard"),
    ("ridge_gated", "ridge_ungated"),
    ("lw_gated", "lw_ungated"),
    ("contextguard_adaptive", "ridge_gated"),
    ("contextguard_adaptive", "lw_gated"),
    ("contextguard_crossfit", "contextguard_adaptive"),
    ("contextguard_adaptive", "ridge_ungated"),
    ("contextguard_adaptive", "lw_ungated"),
)


def comparisons(table: dict) -> dict:
    out = {}
    for m in VS_TOPOLOGY:
        out[f"{m}__vs__topology"] = compare(table, m, "topology")
    for m, b in PAIRS:
        out[f"{m}__vs__{b}"] = compare(table, m, b)
    return out


def main() -> None:
    result = {"primary": comparisons(metrics("primary")), "tirex": comparisons(metrics("tirex"))}
    for name, comp in result.items():
        print(f"[{name}]  geometric MSE change [95% dataset-cluster CI], MSE wins/losses")
        for key, s in comp.items():
            e, lo, hi = effect(s)
            m = s["metrics"]["mse"]
            print(
                f"  {key:52s} {e:+7.2f}% [{lo:+7.2f}, {hi:+7.2f}]  {m['wins']}/{m['losses']}  (n={s['paired_cells']})"
            )
    write_json(result, RESULTS / "summaries" / "estimators.json")


if __name__ == "__main__":
    main()
