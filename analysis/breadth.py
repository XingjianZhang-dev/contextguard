#!/usr/bin/env python3
"""Held-out configurations, a third forecaster and native missing-value input (Section 5.8, Table 6, Fig. 5C).

Sets: six hourly GIFT-Eval configurations with Chronos-2, TimesFM 2.5 and TiRex (180 cells), and TiRex on
the eight primary configurations (80 cells). Native missing-value input passes the failed channel to the
forecaster as missing values instead of repairing it.

Output: ``results/summaries/breadth.json``.
"""

from __future__ import annotations

from common import RESULTS, compare, effect, metrics, write_json


def native_summary(native: dict, reference: dict, model: str) -> dict:
    """Native missing-value input against topology and ContextGuard on the same cells."""
    cells = sorted(
        k[:4]
        for k in native
        if k[1] == model and k[4] == "nan_native" and k[:4] + ("contextguard_adaptive",) in reference
    )
    table = {
        **{k: v for k, v in reference.items() if k[1] == model},
        **{c + ("nan_native",): native[c + ("nan_native",)] for c in cells},
    }
    cg = [reference[c + ("contextguard_adaptive",)] for c in cells]
    nat = [native[c + ("nan_native",)] for c in cells]
    return {
        "cells": len(cells),
        "contextguard_lower_mse": int(sum(g["mse"] < n["mse"] for g, n in zip(cg, nat))),
        "native_width_at_least_10x": int(sum(n["width80"] >= 10 * g["width80"] for g, n in zip(cg, nat))),
        "native_vs_topology": compare(table, "nan_native", "topology"),
        "contextguard_vs_native": compare(table, "contextguard_adaptive", "nan_native"),
    }


def main() -> None:
    held, tirex, primary = metrics("heldout"), metrics("tirex"), metrics("primary")
    native = metrics("native")
    out = {
        "heldout": {"all": compare(held, "contextguard_adaptive", "topology")},
        "tirex_primary": compare(tirex, "contextguard_adaptive", "topology"),
        "gate": {
            "heldout": compare(held, "contextguard_adaptive", "contextguard"),
            "tirex_primary": compare(tirex, "contextguard_adaptive", "contextguard"),
        },
    }
    for mod in ("chronos2", "timesfm25", "tirex"):
        out["heldout"][mod] = compare(held, "contextguard_adaptive", "topology", where=lambda c, m=mod: c[1] == m)
    out["native_missing"] = {
        "heldout_chronos2": native_summary(held, held, "chronos2"),
        "heldout_tirex": native_summary(native, held, "tirex"),
        "primary_chronos2": native_summary(native, primary, "chronos2"),
        "primary_tirex": native_summary(native, tirex, "tirex"),
    }

    def fmt(s: dict) -> str:
        return " | ".join(
            f"{loss} {effect(s, loss)[0]:+.2f}% [{effect(s, loss)[1]:+.2f}, {effect(s, loss)[2]:+.2f}]"
            for loss in ("mse", "mae", "quantile_loss")
        )

    print(f"held-out, all forecasters (n={out['heldout']['all']['paired_cells']}): {fmt(out['heldout']['all'])}")
    for mod in ("chronos2", "timesfm25", "tirex"):
        print(f"  {mod:9s}: {fmt(out['heldout'][mod])}")
    print(f"TiRex, primary (n={out['tirex_primary']['paired_cells']}): {fmt(out['tirex_primary'])}")
    print(f"gate, held-out: {fmt(out['gate']['heldout'])}")
    print(f"gate, TiRex primary: {fmt(out['gate']['tirex_primary'])}")
    for k, v in out["native_missing"].items():
        print(
            f"native input, {k}: ContextGuard lower MSE in {v['contextguard_lower_mse']}/{v['cells']} cells; "
            f"native 80% width >= 10x in {v['native_width_at_least_10x']}/{v['cells']}; "
            f"native vs topology MSE {effect(v['native_vs_topology'])[0]:+.2f}%"
        )
    write_json(out, RESULTS / "summaries" / "breadth.json")


if __name__ == "__main__":
    main()
