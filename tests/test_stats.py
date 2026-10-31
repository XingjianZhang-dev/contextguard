"""Paired statistics: identical to the original audit program, and basic properties."""

import sys

import numpy as np
import pytest
from conftest import ROOT

from contextguard.stats import holm_adjust, summarize

sys.path.insert(0, str(ROOT / "experiments" / "original" / "scripts"))


def rows(seed=0, n_datasets=6, per=10):
    rng = np.random.default_rng(seed)
    out = []
    for d in range(n_datasets):
        for _ in range(per):
            row = {"dataset": f"d{d}"}
            for m in ("mse", "mae", "quantile_loss"):
                lr = rng.normal(-0.3, 0.2)
                row[f"log_{m}_ratio"] = lr
                row[f"{m}_difference"] = np.expm1(lr)
            for label in ("baseline", "method"):
                row[f"{label}_coverage80"] = rng.uniform(0.5, 0.9)
                row[f"{label}_width80"] = rng.uniform(0.5, 2.0)
            out.append(row)
    return out


def test_summary_equals_the_original_audit():
    audit = pytest.importorskip("audit_formal_pairs")
    r = rows()
    assert summarize(r, 2000, 260827) == audit.summarize(r, 2000, 260827)


def test_holm_is_monotone_and_bounded():
    p = [0.01, 0.04, 0.03, 0.5]
    adj = holm_adjust(p)
    assert all(a >= b for a, b in zip(adj, p))
    assert all(a <= 1 for a in adj)
    order = np.argsort(p)
    assert all(adj[order[i]] <= adj[order[i + 1]] for i in range(len(p) - 1))


def test_effect_sign_and_counts():
    s = summarize(rows(1), 500, 1)
    m = s["metrics"]["mse"]
    assert m["geometric_relative_change_percent"] < 0
    assert m["wins"] + m["ties"] + m["losses"] == s["paired_cells"]
    lo, hi = m["dataset_cluster_bootstrap_ci95_percent"]
    assert lo <= m["geometric_relative_change_percent"] <= hi
