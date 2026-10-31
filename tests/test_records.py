"""Consistency of the run records with the released tables and with the package (skipped when the
records or the GIFT-Eval data have not been downloaded)."""

import csv
import json
import sys

import numpy as np
import pytest
from conftest import ROOT

RECORDS = ROOT / "records" / "revision"
DATA = ROOT / "data" / "GiftEval"
needs_records = pytest.mark.skipif(not RECORDS.is_dir(), reason="run scripts/fetch_records.sh first")
needs_data = pytest.mark.skipif(not DATA.is_dir(), reason="run experiments/download_assets.py first")


@needs_records
@pytest.mark.parametrize("name", ["primary", "heldout", "tirex", "native"])
def test_cell_tables_match_the_manifests(name):
    rows = {
        (r["dataset"], r["model"], r["severity"], r["seed"], r["method"]): r
        for r in csv.DictReader(open(ROOT / "results" / "cells" / f"{name}.csv"))
    }
    manifests = sorted((RECORDS / name).glob("*/manifest.json"))
    assert len(manifests) == len(rows)
    for path in manifests:
        j = json.loads(path.read_text())
        a = j["args"]
        row = rows[(a["dataset_key"], a["model"], str(a["severity"]), str(a["corruption_seed"]), a["repair"])]
        assert float(row["mse"]) == j["metrics"]["mse"]


@needs_records
@needs_data
@pytest.mark.parametrize(
    "config",
    [line.split() for line in open(ROOT / "experiments" / "configs" / "primary.tsv")]
    + [line.split() for line in open(ROOT / "experiments" / "configs" / "heldout.tsv")],
)
def test_package_reproduces_the_recorded_gates(config):
    from datasets import load_from_disk

    sys.path.insert(0, str(ROOT / "experiments"))
    import evaluate

    key, path, horizon = config
    ds = load_from_disk(str(DATA / path)).with_format("numpy")
    recs = [evaluate.base.as_target(ds[i]) for i in range(len(ds))]
    windows = evaluate.base.split_windows(min(r.shape[1] for r in recs), int(horizon))
    *_, shared = evaluate.estimator_states(recs, int(horizon), windows)
    run_set = (
        "primary" if key in {line.split()[0] for line in open(ROOT / "experiments/configs/primary.tsv")} else "heldout"
    )
    recorded = json.loads(next((RECORDS / run_set).glob(f"{key}_*_topology_201/manifest.json")).read_text())
    for estimator in ("contextguard", "contextguard_crossfit", "ridge", "lw"):
        np.testing.assert_allclose(shared["gates"][estimator]["g"], recorded["gates"][estimator]["g"], atol=1e-6)
    assert shared["ridge"]["lambda"] == recorded["ridge"]["lambda"]
    assert shared["gate0_replica_matches_archive"]
