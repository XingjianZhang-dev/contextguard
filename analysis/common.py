"""Shared helpers for the analysis scripts: run-record loading and dataset names."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from contextguard.stats import LOSSES, compare, paired_rows, summarize  # noqa: E402,F401

RECORDS = ROOT / os.environ.get("CONTEXTGUARD_RECORDS", "records")  # relative paths are taken from the repository root
RESULTS = ROOT / "results"
REVISION = RECORDS / "revision"
ORIGINAL = RECORDS / "original"

NAMES = {
    "ett1_15T": "ETT1",
    "ett2_15T": "ETT2",
    "jena_10T": "Jena",
    "bitbrains_fast_5T": "Bitbrains Fast",
    "bitbrains_rnd_5T": "Bitbrains RND",
    "bizitobs_application": "BizITObs App",
    "bizitobs_service": "BizITObs Service",
    "bizitobs_l2c_5T": "BizITObs L2C",
    "ett1_H": "ETT1 (H)",
    "ett2_H": "ETT2 (H)",
    "jena_H": "Jena (H)",
    "bitbrains_fast_H": "Bitbrains Fast (H)",
    "bitbrains_rnd_H": "Bitbrains RND (H)",
    "bizitobs_l2c_H": "BizITObs L2C (H)",
}
FORECASTERS = {"chronos2": "Chronos-2", "timesfm25": "TimesFM 2.5", "tirex": "TiRex"}


class Run:
    """One run directory: the manifest is read eagerly, the window arrays on demand."""

    __slots__ = ("manifest", "path")

    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())

    @property
    def metrics(self) -> dict:
        return self.manifest["metrics"]

    @property
    def gates(self) -> dict:
        return self.manifest["gates"]

    def arrays(self):
        return np.load(self.path / "window_metrics.npz")


@lru_cache(maxsize=None)
def runs(name: str) -> dict[tuple, Run]:
    """``{(dataset, model, severity, seed, method): Run}`` for a revision run set; clean sets are keyed
    ``(dataset, model, method)``."""
    root = REVISION / name
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/fetch_records.sh first")
    out = {}
    for manifest in sorted(root.glob("*/manifest.json")):
        r = Run(manifest.parent)
        a = r.manifest["args"]
        if a["corruption"] == "clean":
            out[(a["dataset_key"], a["model"], a["repair"])] = r
        else:
            out[(a["dataset_key"], a["model"], a["severity"], a["corruption_seed"], a["repair"])] = r
    return out


def metrics(*names: str) -> dict[tuple, dict]:
    """Merged ``{key: metrics}`` of one or more run sets."""
    out = {}
    for n in names:
        out.update({k: r.metrics for k, r in runs(n).items()})
    return out


def cells(table: dict) -> list[tuple]:
    return sorted({k[:4] for k in table})


def effect(summary: dict, loss: str = "mse") -> tuple[float, float, float]:
    m = summary["metrics"][loss]
    lo, hi = m["dataset_cluster_bootstrap_ci95_percent"]
    return m["geometric_relative_change_percent"], lo, hi


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(name: str) -> dict:
    return json.loads((RESULTS / "summaries" / name).read_text())
