"""The package reproduces the evaluator of record (``experiments/original/scripts/run_gift_fault_eval.py``)."""

import sys

import numpy as np
import pytest
from conftest import ROOT

from contextguard import fit_statistics, repair

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "experiments" / "original" / "scripts"))
base = pytest.importorskip("run_gift_fault_eval")

H = 48


def records(seed, n_series=3, channels=6, length=6000):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_series, 2, length)).cumsum(-1) * 0.05
    mix = rng.normal(size=(channels, 2))
    out = []
    for s in range(n_series):
        x = (mix @ latent[s] + 0.3 * rng.normal(size=(channels, length))).astype(np.float32)
        x[rng.random(x.shape) < 0.01] = np.nan  # natural missing values
        out.append(x)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_training_statistics_are_identical(seed):
    recs = records(seed)
    windows = base.split_windows(min(r.shape[1] for r in recs), H)
    means, scales, pm, cov, rel, ends = base.training_statistics(recs, H, windows, 0.05)
    s = fit_statistics([r[:, :e].T for r, e in zip(recs, ends)])
    assert np.array_equal(s.pooled_mean, pm)
    assert np.array_equal(s.covariance, cov)
    assert np.allclose(s.gate, rel, atol=1e-7)
    for a, b in zip(s.means, means):
        assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "corruption,severity",
    [("sensor_dropout", 1), ("sensor_dropout", 3), ("block_missing", 2), ("random_missing", 3), ("clean", 1)],
)
def test_repair_matches_the_reference(corruption, severity):
    recs = records(7)
    windows = base.split_windows(min(r.shape[1] for r in recs), H)
    means, scales, pm, cov, rel, _ = base.training_statistics(recs, H, windows, 0.05)
    desc = base.descriptors_for_split(recs, H, windows, 512, "test", 32)
    ctx = np.asarray([(recs[r][:, f - 512 : f].T - means[r]) / scales[r] for r, f in desc], dtype=np.float32)
    x, mask = base.corrupt_batch(ctx, corruption, severity, 201, 0)
    for ref, mine in (
        ("topology", "topology"),
        ("contextguard_adaptive", "contextguard"),
        ("contextguard", "contextguard_ungated"),
    ):
        expected = base.repair_batch(x, mask, ref, pm, cov, rel, torch.device("cpu"))
        got = repair(x, mask, mine, pm, cov, rel)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
