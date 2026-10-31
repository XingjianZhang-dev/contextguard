import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def ar1(rng, n, phi=0.95):
    x = np.zeros(n)
    e = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.fixture
def sensors():
    """Four synthetic sensors: two noisy copies of a latent signal, one independent channel, and a third
    noisier view of the latent signal."""
    rng = np.random.default_rng(0)
    n, split = 12_000, 9_000
    z, ind = ar1(rng, n), ar1(rng, n)
    x = np.stack(
        [z + 0.2 * rng.normal(size=n), z + 0.2 * rng.normal(size=n), ind, 0.8 * z + 0.3 * rng.normal(size=n)], axis=1
    )
    return x.astype(np.float32), split
