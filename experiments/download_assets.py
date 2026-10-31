#!/usr/bin/env python3
"""Download the pinned GIFT-Eval configurations and forecaster checkpoints and verify their hashes.

Datasets go to ``data/GiftEval`` and checkpoints to ``models/``; every dataset fingerprint and model
SHA-256 must equal the value in ``configs/assets.json`` (and therefore in every run manifest).
TiRex is fetched into the Hugging Face cache at its pinned revision on first use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "original" / "scripts"))
from run_gift_fault_eval import dataset_fingerprint, sha256  # noqa: E402


def main() -> int:
    assets = json.loads((HERE / "configs" / "assets.json").read_text())
    gift = assets["gift_eval"]
    configs = list(gift["fingerprints"])
    snapshot_download(
        gift["repository"],
        repo_type="dataset",
        revision=gift["revision"],
        local_dir="data/GiftEval",
        allow_patterns=[f"{c}/*" for c in configs],
        max_workers=8,
    )
    ok = True
    for c in configs:
        match = dataset_fingerprint(Path("data/GiftEval") / c) == gift["fingerprints"][c]
        ok &= match
        print(f"{'ok  ' if match else 'FAIL'} data/GiftEval/{c}")
    for name in ("chronos2", "timesfm25"):
        m = assets["models"][name]
        snapshot_download(m["repository"], revision=m["revision"], local_dir=m["local_dir"])
        match = sha256(Path(m["local_dir"]) / m["file"]) == m["sha256"]
        ok &= match
        print(f"{'ok  ' if match else 'FAIL'} {m['local_dir']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
