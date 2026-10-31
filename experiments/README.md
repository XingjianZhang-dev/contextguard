# Experiments

* `evaluate.py`: revision evaluator. It imports the original evaluator (`original/scripts/run_gift_fault_eval.py`)
  for data handling, corruption, repair and the Chronos-2 / TimesFM 2.5 wrappers, and adds the classical
  estimators, the cross-fitted gate, TiRex, native missing-value input, and per-channel logging.
* `run_revision.sh {primary|heldout|tirex|native|all}`: the four revision experiment sets (3,184 runs).
* `check_native_timesfm.py`: native missing-value input for TimesFM 2.5.
* `setup_env.sh`, `requirements*.txt`: pinned environments (Python 3.12, PyTorch 2.8.0, CUDA 12.8).
* `download_assets.py`, `configs/assets.json`: pinned data and checkpoints with their hashes.
* `configs/primary.tsv`, `configs/heldout.tsv`: configuration key, GIFT-Eval path, forecast horizon.
* `original/`: code of record of the original study.

See [../docs/reproducing.md](../docs/reproducing.md) for the complete workflow.
