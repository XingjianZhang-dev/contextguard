# Original study

Code of record for Sections 5.1-5.4 and 5.9-5.10, unchanged from the original release
(10.5281/zenodo.22124599) except for repository-relative paths in the configurations.

* `scripts/run_gift_fault_eval.py`: evaluator (data handling, corruption generator, repair methods,
  Chronos-2 and TimesFM 2.5 wrappers, HELIX loader). The revision evaluator imports it unchanged.
* `scripts/run_gift_matrix.py`: resumable matrix runner for the configurations in `configs/`.
* `scripts/train_helix_imputer.py`, `scripts/run_helix_training_matrix.py`: training-only HELIX baseline.
* `scripts/audit_*.py`, `scripts/summarize_gift_pairs.py`, `scripts/filter_duplicate_targets.py`:
  statistical reports, data-split and duplicate audits, manifest consistency and reproduction checks.
* `configs/`: frozen experiment matrices; `requirements.txt` and `environment-freeze.txt`: software
  environment of the original runs.

Run from the repository root, for example
`python experiments/original/scripts/run_gift_matrix.py --resume --config experiments/original/configs/gift_sensor_v4_formal_test.json`.
