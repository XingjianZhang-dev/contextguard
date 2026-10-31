# Reproducing the paper

Every number, table and figure of the paper is produced by a script in this repository from the archived
run records. Section numbers refer to the revised manuscript.

## Where each result comes from

| Paper item | Script | Output in `results/` |
|---|---|---|
| Fig. 1 (method) | `analysis/figures/fig1_method.py` | `figures/fig1_method.pdf` |
| Sections 5.1–5.4, 5.9–5.10 (original study: primary effect, HELIX, ablation, gate sensitivity, severity, context length, safety, runtime) | audit programs in `experiments/original/scripts/`, run by `analysis/original_study.py` | `summaries/original/*.json`, `summaries/severity_and_boundary.csv`, `summaries/context_sensitivity.csv` |
| Fig. 2 (primary result) | `analysis/figures/figures.py` | `figures/fig2_primary.pdf` |
| Section 5.5, estimators table, Fig. 3 | `analysis/estimators.py` | `summaries/estimators.json`, `tables/table_estimators.tex`, `figures/fig3_estimators_gate.pdf` |
| Section 5.6, gate table (supplement) | `analysis/gate_validation.py` | `summaries/gate_validation.json`, `tables/table_s_gate.tex` |
| Section 5.7, recovery table, Fig. 4, channel table (supplement) | `analysis/determinants.py`, `analysis/test_context_consistency.py` | `summaries/determinants.json`, `summaries/test_context_consistency.json`, `tables/table_recovery.tex`, `tables/table_s_channels.tex`, `figures/fig4_determinants.pdf` |
| Section 5.8, breadth table, Fig. 5C | `analysis/breadth.py`, `experiments/check_native_timesfm.py` | `summaries/breadth.json`, `summaries/native_timesfm25.json`, `tables/table_breadth.tex`, `figures/fig5_robustness_native.pdf` |
| Fig. 5A–B | `analysis/figures/figures.py` | `figures/fig5_robustness_native.pdf` |
| Holm-adjusted tests (supplement) | `analysis/tables.py` | `tables/table_s_holm.tex` |
| Agreement of the revision reruns with the original records (supplement) | `analysis/hardware_consistency.py` | `summaries/hardware_consistency.json` |
| Graphical abstract | `analysis/figures/graphical_abstract.py` (needs data, Chronos-2 and a GPU) | `figures/graphical_abstract.pdf` |

`tests/test_results.py` checks the headline numbers of the paper against the summaries, and
`tests/test_records.py` checks the summaries and the package against the run records and the data.

## Level 1: tables and figures (seconds)

```bash
pip install -e ".[analysis,test]"
make tables figures
```

## Level 2: summaries from the run records (about 10 minutes, CPU only)

```bash
make records        # downloads contextguard_records_v3.0.0.zip from Zenodo and verifies its checksum
make summaries tables figures
make test
```

`make consistency` additionally recomputes the test-context consistency (Section 5.7); it needs the
GIFT-Eval data, which `python experiments/download_assets.py` downloads and verifies.

## Level 3: full rerun

Requires an NVIDIA GPU with CUDA 12.8 and [uv](https://github.com/astral-sh/uv).

```bash
experiments/setup_env.sh                 # .venv (Chronos-2, TimesFM 2.5) and .venv-tirex (TiRex)
.venv/bin/python experiments/download_assets.py
experiments/run_revision.sh all          # 3,184 runs into logs/revision/
.venv/bin/python experiments/check_native_timesfm.py
```

Each run takes about 1.5 GB of GPU memory, and all runs are independent and resumable, so sets can be
distributed across processes or machines (`PARALLEL=1`).

The original study is rerun with the matrix runner of the original release, for example

```bash
.venv/bin/python experiments/original/scripts/run_gift_matrix.py --resume \
    --config experiments/original/configs/gift_sensor_v4_formal_test.json
```

Reruns are written to `logs/revision/<set>` and `logs/original/<experiment>`, the same layout as
`records/`, so `CONTEXTGUARD_RECORDS=logs make summaries` analyses a fresh rerun instead of the archive.

## Determinism

The corruption of every window depends only on the fault seed and the global window index, so results
do not depend on batch size or on how runs are distributed. All statistics use fixed seeds (bootstrap
seed 260827, 20,000 replicates). Forecasts are deterministic up to floating-point differences between
GPU architectures, which are below 10⁻⁴ relative for every reported metric.
