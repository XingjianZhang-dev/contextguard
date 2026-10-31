# Run records

The run records are archived on Zenodo ([10.5281/zenodo.22124598](https://doi.org/10.5281/zenodo.22124598),
file `contextguard_records_v3.0.0.zip`) and unpack into `records/`:

```
records/
  original/                      1,760 runs of the original study (Sections 5.1-5.4, 5.9-5.10)
    gift_sensor_v4_formal_test/  primary design: 320 runs, 160 paired cells
    gift_helix_formal_test/      trained HELIX imputer on the same cells
    ...                          severity, context, reliability, ablation, random, block, clean, reproduction
  revision/                      3,184 runs of the revision (Sections 5.5-5.8)
    primary/                     8 configurations x {Chronos-2, TimesFM 2.5} x 8 methods      1,280
    primary_clean/               clean-input reference for the recovered fraction                32
    heldout/                     6 hourly configurations x 3 forecasters x 5 methods (+native)  960
    heldout_clean/                                                                               36
    tirex/                       TiRex on the 8 primary configurations x 8 methods              640
    tirex_clean/                                                                                 16
    native/                      native missing-value input (Chronos-2, TiRex)                  220
    native_timesfm25.json        native missing-value input for TimesFM 2.5
```

Every cell of the corrupted sets is one (configuration, forecaster, severity in {1, 3}, fault seed in
201-205). Context length 512, up to 256 official test windows per configuration.

## One run

Each run directory is named `<dataset>_<model>_<corruption>_<severity>_<method>_<seed>` and holds

* `manifest.json`: arguments, dataset fingerprint, model SHA-256, aggregate metrics (`mse`, `mae`,
  `quantile_loss`, `coverage80`, `width80`, `raw_mse`), training boundaries, timings, and the training
  statistics of every estimator: `gates` (per channel `a`, `I`, `g`, `n` for ContextGuard, the cross-fitted
  gate, ridge and Ledoit-Wolf), `ridge` (selected penalty and validation scores) and `lw` (shrinkage).
* `window_metrics.npz`: per-window losses (`mse`, `mae`, `quantile_loss`, `coverage80`, `width80`,
  `raw_mse`), per-window and per-channel losses (`pc_mse`, `pc_mae`, `pc_ql`), the failed-channel mask
  (`dropped`), natural missingness (`natural_any`), reconstruction errors of the failed coordinates for the
  method, the zero fallback and the pooled-mean fallback (`rec_method`, `rec_zero`, `rec_pooled`), and the
  window identity (`record_index`, `forecast_start`).

Losses are computed on standardized values (training mean and scale of each series); `raw_mse` is in
original units.

## Method names

| Record name | Method in the paper |
|---|---|
| `topology` | topology-aware temporal repair (reference) |
| `contextguard_adaptive` | **ContextGuard** (gated) |
| `contextguard` | ContextGuard without the gate (weight one) |
| `contextguard_crossfit` | ContextGuard with the cross-fitted gate |
| `ridge_ungated`, `ridge_gated` | ridge-regression reconstruction without / with the gate |
| `lw_ungated`, `lw_gated` | Ledoit-Wolf reconstruction without / with the gate |
| `nan_native` | failed channel passed to the forecaster as missing values |
| `helix` | trained HELIX imputer (original study) |

The names follow the original release, so that records of both studies can be joined directly.
