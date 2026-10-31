#!/usr/bin/env bash
# Run the revision experiment sets. Usage: experiments/run_revision.sh {primary|heldout|tirex|native|all}
#
#   primary   8 configurations x {Chronos-2, TimesFM 2.5} x 8 methods (1,280 runs) + clean reference (32)
#   heldout   6 hourly configurations x {Chronos-2, TimesFM 2.5, TiRex} x 5 methods, plus native
#             missing-value input for Chronos-2 (960 runs) + clean reference (36)
#   tirex     TiRex on the 8 primary configurations x 8 methods (640 runs) + clean reference (16)
#   native    native missing-value input on the primary configurations (Chronos-2, TiRex) and the
#             held-out configurations (TiRex) (220 runs)
#
# Every cell uses severities {1, 3}, fault seeds 201-205, context 512 and up to 256 test windows.
# Runs are resumable (--resume) and independent, so jobs may be distributed freely; set PARALLEL=1 to
# launch one process per (configuration, forecaster). Outputs go to $OUT (default logs/revision).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-logs/revision}
PARALLEL=${PARALLEL:-0}
ALL8="topology contextguard_adaptive contextguard contextguard_crossfit ridge_ungated ridge_gated lw_ungated lw_gated"
HELD="topology contextguard_adaptive contextguard ridge_gated lw_gated"

python_for() { [ "$1" = tirex ] && echo .venv-tirex/bin/python || echo .venv/bin/python; }
model_path() {
  case $1 in
    chronos2) echo models/amazon-chronos-2 ;;
    timesfm25) echo models/google-timesfm-2.5-200m-pytorch ;;
    tirex) echo NX-AI/TiRex ;;
  esac
}
batch_for() { [ "$1" = tirex ] && echo 256 || echo 32; }

# evaluate <set> <key> <path> <horizon> <model> <methods...>   (+ the clean reference unless set=native)
evaluate() {
  local set=$1 key=$2 path=$3 horizon=$4 model=$5; shift 5
  local common=(--dataset-key "$key" --dataset-path "data/GiftEval/$path" --prediction-length "$horizon"
                --model "$model" --model-path "$(model_path "$model")" --batch-size "$(batch_for "$model")" --resume)
  "$(python_for "$model")" experiments/evaluate.py "${common[@]}" --repairs "$@" --output-root "$OUT/$set"
  if [ "$set" != native ]; then
    "$(python_for "$model")" experiments/evaluate.py "${common[@]}" --corruption clean \
      --repairs topology contextguard_adaptive --output-root "$OUT/${set}_clean"
  fi
}
launch() { if [ "$PARALLEL" = 1 ]; then evaluate "$@" & else evaluate "$@"; fi; }

run_primary() {
  while IFS=$'\t' read -r key path horizon; do
    for model in chronos2 timesfm25; do launch primary "$key" "$path" "$horizon" "$model" $ALL8; done
  done < experiments/configs/primary.tsv
}
run_heldout() {
  while IFS=$'\t' read -r key path horizon; do
    launch heldout "$key" "$path" "$horizon" chronos2 $HELD nan_native
    launch heldout "$key" "$path" "$horizon" timesfm25 $HELD
    launch heldout "$key" "$path" "$horizon" tirex $HELD
  done < experiments/configs/heldout.tsv
}
run_tirex() {
  while IFS=$'\t' read -r key path horizon; do launch tirex "$key" "$path" "$horizon" tirex $ALL8; done \
    < experiments/configs/primary.tsv
}
run_native() {
  while IFS=$'\t' read -r key path horizon; do
    launch native "$key" "$path" "$horizon" chronos2 nan_native
    launch native "$key" "$path" "$horizon" tirex nan_native
  done < experiments/configs/primary.tsv
  while IFS=$'\t' read -r key path horizon; do launch native "$key" "$path" "$horizon" tirex nan_native; done \
    < experiments/configs/heldout.tsv
}

case ${1:-all} in
  primary) run_primary ;;
  heldout) run_heldout ;;
  tirex) run_tirex ;;
  native) run_native ;;
  all) run_primary; run_heldout; run_tirex; run_native ;;
  *) echo "usage: $0 {primary|heldout|tirex|native|all}" >&2; exit 2 ;;
esac
wait
