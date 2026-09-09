#!/usr/bin/env bash
#
# Temporal Generalization (Rolling-Horizon Decay): for each scenario, runs
# run_temporal_decay.py over the parameter grid -- every entry in
# MINING_SETTINGS (configs/screening_mining_settings.yaml) crossed with
# GRANULARITIES below, plus a baseline row per granularity. That YAML is the
# single input: no feasible-config CSV, no notebook export step, no
# real-evaluation ranking. Edit the YAML to change what runs. For each
# resulting config, run_temporal_decay.py mines/fits once on the source
# window's train split (see SOURCE_SPLIT_MODE below) and walks the frozen
# schema/model/threshold forward one window at a time, tracking SHAP/LIME
# importances alongside the metric decay -- mined schemas are cached, so
# rerunning this script only (re)mines whatever isn't already cached.
#
# Usage:
#   src/thesis/shell-scripts/system_eval/run_temporal_decay.sh
#
# Edit the variables below to change the scenario(s), granularities,
# threshold mode, or explanation sampling.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Best-effort conda activation. Default env "thesis" (a dev box); override
# with THESIS_CONDA_ENV for a context where it's named differently -- e.g.
# inside a container whose deps live in base:
#   THESIS_CONDA_ENV=base bash run_temporal_decay.sh
# A missing conda or missing env is only a warning: fall through to whatever
# `python` is already active. PYTHONPATH above means the package needn't be
# pip-installed (run_temporal_decay.py also self-inserts src/ as a backstop).
CONDA_ENV="${THESIS_CONDA_ENV:-thesis}"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null \
    || echo "  [warn] 'conda activate $CONDA_ENV' failed -- using $(command -v python)" >&2
fi

if ! python -c "import thesis, sklearn, numpy, pandas" 2>/dev/null; then
  echo "FATAL: active python ($(command -v python)) can't import the core deps" \
       "(thesis/sklearn/numpy/pandas). Set THESIS_CONDA_ENV to the right env." >&2
  exit 1
fi

SCENARIOS=(cscas)
MINING_SETTINGS="$REPO_ROOT/src/thesis/configs/screening_mining_settings.yaml"
GRANULARITIES=(0.1)  # one granularity keeps the run lean; 0.1 gives the most
                     # horizon windows (finest decay/drift curve). Add 0.25 0.5
                     # back for the cross-granularity comparison. Keep to the
                     # mining grid's MINE_FRACS so every gran has structural backing.
# Every model is crossed with every (grid setting x granularity). logreg and
# xgboost are supervised (fit on the mixed W_src train split); iforest and
# ocsvm are one-class -- fit unsupervised on the benign rows, then Platt-scaled
# against the labels so they score like a classifier (see
# experiments._shared.fit_scored_model). THRESHOLD_MODE="fixed" resolves per
# model to its own operating point (0.5 for the classifiers, the
# contamination cut for the one-class models), so precision/recall/FPR are
# meaningful for all of them without a calibration target.
MODELS=(logreg xgboost iforest ocsvm)
THRESHOLD_MODE="fixed"  # or "calibrated_recall"
CALIBRATED_RECALL_TARGET="0.90"  # only used when THRESHOLD_MODE=calibrated_recall
# Source window W_src:
#   window0        -- window 0 at each config's granularity; walk over
#                     windows 1..n-1 of the whole timeline.
#   baseline_split -- W_src = every alert_group at or before the CSCAS
#                     baseline's split_time (baselines/cscas_base.py); the
#                     walk carves the post-split remainder (the baseline's
#                     own test period) into windows, so the decay curve
#                     lines up one-to-one with the aggregate score the
#                     baseline reports on that same test set. CSCAS only;
#                     --source-split-time defaults to the CSCAS boundary.
SOURCE_SPLIT_MODE="baseline_split"  # or "window0"
# CSCAS_FULL=1 adds one cscas_full feature-set row per (granularity, model):
# the CSCAS paper's own full feature set (5 base cols + SCAS + Similarity +
# SignatureIDSimilarity + 33 attr-similarity columns). A non-deployable
# *reference ceiling* -- SCAS and the offline *Similarity scores can't be
# computed for a fresh alert -- for "does the frozen model decay even with
# the paper's full oracle features?". SCAS is dropped for the one-class
# models (it is itself an anomaly score). CSCAS only.
CSCAS_FULL=1  # 0 to skip the cscas_full arm
# SHAP/LIME per horizon. logreg (LinearExplainer) and xgboost (TreeExplainer)
# get analytic SHAP + LIME. iforest/ocsvm have no analytic SHAP explainer, so
# by default (ONECLASS_SHAP=0) they get LIME only -- the PermutationExplainer
# fallback over every feature at every horizon is what used to make this run
# take hours. Set ONECLASS_SHAP=1 to pay for it.
COMPUTE_EXPLANATIONS=1  # 0 to skip SHAP/LIME entirely (metrics + novelty only)
ONECLASS_SHAP=0         # 1 to also compute (slow) SHAP for iforest/ocsvm
EXPLAIN_SAMPLE_N=50
LIME_NUM_SAMPLES=1000

# If explanations are on but shap/lime aren't importable in this env, drop
# to metrics-only rather than failing the whole run partway through.
if [[ "$COMPUTE_EXPLANATIONS" -eq 1 ]] && ! python -c "import shap, lime" 2>/dev/null; then
  echo "  [warn] shap/lime not importable -- running metrics + novelty only" \
       "(COMPUTE_EXPLANATIONS=0)" >&2
  COMPUTE_EXPLANATIONS=0
fi

LOG_DIR="$REPO_ROOT/artifacts/logs/temporal_decay"
mkdir -p "$LOG_DIR"
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"

total=0
failed=()

for scenario in "${SCENARIOS[@]}"; do
  total=$((total + 1))
  log_file="$LOG_DIR/${RUN_TS}_${scenario}.log"

  echo "[$total] $scenario"

  if [[ ! -f "$MINING_SETTINGS" ]]; then
    echo "    FAILED — no mining-settings grid at $MINING_SETTINGS"
    failed+=("$scenario")
    continue
  fi

  # Built up incrementally (rather than expanding a possibly-empty array)
  # since "${empty_array[@]}" errors under `set -u` on bash <4.4 -- macOS's
  # default /usr/bin/bash is 3.2.
  # -u: unbuffered stdout -- without it, redirecting to $log_file makes
  # Python fully block-buffer stdout (prints only flush every ~8KB or at
  # exit) while warnings.warn() writes straight to unbuffered stderr, so the
  # log looks like it's stuck spewing only sklearn warnings for the whole
  # run with none of the "[n/4] ..."/"Saved →" progress prints showing up
  # until the process exits.
  cmd=(python -u "$REPO_ROOT/src/thesis/scripts/system_eval/run_temporal_decay.py" \
    "$scenario" \
    --mining-settings "$MINING_SETTINGS" \
    --granularities "${GRANULARITIES[@]}" \
    --models "${MODELS[@]}" \
    --threshold-mode "$THRESHOLD_MODE" \
    --source-split-mode "$SOURCE_SPLIT_MODE" \
    --explain-sample-n "$EXPLAIN_SAMPLE_N" \
    --lime-num-samples "$LIME_NUM_SAMPLES")
  if [[ "$THRESHOLD_MODE" == "calibrated_recall" ]]; then
    cmd+=(--calibrated-recall-target "$CALIBRATED_RECALL_TARGET")
  fi
  if [[ "$COMPUTE_EXPLANATIONS" -eq 0 ]]; then
    cmd+=(--no-explanations)
  fi
  if [[ "${ONECLASS_SHAP:-0}" -eq 1 ]]; then
    cmd+=(--oneclass-shap)
  fi
  if [[ "${CSCAS_FULL:-0}" -eq 1 ]]; then
    cmd+=(--cscas-full)
  fi

  "${cmd[@]}" >"$log_file" 2>&1

  if [[ $? -ne 0 ]]; then
    echo "    FAILED — see $log_file"
    failed+=("$scenario")
  else
    grep -E "Temporal decay results|Saved →|Mirrored →" "$log_file" | tail -n 6 | sed 's/^/    /'
  fi
done

echo
echo "============================================================"
echo "  TEMPORAL DECAY SUMMARY: $((total - ${#failed[@]}))/$total succeeded"
echo "============================================================"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed scenarios:"
  for scenario in "${failed[@]}"; do
    echo "  - $scenario"
  done
  exit 1
fi
