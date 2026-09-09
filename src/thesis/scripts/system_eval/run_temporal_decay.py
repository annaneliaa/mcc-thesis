"""
Temporal Generalization (Rolling-Horizon Decay) CLI.

Runs experiments.temporal_decay.run_temporal_decay_experiment for one
scenario: for each shortlisted (feature_set, mining_setting, granularity,
model) config, mines a schema + fits a model once on window 0's train split,
then walks the frozen schema/model/threshold forward one window at a time to
the end of the timeline, tracking SHAP/LIME importances alongside the metric
decay at every step. See experiments/temporal_decay.py's module docstring
for the full experiment design, and thesis.metrics.shortlist for the
shortlist file format (feature_set,mining_setting,granularity,model columns).

By default the shortlist is built on the fly as every entry in the
mining-settings YAML (configs/screening_mining_settings.yaml -- the curated
downstream parameter grid) crossed with --granularities x --models, plus a
baseline row per (granularity, model). That YAML is the single input: no
feasible-config CSV, no notebook export step, no real-evaluation ranking.
To change what runs, edit the YAML. --shortlist overrides this with a
pre-built CSV.

Usage:
  # The parameter grid (configs/screening_mining_settings.yaml) x granularities
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --granularities 0.1 0.25 0.5

  # A different grid file
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --mining-settings path/to/other_grid.yaml --granularities 0.1 0.25 0.5

  # Calibrated-recall threshold instead of flat 0.5, metrics only (no SHAP/LIME)
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --granularities 0.1 0.25 0.5 \\
      --threshold-mode calibrated_recall --calibrated-recall-target 0.9 \\
      --no-explanations

  # Override with a hand-built shortlist CSV
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --shortlist my_shortlist.csv

  # Anchor W_src to the CSCAS baseline's train/test split so the decay curve
  # lines up with the baseline experiment's aggregate score (metrics only)
  python src/thesis/scripts/system_eval/run_temporal_decay.py cscas \\
      --granularities 0.1 0.25 0.5 --models logreg xgboost iforest ocsvm \\
      --source-split-mode baseline_split --no-explanations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thesis.config import GroupingConfig
from thesis.configs import dataset_for_scenario
from thesis.system_eval.temporal_decay import run_temporal_decay_experiment
from thesis.grouping.group_alerts import CSCAS_PREGROUPED_METHOD
from thesis.schemas.experiments import TemporalDecayConfig
from thesis.scripts.system_eval._common import (
    build_shortlist_from_mining_grid,
    cache_dir_for,
)

_HERE = Path(__file__).resolve()
_REPO = next(p for p in _HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_REPO / "src"))

_DEFAULT_MINING_SETTINGS = (
    _REPO / "src" / "thesis" / "configs" / "screening_mining_settings.yaml"
)

# The CSCAS baseline's chronological train/test boundary -- mirrors
# baselines/cscas_base.py's split_time. Used as the default
# --source-split-time when --source-split-mode baseline_split is requested
# for cscas without an explicit instant.
_CSCAS_BASELINE_SPLIT_TIME = "2022-01-26 06:23:21+02:00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal decay: freeze a schema+model mined/trained on window 0's "
        "train split, evaluate + explain one window at a time to the end of the timeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("scenario", help="Scenario name (e.g. cscas, fox).")
    parser.add_argument(
        "--shortlist",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Pre-built shortlist CSV (feature_set,mining_setting,granularity,model). "
            "Optional -- default is every entry in --mining-settings x --granularities "
            "x --models."
        ),
    )
    parser.add_argument(
        "--granularities",
        nargs="+",
        type=float,
        default=None,
        metavar="FRAC",
        help="Granularities to cross with the mining-settings grid. Required unless --shortlist is given.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg"],
        metavar="MODEL",
        help=(
            "Models to cross with the mining-settings grid (ignored with --shortlist). "
            "Supervised (logreg, xgboost, rf, ...) or one-class anomaly (iforest, ocsvm -- "
            "fit on benign rows, Platt-scaled to a probability). Default: logreg"
        ),
    )
    parser.add_argument(
        "--no-baseline",
        action="store_false",
        dest="include_baseline",
        help="Don't add a baseline row per (granularity, model) to the derived shortlist.",
    )
    parser.add_argument(
        "--cscas-full",
        action="store_true",
        dest="include_cscas_full",
        help=(
            "Add a cscas_full feature-set row per (granularity, model): the CSCAS "
            "paper's full feature set (5 base cols + SCAS + Similarity + "
            "SignatureIDSimilarity + 33 attr-similarity columns). A non-deployable "
            "reference ceiling -- SCAS and the offline *Similarity scores can't be "
            "computed for a fresh alert. SCAS is dropped for one-class models "
            "(it's itself an anomaly inlier/outlier score). CSCAS only."
        ),
    )
    parser.add_argument(
        "--cscas-full-symbolic",
        action="store_true",
        dest="include_cscas_full_symbolic",
        help=(
            "Add a cscas_full_symbolic row per (mining setting, granularity, "
            "model): the full CSCAS columns PLUS a schema mined on W_src "
            "(shared base columns encoded once, not doubled). CSCAS only."
        ),
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        dest="train_frac",
        metavar="FRAC",
        help="W_src (window 0) internal train/test split fraction. Default: 0.7",
    )
    parser.add_argument(
        "--source-split-mode",
        choices=["window0", "baseline_split"],
        default="window0",
        dest="source_split_mode",
        help=(
            "What the source window W_src is. 'window0' (default): window 0 at "
            "each config's granularity, walk over windows 1..n-1 of the whole "
            "timeline. 'baseline_split': W_src = every alert_group at or before "
            "--source-split-time (the CSCAS baseline's own train/test boundary), "
            "walk carves the post-split remainder (the baseline's test period) "
            "into windows -- so the decay curve lines up with the baseline's "
            "aggregate score. CSCAS only."
        ),
    )
    parser.add_argument(
        "--source-split-time",
        default=None,
        dest="source_split_time",
        metavar="ISO8601",
        help=(
            "Train/test boundary instant for --source-split-mode baseline_split "
            f"(e.g. '{_CSCAS_BASELINE_SPLIT_TIME}'). Defaults to the CSCAS "
            "baseline's split_time when the scenario is cscas."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "calibrated_recall"],
        default="fixed",
        dest="threshold_mode",
        help="How the frozen decision threshold is chosen from W_src's own train-split data. Default: fixed (0.5)",
    )
    parser.add_argument(
        "--calibrated-recall-target",
        type=float,
        default=0.90,
        dest="calibrated_recall_target",
        metavar="RECALL",
        help="Recall target for --threshold-mode calibrated_recall. Default: 0.90",
    )
    parser.add_argument(
        "--no-explanations",
        action="store_false",
        dest="compute_explanations",
        help="Skip SHAP/LIME importance tracking (metrics only, much faster).",
    )
    parser.add_argument(
        "--oneclass-shap",
        action="store_true",
        dest="oneclass_shap",
        help=(
            "Also compute SHAP for the one-class models (iforest, ocsvm). Off "
            "by default -- they have no analytic explainer, so SHAP falls back "
            "to PermutationExplainer over every feature at every horizon, which "
            "dominates an explanations run (hours vs minutes). Without this "
            "flag the one-class models get LIME importances only."
        ),
    )
    parser.add_argument(
        "--explain-background-n",
        type=int,
        default=100,
        dest="explain_background_n",
        metavar="N",
        help="Rows sampled from W_src's train split as the SHAP/LIME background set. Default: 100",
    )
    parser.add_argument(
        "--explain-sample-n",
        type=int,
        default=50,
        dest="explain_sample_n",
        metavar="N",
        help="Rows sampled from each horizon window to explain. Default: 50",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=1000,
        dest="lime_num_samples",
        metavar="N",
        help="Perturbed samples LIME draws per explained row. Default: 1000",
    )
    parser.add_argument(
        "--mining-settings",
        type=Path,
        default=_DEFAULT_MINING_SETTINGS,
        dest="mining_settings",
        metavar="YAML",
        help=f"Named mining-setting grid axis (same file Experiment 1 used). Default: {_DEFAULT_MINING_SETTINGS}",
    )
    parser.add_argument(
        "--filtered",
        nargs="?",
        const="",
        default=None,
        metavar="METHOD",
        help="AIT-ADS only: use filtered alerts. Optionally a balancing method (e.g. naive50).",
    )
    parser.add_argument(
        "--window-size", type=int, default=2, metavar="W", dest="window_size"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any cached mined schema and re-mine.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        dest="n_jobs",
        metavar="N",
        help="Shortlisted configs to run concurrently (thread pool). Default: 4",
    )
    parser.add_argument("--output-dir", type=Path, default=None, dest="output_dir")
    args = parser.parse_args()

    if args.shortlist is not None:
        shortlist_path = args.shortlist
    else:
        if not args.granularities:
            parser.error("--granularities is required unless --shortlist is given")
        derived = build_shortlist_from_mining_grid(
            args.mining_settings,
            args.granularities,
            args.models,
            args.include_baseline,
            args.include_cscas_full,
            args.include_cscas_full_symbolic,
        )
        derived_dir = (
            _REPO / "artifacts" / "experiments" / "temporal_decay" / args.scenario
        )
        derived_dir.mkdir(parents=True, exist_ok=True)
        shortlist_path = derived_dir / "_derived_shortlist.csv"
        derived.to_csv(shortlist_path, index=False)
        print(
            f"  Shortlist ({len(derived)} configs) = {args.mining_settings.name} "
            f"× granularities={args.granularities} × models={args.models} → {shortlist_path}"
        )

    scenario = args.scenario
    is_cscas = dataset_for_scenario(scenario) == "cscas"

    source_split_time = args.source_split_time
    if args.source_split_mode == "baseline_split":
        if source_split_time is None and is_cscas:
            source_split_time = _CSCAS_BASELINE_SPLIT_TIME
        if source_split_time is None:
            parser.error(
                "--source-split-mode baseline_split needs --source-split-time "
                "(no baseline boundary is known for a non-cscas scenario)"
            )
    grouping = (
        GroupingConfig(mode=CSCAS_PREGROUPED_METHOD)
        if is_cscas
        else GroupingConfig(window_size=args.window_size)
    )
    method = args.filtered if args.filtered else None
    filtered = args.filtered is not None
    alerts_filename = (
        f"alerts_filtered_{method}.json" if method else "alerts_filtered.json"
    )
    alerts_path = (
        _REPO / "artifacts" / "processed-data" / scenario / alerts_filename
        if filtered and not is_cscas
        else None
    )
    cache_dir = cache_dir_for(scenario, filtered, method, args.window_size)
    results_dir = args.output_dir / scenario if args.output_dir is not None else None

    config = TemporalDecayConfig(
        scenario=scenario,
        shortlist_path=shortlist_path,
        train_frac_within_window=args.train_frac,
        source_split_mode=args.source_split_mode,
        source_split_time=source_split_time,
        mining_settings_path=args.mining_settings,
        threshold_mode=args.threshold_mode,
        calibrated_recall_target=args.calibrated_recall_target,
        compute_explanations=args.compute_explanations,
        oneclass_shap=args.oneclass_shap,
        explain_background_n=args.explain_background_n,
        explain_sample_n=args.explain_sample_n,
        lime_num_samples=args.lime_num_samples,
        cache_dir=cache_dir,
        grouping=grouping,
        alerts_json_path=alerts_path,
        results_dir=results_dir,
        force_remine=args.force,
        n_jobs=args.n_jobs,
    )
    out_dir = run_temporal_decay_experiment(config)
    print(f"\n[{scenario}] Temporal decay results → {out_dir}")


if __name__ == "__main__":
    main()
