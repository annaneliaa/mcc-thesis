from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from thesis.system_eval.temporal_decay import (
    WindowScheme,
    _build_decay_summary,
    _cscas_full_schema,
    _group_type_keys,
    _novelty_metrics,
    _resolve_baseline_split_idx,
)
from thesis.schemas.groups import AlertGroup
from thesis.experiments._shared import (
    decide_threshold,
    fit_scored_model,
    metrics_at_threshold,
)
from thesis.metrics.shortlist import load_shortlist
from thesis.pipeline.pipeline import compute_window_bounds


# ---- WindowScheme -----------------------------------------------------------


@pytest.mark.parametrize("gran", [0.1, 0.25, 0.5])
def test_window_scheme_window0_matches_compute_window_bounds(gran):
    n = 1_000_000
    s = WindowScheme("window0", n)
    assert s.n_windows(gran) == compute_window_bounds(n, gran, 0)[2]
    assert s.source_bounds(gran) == compute_window_bounds(n, gran, 0)[:2]
    assert s.target_bounds(gran, 3) == compute_window_bounds(n, gran, 3)[:2]


@pytest.mark.parametrize("gran", [0.1, 0.25, 0.5])
def test_window_scheme_baseline_split_walks_only_the_post_split_tail(gran):
    n, split = 1_000_000, 100_000
    s = WindowScheme("baseline_split", n, split)
    n_fwd = compute_window_bounds(n - split, gran, 0)[2]
    # h=0 anchor + one horizon per post-split window
    assert s.n_windows(gran) == n_fwd + 1
    # W_src is the whole pre-split region
    assert s.source_bounds(gran) == (0, split)
    # h=1 starts exactly at the split; the last horizon ends at the timeline end
    assert s.target_bounds(gran, 1)[0] == split
    assert s.target_bounds(gran, n_fwd)[1] == n
    # horizons are contiguous and never re-enter the training region
    prev_end = split
    for h in range(1, n_fwd + 1):
        start, end = s.target_bounds(gran, h)
        assert start == prev_end
        prev_end = end


def test_resolve_baseline_split_idx_matches_leq_cutoff():
    cutoff_iso = "2022-01-26 06:23:21+02:00"
    cut = int(pd.Timestamp(cutoff_iso).timestamp())

    class _G:
        def __init__(self, ts):
            self.start_ts = ts

    groups = [_G(cut - 100), _G(cut - 1), _G(cut), _G(cut + 1), _G(cut + 100)]
    # groups at or before the cutoff go to train (mirrors `Timestamp <= split_time`)
    assert _resolve_baseline_split_idx(groups, cutoff_iso) == 3


# ---- _build_decay_summary ----------------------------------------------------


def _horizon_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "feature_set": "symbolic",
        "mining_setting": "gr3.0_md4",
        "granularity": 0.1,
        "model": "logreg",
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_build_decay_summary_uses_h0_and_last_horizon_reached():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": 0.99, "fpr": 0.01},
            {"horizon_window_index": 1, "auc": 0.95, "fpr": 0.02},
            {"horizon_window_index": 3, "auc": 0.90, "fpr": 0.03},
        ]
    )
    summary = _build_decay_summary(df)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["h_max"] == 3
    assert row["auc_at_h0"] == pytest.approx(0.99)
    assert row["auc_at_h3"] == pytest.approx(0.90)
    assert row["decay_rate_auc"] == pytest.approx(0.09)
    assert row["fpr_at_h0"] == pytest.approx(0.01)
    assert row["fpr_at_h3"] == pytest.approx(0.03)
    assert row["fpr_drift"] == pytest.approx(0.02)


def test_build_decay_summary_one_row_per_config():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": 0.99, "fpr": 0.01, "model": "logreg"},
            {"horizon_window_index": 2, "auc": 0.91, "fpr": 0.03, "model": "logreg"},
            {"horizon_window_index": 0, "auc": 0.97, "fpr": 0.02, "model": "rf"},
            {"horizon_window_index": 2, "auc": 0.85, "fpr": 0.05, "model": "rf"},
        ]
    )
    summary = _build_decay_summary(df)
    assert len(summary) == 2
    assert set(summary["model"]) == {"logreg", "rf"}


def test_build_decay_summary_nan_when_either_end_is_nan():
    df = _horizon_df(
        [
            {"horizon_window_index": 0, "auc": np.nan, "fpr": 0.01},
            {"horizon_window_index": 1, "auc": 0.90, "fpr": 0.02},
        ]
    )
    summary = _build_decay_summary(df)
    assert np.isnan(summary.iloc[0]["decay_rate_auc"])


def test_build_decay_summary_empty_input():
    assert _build_decay_summary(pd.DataFrame()).empty


# ---- load_shortlist ----------------------------------------------------------


def _write_csv(tmp_path, rows: list[dict]) -> Path:
    path = tmp_path / "shortlist.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_shortlist_valid_round_trip(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "logreg",
            },
            {
                "feature_set": "symbolic",
                "mining_setting": "gr3.0_md3",
                "granularity": 0.2,
                "model": "logreg",
            },
        ],
    )
    configs = load_shortlist(path)
    assert len(configs) == 2
    assert configs[0].feature_set == "baseline"
    assert configs[0].mining_setting is None
    assert configs[1].mining_setting == "gr3.0_md3"
    assert configs[1].granularity == pytest.approx(0.2)


def test_load_shortlist_missing_column_raises(tmp_path):
    path = tmp_path / "shortlist.csv"
    pd.DataFrame(
        [{"feature_set": "baseline", "granularity": 0.2, "model": "logreg"}]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="mining_setting"):
        load_shortlist(path)


def test_load_shortlist_accepts_cscas_full_without_mining_setting(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "cscas_full",
                "mining_setting": "",
                "granularity": 0.25,
                "model": "iforest",
            }
        ],
    )
    configs = load_shortlist(path)
    assert configs[0].feature_set == "cscas_full"
    assert configs[0].mining_setting is None


def test_load_shortlist_cscas_full_with_mining_setting_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "cscas_full",
                "mining_setting": "gr3.0_md3",
                "granularity": 0.25,
                "model": "logreg",
            }
        ],
    )
    with pytest.raises(ValueError, match="must not carry"):
        load_shortlist(path)


def test_load_shortlist_baseline_with_mining_setting_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "gr3.0_md3",
                "granularity": 0.2,
                "model": "logreg",
            }
        ],
    )
    with pytest.raises(ValueError):
        load_shortlist(path)


def test_load_shortlist_symbolic_without_mining_setting_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "symbolic",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "logreg",
            }
        ],
    )
    with pytest.raises(ValueError):
        load_shortlist(path)


def test_load_shortlist_unknown_model_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {
                "feature_set": "baseline",
                "mining_setting": "",
                "granularity": 0.2,
                "model": "not_a_real_model",
            }
        ],
    )
    with pytest.raises(ValueError, match="not_a_real_model"):
        load_shortlist(path)


# ---- fit_scored_model -------------------------------------------------------


def _separable_split(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = np.r_[np.zeros(270), np.ones(30)].astype(int)
    X.loc[y == 1] += 2.5  # attacks shifted away from the benign cloud
    return X, y


@pytest.mark.parametrize("model_name", ["logreg", "iforest", "ocsvm"])
def test_fit_scored_model_exposes_attack_probability(model_name):
    X, y = _separable_split()
    model = fit_scored_model(model_name, X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    # attacks are cleanly separable here -> mean attack proba > mean benign proba
    assert proba[y == 1].mean() > proba[y == 0].mean()


def test_fit_scored_model_returns_none_when_split_cannot_fit():
    X, y = _separable_split()
    benign_only = np.zeros(len(X), dtype=int)
    assert (
        fit_scored_model("logreg", X, benign_only) is None
    )  # supervised needs both classes
    assert (
        fit_scored_model("iforest", X, benign_only) is None
    )  # one-class needs an attack to calibrate


def _imbalanced_split(seed: int = 2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        np.vstack([rng.normal(size=(1500, 6)), rng.normal(size=(40, 6)) + 1.4]),
        columns=[f"f{i}" for i in range(6)],
    )
    y = np.r_[np.zeros(1500), np.ones(40)].astype(int)
    return X, y


@pytest.mark.parametrize("model_name", ["iforest", "ocsvm"])
def test_fixed_threshold_for_one_class_is_the_contamination_cut_not_half(model_name):
    # A flat 0.5 in Platt-probability space predicts everything benign for a
    # one-class model on this imbalance -- "fixed" must instead resolve to the
    # detector's own contamination operating point.
    X, y = _imbalanced_split()
    model = fit_scored_model(model_name, X, y)
    proba = model.predict_proba(X)[:, 1]

    assert model.default_threshold < 0.5
    thr = decide_threshold(y, proba, "fixed", 0.9, model=model)
    assert thr == pytest.approx(model.default_threshold)

    met = metrics_at_threshold(y, proba, thr)
    assert met["tp"] > 0 and met["recall"] > 0  # not the all-benign degenerate case
    # the cut reproduces the detector's own predict()
    hard = (proba >= thr).astype(int)
    assert (hard == (model.inner.predict(X) == -1).astype(int)).mean() > 0.99


def test_fixed_threshold_for_supervised_model_stays_at_half():
    X, y = _separable_split()
    model = fit_scored_model("logreg", X, y)
    assert (
        decide_threshold(y, model.predict_proba(X)[:, 1], "fixed", 0.9, model=model)
        == 0.5
    )


# ---- _cscas_full_schema ----------------------------------------------------


def test_cscas_full_schema_keeps_scas_for_classifiers_drops_it_for_one_class():
    clf = _cscas_full_schema("logreg")
    assert clf.base.kind == "cscas_full"
    assert "scas" in clf.base.features
    assert clf.symbolic is None

    for one_class in ("iforest", "ocsvm"):
        sch = _cscas_full_schema(one_class)
        assert "scas" not in sch.base.features
        assert len(sch.base.features) == len(clf.base.features) - 1


# ---- per-horizon novelty --------------------------------------------------


def _grp(items, category="EXPLOIT", ruleset="ET", proto=6, label="benign"):
    return AlertGroup(
        alert_group_id="x",
        group_id="x",
        method="cscas_pregrouped",
        start_ts=1_642_636_800,
        end_ts=1_642_636_800,
        n_alerts=1,
        group_label=label,
        raw_items=set(items),
        category=category,
        ruleset=ruleset,
        proto=proto,
    )


def test_novelty_metrics_counts_unseen_type_keys():
    train = [_grp(["a", "b"]), _grp(["c"], category="DNS", proto=17)]
    train_items, train_crp = _group_type_keys(train)

    window = [
        _grp(["a", "b"]),  # seen items + crp
        _grp(["d"], label="attack"),  # novel items, seen crp
        _grp(["e"], category="SQL", proto=6, label="attack"),  # novel items + crp
    ]
    nov = _novelty_metrics(window, train_items, train_crp)

    assert nov["n_groups_win"] == 3
    assert nov["n_attack_win"] == 2
    assert nov["n_novel_items"] == 2
    assert nov["frac_novel_items"] == pytest.approx(2 / 3)
    assert nov["n_new_item_types"] == 2
    assert nov["n_novel_items_attack"] == 2
    assert nov["frac_novel_items_attack"] == pytest.approx(1.0)
    assert nov["n_novel_crp"] == 1
    assert nov["n_new_crp_types"] == 1
    assert nov["n_novel_crp_attack"] == 1


def test_novelty_metrics_empty_window_is_zero_counts_nan_fracs():
    nov = _novelty_metrics([], set(), set())
    assert nov["n_groups_win"] == 0
    assert nov["n_novel_items"] == 0
    assert np.isnan(nov["frac_novel_items"])
    assert np.isnan(nov["frac_novel_crp_attack"])


def test_novelty_metrics_attack_fracs_nan_when_window_has_no_attacks():
    train_items, train_crp = _group_type_keys([_grp(["a"])])
    nov = _novelty_metrics([_grp(["z"])], train_items, train_crp)
    assert nov["n_novel_items"] == 1
    assert nov["n_attack_win"] == 0
    assert np.isnan(nov["frac_novel_items_attack"])
