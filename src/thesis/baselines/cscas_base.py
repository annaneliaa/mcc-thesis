"""
Same experimental setup as baselines/cscas.py (split, training-pool
sampling, classifier, seeds) -- the only things that change are (a)
FEATURE_COLS and (b) the eval set.

FEATURE_COLS drops, on top of the paper's own DROP_COLS (Timestamp,
SignatureText, Label, ExtIP, IntIP): SignatureID (nominal identifier, not
a real signal -- same reasoning encoders/baseline.py's
compute_cscas_baseline_features already applies), SCAS (the paper's own
CSCAS outlier/inlier flag -- still used to build the "guided" pool's rows
in _sampling.guided_by_cscas_pool, just no longer handed to the classifier
as a feature), and every column ending in "Similarity" (the bare
`Similarity` column plus all 34 per-field `*Similarity` columns). None of
these are things a real deployment could compute for a fresh alert without
already knowing the answer or running the exact same offline similarity
pipeline CSCAS's paper did -- see Docs/Baselines.md. This is a deliberately
narrower cut than compute_cscas_baseline_features (which still includes
SCAS/similarity/attr_value:* today), so the two have diverged; this
script's FEATURE_COLS is the source of truth for what "the reduced
baselines" actually train on now.

(b) the eval set: unlike cscas.py (the paper-replication anchor, which
stays on the full test set forever), this is an internal-system baseline,
so it scores all three conditions on the shared, frozen evaluation
subsample (see _sampling.get_cscas_eval_subsample) for a fair head-to-head
against the other non-replication baselines.

Run:
    cd src/thesis/baselines
    python cscas_base.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from thesis.baselines._cscas_schema import force_recompute
from thesis.baselines._results import results_exist, save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)

# RandomForestClassifier here is CPU-only -- no GPU/device selection in this
# script -- printed for parity with the torch-based baselines' device line.
print("Using device: cpu")

# 1) Load and sort dataset

df = pd.read_csv("../../../data/cscas/dataset-labeled-anon-ip.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# 2) Verify dataset against papers numbers
assert len(df) == 1_395_324, f"got {len(df)}"
assert df["Label"].sum() == 20_952, f"got {df['Label'].sum()}"
assert df["SCAS"].sum() == 72_672, f"got {df['SCAS'].sum()}"

# 3) Split into train and test sets based on timestamp
split_time = pd.Timestamp("2022-01-26 06:23:21+02:00")

train = df[df["Timestamp"] <= split_time].copy()
test = df[df["Timestamp"] > split_time].copy()

assert len(train) == 139_532, f"got {len(train)}"
assert len(test) == 1_255_792, f"got {len(test)}"
assert train["Label"].sum() == 1_765, f"got {train['Label'].sum()}"
assert test["Label"].sum() == 19_187, f"got {test['Label'].sum()}"

# 4) Define feature columns -- see module docstring for what's dropped and why.
DROP_COLS = [
    "Timestamp",
    "SignatureText",
    "Label",
    "ExtIP",
    "IntIP",
    "SignatureID",
    "SCAS",
]
FEATURE_COLS = [
    c for c in df.columns if c not in DROP_COLS and not c.endswith("Similarity")
]

# Sanity check: should be 5 columns -- SignatureMatchesPerDay, AlertCount,
# Proto, ExtPort, IntPort
assert len(FEATURE_COLS) == 5, f"got {len(FEATURE_COLS)}"
print(f"Feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

if not force_recompute() and all(
    results_exist(n) for n in ("cscas_base", "cscas_base_fulltest")
):
    print(
        "[skip] cscas_base + cscas_base_fulltest already exist (CSCAS_FORCE=1 to re-run)."
    )
    raise SystemExit(0)

# 5) Verify training pools against Table IV (pool construction itself now
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval sets. Primary: the shared, frozen 20k subsample (the grid
# every non-replication baseline lives in). We ALSO score every fitted model
# on the FULL 1.26M-row test set (same fitted models, a second .predict())
# so the reduced 5-feature schema has a cell in the paper's own full-test
# protocol -> results/cscas_base_fulltest.json. results/cscas_base.json is
# unchanged.
eval_df = get_cscas_eval_subsample(test)
X_test = eval_df[FEATURE_COLS].values
y_test = eval_df["Label"].values
X_full = test[FEATURE_COLS].values
y_full = test["Label"].values
print(
    f"Evaluating on shared eval subsample: {len(eval_df)} rows, {int(eval_df['Label'].sum())} positive"
    f"  (+ full test set: {len(test)} rows, {int(test['Label'].sum())} positive)"
)

# 7) Three training-pool conditions
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

REFERENCE = {
    "random": "P=0.669, R=0.963, F1=0.789",
    "class_weighted": None,
    "guided": "P=0.868, R=0.952, F1=0.908",
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}
results_full: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


for condition, build_pool in POOL_BUILDERS.items():
    reference = REFERENCE[condition]
    print(f"\n=== {condition} (my 5-feature reduced base schema) ===")
    if reference:
        print(
            f"    Paper reference (their 42 features incl. SignatureID/SCAS/Similarity): {reference}"
        )

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)

        X_tr = pool[FEATURE_COLS].values
        y_tr = pool["Label"].values

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            class_weight=extra_kwargs.get("class_weight"),
        )
        clf.fit(X_tr, y_tr)

        m_sub = _metrics(y_test, clf.predict(X_test))
        m_full = _metrics(y_full, clf.predict(X_full))
        results[condition].append(m_sub)
        results_full[condition].append(m_full)
        print(
            f"  seed={seed}: subsample  P={m_sub['precision']:.3f} R={m_sub['recall']:.3f} F1={m_sub['f1']:.3f}"
            f"   |  full test  P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}"
        )

    avg = pd.DataFrame(results[condition]).mean()
    print(
        f"  AVERAGE (subsample): P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )


print(
    "\n=== Summary: paper (42 features, full test set) vs mine (5 features, "
    "no SignatureID/SCAS/Similarity, shared eval subsample) ==="
)
for condition, reference in REFERENCE.items():
    avg = pd.DataFrame(results[condition]).mean()
    ref_str = f"paper {reference}  |  " if reference else ""
    print(
        f"{condition:<16}"
        f"{ref_str}"
        f"mine P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )

save_baseline_results(
    name="cscas_base",
    description=(
        "This project's reduced base schema (5 features -- SignatureID, SCAS, "
        "and all Similarity columns removed as unrealistic for a real deployment), "
        "RandomForestClassifier(n_estimators=100), evaluated on the shared eval subsample"
    ),
    results=results,
)
save_baseline_results(
    name="cscas_base_fulltest",
    description=(
        "This project's reduced base schema (5 features), "
        "RandomForestClassifier(n_estimators=100), scored on the FULL 1.26M-row "
        "test set (the CSCAS paper's own protocol -- same fitted models as "
        "cscas_base.json; this is the 5-feature / full-test cell of the grid)"
    ),
    results=results_full,
)
