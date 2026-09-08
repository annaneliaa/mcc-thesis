"""
Reproduces the CSCAS paper's own two baselines (Table IV) -- random
undersampling and guided by CSCAS's SCAS outlier clusters -- using the
paper's own 42 raw feature columns and a RandomForestClassifier, averaged
over 5 seeds. Also runs a third, non-paper condition (class-weighted,
natural-ratio) alongside them, per the project's extended baseline design
(see Docs/Baselines.md).

All three conditions evaluate on the FULL test set, for all three -- this
script is the anchor replication target, and matching the paper's published
F1=0.908 (guided) requires exactly that protocol. The new class-weighted
condition has no published target to match; it just needs to run cleanly.

Run:
    cd src/thesis/baselines
    python cscas.py

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

# 4) Define feature columns
DROP_COLS = ["Timestamp", "SignatureText", "Label", "ExtIP", "IntIP"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

# Sanity check: should be 42 columns
# SignatureID, SignatureMatchesPerDay, AlertCount, Proto,
# ExtPort, IntPort, Similarity, SCAS,
# + 34 AttrSimilarity columns
print(f"Feature count: {len(FEATURE_COLS)}")
print(FEATURE_COLS)

if not force_recompute() and all(
    results_exist(n) for n in ("cscas", "cscas_subsample")
):
    print("[skip] cscas + cscas_subsample already exist (CSCAS_FORCE=1 to re-run).")
    raise SystemExit(0)

# 5) Verify training pools against Table IV (pool construction itself now
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare test sets. The full test set is the replication protocol (see
# docstring). We ALSO score every fitted model on the shared 20k eval
# subsample -- same fitted models, a second .predict() -- so the 42-feature
# RF has a cell in the shared-subsample comparison grid the other baselines
# live in (-> results/cscas_subsample.json). The primary results/cscas.json
# output is unchanged.
X_test = test[FEATURE_COLS].values
y_test = test["Label"].values

eval_sub = get_cscas_eval_subsample(test)
X_sub = eval_sub[FEATURE_COLS].values
y_sub = eval_sub["Label"].values

# 7) Three training-pool conditions
POOL_BUILDERS = {
    "random": lambda seed: random_undersample_pool(train, important, seed),
    "class_weighted": lambda seed: class_weighted_pool(train, seed=seed),
    "guided": lambda seed: guided_by_cscas_pool(train, important, seed),
}

TARGETS = {
    "random": "P=0.669, R=0.963, F1=0.789",
    "class_weighted": None,
    "guided": "P=0.868, R=0.952, F1=0.908",
}

results: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}
results_sub: dict[str, list[dict[str, float]]] = {name: [] for name in POOL_BUILDERS}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


for condition, build_pool in POOL_BUILDERS.items():
    target = TARGETS[condition]
    print(f"\n=== {condition} ===")
    if target:
        print(f"    Paper target: {target}")

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

        m_full = _metrics(y_test, clf.predict(X_test))
        m_sub = _metrics(y_sub, clf.predict(X_sub))
        results[condition].append(m_full)
        results_sub[condition].append(m_sub)
        print(
            f"  seed={seed}: full  P={m_full['precision']:.3f} R={m_full['recall']:.3f} F1={m_full['f1']:.3f}"
            f"   |  subsample  P={m_sub['precision']:.3f} R={m_sub['recall']:.3f} F1={m_sub['f1']:.3f}"
        )

    avg = pd.DataFrame(results[condition]).mean()
    print(
        f"  AVERAGE (full test): P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
    )


# Scenario                          Expected P      Expected R  Expected F1
# random (undersampling)            0.669           0.963       0.789
# guided (by CSCAS)                 0.868           0.952       0.908
# class_weighted                    -- no published target --

save_baseline_results(
    name="cscas",
    description="Paper's own 42 raw features, RandomForestClassifier(n_estimators=100)",
    results=results,
)
save_baseline_results(
    name="cscas_subsample",
    description=(
        "Paper's own 42 raw features, RandomForestClassifier(n_estimators=100), "
        "scored on the shared frozen 20k eval subsample (same fitted models as "
        "cscas.json -- this is the 42-feature / subsample cell of the comparison grid)"
    ),
    results=results_sub,
)
