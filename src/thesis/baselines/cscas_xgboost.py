"""
Same experimental setup as cscas_base.py (base schema, training-pool
sampling, shared eval subsample, 5 seeds) -- swaps RandomForestClassifier
for XGBClassifier, the "modern default tabular baseline" in the project's
baseline design (see Docs/Baselines.md). A straight swap of the RF
pattern: XGBoost, like RF, is tree-based and handles the `-1` "not
applicable" sentinel the same way RF does (splits just treat -1 as a very
low value) -- no scaling, no sentinel imputation needed, unlike
cscas_logreg.py.

Run:
    cd src/thesis/baselines
    python cscas_xgboost.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from thesis.baselines._cscas_schema import (
    SCHEMAS_CLASSIFIER,
    active_schema,
    cscas_feature_cols,
    grid_outputs_done,
    result_name,
    schema_blurb,
)
from thesis.baselines._results import save_baseline_results
from thesis.baselines._sampling import (
    class_weighted_pool,
    get_cscas_eval_subsample,
    guided_by_cscas_pool,
    random_undersample_pool,
)

# XGBClassifier here runs CPU-only (no tree_method/device set to use a GPU)
# -- printed for parity with the torch-based baselines' device line.
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

# 4) Feature schema -- CSCAS_SCHEMA env var picks "base" (5 cols, same as
# cscas_base.py) or "full" (the paper's 42). See _cscas_schema.py.
SCHEMA = active_schema(SCHEMAS_CLASSIFIER)
FEATURE_COLS = cscas_feature_cols(df, schema=SCHEMA)
print(f"Schema: {SCHEMA} -- {len(FEATURE_COLS)} feature columns")
print(FEATURE_COLS)

if grid_outputs_done(["cscas_xgboost"], SCHEMA):
    print(
        f"[skip] cscas_xgboost {SCHEMA} outputs already exist (CSCAS_FORCE=1 to re-run)."
    )
    raise SystemExit(0)

# 5) Verify training pools against Table IV (pool construction itself
# lives in _sampling.py -- these are just the sanity-check counts).
important = train[train["Label"] == 1]
irr_inliers = train[(train["Label"] == 0) & (train["SCAS"] == 0)]
irr_outliers = train[(train["Label"] == 0) & (train["SCAS"] == 1)]

assert len(important) == 1_765, f"got {len(important)}"
assert len(irr_inliers) == 133_614, f"got {len(irr_inliers)}"
assert len(irr_outliers) == 4_153, f"got {len(irr_outliers)}"

# 6) Prepare eval sets -- both cells of the test-set axis.
#   subsample: shared, frozen 20k -- the grid every baseline lives in.
#   fulltest:  all 1.26M test rows -- the CSCAS paper's own protocol.
_eval_sub = get_cscas_eval_subsample(test)
EVAL_SETS = {
    "subsample": (_eval_sub[FEATURE_COLS].values, _eval_sub["Label"].values),
    "fulltest": (test[FEATURE_COLS].values, test["Label"].values),
}
print(
    f"Evaluating on: subsample ({len(_eval_sub)} rows, {int(_eval_sub['Label'].sum())} pos)"
    f"  +  full test ({len(test)} rows, {int(test['Label'].sum())} pos)"
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

# results[eval_set][condition] -> list of per-seed metric dicts
results: dict[str, dict[str, list[dict[str, float]]]] = {
    ek: {name: [] for name in POOL_BUILDERS} for ek in EVAL_SETS
}

for condition, build_pool in POOL_BUILDERS.items():
    reference = REFERENCE[condition]
    print(f"\n=== {condition} (XGBoost, {SCHEMA} schema) ===")
    if reference:
        print(
            f"    Paper reference (RF, 42 numeric features, full test set): {reference}"
        )

    for seed in range(5):
        pool, extra_kwargs = build_pool(seed)

        X_tr = pool[FEATURE_COLS].values
        y_tr = pool["Label"].values

        clf = XGBClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=extra_kwargs.get("scale_pos_weight"),
        )
        clf.fit(X_tr, y_tr)

        row = []
        for ek, (X_ev, y_ev) in EVAL_SETS.items():
            y_pred = clf.predict(X_ev)
            m = {
                "precision": precision_score(y_ev, y_pred),
                "recall": recall_score(y_ev, y_pred),
                "f1": f1_score(y_ev, y_pred),
            }
            results[ek][condition].append(m)
            row.append(f"{ek} F1={m['f1']:.3f}")
        print(f"  seed={seed}: " + "  |  ".join(row))

    for ek in EVAL_SETS:
        avg = pd.DataFrame(results[ek][condition]).mean()
        print(
            f"  AVERAGE [{ek}]: P={avg.precision:.3f} R={avg.recall:.3f} F1={avg.f1:.3f}"
        )


for ek in EVAL_SETS:
    save_baseline_results(
        name=result_name("cscas_xgboost", SCHEMA, ek),
        description=(
            f"{schema_blurb(SCHEMA, len(FEATURE_COLS))}, "
            f"XGBClassifier(n_estimators=100), evaluated on the "
            f"{'shared 20k eval subsample' if ek == 'subsample' else 'full 1.26M-row test set'}"
        ),
        results=results[ek],
    )
