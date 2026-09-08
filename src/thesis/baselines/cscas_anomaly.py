"""
Anomaly-detection counterpart to baselines/cscas_base.py: same raw CSV load,
paper-parity asserts, fixed timestamp split, 5-column reduced FEATURE_COLS,
and shared eval subsample -- but trains a one-class model (OneClassSVM, via
training/model_factory.get_model_factory("ocsvm")) on benign-only rows
instead of a RandomForestClassifier on a class-balanced pool.

This deliberately does not route through experiments/anomaly.py's
AlertGroup/FeatureSchemaRegistry pipeline (a different feature
representation entirely, and CSCAS isn't wired into it) -- it matches this
project's own cscas_*.py baseline convention instead, the same way
cscas_base.py/cscas_bert.py/cscas_zeroshot.py all do.

No pool-condition loop, no seeds: anomaly detection doesn't need class
balance (it's fit on benign rows only, natural count), and OneClassSVM
(kernel='rbf', nu=0.05) is a deterministic convex fit with no random_state
-- there is genuinely nothing to average over, so its per-seed sd is
exactly 0. The IsolationForest siblings (cscas_anomaly_iforest.py /
cscas_mining_anomaly_iforest.py) DO run the 5-seed protocol the trainable
baselines use, since tree bootstrapping is stochastic.

Scoring convention (matches experiments/anomaly.py::_compute_anomaly_metrics):
  scores = -model.decision_function(X_test)   # higher = more anomalous
  y_pred = (model.predict(X_test) == -1)      # 1 = anomaly = attack
AUC is this method's headline metric (score-based, threshold-free); F1/P/R
use the model's own -1/+1 decision boundary.

Run:
    cd src/thesis/baselines
    python cscas_anomaly.py

The data path below is relative to the current working directory (not this
file's location), so it must be run from src/thesis/baselines/.
"""

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from thesis.baselines._cscas_schema import (
    SCHEMAS_ANOMALY,
    active_schema,
    cscas_feature_cols,
    grid_outputs_done,
    result_name,
    schema_blurb,
    sentinel_imputer,
)
from thesis.baselines._results import save_anomaly_results
from thesis.baselines._sampling import get_cscas_eval_subsample
from thesis.training.workload import compute_workload_at_recall

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

# 4) Feature schema -- CSCAS_SCHEMA env var picks "base" (5 cols), or for
# the full schema either "full_noscas" (40 cols, SignatureID + SCAS dropped)
# or "full_scas" (41 cols, SCAS kept). Feeding a one-class detector CSCAS's
# own precomputed outlier flag (SCAS) is circular, so "full_scas" is a
# deliberately-circular diagnostic run. See _cscas_schema.py.
SCHEMA = active_schema(SCHEMAS_ANOMALY)
FEATURE_COLS = cscas_feature_cols(df, schema=SCHEMA)
print(f"Schema: {SCHEMA} -- {len(FEATURE_COLS)} feature columns")
print(FEATURE_COLS)

if grid_outputs_done(["cscas_anomaly_ocsvm"], SCHEMA):
    print(
        f"[skip] cscas_anomaly_ocsvm {SCHEMA} outputs already exist (CSCAS_FORCE=1 to re-run)."
    )
    raise SystemExit(0)

# 4b) OneClassSVM is scale-sensitive (StandardScaler in the pipeline), so
# the -1 "not applicable" sentinel is median-imputed inside the Pipeline
# (sentinel_imputer, fit on the benign train rows only) -- same treatment as
# cscas_logreg.py. See _cscas_schema.py.

# 5) Benign-only training data -- no pool conditions, no undersampling.
train_benign = train[train["Label"] == 0]
print(f"Training on {len(train_benign)} benign-only rows (natural count)")

# 6) Eval sets -- both cells of the test-set axis.
eval_df = get_cscas_eval_subsample(test)
EVAL_FRAMES = {"subsample": eval_df, "fulltest": test}
X_train = train_benign[FEATURE_COLS].values
print(
    f"Evaluating on: subsample ({len(eval_df)} rows, {int(eval_df['Label'].sum())} pos)"
    f"  +  full test ({len(test)} rows, {int(test['Label'].sum())} pos)"
)

# 7) Fit once. Same estimator as model_factory's "ocsvm" (StandardScaler +
# OneClassSVM(kernel='rbf', nu=0.05)), plus a sentinel_imputer() first step
# -- built inline so this script doesn't pull in the classifier factory's
# heavier deps. Deterministic convex fit, no random_state: a genuine single
# run.
model = Pipeline(
    [
        ("impute", sentinel_imputer()),
        ("scaler", StandardScaler()),
        ("clf", OneClassSVM(kernel="rbf", nu=0.05)),
    ]
)
model.fit(X_train)

for ek, frame in EVAL_FRAMES.items():
    X_ev = frame[FEATURE_COLS].values
    y_ev = frame["Label"].values

    scores = -model.decision_function(X_ev)  # higher = more anomalous
    y_pred = (model.predict(X_ev) == -1).astype(int)  # 1 = anomaly = attack

    auc = roc_auc_score(y_ev, scores)
    p = precision_score(y_ev, y_pred, zero_division=0)
    r = recall_score(y_ev, y_pred, zero_division=0)
    f = f1_score(y_ev, y_pred, zero_division=0)
    workload = compute_workload_at_recall(y_ev, scores)

    print(f"\n=== cscas_anomaly_ocsvm [{SCHEMA} / {ek}] ===")
    print(f"AUC={auc:.3f} P={p:.3f} R={r:.3f} F1={f:.3f}  (default nu=0.05 cut)")
    if workload.get("0.90"):
        w = workload["0.90"]
        print(
            f"  @recall>=0.90: P={w['precision']:.3f} FP={int(w['fp'])} "
            f"workload_reduction={w['workload_reduction']:.3f}"
        )

    save_anomaly_results(
        name=result_name("cscas_anomaly_ocsvm", SCHEMA, ek),
        description=(
            "OneClassSVM(kernel='rbf', nu=0.05), median-imputed -1 sentinel + "
            "StandardScaler, fit on benign-only rows of the "
            f"{schema_blurb(SCHEMA, len(FEATURE_COLS))}, evaluated on the "
            f"{'shared 20k eval subsample' if ek == 'subsample' else 'full 1.26M-row test set'}. "
            "No attack rows used in training. precision/recall/f1 at the default "
            "nu=0.05 cut; workload_at_recall is the tuned-threshold view."
        ),
        metrics={"auc": auc, "precision": p, "recall": r, "f1": f},
        workload=workload,
    )
