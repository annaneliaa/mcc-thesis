import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def train_and_evaluate(X, y, n_splits=3, plot=True):

    X = X.reset_index(drop=True)
    y = np.asarray(y)

    assert len(X) == len(y)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
    )

    aucs = []
    all_proba = []
    all_y = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)

        all_proba.extend(proba)
        all_y.extend(y_test)

        print(f"Fold {fold} ROC-AUC: {auc:.3f}")

    mean_auc = float(np.mean(aucs))
    print(f"Mean ROC-AUC: {mean_auc:.3f}")

    # ---------- ROC plot ----------
    if plot:
        fpr, tpr, _ = roc_curve(all_y, all_proba)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Mean AUC = {mean_auc:.3f})")
        plt.show()

    return {
        "model": model,
        "aucs": aucs,
        "mean_auc": mean_auc,
        "y_true": np.array(all_y),
        "proba": np.array(all_proba),
    }

def plot_alert_reduction(y_true, proba):
    thresholds = np.linspace(0, 1, 50)
    reductions = []
    fnrs = []

    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        reductions.append((y_pred == 0).mean())
        fnrs.append(
            ((y_pred == 0) & (y_true == 1)).sum() / max(1, (y_true == 1).sum())
        )

    plt.figure()
    plt.plot(reductions, fnrs)
    plt.xlabel("Alert Reduction")
    plt.ylabel("False Negative Rate")
    plt.title("Alert Reduction vs Missed Attacks")
    plt.show()
