import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ROC curve
def plot_roc(y_true, proba, title_suffix=""):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.legend()
    plt.show()


# Alert reduction vs missed attacks
def plot_alert_reduction(y_true, proba):
    thresholds = np.linspace(0, 1, 50)
    reduction, fnr = [], []

    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        reduction.append((y_pred == 0).mean())
        fnr.append(
            ((y_pred == 0) & (y_true == 1)).sum()
            / max(1, (y_true == 1).sum())
        )

    plt.figure()
    plt.plot(reduction, fnr)
    plt.xlabel("Alert Reduction")
    plt.ylabel("False Negative Rate")
    plt.title("Alert Reduction vs Missed Attacks")
    plt.show()


# Feature importance (logistic regression)
def plot_feature_importance(model, X):
    coef = pd.Series(model.coef_[0], index=X.columns)
    coef.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Feature Importance (Logistic Regression)")
    plt.show()


# Prediction confidence distribution
def plot_confidence_distribution(proba):
    plt.figure()
    plt.hist(proba, bins=50)
    plt.xlabel("Predicted attack probability")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution")
    plt.show()


# Error analysis by alert category
def plot_top_error_categories(df_used, y_true, proba, top_k=10):
    df_err = df_used.copy()
    df_err["pred"] = (proba >= 0.5).astype(int)
    df_err["y"] = y_true

    errors = df_err[df_err["pred"] != df_err["y"]]
    errors["category"].value_counts().head(top_k).plot(kind="bar")
    plt.title("Most Misclassified Alert Categories")
    plt.show()
