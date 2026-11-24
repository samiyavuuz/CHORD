"""Evaluation metrics. OOD scores: higher = more OOD.

Standard OOD metrics (threshold-free with respect to any calibration set):
- AUROC
- FPR95: fraction of OOD slides accepted at the threshold where 95% of the cohort's own ID slides are accepted
- AUPR-In: average precision with ID as the positive class
- AUPR-Out: average precision with OOD as the positive class
"""

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve

OOD_METRICS = ("auroc", "fpr95", "aupr_in", "aupr_out")


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean recall over the classes present in y_true (single class: plain accuracy)."""
    return float(balanced_accuracy_score(y_true, y_pred))


def ood_metrics(score_id: np.ndarray, score_ood: np.ndarray) -> dict:
    if len(score_id) == 0 or len(score_ood) == 0:
        return {k: float("nan") for k in OOD_METRICS}
    is_ood = np.r_[np.zeros(len(score_id)), np.ones(len(score_ood))]
    s = np.r_[score_id, score_ood]
    # ID as positive class, ranked by -score; FPR95 = first ROC point reaching TPR >= 0.95
    fpr, tpr, _ = roc_curve(1 - is_ood, -s)
    return {"auroc": float(roc_auc_score(is_ood, s)),
            "fpr95": float(fpr[np.searchsorted(tpr, 0.95, side="left")]),
            "aupr_in": float(average_precision_score(1 - is_ood, -s)),
            "aupr_out": float(average_precision_score(is_ood, s))}


def auroc(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    return ood_metrics(score_id, score_ood)["auroc"]


def evaluate_cohort(model, u: np.ndarray, role: np.ndarray, id_class: np.ndarray, scorers=("knn", "msp")) -> dict:
    """ID balanced accuracy and, per scorer, near/far OOD metrics."""
    is_id = role == "id"
    out = {"bacc": balanced_accuracy(id_class[is_id], model.predict(u[is_id]))}
    for s in scorers:
        sc = model.score(u, s)
        for g in ("near", "far"):
            for k, v in ood_metrics(sc[is_id], sc[role == g]).items():
                out[f"{s}_{g}_{k}"] = v
    return out
