"""Metrics at a fixed false positive rate.

The threshold is chosen on the validation set (the score above which only the target share of
validation negatives fall) and then applied unchanged to the test set. The realised test false
positive rate is reported next to recall and precision, so any drift is visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(scores_neg: np.ndarray, fpr: float) -> float:
    """Smallest threshold t such that the share of negatives with score > t is at most fpr."""
    s = np.sort(np.asarray(scores_neg))[::-1]
    k = int(np.floor(fpr * len(s)))
    if k >= len(s):
        return -np.inf
    return float(s[k])


def at_threshold(y: np.ndarray, scores: np.ndarray, t: float) -> dict[str, float]:
    pred = scores > t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return {
        "recall": tp / pos if pos else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "flagged": int(pred.sum()),
    }


def evaluate(y_val, s_val, y_test, s_test, fprs=(0.01, 0.05)) -> dict[str, float]:
    y_val, s_val, y_test, s_test = map(np.asarray, (y_val, s_val, y_test, s_test))
    out = {
        "pr_auc": float(average_precision_score(y_test, s_test)),
        "roc_auc": float(roc_auc_score(y_test, s_test)),
        "test_positive_rate": float(y_test.mean()),
    }
    for f in fprs:
        t = threshold_at_fpr(s_val[y_val == 0], f)
        m = at_threshold(y_test, s_test, t)
        tag = f"{round(f * 100)}pct"
        for k, v in m.items():
            out[f"{k}_at_fpr{tag}"] = v
    return out
