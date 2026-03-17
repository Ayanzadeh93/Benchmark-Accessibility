"""Metric helpers for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CorrMetrics:
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    kendall_tau: float
    kendall_p: float


def correlation_metrics(y_true: List[float], y_pred: List[float]) -> Optional[CorrMetrics]:
    """Compute rank/linear correlation metrics.

    Returns None if insufficient data.
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return None
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if np.allclose(yt, yt[0]) or np.allclose(yp, yp[0]):
        return None
    sp = stats.spearmanr(yt, yp)
    pe = stats.pearsonr(yt, yp)
    ke = stats.kendalltau(yt, yp)
    return CorrMetrics(
        spearman_r=float(sp.statistic),
        spearman_p=float(sp.pvalue),
        pearson_r=float(pe.statistic),
        pearson_p=float(pe.pvalue),
        kendall_tau=float(ke.statistic),
        kendall_p=float(ke.pvalue),
    )


def confusion_matrix(labels: List[str], preds: List[str], classes: List[str]) -> np.ndarray:
    """Compute confusion matrix with a fixed class order."""
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for y, p in zip(labels, preds):
        if y not in idx or p not in idx:
            continue
        m[idx[y], idx[p]] += 1
    return m


def f1_macro_from_confusion(m: np.ndarray) -> float:
    """Compute macro-F1 from confusion matrix."""
    # per-class: precision = tp/(tp+fp), recall = tp/(tp+fn)
    f1s: List[float] = []
    for i in range(m.shape[0]):
        tp = float(m[i, i])
        fp = float(m[:, i].sum() - tp)
        fn = float(m[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def accuracy_from_confusion(m: np.ndarray) -> float:
    denom = float(m.sum())
    if denom <= 0:
        return 0.0
    return float(np.trace(m) / denom)

