"""
Evaluation and Calibration Metrics for SafeLens.
Includes ECE (Expected Calibration Error), Brier Score, ROC-AUC, PR-AUC, F1, and Reliability Diagram Bins.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def compute_brier_score(y_true: List[int], y_probs: List[float]) -> float:
    """
    Compute Brier Score (mean squared probability error).
    BS = (1/N) * sum((p_i - y_i)^2)
    Ranges from 0.0 (perfect) to 1.0 (completely inaccurate).
    """
    if not y_true or len(y_true) != len(y_probs):
        return 0.0
    arr_t = np.array(y_true, dtype=float)
    arr_p = np.array(y_probs, dtype=float)
    return float(np.mean((arr_p - arr_t) ** 2))


def compute_ece_and_bins(
    y_true: List[int],
    y_probs: List[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and binning stats for Reliability Diagrams.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if not y_true or len(y_true) != len(y_probs):
        return {
            "ece": 0.0,
            "mce": 0.0,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
            "bin_edges": [],
        }

    arr_t = np.array(y_true, dtype=float)
    arr_p = np.array(y_probs, dtype=float)
    n = len(arr_t)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    ece = 0.0
    mce = 0.0  # Maximum Calibration Error
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        # Upper edge included in last bin
        if upper == 1.0:
            mask = (arr_p >= lower) & (arr_p <= upper)
        else:
            mask = (arr_p >= lower) & (arr_p < upper)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            acc = float(np.mean(arr_t[mask]))
            conf = float(np.mean(arr_p[mask]))
            diff = abs(acc - conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
            bin_accuracies.append(round(acc, 4))
            bin_confidences.append(round(conf, 4))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(round(float((lower + upper) / 2.0), 4))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": [round(float(e), 2) for e in bin_edges],
    }


def compute_classification_metrics(
    y_true: List[int],
    y_probs: List[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute Precision, Recall, F1, Accuracy, FPR, and confusion matrix.
    """
    if not y_true or len(y_true) != len(y_probs):
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "false_positive_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_probs):
        pred = 1 if yp >= threshold else 0
        if pred == 1 and yt == 1:
            tp += 1
        elif pred == 1 and yt == 0:
            fp += 1
        elif pred == 0 and yt == 0:
            tn += 1
        else:
            fn += 1

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_full_evaluation_summary(
    y_true: List[int],
    y_probs: List[float],
    y_raw_probs: Optional[List[float]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Generate a complete evaluation report comparing raw vs calibrated confidence.
    """
    cls_metrics = compute_classification_metrics(y_true, y_probs, threshold=threshold)
    cal_ece = compute_ece_and_bins(y_true, y_probs)
    brier = compute_brier_score(y_true, y_probs)

    summary: Dict[str, Any] = {
        "sample_count": len(y_true),
        "metrics": cls_metrics,
        "calibration": {
            "brier_score": round(brier, 4),
            "expected_calibration_error": cal_ece["ece"],
            "max_calibration_error": cal_ece["mce"],
            "reliability_diagram": cal_ece,
        },
    }

    if y_raw_probs is not None and len(y_raw_probs) == len(y_true):
        raw_brier = compute_brier_score(y_true, y_raw_probs)
        raw_ece = compute_ece_and_bins(y_true, y_raw_probs)
        summary["comparison_vs_uncalibrated"] = {
            "raw_brier_score": round(raw_brier, 4),
            "calibrated_brier_score": round(brier, 4),
            "brier_improvement_pct": round(((raw_brier - brier) / max(1e-5, raw_brier)) * 100, 2),
            "raw_ece": raw_ece["ece"],
            "calibrated_ece": cal_ece["ece"],
            "ece_improvement_pct": round(((raw_ece["ece"] - cal_ece["ece"]) / max(1e-5, raw_ece["ece"])) * 100, 2),
        }

    return summary
