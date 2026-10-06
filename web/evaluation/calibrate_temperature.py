"""
Temperature Scaling & Platt Scaling Optimizer for SafeLens.

Optimizes Temperature (T) and Platt parameters (a, b) on validation logits
to minimize Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE).
"""

import math
from typing import List, Tuple, Dict, Any
import numpy as np


def optimize_temperature(
    logits: List[float],
    labels: List[int],
    initial_t: float = 1.35,
) -> Dict[str, Any]:
    """
    Find optimal temperature T by minimizing Negative Log-Likelihood (NLL).

    Args:
        logits: List of raw logit differences z_i = logp(true) - logp(false)
        labels: List of binary ground truth labels y_i in {0, 1}
        initial_t: Initial guess for temperature

    Returns:
        Dict with optimal_temperature, initial_nll, calibrated_nll, ece_before, ece_after
    """
    if not logits or len(logits) != len(labels):
        return {"optimal_temperature": 1.0, "error": "Invalid input lengths"}

    z = np.array(logits, dtype=float)
    y = np.array(labels, dtype=float)

    def nll(temp: float) -> float:
        t_val = max(1e-4, float(temp))
        scaled_z = z / t_val
        # Numerically stable sigmoid NLL
        loss = np.maximum(scaled_z, 0) - scaled_z * y + np.log(1.0 + np.exp(-np.abs(scaled_z)))
        return float(np.mean(loss))

    # Grid search + golden-section refinement for robust 1D optimization without external C-extensions
    best_t = initial_t
    best_loss = nll(best_t)

    for cand_t in np.linspace(0.1, 5.0, 500):
        loss_val = nll(cand_t)
        if loss_val < best_loss:
            best_loss = loss_val
            best_t = float(cand_t)

    # Refine around best_t
    for cand_t in np.linspace(max(0.05, best_t - 0.1), best_t + 0.1, 200):
        loss_val = nll(cand_t)
        if loss_val < best_loss:
            best_loss = loss_val
            best_t = float(cand_t)

    from .metrics import compute_ece_and_bins, compute_brier_score

    raw_probs = [float(1.0 / (1.0 + math.exp(-item))) for item in logits]
    cal_probs = [float(1.0 / (1.0 + math.exp(-item / best_t))) for item in logits]

    ece_before = compute_ece_and_bins(labels, raw_probs)["ece"]
    ece_after = compute_ece_and_bins(labels, cal_probs)["ece"]
    brier_before = compute_brier_score(labels, raw_probs)
    brier_after = compute_brier_score(labels, cal_probs)

    return {
        "optimal_temperature": round(best_t, 4),
        "nll_uncalibrated": round(nll(1.0), 4),
        "nll_calibrated": round(best_loss, 4),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "ece_reduction_pct": round(((ece_before - ece_after) / max(1e-5, ece_before)) * 100, 2),
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
    }


def optimize_platt_scaling(
    logits: List[float],
    labels: List[int],
) -> Dict[str, Any]:
    """
    Fits Platt scaling parameters a and b: P_cal = σ(a * z + b)
    """
    if not logits or len(logits) != len(labels):
        return {"a": 1.0, "b": 0.0, "error": "Invalid inputs"}

    z = np.array(logits, dtype=float)
    y = np.array(labels, dtype=float)

    best_loss = float("inf")
    best_a, best_b = 1.0, 0.0

    for a_cand in np.linspace(0.1, 3.0, 100):
        for b_cand in np.linspace(-2.0, 2.0, 100):
            scaled_z = a_cand * z + b_cand
            loss = float(np.mean(np.maximum(scaled_z, 0) - scaled_z * y + np.log(1.0 + np.exp(-np.abs(scaled_z)))))
            if loss < best_loss:
                best_loss = loss
                best_a = float(a_cand)
                best_b = float(b_cand)

    return {
        "a": round(best_a, 4),
        "b": round(best_b, 4),
        "temperature_equivalent": round(1.0 / best_a, 4) if best_a > 0 else None,
        "min_nll": round(best_loss, 4),
    }
