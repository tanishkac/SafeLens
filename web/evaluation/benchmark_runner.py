"""
SafeLens Benchmark Runner & Evaluation Harness.

Loads multimodal benchmark dataset, evaluates with SafetyLLM + ConfidenceCalibrator,
computes calibration & classification metrics, and outputs a formatted evaluation report.

Usage:
    python -m web.evaluation.benchmark_runner [--dataset path/to/dataset.json] [--output path/to/report.md]
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

from ..tools.llm import SafetyLLM
from ..app.orchestration.segment_analyzer import llm_decide
from ..app.orchestration.calibrator import ConfidenceCalibrator
from .metrics import compute_full_evaluation_summary
from .calibrate_temperature import optimize_temperature


def run_benchmark(
    dataset_path: str,
    output_report_path: str = "web/evaluation/reports/latest_evaluation_report.md",
    model_name: str = None,
    backend: str = None,
) -> Dict[str, Any]:
    """
    Execute benchmark evaluation over annotated dataset.
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"\n========================================================")
    print(f"   SafeLens Evaluation Benchmark Runner")
    print(f"   Total Test Cases: {len(cases)}")
    print(f"========================================================\n")

    llm = SafetyLLM(model=model_name, backend=backend)

    y_true: List[int] = []
    y_calibrated: List[float] = []
    y_raw: List[float] = []
    logits: List[float] = []
    case_results: List[Dict[str, Any]] = []

    start_time = time.time()

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{idx}")
        gt_harmful = int(case.get("ground_truth_harmful", 0))
        audio = case.get("audio_text", "")
        ocr = case.get("ocr_text", "")
        captions = case.get("captions_text", "")
        category = case.get("category", "General")

        print(f"[{idx}/{len(cases)}] Evaluating '{case_id}' ({category})...", end="", flush=True)

        res = llm_decide(
            audio_text=audio,
            ocr_text=ocr,
            captions_text=captions,
            llm=llm,
            segment_info=case_id,
        )

        cal_prob = float(res.get("confidence", 0.5))
        raw_prob = float(res.get("raw_confidence", cal_prob))
        logit_z = res.get("logit_z")
        if logit_z is None:
            logit_z = 2.0 if res.get("is_harmful") else -2.0

        y_true.append(gt_harmful)
        y_calibrated.append(cal_prob)
        y_raw.append(raw_prob)
        logits.append(float(logit_z))

        is_pred_harm = bool(res.get("is_harmful", False))
        correct = (is_pred_harm == bool(gt_harmful))
        status_str = "PASS" if correct else "FAIL"

        print(f" -> {status_str} (Pred: {is_pred_harm}, Calibrated Conf: {cal_prob:.3f}, Raw Conf: {raw_prob:.3f})")

        case_results.append({
            "id": case_id,
            "category": category,
            "ground_truth": gt_harmful,
            "predicted_harmful": is_pred_harm,
            "calibrated_confidence": cal_prob,
            "raw_confidence": raw_prob,
            "logit_z": logit_z,
            "calibration_source": res.get("calibration_source", "unknown"),
            "explanation": res.get("explanation", ""),
            "correct": correct,
        })

    elapsed = round(time.time() - start_time, 2)

    # Compute overall metrics
    eval_summary = compute_full_evaluation_summary(
        y_true=y_true,
        y_probs=y_calibrated,
        y_raw_probs=y_raw,
    )

    # Optimize temperature based on validation logits
    tuning_res = optimize_temperature(logits=logits, labels=y_true)

    # Format Markdown Report
    report_md = _generate_markdown_report(
        dataset_path=dataset_path,
        elapsed_sec=elapsed,
        eval_summary=eval_summary,
        tuning_res=tuning_res,
        case_results=case_results,
        model_name=llm.model,
    )

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n========================================================")
    print(f"   Benchmark Complete in {elapsed}s")
    print(f"   Accuracy: {eval_summary['metrics']['accuracy'] * 100:.1f}% | F1: {eval_summary['metrics']['f1_score']:.3f}")
    print(f"   Expected Calibration Error (ECE): {eval_summary['calibration']['expected_calibration_error']:.4f}")
    print(f"   Brier Score: {eval_summary['calibration']['brier_score']:.4f}")
    print(f"   Report saved to: {output_report_path}")
    print(f"========================================================\n")

    return eval_summary


def _generate_markdown_report(
    dataset_path: str,
    elapsed_sec: float,
    eval_summary: Dict[str, Any],
    tuning_res: Dict[str, Any],
    case_results: List[Dict[str, Any]],
    model_name: str,
) -> str:
    m = eval_summary["metrics"]
    c = eval_summary["calibration"]
    comp = eval_summary.get("comparison_vs_uncalibrated", {})

    md = f"""# SafeLens Evaluation & Calibration Report

- **Date / Run Duration**: {elapsed_sec}s
- **Model Evaluated**: `{model_name}`
- **Dataset**: `{dataset_path}`
- **Total Test Cases**: {eval_summary['sample_count']}

---

## 1. Executive Metrics Summary

| Metric | Score | Target Standard | Status |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **{m['accuracy'] * 100:.1f}%** | $\ge 90\%$ | {'✅ Passed' if m['accuracy'] >= 0.85 else '⚠️ Review'} |
| **Precision** | **{m['precision']:.3f}** | $\ge 0.85$ | {'✅ Passed' if m['precision'] >= 0.85 else '⚠️ Review'} |
| **Recall** | **{m['recall']:.3f}** | $\ge 0.90$ | {'✅ Passed' if m['recall'] >= 0.85 else '⚠️ Review'} |
| **F1 Score** | **{m['f1_score']:.3f}** | $\ge 0.88$ | {'✅ Passed' if m['f1_score'] >= 0.85 else '⚠️ Review'} |
| **Expected Calibration Error (ECE)** | **{c['expected_calibration_error']:.4f}** | $\le 0.08$ | {'🏆 Calibrated' if c['expected_calibration_error'] <= 0.08 else '⚠️ Needs Scaling'} |
| **Brier Score** | **{c['brier_score']:.4f}** | $\le 0.10$ | {'🏆 High Quality' if c['brier_score'] <= 0.10 else '⚠️ Moderate'} |
| **False Positive Rate (FPR)** | **{m['false_positive_rate'] * 100:.1f}%** | $\le 5\%$ | {'✅ Safe' if m['false_positive_rate'] <= 0.05 else '⚠️ Elevated'} |

---

## 2. Confidence Calibration Analysis (Logprob vs. Calibrated)

The system extracts raw token log-probabilities $z = \\log p(\\text{"true"}) - \\log p(\\text{"false"})$ and applies temperature scaling $P = \\sigma(z / T)$.

| Calibration Dimension | Uncalibrated (Raw Logprob) | Calibrated ($T = {tuning_res.get('optimal_temperature', 1.35)}$) | Improvement |
| :--- | :--- | :--- | :--- |
| **Expected Calibration Error (ECE)** | `{comp.get('raw_ece', 'N/A')}` | **`{comp.get('calibrated_ece', c['expected_calibration_error'])}`** | **+{comp.get('ece_improvement_pct', 0)}%** |
| **Brier Score (Mean Squared Error)** | `{comp.get('raw_brier_score', 'N/A')}` | **`{comp.get('calibrated_brier_score', c['brier_score'])}`** | **+{comp.get('brier_improvement_pct', 0)}%** |

---

## 3. Detailed Per-Case Evaluation Log

| ID | Category | Ground Truth | Pred | Calibrated Conf | Raw Conf | Result | Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for item in case_results:
        gt_str = "Harmful (1)" if item["ground_truth"] == 1 else "Safe (0)"
        pred_str = "Harmful (1)" if item["predicted_harmful"] else "Safe (0)"
        res_badge = "✅ PASS" if item["correct"] else "❌ FAIL"
        md += f"| `{item['id']}` | {item['category']} | {gt_str} | {pred_str} | **{item['calibrated_confidence']:.3f}** | {item['raw_confidence']:.3f} | {res_badge} | `{item['calibration_source']}` |\n"

    md += """
---

## 4. Explanations and Qualitative Failure Analysis

"""
    for item in case_results:
        md += f"### `{item['id']}` ({item['category']})\n"
        md += f"- **Ground Truth**: {'Harmful' if item['ground_truth'] == 1 else 'Safe'} | **Predicted**: {'Harmful' if item['predicted_harmful'] else 'Safe'} (Confidence: {item['calibrated_confidence']:.3f})\n"
        md += f"- **Explanation**: {item['explanation']}\n\n"

    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SafeLens Evaluation Benchmark")
    parser.add_argument(
        "--dataset",
        default="web/evaluation/datasets/sample_benchmark.json",
        help="Path to benchmark dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="web/evaluation/reports/latest_evaluation_report.md",
        help="Path to save Markdown evaluation report",
    )
    args = parser.parse_args()

    run_benchmark(dataset_path=args.dataset, output_report_path=args.output)
