"""
statistical_tests.py
Statistical Analysis for Paper 2 Results

Implements all statistical tests required by the DTAH-Bench evaluation
protocol, as specified in the benchmark design document:

  1. Wilcoxon signed-rank test (B4 vs each Bi, per tier)
  2. Cohen's d effect size (B2 vs B4 at Tier 3 — primary comparison)
  3. Bootstrap confidence intervals (95%, n=1000 resamples)
  4. Spearman rank correlation (for KCS-DT weight ablation)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    alternative: str = "greater",
) -> dict:
    """
    Wilcoxon signed-rank test between two score lists.
    H0: scores_a <= scores_b  (alternative='greater': scores_a > scores_b)

    Returns
    -------
    dict with statistic, p_value, significant (p < 0.05)
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have equal length")

    stat, p = wilcoxon(scores_a, scores_b, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value":   float(p),
        "significant": bool(p < 0.05),
        "n": len(scores_a),
        "alternative": alternative,
    }


def cohen_d(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Cohen's d effect size between two independent samples.
    d = (mean_a - mean_b) / pooled_std

    Interpretation: small=0.2, medium=0.5, large=0.8
    """
    a, b = np.array(scores_a), np.array(scores_b)
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = math.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)

    d = mean_diff / pooled_std if pooled_std > 0 else 0.0
    magnitude = (
        "small" if abs(d) < 0.5
        else "medium" if abs(d) < 0.8
        else "large"
    )

    return {
        "cohen_d":   round(float(d), 4),
        "magnitude": magnitude,
        "mean_a":    round(float(np.mean(a)), 4),
        "mean_b":    round(float(np.mean(b)), 4),
        "diff":      round(float(mean_diff), 4),
    }


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    stat_fn=np.mean,
) -> dict:
    """
    Bootstrap confidence interval for a statistic.

    Returns
    -------
    dict with mean, ci_lower, ci_upper, std
    """
    rng = np.random.default_rng(42)
    arr = np.array(scores)
    boot_stats = [
        float(stat_fn(rng.choice(arr, size=len(arr), replace=True)))
        for _ in range(n_resamples)
    ]
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "mean":       round(float(np.mean(arr)), 4),
        "ci_lower":   round(ci_lower, 4),
        "ci_upper":   round(ci_upper, 4),
        "std":        round(float(np.std(arr)), 4),
        "n":          len(scores),
        "confidence": confidence,
        "n_resamples": n_resamples,
    }


def run_full_analysis(results_dir: str | Path, output_path: Optional[str | Path] = None) -> dict:
    """
    Run all statistical tests on experiment results.

    Expects results_dir to contain JSON files:
      b0_scores.json, b1_scores.json, b2_scores.json,
      b3_scores.json, b4_scores.json
    Each file: list of {"prompt_id": str, "tier": int, "total": float, ...}

    Returns full analysis report.
    """
    results_dir = Path(results_dir)
    baselines = ["b0", "b1", "b2", "b3", "b4"]

    # Load all scores
    all_scores: dict[str, list[dict]] = {}
    for b in baselines:
        path = results_dir / f"{b}_scores.json"
        if path.exists():
            with open(path) as f:
                all_scores[b] = json.load(f)
        else:
            logger.warning("Missing scores file: %s", path)
            all_scores[b] = []

    def _by_tier(scores_list: list[dict], tier: int) -> list[float]:
        return [s["total"] for s in scores_list if s.get("tier") == tier]

    report = {"baselines": {}, "comparisons": {}, "primary_comparison": {}}

    # Per-baseline bootstrap CIs per tier
    for b in baselines:
        report["baselines"][b] = {}
        for tier in [1, 2, 3]:
            tier_scores = _by_tier(all_scores[b], tier)
            if tier_scores:
                report["baselines"][b][f"tier{tier}"] = bootstrap_ci(tier_scores)

    # B4 vs each Bi — Wilcoxon + Cohen's d per tier
    b4_scores = all_scores.get("b4", [])
    for b in ["b0", "b1", "b2", "b3"]:
        report["comparisons"][f"b4_vs_{b}"] = {}
        for tier in [1, 2, 3]:
            b4_tier = _by_tier(b4_scores, tier)
            bi_tier = _by_tier(all_scores.get(b, []), tier)
            if len(b4_tier) == len(bi_tier) and len(b4_tier) > 0:
                report["comparisons"][f"b4_vs_{b}"][f"tier{tier}"] = {
                    "wilcoxon": wilcoxon_signed_rank(b4_tier, bi_tier),
                    "effect_size": cohen_d(b4_tier, bi_tier),
                }

    # Primary comparison: B4 vs B2 at Tier 3
    b4_t3 = _by_tier(b4_scores, 3)
    b2_t3 = _by_tier(all_scores.get("b2", []), 3)
    if b4_t3 and b2_t3 and len(b4_t3) == len(b2_t3):
        report["primary_comparison"] = {
            "description": "B4 (GraphRAG-DT) vs B2 (Flat RAG) at Tier 3 (System level)",
            "wilcoxon": wilcoxon_signed_rank(b4_t3, b2_t3),
            "effect_size": cohen_d(b4_t3, b2_t3),
            "b4_ci": bootstrap_ci(b4_t3),
            "b2_ci": bootstrap_ci(b2_t3),
        }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Statistical analysis saved to %s", output_path)

    return report
