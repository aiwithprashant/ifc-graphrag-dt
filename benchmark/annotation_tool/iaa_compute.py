"""
iaa_compute.py
Inter-Annotator Agreement (IAA) Computation

Computes Cohen's kappa for each KCS-DT sub-score across annotators.
Target: κ ≥ 0.75 on all sub-scores before locking DTAH-Bench.

If κ < 0.70 for any sub-score, flags prompts needing re-annotation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RUBRIC_SCALE = [0.0, 0.25, 0.50, 0.75, 1.00]  # 5-point rubric
SUBSCORES = ["entity", "relation", "attribute", "containment", "connectivity"]
KAPPA_THRESHOLD = 0.75
KAPPA_MINIMUM   = 0.70


def cohen_kappa(ratings_a: list, ratings_b: list) -> float:
    """Compute Cohen's kappa for two lists of ordinal ratings."""
    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore
        return float(cohen_kappa_score(ratings_a, ratings_b, weights="linear"))
    except ImportError:
        # Fallback: simple unweighted kappa
        n = len(ratings_a)
        if n == 0:
            return 0.0
        agree = sum(a == b for a, b in zip(ratings_a, ratings_b))
        p_o = agree / n
        cats = set(ratings_a) | set(ratings_b)
        p_e = sum(
            (ratings_a.count(c) / n) * (ratings_b.count(c) / n)
            for c in cats
        )
        return (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 1.0


def compute_iaa(annotations_path: str | Path, output_path: Optional[str | Path] = None) -> dict:
    """
    Compute IAA from a batch annotation file.

    Annotation file format (JSON list):
    [
      {
        "prompt_id": "T2-MEP-001",
        "annotator_id": "A1",
        "entity": 1.0,
        "relation": 0.75,
        "attribute": 1.0,
        "containment": 0.75,
        "connectivity": 0.5
      },
      ...
    ]

    Returns kappa per sub-score and overall IAA report.
    """
    annotations_path = Path(annotations_path)
    with open(annotations_path) as f:
        annotations = json.load(f)

    # Group by prompt_id
    by_prompt: dict[str, dict[str, dict]] = {}
    for ann in annotations:
        pid = ann["prompt_id"]
        aid = ann["annotator_id"]
        if pid not in by_prompt:
            by_prompt[pid] = {}
        by_prompt[pid][aid] = ann

    # Find prompt IDs with at least 2 annotators
    multi_annotated = {
        pid: anns for pid, anns in by_prompt.items()
        if len(anns) >= 2
    }

    if len(multi_annotated) < 5:
        logger.warning("Too few multi-annotated prompts for reliable IAA: %d", len(multi_annotated))

    # Align annotator pairs
    annotator_ids = sorted({ann["annotator_id"] for ann in annotations})
    if len(annotator_ids) < 2:
        return {"error": "Need at least 2 annotators"}

    # Compute kappa per sub-score across all prompt pairs
    kappas = {}
    flagged_prompts = []

    for subscore in SUBSCORES:
        ratings_a, ratings_b = [], []
        for pid, anns in multi_annotated.items():
            ann_list = list(anns.values())
            ratings_a.append(ann_list[0].get(subscore, 0.0))
            ratings_b.append(ann_list[1].get(subscore, 0.0))

        kappa = cohen_kappa(ratings_a, ratings_b)
        kappas[subscore] = round(kappa, 4)

        # Flag prompts with low agreement on this sub-score
        for pid, anns in multi_annotated.items():
            ann_list = list(anns.values())
            v_a = ann_list[0].get(subscore, 0.0)
            v_b = ann_list[1].get(subscore, 0.0)
            if abs(v_a - v_b) > 0.5:
                flagged_prompts.append({
                    "prompt_id": pid,
                    "subscore": subscore,
                    "annotator_values": [v_a, v_b],
                })

    overall_kappa = sum(kappas.values()) / len(kappas)
    needs_revision = [k for k, v in kappas.items() if v < KAPPA_THRESHOLD]
    failed_minimum = [k for k, v in kappas.items() if v < KAPPA_MINIMUM]

    report = {
        "n_prompts_evaluated":   len(multi_annotated),
        "n_annotators":          len(annotator_ids),
        "kappa_per_subscore":    kappas,
        "overall_kappa":         round(overall_kappa, 4),
        "target_met":            all(v >= KAPPA_THRESHOLD for v in kappas.values()),
        "needs_rubric_revision": needs_revision,
        "failed_minimum":        failed_minimum,
        "flagged_prompts":       flagged_prompts[:20],  # cap at 20 for report
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("IAA report saved: %s", output_path)

    logger.info("Overall κ = %.3f | Target met: %s", overall_kappa, report["target_met"])
    for k, v in kappas.items():
        status = "✓" if v >= KAPPA_THRESHOLD else ("⚠" if v >= KAPPA_MINIMUM else "✗")
        logger.info("  %s %s: κ = %.3f", status, k, v)

    return report
