"""
spec_evaluator.py
Stage A Evaluator — Specification Correctness

Stage A answers: "Did the retriever + spec generator produce the
correct IFC subgraph and scene specification?"

Stage A is evaluated PROGRAMMATICALLY against the ground-truth
IFC subgraph annotation — no human scorer required.

Stage A Score = 0.40·EntityScore + 0.40·RelationScore + 0.20·AttributeScore
"""

from __future__ import annotations
import json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class StageAScore:
    entity_score:   float
    relation_score: float
    attribute_score: float
    total:          float
    prompt_id:      str = ""
    failure_codes:  list = None

    def __post_init__(self):
        if self.failure_codes is None:
            self.failure_codes = []

    def to_dict(self) -> dict:
        return {
            "prompt_id":      self.prompt_id,
            "stage":          "A",
            "entity_score":   round(self.entity_score, 4),
            "relation_score": round(self.relation_score, 4),
            "attribute_score": round(self.attribute_score, 4),
            "total":          round(self.total, 4),
            "failure_codes":  self.failure_codes,
        }


class StageAEvaluator:
    """
    Evaluates specification correctness (Stage A of DTAH-Eval).

    Compares the predicted scene specification JSON against the
    ground-truth IFC subgraph annotation JSON.
    """

    WEIGHTS = {"entity": 0.40, "relation": 0.40, "attribute": 0.20}

    def evaluate(
        self,
        predicted_spec: dict,
        ground_truth: dict,
        prompt_id: str = "",
    ) -> StageAScore:
        e, e_codes = self._score_entities(predicted_spec, ground_truth)
        r, r_codes = self._score_relations(predicted_spec, ground_truth)
        a, a_codes = self._score_attributes(predicted_spec, ground_truth)

        total = (self.WEIGHTS["entity"] * e
                 + self.WEIGHTS["relation"] * r
                 + self.WEIGHTS["attribute"] * a)

        return StageAScore(
            entity_score=e, relation_score=r, attribute_score=a,
            total=total, prompt_id=prompt_id,
            failure_codes=e_codes + r_codes + a_codes,
        )

    def evaluate_batch(
        self,
        predicted_specs: list[dict],
        ground_truths: list[dict],
        prompt_ids: Optional[list[str]] = None,
    ) -> list[StageAScore]:
        ids = prompt_ids or [""] * len(predicted_specs)
        return [
            self.evaluate(ps, gt, pid)
            for ps, gt, pid in zip(predicted_specs, ground_truths, ids)
        ]

    # ── Sub-scores ────────────────────────────────────────────────────────────

    def _score_entities(self, pred: dict, gt: dict) -> tuple[float, list]:
        gt_types  = {e["ifc_type"] for e in gt.get("entities", [])}
        pred_types = {e.get("ifc_type", "") for e in pred.get("entities", [])}
        if not gt_types:
            return 1.0, []
        missing = gt_types - pred_types
        codes = [f"EA-1:{t}" for t in missing]
        score = len(gt_types & pred_types) / len(gt_types)
        return score, codes

    def _score_relations(self, pred: dict, gt: dict) -> tuple[float, list]:
        gt_rels   = {(r["type"], r["from"].split(".")[0], r["to"].split(".")[0])
                     for r in gt.get("relations", [])}
        pred_rels = {(r["type"], r["from"].split(".")[0], r["to"].split(".")[0])
                     for r in pred.get("relations", [])}
        if not gt_rels:
            return 1.0, []
        missing = gt_rels - pred_rels
        codes = [f"EA-2:{r[0]}" for r in missing]
        score = len(gt_rels & pred_rels) / len(gt_rels)
        return score, codes

    def _score_attributes(self, pred: dict, gt: dict) -> tuple[float, list]:
        gt_attrs   = gt.get("attributes", {})
        pred_attrs = pred.get("attributes", {})
        if not gt_attrs:
            return 1.0, []
        total, correct, codes = 0, 0, []
        for eid, gt_vals in gt_attrs.items():
            pred_vals = pred_attrs.get(eid, {})
            for field, gt_val in gt_vals.items():
                total += 1
                if str(pred_vals.get(field, "")).lower() == str(gt_val).lower():
                    correct += 1
                else:
                    codes.append(f"EA-5:{eid}.{field}")
        return (correct / total if total > 0 else 1.0), codes
