"""
spec_score.py
Stage A Specification Score

Programmatic scorer for Stage A of DTAH-Eval.
Measures how well the retrieved IFC subgraph and generated scene spec
match the ground-truth annotation.

Stage A Score = 0.40·EntityScore + 0.40·RelationScore + 0.20·AttributeScore

This is evaluated without human annotators — purely programmatic
comparison against the ground-truth JSON annotations.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpecScore:
    entity_score:    float
    relation_score:  float
    attribute_score: float
    total:           float
    prompt_id:       str = ""
    missing_entities:  list = field(default_factory=list)
    missing_relations: list = field(default_factory=list)
    wrong_attributes:  list = field(default_factory=list)

    WEIGHTS = {"entity": 0.40, "relation": 0.40, "attribute": 0.20}

    def to_dict(self) -> dict:
        return {
            "prompt_id":        self.prompt_id,
            "stage":            "A",
            "entity_score":     round(self.entity_score, 4),
            "relation_score":   round(self.relation_score, 4),
            "attribute_score":  round(self.attribute_score, 4),
            "total":            round(self.total, 4),
            "missing_entities": self.missing_entities,
            "missing_relations":self.missing_relations,
            "wrong_attributes": self.wrong_attributes,
        }

    def failure_code(self) -> str:
        """Return dominant Stage A error code."""
        if self.entity_score < 0.5:
            return "EA-1"   # entity miss
        if self.relation_score < 0.5:
            # Distinguish multi-hop vs direct relation failure
            mh = any("ContainedIn" in r or "Port" in r for r in self.missing_relations)
            return "EA-3" if mh else "EA-2"
        if self.attribute_score < 0.5:
            return "EA-5"
        return "OK"


class SpecScorer:
    """
    Computes Stage A Specification Score between a predicted spec
    and the ground-truth annotation.

    Both inputs follow spec_schema.json format.
    """

    WEIGHTS = {"entity": 0.40, "relation": 0.40, "attribute": 0.20}

    def score(
        self,
        predicted: dict,
        ground_truth: dict,
        prompt_id: str = "",
    ) -> SpecScore:
        e,  missing_e  = self._entity_score(predicted, ground_truth)
        r,  missing_r  = self._relation_score(predicted, ground_truth)
        a,  wrong_a    = self._attribute_score(predicted, ground_truth)

        total = (self.WEIGHTS["entity"]   * e
               + self.WEIGHTS["relation"] * r
               + self.WEIGHTS["attribute"]* a)

        return SpecScore(
            entity_score=e, relation_score=r, attribute_score=a,
            total=total, prompt_id=prompt_id,
            missing_entities=missing_e,
            missing_relations=missing_r,
            wrong_attributes=wrong_a,
        )

    def score_batch(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        prompt_ids: Optional[list[str]] = None,
    ) -> list[SpecScore]:
        ids = prompt_ids or [""] * len(predictions)
        return [self.score(p, g, pid)
                for p, g, pid in zip(predictions, ground_truths, ids)]

    # ── Sub-scores ────────────────────────────────────────────────────────────

    def _entity_score(self, pred: dict, gt: dict) -> tuple[float, list]:
        gt_types   = {e["ifc_type"] for e in gt.get("entities", [])}
        pred_types = {e.get("ifc_type","") for e in pred.get("entities", [])}
        if not gt_types:
            return 1.0, []
        missing = sorted(gt_types - pred_types)
        score   = len(gt_types & pred_types) / len(gt_types)
        return score, missing

    def _relation_score(self, pred: dict, gt: dict) -> tuple[float, list]:
        def normalise(rels):
            return {(r["type"],
                     r["from"].split(".")[0].split("[")[0],
                     r["to"].split(".")[0].split("[")[0])
                    for r in rels}

        gt_rels   = normalise(gt.get("relations", []))
        pred_rels = normalise(pred.get("relations", []))
        if not gt_rels:
            return 1.0, []
        missing = sorted(str(r) for r in gt_rels - pred_rels)
        score   = len(gt_rels & pred_rels) / len(gt_rels)
        return score, missing

    def _attribute_score(self, pred: dict, gt: dict) -> tuple[float, list]:
        gt_attrs   = gt.get("attributes", {})
        pred_attrs = pred.get("attributes", {})
        if not gt_attrs:
            return 1.0, []

        total, correct, wrong = 0, 0, []
        for eid, gt_vals in gt_attrs.items():
            pred_vals = pred_attrs.get(eid, {})
            for field_name, gt_val in gt_vals.items():
                total += 1
                pred_val = pred_vals.get(field_name, "")
                if str(gt_val).lower().strip() == str(pred_val).lower().strip():
                    correct += 1
                else:
                    wrong.append(f"{eid}.{field_name}: expected={gt_val} got={pred_val}")
        return (correct / total if total > 0 else 1.0), wrong
