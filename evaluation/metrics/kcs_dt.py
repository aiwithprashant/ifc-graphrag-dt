"""
kcs_dt.py
KCS-DT: Digital Twin Constraint Satisfaction Metric

Formal definition (Paper 2, Section 3):
  KCS-DT = 0.20·E + 0.35·R + 0.15·A + 0.15·Cn + 0.15·Cv

Components
----------
E   Entity Correctness      — required IFC entity types present in output
R   Relation Correctness     — required IFC relations correctly instantiated
A   Attribute Correctness    — material, dimensions, flow direction correct
Cn  Containment Correctness  — entities placed in correct spatial containers
Cv  Connectivity Correctness — port-to-port and system connections correct

Weights are justified empirically: Paper 1 identified relation precision
as the dominant failure mode in relational prompts (highest weight = 0.35).
Cn and Cv are DT-specific sub-scores absent from prior text-to-3D metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default weight vector ─────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "entity":       0.20,
    "relation":     0.35,
    "attribute":    0.15,
    "containment":  0.15,
    "connectivity": 0.15,
}


@dataclass
class KCSDTScore:
    """
    Holds all five sub-scores and the weighted total KCS-DT score.

    Attributes
    ----------
    entity       : float  E  ∈ [0, 1]
    relation     : float  R  ∈ [0, 1]
    attribute    : float  A  ∈ [0, 1]
    containment  : float  Cn ∈ [0, 1]
    connectivity : float  Cv ∈ [0, 1]
    total        : float  KCS-DT ∈ [0, 1]
    weights      : dict   Weight vector used
    prompt_id    : str    Optional prompt identifier
    """
    entity:       float
    relation:     float
    attribute:    float
    containment:  float
    connectivity: float
    total:        float
    weights:      dict
    prompt_id:    str = ""

    def to_dict(self) -> dict:
        return {
            "prompt_id":    self.prompt_id,
            "entity":       round(self.entity, 4),
            "relation":     round(self.relation, 4),
            "attribute":    round(self.attribute, 4),
            "containment":  round(self.containment, 4),
            "connectivity": round(self.connectivity, 4),
            "total":        round(self.total, 4),
            "weights":      self.weights,
        }

    def __repr__(self) -> str:
        return (
            f"KCSDTScore("
            f"E={self.entity:.3f}, R={self.relation:.3f}, "
            f"A={self.attribute:.3f}, Cn={self.containment:.3f}, "
            f"Cv={self.connectivity:.3f} → total={self.total:.3f})"
        )


class KCSDTScorer:
    """
    Computes KCS-DT between a predicted scene specification and
    the ground-truth IFC annotation.

    Both `prediction` and `ground_truth` are expected as dicts
    following the scene specification schema in spec_schema.json.

    Schema keys used
    ----------------
    ground_truth / prediction:
      "entities"    : list[{"ifc_type": str, "name": str, ...}]
      "relations"   : list[{"type": str, "from": str, "to": str}]
      "attributes"  : dict[entity_name, {"material": str, ...}]
      "containment" : list[{"entity": str, "container": str}]
      "connectivity": list[{"from": str, "to": str, "port_type": str}]
    """

    def __init__(self, weights: Optional[dict] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def score(
        self,
        prediction: dict,
        ground_truth: dict,
        prompt_id: str = "",
    ) -> KCSDTScore:
        """
        Compute KCS-DT score.

        Parameters
        ----------
        prediction   : dict  Predicted scene specification
        ground_truth : dict  Ground-truth IFC annotation
        prompt_id    : str   Optional identifier for logging

        Returns
        -------
        KCSDTScore with all sub-scores and weighted total
        """
        e  = self._score_entity(prediction, ground_truth)
        r  = self._score_relation(prediction, ground_truth)
        a  = self._score_attribute(prediction, ground_truth)
        cn = self._score_containment(prediction, ground_truth)
        cv = self._score_connectivity(prediction, ground_truth)

        total = (
            self.weights["entity"]       * e
            + self.weights["relation"]     * r
            + self.weights["attribute"]    * a
            + self.weights["containment"]  * cn
            + self.weights["connectivity"] * cv
        )

        score = KCSDTScore(
            entity=e, relation=r, attribute=a,
            containment=cn, connectivity=cv,
            total=total, weights=self.weights,
            prompt_id=prompt_id,
        )
        logger.debug("KCS-DT [%s]: %s", prompt_id, score)
        return score

    def score_batch(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
        prompt_ids: Optional[list[str]] = None,
    ) -> list[KCSDTScore]:
        """Score a batch of predictions against ground truths."""
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")

        ids = prompt_ids or [""] * len(predictions)
        return [
            self.score(pred, gt, pid)
            for pred, gt, pid in zip(predictions, ground_truths, ids)
        ]

    def aggregate(self, scores: list[KCSDTScore]) -> dict:
        """Compute mean and std for each sub-score across a batch."""
        import statistics

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "min":  round(min(vals), 4),
                "max":  round(max(vals), 4),
            }

        return {
            "n": len(scores),
            "entity":       _stats([s.entity for s in scores]),
            "relation":     _stats([s.relation for s in scores]),
            "attribute":    _stats([s.attribute for s in scores]),
            "containment":  _stats([s.containment for s in scores]),
            "connectivity": _stats([s.connectivity for s in scores]),
            "total":        _stats([s.total for s in scores]),
        }

    # ── Sub-score methods ─────────────────────────────────────────────────────

    def _score_entity(self, pred: dict, gt: dict) -> float:
        """
        E: Fraction of required IFC entity types present in prediction.
        Score = |pred_types ∩ gt_types| / |gt_types|
        """
        gt_types  = {e["ifc_type"] for e in gt.get("entities", [])}
        pred_types = {e["ifc_type"] for e in pred.get("entities", [])}

        if not gt_types:
            return 1.0

        return len(gt_types & pred_types) / len(gt_types)

    def _score_relation(self, pred: dict, gt: dict) -> float:
        """
        R: Fraction of required IFC relations correctly instantiated.
        A relation is correct if its type, source entity type, and
        target entity type all match a ground-truth relation.

        Uses type-level matching (not instance-level) to avoid
        false negatives from GlobalId differences.
        """
        gt_rels  = self._normalise_relations(gt.get("relations", []))
        pred_rels = self._normalise_relations(pred.get("relations", []))

        if not gt_rels:
            return 1.0

        matched = sum(1 for r in gt_rels if r in pred_rels)
        return matched / len(gt_rels)

    def _score_attribute(self, pred: dict, gt: dict) -> float:
        """
        A: Fraction of required attribute values correctly assigned.
        Compares material, dimensions, flow direction across entities.
        """
        gt_attrs   = gt.get("attributes", {})
        pred_attrs = pred.get("attributes", {})

        if not gt_attrs:
            return 1.0

        total_fields = 0
        correct_fields = 0

        for entity_name, gt_vals in gt_attrs.items():
            pred_vals = pred_attrs.get(entity_name, {})
            for field_name, gt_val in gt_vals.items():
                total_fields += 1
                pred_val = pred_vals.get(field_name)
                if self._attribute_match(gt_val, pred_val):
                    correct_fields += 1

        return correct_fields / total_fields if total_fields > 0 else 1.0

    def _score_containment(self, pred: dict, gt: dict) -> float:
        """
        Cn: Whether entities are placed in correct spatial containers.
        Checks IfcRelContainedInSpatialStructure correctness.
        Score = |correct containment pairs| / |required containment pairs|
        """
        gt_containment   = {(c["entity"], c["container"]) for c in gt.get("containment", [])}
        pred_containment = {(c["entity"], c["container"]) for c in pred.get("containment", [])}

        if not gt_containment:
            return 1.0

        # Partial credit: match on entity type only if container type also matches
        matched = self._fuzzy_pair_match(gt_containment, pred_containment)
        return matched / len(gt_containment)

    def _score_connectivity(self, pred: dict, gt: dict) -> float:
        """
        Cv: Whether port-to-port and system-level connections are topologically correct.
        Score = |correct connectivity pairs| / |required connectivity pairs|
        """
        gt_conn   = {(c["from"], c["to"]) for c in gt.get("connectivity", [])}
        pred_conn = {(c["from"], c["to"]) for c in pred.get("connectivity", [])}

        if not gt_conn:
            return 1.0

        matched = self._fuzzy_pair_match(gt_conn, pred_conn)
        return matched / len(gt_conn)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_relations(relations: list[dict]) -> set[tuple]:
        """Convert relation list to a set of (type, from_type, to_type) tuples."""
        normalised = set()
        for r in relations:
            rel_type  = r.get("type", "")
            from_part = r.get("from", "").split(".")[0]  # strip port suffix
            to_part   = r.get("to", "").split(".")[0]
            normalised.add((rel_type, from_part, to_part))
        return normalised

    @staticmethod
    def _attribute_match(gt_val: object, pred_val: object) -> bool:
        """Flexible attribute matching: exact for strings, tolerance for numerics."""
        if pred_val is None:
            return False
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            return abs(float(gt_val) - float(pred_val)) / (abs(float(gt_val)) + 1e-9) < 0.1
        return str(gt_val).lower().strip() == str(pred_val).lower().strip()

    @staticmethod
    def _fuzzy_pair_match(gt_pairs: set[tuple], pred_pairs: set[tuple]) -> int:
        """
        Count how many gt_pairs have a matching pred_pair.
        Matching is type-level: IFC type prefixes must agree.
        """
        matched = 0
        for gt_pair in gt_pairs:
            for pred_pair in pred_pairs:
                if (
                    gt_pair[0].split("[")[0] == pred_pair[0].split("[")[0]
                    and gt_pair[1].split("[")[0] == pred_pair[1].split("[")[0]
                ):
                    matched += 1
                    break
        return matched

    def _validate_weights(self) -> None:
        required = {"entity", "relation", "attribute", "containment", "connectivity"}
        if set(self.weights.keys()) != required:
            raise ValueError(f"Weights must contain exactly: {required}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0 (got {total:.4f})")
