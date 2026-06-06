"""
mesh_evaluator.py
Stage B Evaluator — Generation Fidelity

Stage B answers: "Given a correct specification, did the 3D generator
faithfully produce it?"

Stage B is evaluated using KCS-DT applied to the generated mesh output,
scored by expert annotators using the structured rubric from the
DTAH-Bench annotation protocol.

CRITICAL DESIGN NOTE:
Stage B uses the GROUND-TRUTH specification as input (not the
GraphRAG-generated one). This isolates generation fidelity from
retrieval quality — the core diagnostic insight of DTAH-Eval.
"""

from __future__ import annotations
import json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Stage B rubric scores (expert annotator scale)
RUBRIC_SCORES = {
    "fully_correct":   1.00,
    "mostly_correct":  0.75,
    "partially_correct": 0.50,
    "mostly_incorrect": 0.25,
    "incorrect":       0.00,
}


@dataclass
class StageBScore:
    """KCS-DT sub-scores from expert annotation at Stage B."""
    entity:       float
    relation:     float
    attribute:    float
    containment:  float
    connectivity: float
    total:        float
    prompt_id:    str = ""
    annotator_id: str = ""
    failure_codes: list = None

    def __post_init__(self):
        if self.failure_codes is None:
            self.failure_codes = []

    def to_dict(self) -> dict:
        return {
            "prompt_id":    self.prompt_id,
            "stage":        "B",
            "annotator_id": self.annotator_id,
            "entity":       round(self.entity, 4),
            "relation":     round(self.relation, 4),
            "attribute":    round(self.attribute, 4),
            "containment":  round(self.containment, 4),
            "connectivity": round(self.connectivity, 4),
            "total":        round(self.total, 4),
            "failure_codes": self.failure_codes,
        }


class StageBEvaluator:
    """
    Computes Stage B score from expert annotation input.

    In practice, annotators fill in a structured JSON rubric for each
    generated mesh. This class reads those annotations and computes
    the weighted KCS-DT Stage B score.
    """

    KCS_WEIGHTS = {
        "entity": 0.20, "relation": 0.35, "attribute": 0.15,
        "containment": 0.15, "connectivity": 0.15,
    }

    def from_annotation(self, annotation: dict, prompt_id: str = "") -> StageBScore:
        """
        Compute Stage B score from a completed annotator rubric dict.

        annotation keys: entity, relation, attribute, containment,
        connectivity — each with value from RUBRIC_SCORES keys.
        """
        def _parse(val):
            if isinstance(val, (int, float)):
                return float(val)
            return RUBRIC_SCORES.get(str(val).lower().replace(" ", "_"), 0.0)

        e  = _parse(annotation.get("entity", "incorrect"))
        r  = _parse(annotation.get("relation", "incorrect"))
        a  = _parse(annotation.get("attribute", "incorrect"))
        cn = _parse(annotation.get("containment", "incorrect"))
        cv = _parse(annotation.get("connectivity", "incorrect"))

        total = (self.KCS_WEIGHTS["entity"] * e
                 + self.KCS_WEIGHTS["relation"] * r
                 + self.KCS_WEIGHTS["attribute"] * a
                 + self.KCS_WEIGHTS["containment"] * cn
                 + self.KCS_WEIGHTS["connectivity"] * cv)

        failure_codes = []
        if e < 0.5:   failure_codes.append("EB-1:missing_object")
        if r < 0.5:   failure_codes.append("EB-2:topology_violation")
        if cn < 0.5:  failure_codes.append("EB-4:containment_violation")
        if cv < 0.5:  failure_codes.append("EB-2:topology_violation")

        return StageBScore(
            entity=e, relation=r, attribute=a, containment=cn, connectivity=cv,
            total=total, prompt_id=prompt_id,
            annotator_id=annotation.get("annotator_id", ""),
            failure_codes=failure_codes,
        )

    def load_and_score(self, annotation_file: Path) -> list[StageBScore]:
        """Load a batch annotation JSON file and compute all Stage B scores."""
        with open(annotation_file) as f:
            annotations = json.load(f)
        return [
            self.from_annotation(a, a.get("prompt_id", ""))
            for a in annotations
        ]
