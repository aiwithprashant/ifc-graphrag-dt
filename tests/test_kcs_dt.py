"""Tests for KCS-DT scorer."""
import pytest
import sys
sys.path.insert(0, '.')
from evaluation.metrics.kcs_dt import KCSDTScorer, KCSDTScore, DEFAULT_WEIGHTS

GT = {
    "entities": [
        {"ifc_type": "IfcPump"}, {"ifc_type": "IfcPipeSegment"}, {"ifc_type": "IfcValve"}
    ],
    "relations": [
        {"type": "IfcRelConnectsPortToElement", "from": "IfcPump.OutletPort", "to": "IfcPipeSegment"},
        {"type": "IfcRelConnects", "from": "IfcPipeSegment", "to": "IfcValve"},
    ],
    "attributes": {
        "pump_0": {"material": "cast_iron", "flow_direction": "horizontal"}
    },
    "containment": [{"entity": "IfcPump", "container": "IfcSpace"}],
    "connectivity": [{"from": "IfcPump.OutletPort", "to": "IfcPipeSegment"}],
}

def test_perfect_score():
    scorer = KCSDTScorer()
    score = scorer.score(GT, GT)
    assert score.total == pytest.approx(1.0, abs=1e-4)
    assert score.entity == pytest.approx(1.0)
    assert score.relation == pytest.approx(1.0)

def test_empty_prediction():
    scorer = KCSDTScorer()
    pred = {"entities": [], "relations": [], "attributes": {},
            "containment": [], "connectivity": []}
    score = scorer.score(pred, GT)
    assert score.entity == pytest.approx(0.0)
    assert score.relation == pytest.approx(0.0)
    assert score.total < 0.1

def test_partial_entity_score():
    scorer = KCSDTScorer()
    pred = {**GT, "entities": [{"ifc_type": "IfcPump"}]}  # only 1 of 3
    score = scorer.score(pred, GT)
    assert score.entity == pytest.approx(1/3, abs=0.01)

def test_weights_sum_to_one():
    scorer = KCSDTScorer()
    assert sum(scorer.weights.values()) == pytest.approx(1.0, abs=1e-6)

def test_invalid_weights_raise():
    with pytest.raises(ValueError):
        KCSDTScorer(weights={"entity": 0.5, "relation": 0.5})  # missing keys

def test_batch_scoring():
    scorer = KCSDTScorer()
    scores = scorer.score_batch([GT, GT], [GT, GT], ["T1", "T2"])
    assert len(scores) == 2
    assert all(s.total == pytest.approx(1.0, abs=1e-4) for s in scores)

def test_aggregate():
    scorer = KCSDTScorer()
    scores = scorer.score_batch([GT, GT], [GT, GT])
    agg = scorer.aggregate(scores)
    assert agg["total"]["mean"] == pytest.approx(1.0, abs=1e-4)
    assert agg["n"] == 2
