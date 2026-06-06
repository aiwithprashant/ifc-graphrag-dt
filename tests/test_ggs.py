"""Tests for GGS scorer."""
import pytest, sys
sys.path.insert(0, '.')
import networkx as nx
from evaluation.metrics.ggs import GGSScorer, GGSScore

def _make_graph(node_types, edge_triples):
    G = nx.DiGraph()
    for i, t in enumerate(node_types):
        G.add_node(str(i), ifc_type=t, name=f"node_{i}")
    for src, rel, tgt in edge_triples:
        G.add_edge(src, tgt, relation_type=rel)
    return G

GT_G = _make_graph(
    ["IfcPump", "IfcPipeSegment", "IfcValve"],
    [("0", "IfcRelConnectsPortToElement", "1"),
     ("1", "IfcRelConnects", "2")]
)

def test_perfect_retrieval():
    scorer = GGSScorer()
    score = scorer.score(GT_G, GT_G)
    assert score.node_recall == pytest.approx(1.0)
    assert score.edge_recall == pytest.approx(1.0)

def test_empty_retrieval():
    scorer = GGSScorer()
    empty = nx.DiGraph()
    score = scorer.score(empty, GT_G)
    assert score.node_recall == pytest.approx(0.0)
    assert score.edge_recall == pytest.approx(0.0)
    assert score.total < 0.1

def test_partial_retrieval():
    scorer = GGSScorer()
    partial = _make_graph(["IfcPump"], [])  # only seed, no edges
    score = scorer.score(partial, GT_G)
    assert 0.0 < score.node_recall < 1.0
    assert score.edge_recall == pytest.approx(0.0)

def test_failure_mode_entity_miss():
    scorer = GGSScorer()
    empty = nx.DiGraph()
    score = scorer.score(empty, GT_G)
    assert score.failure_mode() == "entity_miss"

def test_weights_sum_to_one():
    scorer = GGSScorer()
    assert sum(scorer.weights.values()) == pytest.approx(1.0, abs=1e-6)
