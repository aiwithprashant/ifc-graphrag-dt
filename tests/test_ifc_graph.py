"""Tests for IFC graph builder and k-hop traversal (CPU, no IFC file needed)."""
import pytest, sys
sys.path.insert(0, '.')
import networkx as nx
from pipeline.layer1_retriever.khop_traversal import KHopTraversal, TraversalResult

def _make_test_graph():
    """Build a small synthetic IFC-like graph for testing."""
    G = nx.DiGraph()
    nodes = [
        ("A", {"ifc_type": "IfcSpace",       "name": "PumpRoom"}),
        ("B", {"ifc_type": "IfcPump",         "name": "P-01"}),
        ("C", {"ifc_type": "IfcPipeSegment",  "name": "Pipe-01"}),
        ("D", {"ifc_type": "IfcValve",        "name": "V-01"}),
        ("E", {"ifc_type": "IfcPipeSegment",  "name": "Pipe-02"}),
    ]
    for nid, data in nodes:
        G.add_node(nid, **data)
    G.add_edge("A", "B", relation_type="IfcRelContainedInSpatialStructure", category="containment")
    G.add_edge("B", "C", relation_type="IfcRelConnectsPortToElement", category="port_connectivity")
    G.add_edge("C", "D", relation_type="IfcRelConnects", category="connectivity")
    G.add_edge("D", "E", relation_type="IfcRelConnects", category="connectivity")
    return G

G = _make_test_graph()

def test_depth_1_traversal():
    traversal = KHopTraversal(G, max_depth=1)
    result = traversal.traverse(["B"])
    assert "B" in result.subgraph.nodes
    assert "C" in result.subgraph.nodes  # direct neighbour
    assert "D" not in result.subgraph.nodes  # 2 hops away

def test_depth_3_traversal():
    traversal = KHopTraversal(G, max_depth=3)
    result = traversal.traverse(["B"])
    assert "D" in result.subgraph.nodes  # 2 hops
    assert "E" in result.subgraph.nodes  # 3 hops

def test_bidirectional_traversal():
    traversal = KHopTraversal(G, max_depth=1, bidirectional=True)
    result = traversal.traverse(["B"])
    # Bidirectional: B can reach A (containment, reverse direction)
    assert "A" in result.subgraph.nodes

def test_invalid_seed():
    traversal = KHopTraversal(G, max_depth=2)
    result = traversal.traverse(["INVALID_ID"])
    assert result.node_count == 0

def test_result_summary():
    traversal = KHopTraversal(G, max_depth=2)
    result = traversal.traverse(["B"])
    summary = result.summary()
    assert "nodes_retrieved" in summary
    assert "edges_retrieved" in summary
    assert summary["nodes_retrieved"] > 0

def test_entity_list():
    traversal = KHopTraversal(G, max_depth=1)
    result = traversal.traverse(["B"])
    entities = result.to_entity_list()
    assert isinstance(entities, list)
    assert all("ifc_type" in e for e in entities)
