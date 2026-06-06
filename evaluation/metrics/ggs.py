"""
ggs.py
Graph Grounding Score (GGS) — Retrieval Layer Metric

Measures how well the GraphRAG retriever recovered the ground-truth
IFC subgraph. Paired with KCS-DT to give a complete two-layer
measurement stack for DTAH-Eval.

  GGS = w_n · NodeRecall + w_e · EdgeRecall + w_p · PathRecall

  w_n = 0.35  (nodes are necessary but not sufficient)
  w_e = 0.45  (edges encode the relational structure — highest weight)
  w_p = 0.20  (path recall rewards multi-hop traversal depth)

GGS answers: "Did GraphRAG recover the right IFC subgraph?"
KCS-DT answers: "Did the generator produce a DT-ready asset?"

High GGS + Low KCS-DT → generation failure (fix Layer 3)
Low  GGS + Low KCS-DT → retrieval failure (fix Layer 1)
High GGS + High KCS-DT → full pipeline success
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

GGS_WEIGHTS = {
    "node_recall": 0.35,
    "edge_recall": 0.45,
    "path_recall": 0.20,
}


@dataclass
class GGSScore:
    """Holds the three GGS sub-scores and the weighted total."""
    node_recall: float
    edge_recall: float
    path_recall: float
    total:       float
    weights:     dict
    prompt_id:   str = ""

    def to_dict(self) -> dict:
        return {
            "prompt_id":   self.prompt_id,
            "node_recall": round(self.node_recall, 4),
            "edge_recall": round(self.edge_recall, 4),
            "path_recall": round(self.path_recall, 4),
            "total":       round(self.total, 4),
            "weights":     self.weights,
        }

    def failure_mode(self) -> str:
        """Diagnose dominant failure mode."""
        if self.node_recall < 0.5:
            return "entity_miss"
        if self.edge_recall < 0.5:
            return "relation_miss"
        if self.path_recall < 0.5:
            return "multi_hop_failure"
        return "none"

    def __repr__(self) -> str:
        return (
            f"GGSScore(N={self.node_recall:.3f}, E={self.edge_recall:.3f}, "
            f"P={self.path_recall:.3f} → total={self.total:.3f})"
        )


class GGSScorer:
    """
    Computes Graph Grounding Score between a retrieved subgraph
    and the ground-truth IFC subgraph annotation.

    Parameters
    ----------
    weights : dict | None
        Custom weight vector. Defaults to GGS_WEIGHTS.
    max_path_length : int
        Maximum path length to consider for path recall (default: 4).
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        max_path_length: int = 4,
    ):
        self.weights = weights or GGS_WEIGHTS.copy()
        self.max_path_length = max_path_length
        self._validate_weights()

    # ── Public API ────────────────────────────────────────────────────────────

    def score(
        self,
        retrieved: nx.DiGraph,
        ground_truth: nx.DiGraph,
        prompt_id: str = "",
    ) -> GGSScore:
        """
        Compute GGS between a retrieved and ground-truth subgraph.

        Matching is type-based (IFC type strings), not instance-based
        (GlobalIds), to avoid false negatives from ID differences between
        reference models and generated outputs.

        Parameters
        ----------
        retrieved    : nx.DiGraph  Subgraph from KHopTraversal
        ground_truth : nx.DiGraph  Annotated ground-truth subgraph

        Returns
        -------
        GGSScore
        """
        n = self._node_recall(retrieved, ground_truth)
        e = self._edge_recall(retrieved, ground_truth)
        p = self._path_recall(retrieved, ground_truth)

        total = (
            self.weights["node_recall"] * n
            + self.weights["edge_recall"] * e
            + self.weights["path_recall"] * p
        )

        score = GGSScore(
            node_recall=n, edge_recall=e, path_recall=p,
            total=total, weights=self.weights, prompt_id=prompt_id,
        )
        logger.debug("GGS [%s]: %s", prompt_id, score)
        return score

    def score_batch(
        self,
        retrieved_graphs: list[nx.DiGraph],
        ground_truth_graphs: list[nx.DiGraph],
        prompt_ids: Optional[list[str]] = None,
    ) -> list[GGSScore]:
        ids = prompt_ids or [""] * len(retrieved_graphs)
        return [
            self.score(r, g, pid)
            for r, g, pid in zip(retrieved_graphs, ground_truth_graphs, ids)
        ]

    def aggregate(self, scores: list[GGSScore]) -> dict:
        import statistics

        def _stats(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0}
            return {
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            }

        return {
            "n": len(scores),
            "node_recall": _stats([s.node_recall for s in scores]),
            "edge_recall": _stats([s.edge_recall for s in scores]),
            "path_recall": _stats([s.path_recall for s in scores]),
            "total":       _stats([s.total for s in scores]),
            "failure_modes": {
                fm: sum(1 for s in scores if s.failure_mode() == fm)
                for fm in ["entity_miss", "relation_miss", "multi_hop_failure", "none"]
            },
        }

    # ── Sub-scores ────────────────────────────────────────────────────────────

    def _node_recall(self, retrieved: nx.DiGraph, gt: nx.DiGraph) -> float:
        """Fraction of ground-truth node types present in retrieved graph."""
        gt_types   = self._node_types(gt)
        ret_types  = self._node_types(retrieved)
        if not gt_types:
            return 1.0
        return len(gt_types & ret_types) / len(gt_types)

    def _edge_recall(self, retrieved: nx.DiGraph, gt: nx.DiGraph) -> float:
        """Fraction of ground-truth edge type pairs present in retrieved graph."""
        gt_edges  = self._edge_type_pairs(gt)
        ret_edges = self._edge_type_pairs(retrieved)
        if not gt_edges:
            return 1.0
        matched = sum(1 for e in gt_edges if e in ret_edges)
        return matched / len(gt_edges)

    def _path_recall(self, retrieved: nx.DiGraph, gt: nx.DiGraph) -> float:
        """
        Fraction of ground-truth paths (up to max_path_length) that exist
        in the retrieved graph. Paths are compared as sequences of node types.
        """
        gt_paths  = self._enumerate_paths(gt)
        ret_paths = self._enumerate_paths(retrieved)
        if not gt_paths:
            return 1.0
        matched = sum(1 for p in gt_paths if p in ret_paths)
        return matched / len(gt_paths)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _node_types(graph: nx.DiGraph) -> set[str]:
        return {
            data.get("ifc_type", "unknown")
            for _, data in graph.nodes(data=True)
        }

    @staticmethod
    def _edge_type_pairs(graph: nx.DiGraph) -> set[tuple]:
        pairs = set()
        for src, tgt, data in graph.edges(data=True):
            src_type = graph.nodes[src].get("ifc_type", "?")
            tgt_type = graph.nodes[tgt].get("ifc_type", "?")
            rel_type = data.get("relation_type", "?")
            pairs.add((rel_type, src_type, tgt_type))
        return pairs

    def _enumerate_paths(self, graph: nx.DiGraph) -> set[tuple]:
        """Enumerate all simple paths up to max_path_length as type sequences."""
        paths = set()
        nodes = list(graph.nodes())
        for source in nodes:
            for target in nodes:
                if source == target:
                    continue
                try:
                    for path in nx.all_simple_paths(
                        graph, source, target,
                        cutoff=self.max_path_length
                    ):
                        type_seq = tuple(
                            graph.nodes[n].get("ifc_type", "?") for n in path
                        )
                        paths.add(type_seq)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return paths

    def _validate_weights(self) -> None:
        required = {"node_recall", "edge_recall", "path_recall"}
        if set(self.weights.keys()) != required:
            raise ValueError(f"GGS weights must contain exactly: {required}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"GGS weights must sum to 1.0 (got {total:.4f})")
