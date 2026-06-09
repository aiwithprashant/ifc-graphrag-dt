"""
khop_traversal.py
Layer 1 — k-hop GraphRAG Traversal

Given seed entity node(s) and a depth k, traverses the IFC property
graph and returns the induced subgraph containing all nodes reachable
within k hops. This is the core operation that gives GraphRAG its
advantage over flat RAG for system-level prompts.

Theoretical basis (from Paper 2 Section 4):
  For a prompt requiring k-hop relational grounding,
  Flat RAG recovers O(1) neighbourhood information,
  GraphRAG recovers O(k) path information.
  For Tier 3 system prompts, k >= 3 empirically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result of a k-hop graph traversal."""
    seed_ids: list[str]
    max_depth: int
    subgraph: nx.DiGraph
    visited_by_depth: dict[int, list[str]] = field(default_factory=dict)
    relation_type_counts: dict[str, int] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return self.subgraph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.subgraph.number_of_edges()

    def to_entity_list(self) -> list[dict]:
        """Return nodes as a list of entity dicts (for spec generator input)."""
        return [
            {
                "global_id": nid,
                **{k: v for k, v in data.items() if k != "psets"},
            }
            for nid, data in self.subgraph.nodes(data=True)
        ]

    def to_relation_list(self) -> list[dict]:
        """Return edges as a list of relation dicts."""
        return [
            {"source": src, "target": tgt, **data}
            for src, tgt, data in self.subgraph.edges(data=True)
        ]

    def summary(self) -> dict:
        return {
            "seeds": self.seed_ids,
            "max_depth": self.max_depth,
            "nodes_retrieved": self.node_count,
            "edges_retrieved": self.edge_count,
            "depth_distribution": {
                str(d): len(nodes)
                for d, nodes in self.visited_by_depth.items()
            },
            "relation_types": self.relation_type_counts,
        }


class KHopTraversal:
    """
    Performs k-hop neighbourhood traversal on the IFC property graph.

    Traversal is bidirectional by default (follows both in- and out-edges)
    to ensure that containment relations (which point container → contained)
    are captured when traversing from a contained entity.

    Parameters
    ----------
    graph : nx.DiGraph
        The full IFC property graph from IFCGraphBuilder.
    max_depth : int
        Maximum number of hops from seed nodes (default: 4).
        Tier 1 prompts typically need k=1.
        Tier 2 prompts typically need k=2.
        Tier 3 prompts typically need k=3-4.
    relation_filter : list[str] | None
        If provided, only traverse edges of these relation types.
        If None, all relation types are traversed.
    bidirectional : bool
        Whether to follow edges in both directions (default: True).
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        max_depth: int = 4,
        relation_filter: Optional[list[str]] = None,
        bidirectional: bool = True,
    ):
        self.graph = graph
        self.max_depth = max_depth
        self.relation_filter = set(relation_filter) if relation_filter else None
        self.bidirectional = bidirectional

        if bidirectional:
            self._undirected = graph.to_undirected(as_view=True)

    def traverse(self, seed_ids: list[str]) -> TraversalResult:
        """
        Traverse from seed nodes up to max_depth hops.

        Parameters
        ----------
        seed_ids : list[str]
            GlobalId values of seed IFC entities.

        Returns
        -------
        TraversalResult containing the induced subgraph and metadata.
        """
        valid_seeds = [sid for sid in seed_ids if sid in self.graph]
        if not valid_seeds:
            logger.warning("No valid seed IDs found in graph. Check GlobalId values.")
            return TraversalResult(
                seed_ids=seed_ids,
                max_depth=self.max_depth,
                subgraph=nx.DiGraph(),
            )

        visited: set[str] = set()
        frontier: set[str] = set(valid_seeds)
        visited_by_depth: dict[int, list[str]] = {0: list(frontier)}
        visited.update(frontier)

        for depth in range(1, self.max_depth + 1):
            next_frontier: set[str] = set()
            for node_id in frontier:
                neighbours = self._get_neighbours(node_id)
                for neighbour in neighbours:
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
            visited.update(next_frontier)
            visited_by_depth[depth] = list(next_frontier)
            frontier = next_frontier
            if not frontier:
                logger.debug("Traversal exhausted at depth %d", depth)
                break

        subgraph = self.graph.subgraph(visited).copy()

        # Count relation types in subgraph
        rel_counts: dict[str, int] = {}
        for _, _, data in subgraph.edges(data=True):
            rt = data.get("relation_type", "unknown")
            rel_counts[rt] = rel_counts.get(rt, 0) + 1

        logger.info(
            "Traversal: %d seeds -> %d nodes, %d edges at depth %d",
            len(valid_seeds),
            subgraph.number_of_nodes(),
            subgraph.number_of_edges(),
            self.max_depth,
        )

        return TraversalResult(
            seed_ids=valid_seeds,
            max_depth=self.max_depth,
            subgraph=subgraph,
            visited_by_depth=visited_by_depth,
            relation_type_counts=rel_counts,
        )

    def traverse_from_type(self, ifc_type: str) -> TraversalResult:
        """
        Traverse from all entities of a given IFC type.
        Useful for system-level queries (e.g. all IfcPump nodes).
        """
        seed_ids = [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get("ifc_type", "") == ifc_type
        ]
        logger.info("Found %d seed nodes of type %s", len(seed_ids), ifc_type)
        return self.traverse(seed_ids)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_neighbours(self, node_id: str) -> list[str]:
        """Return neighbouring node IDs, filtered by relation type if set."""
        neighbours: list[str] = []

        if self.bidirectional:
            graph_to_use = self._undirected
        else:
            graph_to_use = self.graph

        for neighbour in graph_to_use.neighbors(node_id):
            if self.relation_filter is not None:
                edge_data = self._get_edge_data(node_id, neighbour)
                if edge_data is None:
                    continue
                rel_type = edge_data.get("relation_type", "")
                if rel_type not in self.relation_filter:
                    continue
            neighbours.append(neighbour)

        return neighbours

    def _get_edge_data(self, a: str, b: str) -> Optional[dict]:
        """Get edge data between two nodes (either direction)."""
        if self.graph.has_edge(a, b):
            return self.graph[a][b]
        if self.graph.has_edge(b, a):
            return self.graph[b][a]
        return None
