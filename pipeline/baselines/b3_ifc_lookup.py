"""
b3_ifc_lookup.py — Baseline implementation for Paper 2 ablation study.
B3: IFC Direct Lookup. Schema lookup by entity name, no k-hop traversal.
"""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

class B3IFCLookup:
    """
    IFC Direct Lookup baseline — looks up entity schema by name,
    returns direct neighbours only (depth=1), no multi-hop traversal.
    Tests retrieval without GraphRAG's k-hop advantage.
    """

    def __init__(self, config: dict):
        self.config = config

    def run(self, prompt: str, graph: "nx.DiGraph", ifc_config: dict, prompt_id: str = "") -> dict:
        """Extract IFC entity names from prompt, look up direct schema neighbours."""
        logger.info("[B3] IFC direct lookup for: %s", prompt[:60])

        # Simple keyword extraction for IFC types from prompt
        all_types = []
        for category_types in ifc_config.get("entity_categories", {}).values():
            all_types.extend(category_types)

        prompt_lower = prompt.lower()
        matched_types = [
            t for t in all_types
            if t.replace("Ifc", "").lower() in prompt_lower
        ]

        # Look up matched types in graph — depth 1 only
        entities = []
        relations = []
        seen_nodes = set()

        for node_id, data in graph.nodes(data=True):
            if data.get("ifc_type", "") in matched_types:
                if node_id not in seen_nodes:
                    entities.append({"id": node_id, "ifc_type": data["ifc_type"],
                                     "name": data.get("name", "")})
                    seen_nodes.add(node_id)
                    # Only direct neighbours (depth=1, no further traversal)
                    for _, nbr in graph.out_edges(node_id):
                        if nbr not in seen_nodes:
                            nbr_data = graph.nodes[nbr]
                            entities.append({"id": nbr, "ifc_type": nbr_data.get("ifc_type", ""),
                                             "name": nbr_data.get("name", "")})
                            seen_nodes.add(nbr)
                        edge_data = graph[node_id][nbr]
                        relations.append({"type": edge_data.get("relation_type", ""),
                                          "from": node_id, "to": nbr})

        return {
            "prompt_id": prompt_id, "prompt": prompt, "baseline": "B3_ifc_lookup",
            "entities": entities, "relations": relations, "attributes": {},
            "containment": [], "connectivity": [], "constraints": [],
            "generation_prompt": f"{prompt}, technical 3D render, white background",
            "matched_ifc_types": matched_types,
        }
