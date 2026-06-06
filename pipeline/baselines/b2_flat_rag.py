"""
b2_flat_rag.py — Baseline implementation for Paper 2 ablation study.
B2: Flat RAG. Dense vector retrieval from IFC docs without graph structure.
"""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

class B2FlatRAG:
    """
    Flat RAG baseline — dense retrieval from IFC documentation text
    without graph structure. Nodes retrieved but edges not preserved.
    This is the critical comparison against B4 (GraphRAG-DT).
    """

    def __init__(self, config: dict, graph_embedder=None):
        self.config = config
        self.embedder = graph_embedder  # Same embedder as B4, but no traversal
        self.top_k = config.get("retriever", {}).get("top_k_seeds", 10)

    def run(self, prompt: str, graph: "nx.DiGraph", prompt_id: str = "") -> dict:
        """Retrieve top-k nodes by similarity only — no edge traversal."""
        logger.info("[B2] Flat RAG baseline for: %s", prompt[:60])

        if self.embedder is None:
            return self._empty_spec(prompt, prompt_id)

        # Retrieve seed nodes only — no k-hop expansion
        seeds = self.embedder.retrieve_seeds(prompt, top_k=self.top_k)
        seed_ids = [s["node_id"] for s in seeds]

        # Return only the seed nodes with NO edges — flat retrieval
        flat_entities = []
        for seed in seeds:
            flat_entities.append({
                "id": seed["node_id"],
                "ifc_type": seed["ifc_type"],
                "similarity_score": seed["score"],
            })

        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "baseline": "B2_flat_rag",
            "entities": flat_entities,
            "relations": [],        # KEY DIFFERENCE: no edges recovered
            "attributes": {},
            "containment": [],
            "connectivity": [],
            "constraints": [],
            "generation_prompt": f"{prompt}, technical 3D render, white background",
        }

    @staticmethod
    def _empty_spec(prompt, prompt_id):
        return {"prompt_id": prompt_id, "prompt": prompt, "baseline": "B2_flat_rag",
                "entities": [], "relations": [], "attributes": {},
                "containment": [], "connectivity": [], "constraints": [],
                "generation_prompt": prompt}
