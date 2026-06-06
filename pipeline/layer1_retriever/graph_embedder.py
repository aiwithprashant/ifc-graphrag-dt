"""
graph_embedder.py
Layer 1 — Graph Embedder

Embeds IFC entity descriptions using sentence-transformers and builds
a FAISS index for fast similarity-based seed node selection.

Workflow:
  1. For each node in the IFC graph, build a text description from
     its ifc_type, name, attributes, and property sets.
  2. Encode descriptions with a sentence-transformer model.
  3. Index embeddings in FAISS for ANN search.
  4. Given a natural-language prompt, retrieve top-k seed nodes
     whose descriptions are most similar to the prompt.

These seed nodes are then passed to KHopTraversal to expand
into a full IFC subgraph.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# Default embedding model — lightweight, good for domain text
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_node_text(node_id: str, data: dict) -> str:
    """
    Build a natural-language description of an IFC entity for embedding.

    Example output:
      "IfcPump named PrimaryPump: material=cast_iron, flow_rate=medium,
       orientation=horizontal. Pset_PumpTypeCommon: FlowRate=50m3h"
    """
    parts = []

    ifc_type = data.get("ifc_type", "IfcObject")
    name = data.get("name", "")
    description = data.get("description", "")

    if name:
        parts.append(f"{ifc_type} named {name}")
    else:
        parts.append(ifc_type)

    if description:
        parts.append(f"description: {description}")

    attrs = data.get("attributes", {})
    if attrs:
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v is not None)
        if attr_str:
            parts.append(attr_str)

    psets = data.get("psets", {})
    for pset_name, props in psets.items():
        if props:
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            parts.append(f"{pset_name}: {prop_str}")

    return ". ".join(parts)


class GraphEmbedder:
    """
    Embeds IFC graph nodes and retrieves seed nodes for a given prompt.

    Parameters
    ----------
    model_name : str
        HuggingFace sentence-transformers model name.
    device : str
        'cpu' or 'cuda'. Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._index = None
        self._node_ids: list[str] = []
        self._node_texts: list[str] = []
        self._embeddings: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, graph: nx.DiGraph) -> "GraphEmbedder":
        """
        Build embeddings for all nodes in the graph and index them.

        Parameters
        ----------
        graph : nx.DiGraph
            The IFC property graph from IFCGraphBuilder.

        Returns
        -------
        self (for chaining)
        """
        self._load_model()
        self._node_ids = []
        self._node_texts = []

        for node_id, data in graph.nodes(data=True):
            self._node_ids.append(node_id)
            self._node_texts.append(_build_node_text(node_id, data))

        logger.info("Embedding %d IFC entity nodes...", len(self._node_texts))
        self._embeddings = self._model.encode(
            self._node_texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self._build_faiss_index()
        logger.info("Embeddings built and indexed.")
        return self

    def retrieve_seeds(
        self,
        prompt: str,
        top_k: int = 5,
        ifc_type_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve top-k seed nodes most similar to the prompt.

        Parameters
        ----------
        prompt : str
            Natural language prompt (e.g. "Generate a pump connected to two pipes")
        top_k : int
            Number of seed nodes to return.
        ifc_type_filter : list[str] | None
            If provided, only return nodes of these IFC types.

        Returns
        -------
        list of dicts: [{"node_id": str, "score": float, "ifc_type": str, "text": str}]
        """
        if self._model is None or self._index is None:
            raise RuntimeError("Call fit() before retrieve_seeds().")

        query_embedding = self._model.encode(
            [prompt],
            normalize_embeddings=True,
        )

        # Search more candidates if filtering by type
        search_k = top_k * 10 if ifc_type_filter else top_k
        scores, indices = self._index.search(query_embedding.astype(np.float32), search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._node_ids):
                continue
            node_id = self._node_ids[idx]
            # We need ifc_type from stored text — extract from text prefix
            node_text = self._node_texts[idx]
            ifc_type = node_text.split(" ")[0]

            if ifc_type_filter and ifc_type not in ifc_type_filter:
                continue

            results.append({
                "node_id": node_id,
                "score": float(score),
                "ifc_type": ifc_type,
                "text": node_text,
            })
            if len(results) >= top_k:
                break

        logger.debug("Retrieved %d seed nodes for prompt: %s", len(results), prompt[:60])
        return results

    def retrieve_by_type(self, ifc_types: list[str]) -> list[str]:
        """
        Return all node IDs matching any of the given IFC types.
        Used for type-driven seed selection (bypasses embedding search).
        """
        results = []
        for node_id, text in zip(self._node_ids, self._node_texts):
            node_type = text.split(" ")[0]
            if node_type in ifc_types:
                results.append(node_id)
        return results

    def save(self, output_dir: str | Path) -> None:
        """Save embeddings, index, and metadata to disk."""
        import faiss  # type: ignore

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(output_dir / "faiss.index"))
        np.save(output_dir / "embeddings.npy", self._embeddings)

        with open(output_dir / "metadata.pkl", "wb") as f:
            pickle.dump({
                "node_ids": self._node_ids,
                "node_texts": self._node_texts,
                "model_name": self.model_name,
            }, f)

        logger.info("Embedder saved to %s", output_dir)

    @classmethod
    def load(cls, output_dir: str | Path, device: Optional[str] = None) -> "GraphEmbedder":
        """Load a previously saved embedder from disk."""
        import faiss  # type: ignore

        output_dir = Path(output_dir)

        with open(output_dir / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)

        embedder = cls(model_name=meta["model_name"], device=device)
        embedder._node_ids = meta["node_ids"]
        embedder._node_texts = meta["node_texts"]
        embedder._embeddings = np.load(output_dir / "embeddings.npy")
        embedder._index = faiss.read_index(str(output_dir / "faiss.index"))
        embedder._load_model()

        logger.info(
            "Embedder loaded: %d nodes from %s",
            len(embedder._node_ids),
            output_dir,
        )
        return embedder

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            ) from e

        device = self.device or ("cuda" if self._cuda_available() else "cpu")
        logger.info("Loading embedding model %s on %s", self.model_name, device)
        self._model = SentenceTransformer(self.model_name, device=device)

    def _build_faiss_index(self) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise ImportError(
                "faiss-cpu required. Install: pip install faiss-cpu"
            ) from e

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalised vecs
        self._index.add(self._embeddings.astype(np.float32))
        logger.info("FAISS index built: %d vectors, dim=%d", self._index.ntotal, dim)

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
