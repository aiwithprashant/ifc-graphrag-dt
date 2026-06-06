"""
b0_no_grounding.py — Baseline implementation for Paper 2 ablation study.
B0: No grounding. Vanilla text-to-3D with no IFC retrieval.
"""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

class B0NoGrounding:
    """Vanilla text-to-3D with no IFC grounding (floor baseline)."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs/meshes"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, prompt: str, prompt_id: str = "") -> dict:
        """Return an empty spec — no grounding applied."""
        logger.info("[B0] Running no-grounding baseline for: %s", prompt[:60])
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "baseline": "B0_no_grounding",
            "entities": [],
            "relations": [],
            "attributes": {},
            "containment": [],
            "connectivity": [],
            "constraints": [],
            "generation_prompt": (
                f"{prompt}, photorealistic, technical 3D render, "
                "white background, studio lighting"
            ),
        }
