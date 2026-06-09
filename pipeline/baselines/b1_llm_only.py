"""
b1_llm_only.py — Baseline implementation for Paper 2 ablation study.
B1: Prompt-only LLM. LLM generates spec from prompt alone, no retrieval.
"""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

SPEC_PROMPT = """You are a 3D scene planner. Given a prompt, produce a JSON scene specification.
Output ONLY valid JSON with keys: entities, relations, attributes, containment, connectivity, constraints, generation_prompt.
No markdown, no preamble."""

class B1LLMOnly:
    """Prompt-only LLM baseline — no retrieval, just chain-of-thought reasoning."""

    def __init__(self, config: dict):
        self.config = config
        self.llm_provider = config.get("llm_provider", "anthropic")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self._client = None

    def run(self, prompt: str, prompt_id: str = "") -> dict:
        logger.info("[B1] LLM-only baseline for: %s", prompt[:60])
        if self.llm_provider == "deterministic":
            return self._deterministic_spec(prompt, prompt_id)
        response = self._call_llm(prompt)
        import re, json as _json
        clean = re.sub(r"```(?:json)?", "", response).strip().rstrip("`")
        try:
            spec = _json.loads(clean)
        except Exception:
            spec = {"entities": [], "relations": [], "attributes": {},
                    "containment": [], "connectivity": [], "constraints": [],
                    "generation_prompt": prompt}
        spec.update({"prompt_id": prompt_id, "prompt": prompt, "baseline": "B1_llm_only"})
        return spec

    @staticmethod
    def _deterministic_spec(prompt: str, prompt_id: str) -> dict:
        """Offline prompt-only approximation used for reproducible smoke runs."""
        keyword_types = {
            "pump": "IfcPump",
            "valve": "IfcValve",
            "pipe": "IfcPipeSegment",
            "fan": "IfcFan",
            "duct": "IfcDuctSegment",
            "beam": "IfcBeam",
            "column": "IfcColumn",
            "wall": "IfcWall",
            "slab": "IfcSlab",
            "space": "IfcSpace",
            "room": "IfcSpace",
        }
        prompt_lower = prompt.lower()
        entity_types = []
        for keyword, ifc_type in keyword_types.items():
            if keyword in prompt_lower and ifc_type not in entity_types:
                entity_types.append(ifc_type)
        entities = [
            {"id": f"entity_{index}", "ifc_type": ifc_type}
            for index, ifc_type in enumerate(entity_types)
        ]
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "baseline": "B1_llm_only_offline",
            "entities": entities,
            "relations": [],
            "attributes": {},
            "containment": [],
            "connectivity": [],
            "constraints": [],
            "generation_prompt": prompt,
        }

    def _call_llm(self, prompt: str) -> str:
        if self.llm_provider == "anthropic":
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic()
            msg = self._client.messages.create(
                model=self.model, max_tokens=1500,
                system=SPEC_PROMPT,
                messages=[{"role": "user", "content": f"PROMPT: {prompt}"}],
            )
            return msg.content[0].text
        raise ValueError(f"Unknown provider: {self.llm_provider}")
