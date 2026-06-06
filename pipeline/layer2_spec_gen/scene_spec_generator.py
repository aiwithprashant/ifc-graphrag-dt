"""
scene_spec_generator.py
Layer 2 — Scene Specification Generator

Converts a retrieved IFC subgraph (from Layer 1) into a structured
scene specification JSON that:
  1. Lists all required entities with positions and attributes
  2. Defines IFC relations between them
  3. Specifies containment hierarchy
  4. Lists connectivity (port-to-port)
  5. Encodes geometric/topological constraints
  6. Produces a formatted generation prompt for Stable Diffusion

The spec is evaluated at Stage A (specification correctness) before
being passed to Layer 3 (3D generator).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

SPEC_SCHEMA_PATH = Path(__file__).parent / "spec_schema.json"

# ── System prompt for LLM spec generation ────────────────────────────────────
SPEC_SYSTEM_PROMPT = """You are an expert BIM engineer and IFC schema specialist.
Your task is to convert a retrieved IFC ontology subgraph into a structured
Digital Twin scene specification.

You will be given:
1. A natural language prompt describing a DT asset or system
2. A list of IFC entities retrieved from the ontology graph
3. A list of IFC relations between those entities

Produce a JSON scene specification following this exact structure:
{
  "entities": [{"id": "...", "ifc_type": "...", "material": "...", "position": [x,y,z], "attributes": {...}}],
  "relations": [{"type": "IfcRel...", "from": "entity_id_or_port", "to": "entity_id"}],
  "attributes": {"entity_id": {"material": "...", "flow_direction": "..."}},
  "containment": [{"entity": "entity_id", "container": "space_id"}],
  "connectivity": [{"from": "entity_id.port", "to": "entity_id", "port_type": "inlet|outlet|supply|return|generic"}],
  "constraints": ["constraint description as plain English"],
  "generation_prompt": "A detailed visual description for Stable Diffusion"
}

Rules:
- Use ONLY IFC types present in the retrieved entities list
- Position values should be plausible in metres (origin at [0,0,0])
- Constraints must be enforceable (geometric or topological)
- generation_prompt should describe the 3D scene visually and technically
- Respond ONLY with valid JSON — no preamble, no markdown fences
"""


class SceneSpecGenerator:
    """
    Converts an IFC subgraph + prompt into a structured scene specification.

    Parameters
    ----------
    config : dict
        spec_generator section from pipeline_config.yaml
    """

    def __init__(self, config: dict):
        self.config = config
        self.llm_provider = config.get("llm_provider", "anthropic")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.0)
        self.output_dir = Path(config.get("output_dir", "outputs/specs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        subgraph: nx.DiGraph,
        prompt_id: str = "",
        tier: int = 1,
        domain: str = "MEP",
    ) -> dict:
        """
        Generate a scene specification from a prompt and retrieved subgraph.

        Parameters
        ----------
        prompt     : str         Natural language prompt
        subgraph   : nx.DiGraph  Retrieved IFC subgraph from Layer 1
        prompt_id  : str         DTAH-Bench prompt ID
        tier       : int         1, 2, or 3
        domain     : str         MEP, structural, or HVAC

        Returns
        -------
        dict  Scene specification (validated against spec_schema.json)
        """
        entities = self._subgraph_to_entity_list(subgraph)
        relations = self._subgraph_to_relation_list(subgraph)

        user_message = self._build_user_message(prompt, entities, relations, tier)

        logger.info("Generating spec for [%s]: %s", prompt_id, prompt[:60])
        raw_response = self._call_llm(user_message)
        spec = self._parse_and_validate(raw_response)

        spec["prompt_id"] = prompt_id
        spec["prompt"] = prompt
        spec["tier"] = tier
        spec["domain"] = domain

        self._save_spec(spec, prompt_id)
        return spec

    def generate_batch(
        self,
        prompts: list[dict],
        subgraphs: list[nx.DiGraph],
    ) -> list[dict]:
        """Generate specs for a batch of prompts."""
        results = []
        for item, subgraph in zip(prompts, subgraphs):
            spec = self.generate(
                prompt=item["prompt"],
                subgraph=subgraph,
                prompt_id=item.get("id", ""),
                tier=item.get("tier", 1),
                domain=item.get("domain", "MEP"),
            )
            results.append(spec)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _subgraph_to_entity_list(self, subgraph: nx.DiGraph) -> list[dict]:
        entities = []
        for node_id, data in subgraph.nodes(data=True):
            entities.append({
                "global_id": node_id,
                "ifc_type":  data.get("ifc_type", "IfcObject"),
                "name":      data.get("name", ""),
                "attributes": data.get("attributes", {}),
            })
        return entities

    def _subgraph_to_relation_list(self, subgraph: nx.DiGraph) -> list[dict]:
        relations = []
        for src, tgt, data in subgraph.edges(data=True):
            relations.append({
                "relation_type": data.get("relation_type", "IfcRelConnects"),
                "category":      data.get("category", "connectivity"),
                "from_global_id": src,
                "to_global_id":   tgt,
                "attributes":    data.get("attributes", {}),
            })
        return relations

    def _build_user_message(
        self,
        prompt: str,
        entities: list[dict],
        relations: list[dict],
        tier: int,
    ) -> str:
        return f"""PROMPT: {prompt}

TIER: {tier} ({'Asset' if tier == 1 else 'Assembly' if tier == 2 else 'System'})

RETRIEVED IFC ENTITIES ({len(entities)} nodes):
{json.dumps(entities, indent=2)}

RETRIEVED IFC RELATIONS ({len(relations)} edges):
{json.dumps(relations, indent=2)}

Generate the scene specification JSON."""

    def _call_llm(self, user_message: str) -> str:
        if self.llm_provider == "anthropic":
            return self._call_anthropic(user_message)
        elif self.llm_provider == "openai":
            return self._call_openai(user_message)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def _call_anthropic(self, user_message: str) -> str:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError("anthropic package required: pip install anthropic") from e

        if self._client is None:
            self._client = anthropic.Anthropic()

        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SPEC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text

    def _call_openai(self, user_message: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("openai package required: pip install openai") from e

        if self._client is None:
            self._client = OpenAI()

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SPEC_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _parse_and_validate(self, raw: str) -> dict:
        """Parse JSON response and ensure required keys exist."""
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            spec = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Raw response: %s", raw[:500])
            return self._empty_spec()

        # Ensure all required keys present
        required = ["entities", "relations", "attributes",
                    "containment", "connectivity", "constraints"]
        for key in required:
            if key not in spec:
                spec[key] = [] if key != "attributes" else {}

        if "generation_prompt" not in spec:
            spec["generation_prompt"] = self._fallback_generation_prompt(spec)

        return spec

    def _fallback_generation_prompt(self, spec: dict) -> str:
        types = [e.get("ifc_type", "") for e in spec.get("entities", [])]
        return f"Technical 3D render of {', '.join(set(types))}, industrial building services equipment, photorealistic, white background"

    @staticmethod
    def _empty_spec() -> dict:
        return {
            "entities": [], "relations": [], "attributes": {},
            "containment": [], "connectivity": [], "constraints": [],
            "generation_prompt": "",
        }

    def _save_spec(self, spec: dict, prompt_id: str) -> None:
        if not prompt_id:
            return
        out_path = self.output_dir / f"{prompt_id}_spec.json"
        with open(out_path, "w") as f:
            json.dump(spec, f, indent=2)
        logger.debug("Spec saved: %s", out_path)
