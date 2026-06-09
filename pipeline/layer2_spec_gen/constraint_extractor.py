"""
constraint_extractor.py
Layer 2 — Constraint Extractor

Extracts geometric and topological constraints from a retrieved IFC
subgraph and appends them to the scene specification. These constraints
are passed to Layer 3 to condition 3D generation.

Constraint types:
  - Spatial: "pump inside pump_room bounding box"
  - Topological: "pipe_endpoint touches pump_port"
  - Material: "valve body material = stainless_steel"
  - Dimensional: "pump dimensions plausible for 50 m3/h flow rate"
  - Connectivity: "valve lies between two pipe segments"
"""

from __future__ import annotations
import logging
from typing import Optional
import networkx as nx

logger = logging.getLogger(__name__)

# IFC relation → constraint template
RELATION_CONSTRAINTS = {
    "IfcRelConnectsPortToElement": (
        "port of {source} connects directly to {target} — endpoints must touch"
    ),
    "IfcRelConnects": (
        "{source} physically connects to {target} — no gap between objects"
    ),
    "IfcRelContainedInSpatialStructure": (
        "{target} must be fully contained within bounding box of {source}"
    ),
    "IfcRelAggregates": (
        "{target} is a sub-component of {source} — must be spatially nested"
    ),
    "IfcRelAssignsToGroup": (
        "{target} belongs to system {source} — must be reachable via connectivity"
    ),
}


class ConstraintExtractor:
    """
    Extracts enforceable constraints from an IFC subgraph.

    Parameters
    ----------
    tier : int
        DTAH-Bench tier (1/2/3). Higher tiers produce more constraints.
    """

    def __init__(self, tier: int = 2):
        self.tier = tier

    def extract(
        self,
        subgraph: nx.DiGraph,
        spec: dict,
        prompt: str = "",
    ) -> list[str]:
        """
        Extract constraints from the subgraph and existing spec.

        Returns a list of plain-English constraint strings for injection
        into the scene specification and Stable Diffusion prompt.
        """
        constraints = []

        # 1. Relation-derived constraints
        constraints.extend(self._relation_constraints(subgraph))

        # 2. Containment constraints (Tier 3 only)
        if self.tier >= 3:
            constraints.extend(self._containment_constraints(subgraph))

        # 3. Connectivity chain constraints (Tier 2+)
        if self.tier >= 2:
            constraints.extend(self._connectivity_chain_constraints(subgraph))

        # 4. Material consistency constraints from spec attributes
        constraints.extend(self._material_constraints(spec))

        # 5. Scale plausibility from IFC entity type
        constraints.extend(self._scale_constraints(subgraph))

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for c in constraints:
            if c not in seen:
                seen.add(c)
                deduped.append(c)

        logger.debug("Extracted %d constraints for tier %d", len(deduped), self.tier)
        return deduped

    # ── Constraint generators ─────────────────────────────────────────────────

    def _relation_constraints(self, subgraph: nx.DiGraph) -> list[str]:
        constraints = []
        for src, tgt, data in subgraph.edges(data=True):
            rel_type = data.get("relation_type", "")
            template = RELATION_CONSTRAINTS.get(rel_type)
            if template:
                src_type = subgraph.nodes[src].get("ifc_type", src)
                tgt_type = subgraph.nodes[tgt].get("ifc_type", tgt)
                constraints.append(
                    template.format(source=src_type, target=tgt_type)
                )
        return constraints

    def _containment_constraints(self, subgraph: nx.DiGraph) -> list[str]:
        constraints = []
        spaces = [n for n, d in subgraph.nodes(data=True)
                  if "Space" in d.get("ifc_type", "") or "Storey" in d.get("ifc_type", "")]
        if spaces:
            non_spaces = [n for n, d in subgraph.nodes(data=True)
                         if "Space" not in d.get("ifc_type","") and
                            "Storey" not in d.get("ifc_type","")]
            if non_spaces:
                space_name = subgraph.nodes[spaces[0]].get("ifc_type", "spatial container")
                constraints.append(
                    f"all equipment must be spatially located within {space_name}"
                )
        return constraints

    def _connectivity_chain_constraints(self, subgraph: nx.DiGraph) -> list[str]:
        constraints = []
        # Find pipe segments and check they form connected chains
        pipe_nodes = [n for n, d in subgraph.nodes(data=True)
                      if "Pipe" in d.get("ifc_type","") or "Duct" in d.get("ifc_type","")]
        if len(pipe_nodes) >= 2:
            constraints.append(
                "pipe/duct segments must form a continuous connected network "
                "with no floating endpoints"
            )
        return constraints

    def _material_constraints(self, spec: dict) -> list[str]:
        constraints = []
        attrs = spec.get("attributes", {})
        seen_materials = set()
        for eid, vals in attrs.items():
            mat = vals.get("material", "")
            if mat and mat not in seen_materials:
                seen_materials.add(mat)
                constraints.append(
                    f"objects with material={mat} must have visually consistent "
                    f"surface appearance (e.g. metallic sheen for steel/cast_iron)"
                )
        return constraints

    def _scale_constraints(self, subgraph: nx.DiGraph) -> list[str]:
        """Add scale plausibility constraints based on IFC entity type."""
        SCALE_HINTS = {
            "IfcPump":           "pump body approximately 0.3–1.5m in longest dimension",
            "IfcValve":          "valve body approximately 0.1–0.5m in longest dimension",
            "IfcPipeSegment":    "pipe diameter consistent with connected equipment ports",
            "IfcDuctSegment":    "duct cross-section consistent with connected AHU/fan",
            "IfcColumn":         "column cross-section proportional to height (slenderness ratio 10–20)",
            "IfcBeam":           "beam depth approximately span/15 to span/20",
            "IfcFan":            "fan casing approximately 0.3–1.2m diameter",
            "IfcChiller":        "chiller unit approximately 1.5–4m long",
            "IfcAirToAirHeatRecovery": "AHU unit approximately 1.5–5m long",
        }
        constraints = []
        seen = set()
        for _, data in subgraph.nodes(data=True):
            ifc_type = data.get("ifc_type","")
            if ifc_type in SCALE_HINTS and ifc_type not in seen:
                constraints.append(SCALE_HINTS[ifc_type])
                seen.add(ifc_type)
        return constraints
