"""
ifc_graph_builder.py
Layer 1 — IFC Graph Builder

Loads an IFC file using ifcopenshell and constructs a typed NetworkX
property graph G = (V, E, τ) where:
  V = IFC entity instances (nodes)
  E = IFC relation instances (edges)
  τ = relation type label (IfcRelConnects, IfcRelContainedIn, etc.)

This graph is the foundation for k-hop GraphRAG traversal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ── IFC relation types we care about, mapped to semantic category ─────────────
IFC_RELATION_CATEGORIES: dict[str, str] = {
    "IfcRelContainedInSpatialStructure": "containment",
    "IfcRelAggregates":                  "aggregation",
    "IfcRelConnects":                    "connectivity",
    "IfcRelConnectsPortToElement":       "port_connectivity",
    "IfcRelAssignsToGroup":              "system_membership",
    "IfcRelAssigns":                     "assignment",
    "IfcRelVoidsElement":                "void",
    "IfcRelFillsElement":                "fill",
    "IfcRelAssociatesMaterial":          "material",
    "IfcRelDefinesByProperties":         "property",
    "IfcRelDefinesByType":               "type_definition",
}


@dataclass
class IFCGraphNode:
    """Represents a single IFC entity as a graph node."""
    global_id: str
    ifc_type: str
    name: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    psets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "ifc_type": self.ifc_type,
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "psets": self.psets,
        }


@dataclass
class IFCGraphEdge:
    """Represents a single IFC relation as a typed graph edge."""
    relation_id: str
    relation_type: str
    category: str
    source_id: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "category": self.category,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "attributes": self.attributes,
        }


class IFCGraphBuilder:
    """
    Builds a typed NetworkX DiGraph from an IFC file.

    Usage
    -----
    builder = IFCGraphBuilder("path/to/model.ifc")
    G = builder.build()
    builder.save_graph("output/graph.json")
    """

    def __init__(self, ifc_path: str | Path):
        self.ifc_path = Path(ifc_path)
        self.model = None
        self.G: nx.DiGraph = nx.DiGraph()
        self._node_map: dict[str, IFCGraphNode] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> nx.DiGraph:
        """Load IFC file and construct the full property graph."""
        self._load_model()
        self._add_entity_nodes()
        self._add_relation_edges()
        logger.info(
            "Graph built: %d nodes, %d edges",
            self.G.number_of_nodes(),
            self.G.number_of_edges(),
        )
        return self.G

    def get_node(self, global_id: str) -> Optional[IFCGraphNode]:
        return self._node_map.get(global_id)

    def get_subgraph(self, global_ids: list[str]) -> nx.DiGraph:
        """Return induced subgraph for a set of node global IDs."""
        return self.G.subgraph(global_ids).copy()

    def save_graph(self, output_path: str | Path) -> None:
        """Serialise graph to JSON (node-link format)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Graph saved to %s", output_path)

    @classmethod
    def load_graph(cls, json_path: str | Path) -> nx.DiGraph:
        """Load a previously saved graph from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return nx.node_link_graph(data)

    def summary(self) -> dict:
        """Return a summary of the graph contents."""
        type_counts: dict[str, int] = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("ifc_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, data in self.G.edges(data=True):
            r = data.get("relation_type", "unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "entity_type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "relation_type_counts": dict(sorted(rel_counts.items(), key=lambda x: -x[1])),
            "is_dag": nx.is_directed_acyclic_graph(self.G),
        }

    # ── Internal methods ──────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            import ifcopenshell  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ifcopenshell is required. Install with: pip install ifcopenshell"
            ) from e

        if not self.ifc_path.exists():
            raise FileNotFoundError(f"IFC file not found: {self.ifc_path}")

        logger.info("Loading IFC file: %s", self.ifc_path)
        self.model = ifcopenshell.open(str(self.ifc_path))
        logger.info("IFC schema: %s", self.model.schema)

    def _add_entity_nodes(self) -> None:
        """Add one node per IFC product/object entity."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build() first.")

        entity_types = [
            "IfcProduct",
            "IfcObject",
            "IfcContext",
            "IfcGroup",
            "IfcSystem",
        ]

        added = 0
        for entity_type in entity_types:
            for entity in self._by_type(entity_type):
                global_id = getattr(entity, "GlobalId", None)
                if global_id is None or global_id in self._node_map:
                    continue

                node = IFCGraphNode(
                    global_id=global_id,
                    ifc_type=entity.is_a(),
                    name=getattr(entity, "Name", "") or "",
                    description=getattr(entity, "Description", "") or "",
                    attributes=self._extract_attributes(entity),
                    psets=self._extract_psets(entity),
                )
                self._node_map[global_id] = node
                self.G.add_node(global_id, **node.to_dict())
                added += 1

        logger.info("Added %d entity nodes", added)

    def _add_relation_edges(self) -> None:
        """Add typed directed edges for each tracked IFC relation type."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        added = 0
        for rel_type, category in IFC_RELATION_CATEGORIES.items():
            for rel in self._by_type(rel_type):
                edges = self._extract_edges_from_relation(rel, rel_type, category)
                for edge in edges:
                    if edge.source_id not in self.G or edge.target_id not in self.G:
                        continue
                    self.G.add_edge(
                        edge.source_id,
                        edge.target_id,
                        **edge.to_dict(),
                    )
                    added += 1

        logger.info("Added %d relation edges", added)

    def _by_type(self, entity_type: str) -> list[Any]:
        """Return entities of a type, skipping types absent from the IFC schema."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        try:
            return self.model.by_type(entity_type)
        except RuntimeError as exc:
            if "not found in schema" not in str(exc):
                raise
            logger.debug(
                "Skipping unsupported IFC type %s for schema %s",
                entity_type,
                self.model.schema,
            )
            return []

    def _extract_edges_from_relation(
        self, rel: Any, rel_type: str, category: str
    ) -> list[IFCGraphEdge]:
        """Extract (source, target) pairs from a given IFC relation instance."""
        edges = []
        rel_id = getattr(rel, "GlobalId", str(id(rel)))

        # Spatial containment: space → product
        if rel_type == "IfcRelContainedInSpatialStructure":
            container = getattr(rel, "RelatingStructure", None)
            products = getattr(rel, "RelatedElements", []) or []
            if container and hasattr(container, "GlobalId"):
                for product in products:
                    if hasattr(product, "GlobalId"):
                        edges.append(IFCGraphEdge(
                            relation_id=rel_id, relation_type=rel_type,
                            category=category,
                            source_id=container.GlobalId,
                            target_id=product.GlobalId,
                        ))

        # Aggregation: whole → parts
        elif rel_type == "IfcRelAggregates":
            whole = getattr(rel, "RelatingObject", None)
            parts = getattr(rel, "RelatedObjects", []) or []
            if whole and hasattr(whole, "GlobalId"):
                for part in parts:
                    if hasattr(part, "GlobalId"):
                        edges.append(IFCGraphEdge(
                            relation_id=rel_id, relation_type=rel_type,
                            category=category,
                            source_id=whole.GlobalId,
                            target_id=part.GlobalId,
                        ))

        # Port-to-element connectivity
        elif rel_type == "IfcRelConnectsPortToElement":
            port = getattr(rel, "RelatingPort", None)
            element = getattr(rel, "RelatedElement", None)
            if port and element and hasattr(port, "GlobalId") and hasattr(element, "GlobalId"):
                edges.append(IFCGraphEdge(
                    relation_id=rel_id, relation_type=rel_type,
                    category=category,
                    source_id=port.GlobalId,
                    target_id=element.GlobalId,
                    attributes={"flow_direction": getattr(port, "FlowDirection", None)},
                ))

        # Generic connects
        elif rel_type == "IfcRelConnects":
            source = getattr(rel, "RelatingElement", None)
            target = getattr(rel, "RelatedElement", None)
            if source and target and hasattr(source, "GlobalId") and hasattr(target, "GlobalId"):
                edges.append(IFCGraphEdge(
                    relation_id=rel_id, relation_type=rel_type,
                    category=category,
                    source_id=source.GlobalId,
                    target_id=target.GlobalId,
                ))

        # System membership
        elif rel_type == "IfcRelAssignsToGroup":
            group = getattr(rel, "RelatingGroup", None)
            objects = getattr(rel, "RelatedObjects", []) or []
            if group and hasattr(group, "GlobalId"):
                for obj in objects:
                    if hasattr(obj, "GlobalId"):
                        edges.append(IFCGraphEdge(
                            relation_id=rel_id, relation_type=rel_type,
                            category=category,
                            source_id=group.GlobalId,
                            target_id=obj.GlobalId,
                        ))

        # Voids element (openings)
        elif rel_type == "IfcRelVoidsElement":
            building_element = getattr(rel, "RelatingBuildingElement", None)
            opening = getattr(rel, "RelatedOpeningElement", None)
            if building_element and opening and hasattr(building_element, "GlobalId") and hasattr(opening, "GlobalId"):
                edges.append(IFCGraphEdge(
                    relation_id=rel_id, relation_type=rel_type,
                    category=category,
                    source_id=building_element.GlobalId,
                    target_id=opening.GlobalId,
                ))

        return edges

    def _extract_attributes(self, entity: Any) -> dict[str, Any]:
        """Extract scalar attributes from an IFC entity."""
        attrs = {}
        skip = {"GlobalId", "OwnerHistory", "Name", "Description"}
        for attr_name in entity.get_info():
            if attr_name in skip:
                continue
            try:
                val = getattr(entity, attr_name, None)
                if val is None or hasattr(val, "is_a"):
                    continue
                if isinstance(val, (str, int, float, bool)):
                    attrs[attr_name] = val
            except Exception:
                pass
        return attrs

    def _extract_psets(self, entity: Any) -> dict[str, dict[str, Any]]:
        """Extract property sets from an IFC entity."""
        psets: dict[str, dict[str, Any]] = {}
        try:
            for definition in getattr(entity, "IsDefinedBy", []) or []:
                if not definition.is_a("IfcRelDefinesByProperties"):
                    continue
                pset = getattr(definition, "RelatingPropertyDefinition", None)
                if pset is None or not pset.is_a("IfcPropertySet"):
                    continue
                pset_name = getattr(pset, "Name", "Unknown") or "Unknown"
                props = {}
                for prop in getattr(pset, "HasProperties", []) or []:
                    prop_name = getattr(prop, "Name", None)
                    prop_val = getattr(prop, "NominalValue", None)
                    if prop_name and prop_val is not None:
                        props[prop_name] = getattr(prop_val, "wrappedValue", str(prop_val))
                psets[pset_name] = props
        except Exception:
            pass
        return psets
