"""Audit whether an IFC reference graph covers DTAH-Bench entity types."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from benchmark.dtah_bench import DTAHBench
from pipeline.layer1_retriever.ifc_graph_builder import IFCGraphBuilder


def normalize_ifc_type(value: str) -> str:
    """Remove benchmark instance qualifiers such as ``IfcPump[duty]``."""
    return re.split(r"\[", value, maxsplit=1)[0]


def audit_reference_coverage(
    graph_path: str | Path = "outputs/graphs/ifc_graph.json",
) -> dict:
    """Return benchmark entity-type coverage for a cached IFC graph."""
    graph = IFCGraphBuilder.load_graph(graph_path)
    graph_types = {
        data.get("ifc_type", "")
        for _, data in graph.nodes(data=True)
        if data.get("ifc_type")
    }
    required_types = {
        normalize_ifc_type(prompt["ifc_entity"])
        for prompt in DTAHBench().load_all()
        if isinstance(prompt.get("ifc_entity"), str)
    }
    covered = sorted(required_types & graph_types)
    missing = sorted(required_types - graph_types)
    return {
        "graph_path": str(graph_path),
        "graph_type_count": len(graph_types),
        "required_type_count": len(required_types),
        "covered_type_count": len(covered),
        "coverage": len(covered) / len(required_types) if required_types else 1.0,
        "covered_types": covered,
        "missing_types": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="outputs/graphs/ifc_graph.json")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.8,
        help="Exit with failure when coverage is below this fraction",
    )
    args = parser.parse_args()

    report = audit_reference_coverage(args.graph)
    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["coverage"] < args.min_coverage:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
