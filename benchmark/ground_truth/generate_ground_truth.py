"""
generate_ground_truth.py
Ground Truth Generator for DTAH-Bench

Automatically generates ground-truth IFC subgraph annotations and
scene specifications for all 150 DTAH-Bench prompts by:
  1. Running the full IFC-GraphRAG-DT pipeline (Layer 1 + Layer 2)
     using the IFC reference model
  2. Saving retrieved subgraphs as ground-truth IFC annotations
  3. Saving LLM-generated specs as ground-truth scene specs
  4. Producing annotation JSON stubs for expert Stage B review

This replaces the need for fully manual annotation for Stage A.
Stage B (mesh evaluation) still requires expert human annotation.

Usage:
    python benchmark/ground_truth/generate_ground_truth.py \\
        --ifc benchmark/ifc_reference_models/duplex.ifc \\
        --tier 1 2 3 --pilot
"""

from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add repo root to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_ground_truth(
    ifc_path: str,
    tiers: list[int],
    pilot: bool,
    anthropic_key: str,
    output_dir: str = "benchmark/ground_truth",
) -> None:
    from pipeline.layer1_retriever.ifc_graph_builder import IFCGraphBuilder
    from pipeline.layer1_retriever.graph_embedder import GraphEmbedder
    from pipeline.layer1_retriever.khop_traversal import KHopTraversal
    from pipeline.layer2_spec_gen.scene_spec_generator import SceneSpecGenerator
    from benchmark.dtah_bench import DTAHBench
    import networkx as nx

    os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    TIER_DEPTHS = {1: 1, 2: 2, 3: 4}

    # Output dirs
    subgraph_dir = Path(output_dir) / "ifc_subgraphs"
    spec_dir     = Path(output_dir) / "scene_specs"
    annot_dir    = Path(output_dir) / "annotations"
    for d in [subgraph_dir, spec_dir, annot_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Build graph and embedder
    graph_cache = Path("outputs/graphs/ifc_graph.json")
    emb_cache   = Path("outputs/embedders/graph_embedder")
    Path("outputs/graphs").mkdir(parents=True, exist_ok=True)
    Path("outputs/embedders").mkdir(parents=True, exist_ok=True)

    if graph_cache.exists():
        logger.info("Loading cached IFC graph...")
        G = IFCGraphBuilder.load_graph(graph_cache)
    else:
        logger.info("Building IFC graph from %s...", ifc_path)
        builder = IFCGraphBuilder(ifc_path)
        G = builder.build()
        builder.save_graph(graph_cache)
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if (emb_cache / "faiss.index").exists():
        logger.info("Loading cached embedder...")
        embedder = GraphEmbedder.load(emb_cache)
    else:
        logger.info("Building embeddings (this takes a few minutes)...")
        embedder = GraphEmbedder()
        embedder.fit(G)
        embedder.save(emb_cache)

    spec_config = {
        "llm_provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 2048,
        "temperature": 0.0,
        "output_dir": str(spec_dir),
    }
    spec_gen = SceneSpecGenerator(spec_config)
    bench    = DTAHBench(pilot_mode=pilot)

    total, done, skipped = 0, 0, 0
    for tier in tiers:
        prompts = bench.load_tier(tier)
        for p in prompts:
            pid    = p["id"]
            prompt = p["prompt"]
            domain = p.get("domain", "MEP")
            total += 1

            # Skip if already generated
            spec_path = spec_dir / f"{pid}_gt_spec.json"
            sg_path   = subgraph_dir / f"{pid}_subgraph.json"
            if spec_path.exists() and sg_path.exists():
                logger.info("[SKIP] %s — already exists", pid)
                skipped += 1
                continue

            logger.info("[%d/%d] Generating GT for %s: %s",
                        done+1, total-skipped, pid, prompt[:60])

            # Layer 1: retrieve subgraph
            seeds    = embedder.retrieve_seeds(prompt, top_k=5)
            seed_ids = [s["node_id"] for s in seeds]
            depth    = TIER_DEPTHS[tier]
            trav     = KHopTraversal(G, max_depth=depth, bidirectional=True)
            result   = trav.traverse(seed_ids)
            subgraph = result.subgraph

            # Save subgraph as ground truth
            sg_data = nx.node_link_data(subgraph)
            with open(sg_path, "w") as f:
                json.dump(sg_data, f, indent=2, default=str)

            # Layer 2: generate spec (this becomes ground-truth spec)
            try:
                spec = spec_gen.generate(
                    prompt=prompt, subgraph=subgraph,
                    prompt_id=pid, tier=tier, domain=domain,
                )
                spec["ground_truth"] = True
                spec["generation_method"] = "IFC-GraphRAG-DT-auto"
                with open(spec_path, "w") as f:
                    json.dump(spec, f, indent=2)

                # Create annotation stub for Stage B expert review
                annot_stub = {
                    "prompt_id":    pid,
                    "prompt":       prompt,
                    "tier":         tier,
                    "domain":       domain,
                    "annotator_id": "",           # to be filled by expert
                    "annotation_date": "",
                    "entity":       None,         # expert fills: 0.0/0.25/0.5/0.75/1.0
                    "relation":     None,
                    "attribute":    None,
                    "containment":  None,
                    "connectivity": None,
                    "failure_codes": [],
                    "notes":        "",
                    "mesh_path":    f"outputs/meshes/{pid}.obj",
                    "spec_path":    str(spec_path),
                }
                annot_path = annot_dir / f"{pid}_annotation_stub.json"
                with open(annot_path, "w") as f:
                    json.dump(annot_stub, f, indent=2)

                done += 1
                logger.info("  ✓ Saved: subgraph=%d nodes, spec=%d entities",
                            subgraph.number_of_nodes(), len(spec.get("entities",[])))

            except Exception as e:
                logger.error("  ✗ Failed for %s: %s", pid, e)

    logger.info("\n" + "="*55)
    logger.info("Ground truth generation complete:")
    logger.info("  Total prompts:  %d", total)
    logger.info("  Generated:      %d", done)
    logger.info("  Skipped:        %d", skipped)
    logger.info("  Subgraphs:      %s", subgraph_dir)
    logger.info("  Specs:          %s", spec_dir)
    logger.info("  Annotation stubs: %s", annot_dir)
    logger.info("="*55)
    logger.info("NEXT STEP: Expert annotators fill in the annotation stubs")
    logger.info("  under benchmark/ground_truth/annotations/")
    logger.info("  Then run: python -m evaluation.run_eval --stage-b --annotations ...")


def main():
    parser = argparse.ArgumentParser(description="Generate DTAH-Bench ground truth annotations")
    parser.add_argument("--ifc",    default="benchmark/ifc_reference_models/duplex.ifc")
    parser.add_argument("--tiers",  nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pilot",  action="store_true", help="Use DTAH-Bench-50 pilot only")
    parser.add_argument("--key",    default=os.environ.get("ANTHROPIC_API_KEY",""),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    if not args.key:
        print("ERROR: Anthropic API key required. Set --key or ANTHROPIC_API_KEY env var.")
        sys.exit(1)

    generate_ground_truth(
        ifc_path=args.ifc,
        tiers=args.tiers,
        pilot=args.pilot,
        anthropic_key=args.key,
    )


if __name__ == "__main__":
    main()
