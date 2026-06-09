"""
run_pipeline.py
IFC-GraphRAG-DT — Main Pipeline Orchestrator

Ties all three layers together into an end-to-end pipeline:
  Layer 1: IFC graph retrieval + k-hop traversal
  Layer 2: Scene specification generation
  Layer 3: Conditioned 3D generation + mesh post-processing

Also runs all baselines for comparison and computes GGS + KCS-DT scores.

Usage (CLI):
  graphrag-dt --prompt "Generate a pump connected to two pipes" --config pipeline/configs/pipeline_config.yaml
  graphrag-dt --benchmark --tier 1 --pilot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a single pipeline run."""
    prompt_id:    str
    prompt:       str
    tier:         int
    spec:         dict
    mesh_path:    Optional[Path]
    stage_a:      Optional[dict]   # GGS + spec score
    stage_b:      Optional[dict]   # KCS-DT
    baseline:     str = "B4_graphrag_dt"

    def to_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt":    self.prompt,
            "tier":      self.tier,
            "baseline":  self.baseline,
            "spec_entities": len(self.spec.get("entities", [])),
            "spec_relations": len(self.spec.get("relations", [])),
            "mesh_path": str(self.mesh_path) if self.mesh_path else None,
            "stage_a":   self.stage_a,
            "stage_b":   self.stage_b,
        }


class IFCGraphRAGDT:
    """
    Main pipeline class for IFC-GraphRAG-DT.

    Parameters
    ----------
    config_path : str | Path
        Path to pipeline_config.yaml
    dry_run : bool
        If True, skip 3D generation (Layer 3) — useful for local CPU testing
    """

    def __init__(
        self,
        config_path: str | Path = "pipeline/configs/pipeline_config.yaml",
        dry_run: bool = False,
        offline: bool = False,
    ):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.offline = offline
        self.config = self._load_config()
        if self.offline:
            self.config.setdefault("spec_generator", {})["llm_provider"] = "deterministic"
        self._setup_logging()

        # Lazy-loaded components
        self._graph = None
        self._embedder = None
        self._traversal = None
        self._spec_gen = None
        self._sd = None
        self._triposr = None
        self._postproc = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, prompt: str, prompt_id: str = "", tier: int = 2) -> PipelineResult:
        """Run the full pipeline for a single prompt."""
        logger.info("=" * 60)
        logger.info("Pipeline run: [%s] %s", prompt_id, prompt[:80])

        # Layer 1: Retrieve IFC subgraph
        subgraph, seeds = self._retrieve(prompt, tier)
        logger.info("Layer 1: %d nodes, %d edges retrieved", subgraph.number_of_nodes(), subgraph.number_of_edges())

        # Layer 2: Generate scene specification
        spec = self._generate_spec(prompt, subgraph, prompt_id, tier)
        logger.info("Layer 2: %d entities, %d relations in spec", len(spec.get("entities", [])), len(spec.get("relations", [])))

        # Layer 3: 3D generation (skipped in dry_run)
        mesh_path = None
        if not self.dry_run:
            mesh_path = self._generate_3d(spec, prompt_id)
            logger.info("Layer 3: mesh saved to %s", mesh_path)
        else:
            logger.info("Layer 3: skipped (dry_run=True)")

        return PipelineResult(
            prompt_id=prompt_id, prompt=prompt, tier=tier,
            spec=spec, mesh_path=mesh_path,
            stage_a=None, stage_b=None,
        )

    def run_benchmark(
        self,
        tiers: Optional[list[int]] = None,
        pilot: bool = False,
    ) -> list[PipelineResult]:
        """Run the pipeline on the full DTAH-Bench (or pilot subset)."""
        from benchmark.dtah_bench import DTAHBench

        bench = DTAHBench(pilot_mode=pilot)
        tiers = tiers or [1, 2, 3]
        results = []

        for tier in tiers:
            prompts = bench.load_tier(tier)
            logger.info("Running Tier %d: %d prompts", tier, len(prompts))
            for prompt_data in prompts:
                result = self.run(
                    prompt=prompt_data["prompt"],
                    prompt_id=prompt_data["id"],
                    tier=tier,
                )
                results.append(result)

        self._save_results(results)
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrieve(self, prompt: str, tier: int):
        from pipeline.layer1_retriever.ifc_graph_builder import IFCGraphBuilder
        from pipeline.layer1_retriever.graph_embedder import GraphEmbedder
        from pipeline.layer1_retriever.khop_traversal import KHopTraversal
        import yaml

        rc = self.config["retriever"]

        # Build / load graph
        if self._graph is None:
            cache = Path(rc["graph_cache_path"])
            if cache.exists():
                logger.info("Loading cached IFC graph from %s", cache)
                self._graph = IFCGraphBuilder.load_graph(cache)
            else:
                builder = IFCGraphBuilder(rc["ifc_model_path"])
                self._graph = builder.build()
                builder.save_graph(cache)

        # Build / load embedder
        if self._embedder is None:
            emb_cache = Path(rc["embedder_cache_path"])
            if emb_cache.exists():
                self._embedder = GraphEmbedder.load(emb_cache)
            else:
                self._embedder = GraphEmbedder(rc["embedding_model"])
                self._embedder.fit(self._graph)
                self._embedder.save(emb_cache)

        # Retrieve seeds
        seeds = self._embedder.retrieve_seeds(prompt, top_k=rc["top_k_seeds"])
        seed_ids = [s["node_id"] for s in seeds]

        # Load tier-specific depth
        ifc_config_path = Path("pipeline/configs/ifc_config.yaml")
        max_depth = rc["max_depth"]
        if ifc_config_path.exists():
            with open(ifc_config_path) as f:
                ifc_cfg = yaml.safe_load(f)
            max_depth = ifc_cfg.get("tier_depth_map", {}).get(tier, max_depth)

        traversal = KHopTraversal(
            graph=self._graph,
            max_depth=max_depth,
            bidirectional=rc.get("bidirectional", True),
        )
        result = traversal.traverse(seed_ids)
        return result.subgraph, seeds

    def _generate_spec(self, prompt, subgraph, prompt_id, tier):
        from pipeline.layer2_spec_gen.scene_spec_generator import SceneSpecGenerator
        if self._spec_gen is None:
            self._spec_gen = SceneSpecGenerator(self.config["spec_generator"])
        return self._spec_gen.generate(
            prompt=prompt, subgraph=subgraph,
            prompt_id=prompt_id, tier=tier,
        )

    def _generate_3d(self, spec, prompt_id):
        from pipeline.layer3_generator.sd_conditioner import SDConditioner
        from pipeline.layer3_generator.triposr_runner import TripoSRRunner
        from pipeline.layer3_generator.mesh_postprocess import MeshPostProcessor

        if self._sd is None:
            self._sd = SDConditioner(self.config["generator"])
        if self._triposr is None:
            self._triposr = TripoSRRunner(self.config["generator"])
        if self._postproc is None:
            self._postproc = MeshPostProcessor(self.config["generator"])

        image_paths = self._sd.generate_views(spec, prompt_id)
        mesh_path = self._triposr.reconstruct(image_paths, prompt_id)
        self._postproc.process(mesh_path, prompt_id=prompt_id)
        return mesh_path

    def _save_results(self, results: list[PipelineResult]) -> None:
        out_dir = Path(self.config["evaluation"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "b4_scores.json"
        with open(out_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Results saved: %s", out_path)

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning("Config not found: %s — using defaults", self.config_path)
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        level = getattr(logging, log_cfg.get("level", "INFO"))
        log_file = log_cfg.get("log_file", "outputs/run.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file),
            ],
        )


def main():
    parser = argparse.ArgumentParser(description="IFC-GraphRAG-DT Pipeline")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument("--prompt-id", type=str, default="", help="Prompt ID")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    parser.add_argument("--pilot", action="store_true", help="Use DTAH-Bench-50 pilot")
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--config", type=str, default="pipeline/configs/pipeline_config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Skip 3D generation")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic Layer 2 generation without an LLM API",
    )
    args = parser.parse_args()

    pipeline = IFCGraphRAGDT(
        config_path=args.config,
        dry_run=args.dry_run,
        offline=args.offline,
    )

    if args.benchmark:
        results = pipeline.run_benchmark(tiers=args.tiers, pilot=args.pilot)
        print(f"\nCompleted {len(results)} benchmark runs.")
    elif args.prompt:
        result = pipeline.run(args.prompt, args.prompt_id, args.tier)
        print(json.dumps(result.to_dict(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
