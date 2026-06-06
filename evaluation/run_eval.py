"""
run_eval.py
DTAH-Eval Runner — evaluates pipeline outputs against DTAH-Bench ground truth.

Usage:
  dtah-eval --scores outputs/scores/ --ground-truth benchmark/ground_truth/ --output outputs/results/
  dtah-eval --stage-a --predictions outputs/specs/ --tier 2
  dtah-eval --stats outputs/scores/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_stage_a(predictions_dir: Path, ground_truth_dir: Path, output_dir: Path) -> None:
    """Run Stage A (specification correctness) evaluation over a directory of specs."""
    from evaluation.stage_a.spec_evaluator import StageAEvaluator
    from benchmark.dtah_bench import DTAHBench

    evaluator = StageAEvaluator()
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_files = sorted(predictions_dir.glob("*_spec.json"))
    if not spec_files:
        logger.warning("No *_spec.json files found in %s", predictions_dir)
        return

    scores = []
    for spec_file in spec_files:
        prompt_id = spec_file.stem.replace("_spec", "")
        gt_path = ground_truth_dir / "scene_specs" / f"{prompt_id}_gt_spec.json"

        if not gt_path.exists():
            logger.warning("No ground truth for %s — skipping", prompt_id)
            continue

        with open(spec_file) as f:
            pred = json.load(f)
        with open(gt_path) as f:
            gt = json.load(f)

        score = evaluator.evaluate(pred, gt, prompt_id=prompt_id)
        scores.append(score.to_dict())
        logger.info("[%s] Stage A: %.3f", prompt_id, score.total)

    out_path = output_dir / "stage_a_scores.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)

    if scores:
        mean_total = sum(s["total"] for s in scores) / len(scores)
        print(f"\nStage A complete: {len(scores)} prompts evaluated")
        print(f"Mean Stage A score: {mean_total:.4f}")
        print(f"Results saved: {out_path}")


def run_stage_b(annotations_dir: Path, output_dir: Path) -> None:
    """Run Stage B (generation fidelity) from expert annotation files."""
    from evaluation.stage_b.mesh_evaluator import StageBEvaluator

    evaluator = StageBEvaluator()
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_dir.glob("*_annotation.json"))
    if not annotation_files:
        logger.warning("No annotation files found in %s", annotations_dir)
        return

    all_scores = []
    for ann_file in annotation_files:
        stage_b_scores = evaluator.load_and_score(ann_file)
        all_scores.extend([s.to_dict() for s in stage_b_scores])

    out_path = output_dir / "stage_b_scores.json"
    with open(out_path, "w") as f:
        json.dump(all_scores, f, indent=2)

    if all_scores:
        mean = sum(s["total"] for s in all_scores) / len(all_scores)
        print(f"\nStage B complete: {len(all_scores)} annotations scored")
        print(f"Mean KCS-DT (Stage B): {mean:.4f}")
        print(f"Results saved: {out_path}")


def run_statistics(scores_dir: Path, output_dir: Path) -> None:
    """Run full statistical analysis across all baselines."""
    from evaluation.results.statistical_tests import run_full_analysis

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "statistical_report.json"
    report = run_full_analysis(scores_dir, output_path=out_path)

    print("\nStatistical Analysis Report")
    print("=" * 50)
    primary = report.get("primary_comparison", {})
    if primary:
        print(f"\nPrimary comparison: B4 (GraphRAG) vs B2 (Flat RAG) at Tier 3")
        wlx = primary.get("wilcoxon", {})
        eff = primary.get("effect_size", {})
        print(f"  Wilcoxon p-value: {wlx.get('p_value', 'N/A'):.4f} "
              f"({'significant' if wlx.get('significant') else 'not significant'})")
        print(f"  Cohen's d: {eff.get('cohen_d', 'N/A')} ({eff.get('magnitude', 'N/A')} effect)")
        b4_ci = primary.get("b4_ci", {})
        b2_ci = primary.get("b2_ci", {})
        print(f"  B4 mean KCS-DT: {b4_ci.get('mean', 'N/A')} "
              f"[{b4_ci.get('ci_lower', '')}, {b4_ci.get('ci_upper', '')}]")
        print(f"  B2 mean KCS-DT: {b2_ci.get('mean', 'N/A')} "
              f"[{b2_ci.get('ci_lower', '')}, {b2_ci.get('ci_upper', '')}]")
    print(f"\nFull report saved: {out_path}")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description="DTAH-Eval Runner")
    parser.add_argument("--stage-a", action="store_true", help="Run Stage A evaluation")
    parser.add_argument("--stage-b", action="store_true", help="Run Stage B evaluation")
    parser.add_argument("--stats",   action="store_true", help="Run statistical analysis")
    parser.add_argument("--predictions",    default="outputs/specs",
                        help="Directory of predicted spec JSON files")
    parser.add_argument("--ground-truth",   default="benchmark/ground_truth",
                        help="Ground truth directory")
    parser.add_argument("--annotations",    default="benchmark/ground_truth/annotations",
                        help="Expert annotation JSON files directory")
    parser.add_argument("--scores",         default="outputs/scores",
                        help="Scores directory (for --stats)")
    parser.add_argument("--output",         default="outputs/results",
                        help="Output directory for results")
    args = parser.parse_args()

    if args.stage_a:
        run_stage_a(Path(args.predictions), Path(args.ground_truth), Path(args.output))
    if args.stage_b:
        run_stage_b(Path(args.annotations), Path(args.output))
    if args.stats:
        run_statistics(Path(args.scores), Path(args.output))
    if not any([args.stage_a, args.stage_b, args.stats]):
        parser.print_help()


if __name__ == "__main__":
    main()
