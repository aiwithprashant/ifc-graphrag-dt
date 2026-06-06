# IFC-GraphRAG-DT

**Ontology-Grounded 3D Asset Generation for Building Digital Twins**

> PhD Research · IIT Patna · Paper 2 of 3

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper 1](https://img.shields.io/badge/SSRN-Paper%201-orange.svg)](http://ssrn.com/abstract=6879762)

---

## Overview

Standard text-to-3D pipelines exhibit an **85% performance collapse** on relational prompts
that describe multi-entity, multi-constraint scenes — the exact prompts required for Digital Twin
asset generation. This repository contains the full implementation for Paper 2, which addresses
this bottleneck through graph-structured ontology retrieval.

### Three Contributions

| ID | Contribution | Description |
|----|-------------|-------------|
| **C1** | **DTAH-Bench** | 150-prompt Digital Twin Asset Hierarchy Benchmark (Asset / Assembly / System tiers) |
| **C2** | **KCS-DT + DTAH-Eval** | DT-readiness metric + two-stage diagnostic evaluation protocol |
| **C3** | **IFC-GraphRAG-DT Pipeline** | Graph-retrieval-conditioned 3D generation grounded in IFC ontology |

### PhD Arc

```
Paper 1 (EAAI, under review) → Diagnose the bottleneck
Paper 2 (this repo)          → Solve it at asset level
Paper 3 (planned)            → Scale to scene-level DT construction
```

---

## Repository Structure

```
ifc-graphrag-dt/
├── benchmark/               # C1: DTAH-Bench
│   ├── prompts/             # 150 evaluation prompts (JSON) by tier
│   ├── ground_truth/        # IFC subgraphs, scene specs, annotations
│   ├── ifc_reference_models/# buildingSMART open IFC files
│   └── annotation_tool/     # Rubric schema + IAA computation
│
├── evaluation/              # C2: KCS-DT + DTAH-Eval + GGS
│   ├── metrics/             # kcs_dt.py · ggs.py · spec_score.py
│   ├── stage_a/             # Specification correctness evaluator
│   ├── stage_b/             # Generation fidelity evaluator
│   └── results/             # Scores, figures, statistical tests
│
├── pipeline/                # C3: IFC-GraphRAG-DT
│   ├── layer1_retriever/    # IFC graph builder + k-hop traversal
│   ├── layer2_spec_gen/     # Subgraph → structured scene spec
│   ├── layer3_generator/    # SD + TripoSR conditioned generation
│   ├── baselines/           # B0–B3 baseline implementations
│   └── configs/             # YAML configuration files
│
├── notebooks/               # Google Colab experiment notebooks
│   ├── 00_setup_and_ifc_exploration.ipynb
│   ├── 01_ifc_graph_construction.ipynb
│   ├── 02_graphrag_retriever_dev.ipynb
│   ├── 03_pipeline_dry_run.ipynb
│   ├── 04_baseline_experiments.ipynb
│   └── 05_evaluation_and_figures.ipynb
│
├── tests/                   # Unit tests
├── docs/                    # Architecture, annotation guide, Colab workflow
├── outputs/                 # Generated meshes, specs, scores (gitignored)
├── colab_setup.sh           # One-cell Colab environment setup
├── requirements.txt
└── setup.py
```

---

## Quick Start

### Local (VSCode)

```bash
git clone https://github.com/prashantsrivastava/ifc-graphrag-dt.git
cd ifc-graphrag-dt
pip install -e .
```

### Google Colab

Paste this in your first cell:

```python
!git clone https://github.com/prashantsrivastava/ifc-graphrag-dt.git
!bash ifc-graphrag-dt/colab_setup.sh
```

---

## Benchmark: DTAH-Bench

| Tier | Level | Prompts | IFC Relations |
|------|-------|---------|---------------|
| Tier 1 | Asset | 50 | Entity types only |
| Tier 2 | Assembly | 50 | IfcRelConnects, IfcRelAggregates |
| Tier 3 | System | 50 | IfcRelContainedInSpatialStructure, IfcSystem |

```python
from benchmark.dtah_bench import DTAHBench

bench = DTAHBench()
tier1 = bench.load_tier(1)          # 50 asset-level prompts
pilot = bench.load_pilot()          # DTAH-Bench-50 (15+15+20)
```

---

## Evaluation: KCS-DT

```
KCS-DT = 0.20·E + 0.35·R + 0.15·A + 0.15·Cn + 0.15·Cv
```

| Symbol | Component | Weight |
|--------|-----------|--------|
| E | Entity Correctness | 0.20 |
| R | Relation Correctness | 0.35 |
| A | Attribute Correctness | 0.15 |
| Cn | Containment Correctness | 0.15 |
| Cv | Connectivity Correctness | 0.15 |

```python
from evaluation.metrics.kcs_dt import KCSDTScorer

scorer = KCSDTScorer()
score = scorer.score(prediction=pred_spec, ground_truth=gt_spec)
print(score)  # KCSDTScore(entity=0.91, relation=0.74, ..., total=0.81)
```

---

## Pipeline

```python
from pipeline.run_pipeline import IFCGraphRAGDT

pipeline = IFCGraphRAGDT(config="pipeline/configs/pipeline_config.yaml")
result = pipeline.run(prompt="Generate a pump connected to two pipe segments")

# result.spec      → structured scene specification (JSON)
# result.mesh_path → path to generated .obj file
# result.stage_a   → Stage A score (GGS + spec correctness)
# result.stage_b   → Stage B score (KCS-DT)
```

---

## Baselines

| ID | Baseline | Script |
|----|----------|--------|
| B0 | No grounding | `pipeline/baselines/b0_no_grounding.py` |
| B1 | Prompt-only LLM | `pipeline/baselines/b1_llm_only.py` |
| B2 | Flat RAG | `pipeline/baselines/b2_flat_rag.py` |
| B3 | IFC direct lookup | `pipeline/baselines/b3_ifc_lookup.py` |
| B4 | **IFC-GraphRAG-DT** | `pipeline/run_pipeline.py` |

---

## Citation

If you use DTAH-Bench or KCS-DT in your research, please cite:

```bibtex
@article{srivastava2026ifcgraphragdt,
  title   = {IFC-GraphRAG-DT: Ontology-Grounded 3D Asset Generation
             for Building Digital Twins},
  author  = {Srivastava, Prashant and Misra, Rajiv and Verma, Amit Kumar},
  journal = {Advanced Engineering Informatics},
  year    = {2026},
  note    = {Under preparation}
}
```

**Paper 1 (published):**
```bibtex
@inproceedings{srivastava2025benchmark,
  title     = {Knowledge Grounding as the Bottleneck in Text-to-3D Generation
               for Digital Twin Deployment},
  author    = {Srivastava, Prashant and others},
  booktitle = {IEEE Big Data 2025},
  doi       = {10.1109/BigData66926.2025.11401741}
}
```

---

## License

MIT © 2026 Prashant Srivastava, IIT Patna
