# IFC-GraphRAG-DT — System Architecture

## Overview

IFC-GraphRAG-DT is a three-layer pipeline for ontology-grounded 3D asset generation for building digital twins. The system grounds text-to-3D generation in the IFC ontology using graph-structured retrieval, and evaluates outputs using the KCS-DT metric and two-stage DTAH-Eval protocol.

## Three-Layer Pipeline

```
Natural Language Prompt
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 1: IFC-GraphRAG Retriever                          │
│                                                            │
│  1a. IFCGraphBuilder                                       │
│      ifcopenshell → typed NetworkX DiGraph G=(V,E,τ)      │
│                                                            │
│  1b. GraphEmbedder                                         │
│      sentence-transformers + FAISS → seed node selection  │
│                                                            │
│  1c. KHopTraversal                                         │
│      k-hop bidirectional traversal → IFC subgraph         │
└─────────────────────┬──────────────────────────────────────┘
                      │  IFC Subgraph (V̂, Ê, τ̂)
                      ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 2: Scene Specification Generator                    │
│                                                            │
│  SceneSpecGenerator (LLM-conditioned)                      │
│  IFC subgraph → structured scene spec JSON                 │
│                                                            │
│  Spec contains: entities, relations, attributes,           │
│  containment, connectivity, constraints, SD prompt         │
│                                                            │
│  ← STAGE A evaluated here (GGS + Spec score)              │
└─────────────────────┬──────────────────────────────────────┘
                      │  Scene Specification JSON
                      ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 3: Conditioned 3D Generator                         │
│                                                            │
│  SDConditioner → SDXL multi-view images                    │
│  TripoSRRunner → triplane 3D reconstruction                │
│  MeshPostProcessor → validation + cleaning                 │
│                                                            │
│  ← STAGE B evaluated here (KCS-DT)                        │
└─────────────────────┬──────────────────────────────────────┘
                      │
                      ▼
        3D Mesh + DTAH-Eval Report (Stage A + Stage B)
```

## Graph-Theoretic Justification

Let **G = (V, E, τ)** represent the IFC ontology where V is entity types, E is typed relations, and τ maps each relation to a semantic category.

For a system-level prompt requiring k-hop relational grounding:

- **Flat RAG** recovers O(1) neighbourhood: entities only, no edges
- **GraphRAG** recovers O(k) path information: entities + edges + multi-hop paths

For Tier 3 system prompts (e.g. "pump room with pumps, headers, valves"), k ≥ 3 is required. Flat RAG is structurally insufficient for these cases.

## Evaluation Stack

| Metric | Stage | Measures |
|--------|-------|---------|
| GGS (Graph Grounding Score) | Stage A | Retrieval quality: node recall + edge recall + path recall |
| Spec Score | Stage A | Specification correctness: entity + relation + attribute |
| KCS-DT | Stage B | Generation fidelity: E + R + A + Cn + Cv |

### KCS-DT Formula
```
KCS-DT = 0.20·E + 0.35·R + 0.15·A + 0.15·Cn + 0.15·Cv
```

Relation gets the highest weight (0.35) because Paper 1 empirically identified relation precision as the dominant failure mode.

## Baselines

| ID | Baseline | Key Difference |
|----|----------|----------------|
| B0 | No Grounding | No retrieval, vanilla text-to-3D |
| B1 | Prompt-only LLM | LLM reasoning only, no retrieval |
| B2 | Flat RAG | Retrieves nodes, loses edges |
| B3 | IFC Direct Lookup | Depth-1 only, no k-hop traversal |
| B4 | IFC-GraphRAG-DT | Full pipeline (proposed) |

## Module Map

```
pipeline/
  layer1_retriever/
    ifc_graph_builder.py   ← builds G=(V,E,τ) from .ifc file
    graph_embedder.py      ← FAISS seed retrieval
    khop_traversal.py      ← k-hop subgraph expansion
  layer2_spec_gen/
    scene_spec_generator.py ← LLM spec generation
    spec_schema.json        ← JSON schema
  layer3_generator/
    sd_conditioner.py      ← SDXL multi-view generation
    triposr_runner.py      ← 3D reconstruction
    mesh_postprocess.py    ← mesh cleaning
  baselines/
    b0_no_grounding.py
    b1_llm_only.py
    b2_flat_rag.py
    b3_ifc_lookup.py
  run_pipeline.py          ← orchestrator + CLI

evaluation/
  metrics/
    kcs_dt.py              ← KCS-DT scorer
    ggs.py                 ← Graph Grounding Score
  stage_a/
    spec_evaluator.py      ← programmatic spec evaluation
  stage_b/
    mesh_evaluator.py      ← expert annotation scorer
  results/
    statistical_tests.py   ← Wilcoxon + Cohen's d + bootstrap CI
  run_eval.py              ← evaluation CLI

benchmark/
  dtah_bench.py            ← DataLoader (pilot + full)
  prompts/                 ← 150 JSON prompts (3 tiers)
  ground_truth/            ← IFC subgraph annotations
  annotation_tool/
    iaa_compute.py         ← Cohen's κ IAA computation
```
