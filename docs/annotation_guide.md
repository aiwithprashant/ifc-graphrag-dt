# DTAH-Bench Annotation Guide

## Overview

This guide is for expert annotators scoring generated 3D assets in Stage B of DTAH-Eval. Annotators evaluate whether a generated mesh faithfully instantiates its ground-truth scene specification.

## Annotator Qualifications

- BIM professional certification (Autodesk Certified Professional, buildingSMART, or equivalent), OR
- ≥ 3 years professional IFC modelling experience

## IAA Target

Cohen's κ ≥ 0.75 on all KCS-DT sub-scores. Prompts with κ < 0.70 on any sub-score are re-annotated.

## Scoring Rubric

Each prompt receives a score for five sub-components. Use the following scale:

| Score | Label | Criterion |
|-------|-------|-----------|
| 1.00 | Fully correct | All required elements present, correctly connected, no errors |
| 0.75 | Mostly correct | ≥ 80% correct, minor omissions or geometry artefacts |
| 0.50 | Partially correct | 50–79% correct, some missing/disconnected components |
| 0.25 | Mostly incorrect | < 50% correct, major failures |
| 0.00 | Incorrect | Output does not match intent or is unusable |

## Sub-Score Definitions

**E — Entity Correctness**
Are all required IFC entity types present as distinct 3D objects? (e.g. for T2-MEP-001: pump, two pipe segments)

**R — Relation Correctness**
Are required IFC relations instantiated geometrically? (e.g. pipe endpoints touching pump ports, valve seated on pipe)

**A — Attribute Correctness**
Do material, scale, and orientation match the spec? (e.g. steel appearance, correct proportions, horizontal orientation)

**Cn — Containment Correctness**
Are entities spatially placed inside their designated containers? (e.g. pump inside pump room bounding box)

**Cv — Connectivity Correctness**
Are port-to-port and system-level connections topologically correct? (e.g. pipe-pump-pipe chain with no gaps or crossings)

## Error Taxonomy (Stage B)

| Code | Category | Definition |
|------|----------|-----------|
| EB-1 | Missing object | Required entity absent from mesh |
| EB-2 | Topology violation | Objects not connected as specified |
| EB-3 | Scale error | Proportions implausible for DT standards |
| EB-4 | Containment violation | Object outside designated spatial container |
| EB-5 | Degenerate geometry | Non-manifold, zero-area faces, open surfaces |

## Annotation File Format

Submit annotations as JSON:

```json
[
  {
    "prompt_id": "T2-MEP-001",
    "annotator_id": "A1",
    "entity": 1.0,
    "relation": 0.75,
    "attribute": 1.0,
    "containment": 1.0,
    "connectivity": 0.75,
    "failure_codes": ["EB-2:pipe_endpoint_gap"],
    "notes": "Pipe-pump connection has a small visible gap at inlet port"
  }
]
```

## Calibration Protocol

Before scoring the full set, all annotators score the same 10 calibration prompts independently. IAA is computed. If κ < 0.70 on any sub-score, the rubric is reviewed and a consensus meeting is held before proceeding.
