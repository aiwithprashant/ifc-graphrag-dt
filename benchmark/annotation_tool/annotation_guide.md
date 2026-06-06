# Annotation Tool Guide

See `docs/annotation_guide.md` for the full annotator protocol.

## Quick Reference

Annotation JSON location: `benchmark/ground_truth/annotations/`
IAA computation: `python -m benchmark.annotation_tool.iaa_compute`

## IAA Computation

```python
from benchmark.annotation_tool.iaa_compute import compute_iaa

report = compute_iaa(
    annotations_path="benchmark/ground_truth/annotations/batch1_annotations.json",
    output_path="outputs/results/iaa_report.json"
)
print(f"Overall κ = {report['overall_kappa']}")
print(f"Target met: {report['target_met']}")
```
