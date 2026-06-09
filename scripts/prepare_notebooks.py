"""Normalize notebooks for both local VS Code and Google Colab execution."""

from __future__ import annotations

from pathlib import Path

import nbformat


COMMON_SETUP = """\
import os, sys
from pathlib import Path

try:
    import google.colab  # type: ignore
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if not Path("pipeline").exists():
    if Path("../pipeline").exists():
        os.chdir("..")
    elif Path("ifc-graphrag-dt/pipeline").exists():
        os.chdir("ifc-graphrag-dt")
    elif IS_COLAB:
        !git clone https://github.com/aiwithprashant/ifc-graphrag-dt.git
        !bash ifc-graphrag-dt/colab_setup.sh
        os.chdir("ifc-graphrag-dt")
    else:
        raise RuntimeError("Run this notebook from the ifc-graphrag-dt repository root.")

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

print(f"Working directory: {os.getcwd()}")
"""


def replace_prefix(source: str, anchor: str, prefix: str = COMMON_SETUP) -> str:
    _, separator, suffix = source.partition(anchor)
    if not separator:
        raise ValueError(f"Notebook setup anchor not found: {anchor}")
    return prefix + "\n" + anchor + suffix


def first_code_cell(notebook):
    return next(cell for cell in notebook.cells if cell.cell_type == "code")


def code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def main() -> None:
    notebook_dir = Path("notebooks")
    notebooks = {
        path.name: nbformat.read(path, as_version=4)
        for path in sorted(notebook_dir.glob("*.ipynb"))
    }

    notebooks["00_setup_and_ifc_exploration.ipynb"].cells[1].source = (
        COMMON_SETUP + '\nprint("Setup complete.")'
    )

    for name in [
        "01_ifc_graph_construction.ipynb",
        "02_graphrag_retriever_dev.ipynb",
    ]:
        cell = first_code_cell(notebooks[name])
        cell.source = replace_prefix(cell.source, "import networkx as nx")

    notebook = notebooks["03_pipeline_dry_run.ipynb"]
    cell = first_code_cell(notebook)
    cell.source = replace_prefix(
        cell.source,
        "import json\nfrom pathlib import Path",
        COMMON_SETUP
        + '\nSPEC_PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "deterministic"\n'
        + 'print(f"Layer 2 provider: {SPEC_PROVIDER}")\n',
    )
    code_cells(notebook)[2].source = code_cells(notebook)[2].source.replace(
        "'llm_provider': 'anthropic'", "'llm_provider': SPEC_PROVIDER"
    )

    notebook = notebooks["04_baseline_experiments.ipynb"]
    first_code_cell(notebook).source = COMMON_SETUP + """
import json, time
import torch
from pathlib import Path
import yaml
import numpy as np
from benchmark.dtah_bench import DTAHBench
from pipeline.layer1_retriever.ifc_graph_builder import IFCGraphBuilder
from pipeline.layer1_retriever.graph_embedder import GraphEmbedder
from pipeline.layer1_retriever.khop_traversal import KHopTraversal
from pipeline.layer2_spec_gen.scene_spec_generator import SceneSpecGenerator
from pipeline.baselines.b0_no_grounding import B0NoGrounding
from pipeline.baselines.b1_llm_only import B1LLMOnly
from pipeline.baselines.b2_flat_rag import B2FlatRAG
from pipeline.baselines.b3_ifc_lookup import B3IFCLookup
from evaluation.metrics.kcs_dt import KCSDTScorer

DRIVE_OUT = "outputs"
SPEC_PROVIDER = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "deterministic"
IFC_PATH = "benchmark/ifc_reference_models/duplex.ifc"
GRAPH_CACHE = "outputs/graphs/ifc_graph.json"
EMB_CACHE = "outputs/embedders/graph_embedder"
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Layer 2 provider: {SPEC_PROVIDER}")

for directory in ["outputs/scores", "outputs/specs", "outputs/meshes", "outputs/figures"]:
    os.makedirs(directory, exist_ok=True)
print("All imports OK")
"""
    cells = code_cells(notebook)
    cells[2].source = cells[2].source.replace(
        "'llm_provider': 'anthropic'", "'llm_provider': SPEC_PROVIDER"
    )
    cells[4].source = cells[4].source.replace(
        "dry_run=not HAS_GPU   # skip 3D generation on CPU",
        "dry_run=not HAS_GPU,  # skip 3D generation on CPU\n"
        "    offline=(SPEC_PROVIDER == 'deterministic')",
    )

    notebook = notebooks["05_evaluation_and_figures.ipynb"]
    cell = first_code_cell(notebook)
    cell.source = replace_prefix(
        cell.source,
        "import numpy as np",
        COMMON_SETUP + "\nimport json\n",
    )
    score_cell = code_cells(notebook)[6]
    if "r['score_type'] = 'proxy'" not in score_cell.source:
        score_cell.source = score_cell.source.replace(
            "            r['total'] = scores_by_id.get(r.get('prompt_id',''), 0.0)",
            "            r['total'] = scores_by_id.get(r.get('prompt_id',''), 0.0)\n"
            "            r['score_type'] = 'proxy'",
        )

    notebook = notebooks["06_paper_artifacts.ipynb"]
    first_code_cell(notebook).source = COMMON_SETUP + """
import json, numpy as np
from pathlib import Path

for directory in ["outputs/figures/paper", "outputs/tables", "outputs/results"]:
    os.makedirs(directory, exist_ok=True)
print("Paper artifact directories ready.")
"""

    download_cell = """\
from pipeline.ifc_assets import ensure_duplex_ifc

DUPLEX_PATH = ensure_duplex_ifc()
IFC_PATH = str(DUPLEX_PATH)
print(f"IFC file ready: {DUPLEX_PATH} ({DUPLEX_PATH.stat().st_size / 1024:.1f} KB)")
"""
    code_cells(notebooks["00_setup_and_ifc_exploration.ipynb"])[4].source = download_cell
    for name, index in [
        ("01_ifc_graph_construction.ipynb", 1),
        ("02_graphrag_retriever_dev.ipynb", 1),
        ("03_pipeline_dry_run.ipynb", 1),
        ("04_baseline_experiments.ipynb", 1),
    ]:
        cell = code_cells(notebooks[name])[index]
        marker = "if Path(GRAPH_CACHE).exists():"
        if marker in cell.source:
            suffix = marker + cell.source.partition(marker)[2]
            cell.source = download_cell + "\n" + suffix
        else:
            cell.source = download_cell

    for name, notebook in notebooks.items():
        path = notebook_dir / name
        nbformat.write(notebook, path)
        print(f"Updated {path}")


if __name__ == "__main__":
    main()
