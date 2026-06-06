# Google Colab Workflow

## Overview

All GPU-intensive work (3D generation, large embedding models) runs on Google Colab. Code is developed locally in VSCode, pushed to GitHub, and pulled into Colab for execution. Outputs are saved to Google Drive for persistence across Colab sessions.

## Setup (First Cell of Every Notebook)

```python
# Cell 1 — always run first
!git clone https://github.com/aiwithprashant/ifc-graphrag-dt.git
!bash ifc-graphrag-dt/colab_setup.sh
import sys
sys.path.insert(0, 'ifc-graphrag-dt')
```

If the repo is already cloned, update instead:
```python
import os
if os.path.exists('ifc-graphrag-dt'):
    !cd ifc-graphrag-dt && git pull
else:
    !git clone https://github.com/aiwithprashant/ifc-graphrag-dt.git
!bash ifc-graphrag-dt/colab_setup.sh
```

## Notebook Sequence

| Notebook | Purpose | GPU needed? |
|----------|---------|-------------|
| `00_setup_and_ifc_exploration.ipynb` | Verify setup, explore IFC schema | No |
| `01_ifc_graph_construction.ipynb` | Build IFC property graph, inspect nodes/edges | No |
| `02_graphrag_retriever_dev.ipynb` | Build embedder, test k-hop traversal, validate GGS | No |
| `03_pipeline_dry_run.ipynb` | End-to-end dry run (no 3D gen), Layer 1+2 only | No |
| `04_baseline_experiments.ipynb` | Run all 5 baselines on DTAH-Bench | Yes (T4/A100) |
| `05_evaluation_and_figures.ipynb` | KCS-DT scoring, statistical tests, paper figures | No |

## Output Persistence (Google Drive)

All outputs are saved to `/content/drive/MyDrive/ifc_graphrag_dt_outputs/` which persists across sessions.

```python
# At the start of experiment notebooks:
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = '/content/drive/MyDrive/ifc_graphrag_dt_outputs'

import os
os.makedirs(f'{OUTPUT_DIR}/meshes', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/specs', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/scores', exist_ok=True)
```

## Runtime Selection

- Notebooks 00–03: CPU runtime (free tier is fine)
- Notebooks 04–05: GPU runtime recommended
  - Runtime → Change runtime type → T4 GPU (free) or A100 (Colab Pro)
  - T4 (16GB VRAM): sufficient for SDXL + TripoSR at resolution 256
  - A100 (40GB VRAM): recommended for full resolution and faster throughput

## IFC Reference Models

buildingSMART open IFC reference models must be downloaded separately (too large for GitHub):

```python
# Download in Colab
!wget -O ifc-graphrag-dt/benchmark/ifc_reference_models/duplex.ifc \
    https://github.com/buildingSMART/Sample-Test-Files/raw/master/IFC%202x3/Duplex%20Apartment/Duplex_A_20110907.ifc

!wget -O ifc-graphrag-dt/benchmark/ifc_reference_models/office_building.ifc \
    https://github.com/buildingSMART/Sample-Test-Files/raw/master/IFC%204/Schependomlaan/Design%20model%20IFC4/AR-20160125-Schependomlaan.ifc
```

## VSCode ↔ GitHub ↔ Colab Workflow

```
Local VSCode                GitHub              Google Colab
────────────                ──────              ────────────
Edit code           →  git push  →  git pull (in notebook cell)
Write tests         →  push      →  !pip install -e ifc-graphrag-dt
Design notebooks    →  push      →  Open notebook, Run all
                                    ↓
                                    Outputs → Google Drive
                                    Download results locally
```

## Environment Variables

Set these in Colab before running LLM-dependent cells:

```python
import os
os.environ['ANTHROPIC_API_KEY'] = 'your-key-here'   # for Layer 2 spec gen
# OR
os.environ['OPENAI_API_KEY'] = 'your-key-here'       # if using GPT-4o
```

For repeated use, store in Colab Secrets (left sidebar → key icon).
