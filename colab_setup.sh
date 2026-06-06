#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# colab_setup.sh — IFC-GraphRAG-DT environment setup
# Usage (first cell of every notebook): !bash ifc-graphrag-dt/colab_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

REPO="ifc-graphrag-dt"

echo "==> Cloning / updating repository..."
if [ -d "$REPO" ]; then
  cd "$REPO" && git pull --quiet && cd ..
else
  git clone https://github.com/aiwithprashant/ifc-graphrag-dt.git
fi

echo "==> Installing core package in editable mode..."
pip install -e "$REPO" --quiet

echo "==> Installing ifcopenshell..."
pip install ifcopenshell --quiet

echo "==> Installing sentence-transformers and faiss-cpu..."
pip install sentence-transformers faiss-cpu --quiet

echo "==> Installing scipy and scikit-learn (for statistical tests)..."
pip install scipy scikit-learn seaborn --quiet

echo "==> Installing pyyaml, tqdm, rich, jsonschema..."
pip install pyyaml tqdm rich jsonschema --quiet

echo ""
echo "✓ Base setup complete."
echo "  GPU packages (torch, diffusers, TripoSR) are installed only in Notebook 04."
echo "  Run: import sys; sys.path.insert(0, 'ifc-graphrag-dt'); import os; os.chdir('ifc-graphrag-dt')"
