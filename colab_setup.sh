#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# colab_setup.sh
# Run this script in the first cell of every Colab notebook:
#   !bash colab_setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "==> Cloning / updating ifc-graphrag-dt repository..."
if [ -d "ifc-graphrag-dt" ]; then
  cd ifc-graphrag-dt && git pull && cd ..
else
  git clone https://github.com/prashantsrivastava/ifc-graphrag-dt.git
fi

echo "==> Installing Python package in editable mode..."
pip install -e ifc-graphrag-dt --quiet

echo "==> Installing ifcopenshell (Colab-compatible wheel)..."
pip install ifcopenshell --quiet

echo "==> Installing GPU-aware PyTorch (Colab provides CUDA automatically)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

echo "==> Installing TripoSR..."
pip install git+https://github.com/VAST-AI-Research/TripoSR.git --quiet

echo "==> Installing FAISS GPU build..."
pip install faiss-gpu --quiet

echo "==> Mounting Google Drive for output persistence..."
python3 - << 'PYEOF'
try:
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.makedirs('/content/drive/MyDrive/ifc_graphrag_dt_outputs', exist_ok=True)
    print("Google Drive mounted. Outputs → /content/drive/MyDrive/ifc_graphrag_dt_outputs/")
except ImportError:
    print("Not running in Colab — Drive mount skipped.")
PYEOF

echo ""
echo "✓ Setup complete. You can now import from 'benchmark', 'pipeline', and 'evaluation'."
