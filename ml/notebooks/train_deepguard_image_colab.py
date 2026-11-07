"""
DeepGuard-Image — Google Colab Training Script
================================================
Run this on Google Colab (free T4 GPU) to train the model in ~2-4 hours.

Steps to use:
1. Go to https://colab.research.google.com
2. New notebook → Runtime → Change runtime type → T4 GPU
3. Upload this file OR paste the cells below into Colab
4. Run all cells in order
5. Download ml/checkpoints/deepguard_image_lite.pth when done
6. Place it in your local ml/checkpoints/ folder
7. Copy to backend/checkpoints/ (or update IMAGE_MODEL_PATH in .env)

Expected results on CIFAKE with T4 GPU (~2 h):
  AUC-ROC  : 0.97 - 0.99
  Accuracy : 93% - 97%
  F1 Score : 0.93 - 0.97
  EER      : 0.03 - 0.07
"""

# ============================================================
# CELL 1 — Mount Drive & Clone Repo (or upload files)
# ============================================================
# from google.colab import drive
# drive.mount('/content/drive')
#
# Option A: clone your repo
# !git clone https://github.com/YOUR_USERNAME/deepguard.git
# %cd deepguard
#
# Option B: upload zip of your project
# from google.colab import files
# files.upload()  # upload deepguard.zip
# !unzip deepguard.zip

# ============================================================
# CELL 2 — Install dependencies
# ============================================================
# !pip install torch torchvision loguru scikit-learn matplotlib pillow -q

# ============================================================
# CELL 3 — Download CIFAKE dataset from Kaggle
# ============================================================
# Option A: Kaggle API (fastest)
# !pip install kaggle -q
# from google.colab import files
# files.upload()  # upload kaggle.json
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
# !unzip cifake-real-and-ai-generated-synthetic-images.zip -d data/raw/cifake/
#
# Option B: If you already have it locally, upload the zip to Drive

# ============================================================
# CELL 4 — Prepare dataset
# ============================================================
# !python scripts/prepare_cifake.py

# ============================================================
# CELL 5 — Train (FULL CIFAKE, GPU, ~2 h)
# ============================================================
# !python backend/training/train_image_lite.py \
#     --dataset_root data/processed/cifake \
#     --checkpoint_dir ml/checkpoints \
#     --epochs 30 \
#     --batch_size 128 \
#     --workers 4 \
#     --device cuda \
#     --lr 3e-4

# ============================================================
# CELL 6 — Evaluate and generate evidence
# ============================================================
# !pip install matplotlib grad-cam opencv-python-headless -q
# !python backend/training/evaluate_image.py \
#     --checkpoint ml/checkpoints/deepguard_image_lite.pth \
#     --dataset_root data/processed/cifake \
#     --output_dir ml/evaluation \
#     --device cuda

# ============================================================
# CELL 7 — Download results
# ============================================================
# import shutil
# shutil.make_archive('deepguard_results', 'zip', 'ml')
# from google.colab import files
# files.download('deepguard_results.zip')

# ============================================================
# SELF-CONTAINED VERSION (paste directly into one Colab cell)
# ============================================================

COLAB_NOTEBOOK = """
# @title DeepGuard-Image Training on Colab
# @markdown Run this cell to train the model end-to-end.

import subprocess, sys

# Install deps
subprocess.run([sys.executable, '-m', 'pip', 'install',
    'torch', 'torchvision', 'loguru', 'scikit-learn',
    'matplotlib', 'grad-cam', 'opencv-python-headless', '-q'
])

# Download CIFAKE (requires Kaggle API key)
# subprocess.run(['kaggle', 'datasets', 'download',
#     '-d', 'birdy654/cifake-real-and-ai-generated-synthetic-images'])

# This script is a guide — paste individual cells above into Colab directly.
print("See the cells above — copy them into Colab one by one.")
"""


# ──────────────────────────────────────────────────────────────────────────
# ALTERNATIVELY: Run locally overnight with a tiny subset
# Expected: ~5-6 hours on AMD Ryzen 3 3250U
# ──────────────────────────────────────────────────────────────────────────
#
# python backend/training/train_image_lite.py \
#     --dataset_root data/processed/cifake \
#     --checkpoint_dir ml/checkpoints \
#     --max_per_class 2000 \
#     --epochs 20 \
#     --batch_size 32 \
#     --workers 2
#
# Expected metrics (2000/class, 20 epochs on CPU):
#   AUC-ROC  : 0.88 - 0.93
#   Accuracy : 83% - 90%
#   F1 Score : 0.85 - 0.91
#   (lower than full dataset but proves the pipeline works)

if __name__ == "__main__":
    print(__doc__)
