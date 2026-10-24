"""
create_dummy_checkpoint.py
==========================
Builds a DeepGuard model with random (untrained) weights and saves it to disk
in the exact checkpoint format that the model registry expects.

⚠️  WARNING: DUMMY CHECKPOINT — random weights, not trained.
    This is only useful for:
      • testing the end-to-end inference pipeline without real weights
      • verifying checkpoint format compatibility with the model registry
      • CI/CD smoke tests

Usage:
    python backend/utils/create_dummy_checkpoint.py --model image --output ml/checkpoints/deepguard_image.pth

    # Or let it read the output path from the YAML config:
    python backend/utils/create_dummy_checkpoint.py --model image --config ml/experiments/image_model.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root: python backend/utils/create_dummy_checkpoint.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


# ── Model builders ─────────────────────────────────────────────────────────

SUPPORTED_MODELS = ("image",)


def _build_image_model() -> torch.nn.Module:
    """Import and instantiate DeepGuardImageModel with random init weights."""
    from models.image_model import DeepGuardImageModel
    model = DeepGuardImageModel()
    model.eval()
    return model


MODEL_BUILDERS = {
    "image": _build_image_model,
}


# ── YAML helper ────────────────────────────────────────────────────────────

def _read_canonical_path_from_yaml(config_path: str) -> str | None:
    """
    Extract checkpoint.dir / checkpoint.canonical_name from the YAML config.
    Returns the full path string, or None if unreadable.
    """
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed — cannot read --config. Install with: pip install pyyaml")
        return None

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: config file not found: {config_path}")
        return None

    checkpoint_cfg = cfg.get("checkpoint", {})
    chk_dir  = checkpoint_cfg.get("dir", "ml/checkpoints")
    chk_name = checkpoint_cfg.get("canonical_name", "deepguard_image.pth")
    return str(Path(chk_dir) / chk_name)


# ── Core function ──────────────────────────────────────────────────────────

def create_dummy_checkpoint(model_type: str, output_path: str) -> None:
    """
    Build a model with random weights and save its state_dict to output_path.

    Saved format (matches registry expectations):
        {
            "model_state_dict": <OrderedDict>,  # keys start with 'spatial_branch', etc.
            "epoch": 0,
            "metrics": {"auc": 0.0, "accuracy": 0.0, "f1": 0.0, "eer": 0.5},
            "dummy": True,
        }

    Args:
        model_type:  One of SUPPORTED_MODELS (currently only "image").
        output_path: Where to write the .pth file.
    """
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}"
        )

    print("=" * 60)
    print("⚠️  WARNING: DUMMY CHECKPOINT — random weights, not trained")
    print("=" * 60)
    print(f"  model type : {model_type}")
    print(f"  output path: {output_path}")
    print()

    # Build the model on CPU (no GPU required for weight init)
    print("Building model with random weight initialisation...")
    model = MODEL_BUILDERS[model_type]()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters : {n_params:,}")

    # Verify the state_dict has the expected branch keys
    state_dict = model.state_dict()
    branch_keys = [k for k in state_dict if k.startswith("spatial_branch")]
    if not branch_keys:
        print(
            "Warning: no keys starting with 'spatial_branch' found in state_dict. "
            "The registry may not load this checkpoint correctly."
        )
    else:
        print(f"  branch key sample: {branch_keys[0]}")

    # Create parent directory if needed
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Save in the format the registry accepts
    payload = {
        "model_state_dict": state_dict,
        "epoch": 0,
        "metrics": {
            "auc":      0.0,
            "accuracy": 0.0,
            "f1":       0.0,
            "eer":      0.5,
        },
        "dummy": True,   # sentinel so callers can detect dummy checkpoints at load time
    }

    torch.save(payload, out)
    size_mb = out.stat().st_size / (1024 ** 2)
    print(f"\nSaved to: {out}  ({size_mb:.1f} MB)")
    print()
    print("You can now start the backend — the registry will load these random")
    print("weights and inference will run (results will be meaningless).")
    print()
    print("To train real weights, run:")
    print("  python backend/training/train_image.py \\")
    print("    --config ml/experiments/image_model.yaml \\")
    print("    --dataset_root ml/datasets/processed")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a dummy (randomly initialised) DeepGuard checkpoint. "
            "⚠️  NOT trained — for pipeline testing only."
        )
    )
    parser.add_argument(
        "--model",
        default="image",
        choices=list(SUPPORTED_MODELS),
        help="Which model type to create a dummy checkpoint for (default: image)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .pth path "
            "(e.g. ml/checkpoints/deepguard_image.pth). "
            "If not provided, uses --config to determine the path."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML experiment config to read checkpoint.dir / "
            "checkpoint.canonical_name from (e.g. ml/experiments/image_model.yaml). "
            "Ignored when --output is explicitly set."
        ),
    )
    args = parser.parse_args()

    # Resolve output path: explicit --output > YAML config > hardcoded default
    output = args.output
    if output is None and args.config:
        output = _read_canonical_path_from_yaml(args.config)
        if output:
            print(f"Output path resolved from config: {output}")

    if output is None:
        # Final fallback based on model type
        defaults = {"image": "ml/checkpoints/deepguard_image.pth"}
        output = defaults.get(args.model, f"ml/checkpoints/deepguard_{args.model}.pth")
        print(f"No --output or --config provided — using default: {output}")

    create_dummy_checkpoint(model_type=args.model, output_path=output)


if __name__ == "__main__":
    main()
