from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import torch

from src.config import load_yaml
from src.model import LiteCNNMoEClassifier
from src.transfer import resolve_fixed_experts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    return parser.parse_args()


def main():
    root = Path(parse_args().project_root).resolve()

    required = [
        root / "configs",
        root / "scripts",
        root / "src",
        root / "results",
        root / "logs",
        root / "checkpoints",
    ]
    for path in required:
        path.mkdir(parents=True, exist_ok=True)
        assert path.exists(), f"missing {path}"

    pairs_cfg = root / "configs" / "pairs" / "directed_main_pairs.yaml"
    if not pairs_cfg.exists():
        raise FileNotFoundError(f"directed pair config not found: {pairs_cfg}")

    ref_cfg = sorted((root / "configs" / "generated" / "main_round" / "ref").glob("*.yaml"))
    dyn_cfg = sorted((root / "configs" / "generated" / "main_round" / "dynamic").glob("*.yaml"))
    hyb_cfg = sorted((root / "configs" / "generated" / "main_round" / "hybrid").glob("*.yaml"))
    assert ref_cfg, "no generated ref configs"
    assert dyn_cfg, "no generated dynamic configs"
    assert hyb_cfg, "no generated hybrid configs"

    cfg = load_yaml(dyn_cfg[0])
    model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=cfg["model"]["feature_dim"],
        num_experts=cfg["model"]["moe"]["num_experts"],
        top_k=cfg["model"]["moe"]["top_k"],
        fixed_experts=[],
        hidden_dim=cfg["model"]["moe"]["expert_mlp_hidden_dim"],
        num_classes=cfg["data"]["num_classes"],
    )
    x = torch.randn(2, 3, 32, 32)
    logits, aux = model(x)
    assert logits.shape == (2, cfg["data"]["num_classes"])
    assert aux["selected_idx"].shape[1] == cfg["model"]["moe"]["top_k"]

    print("Smoke test passed.")
    print(f"Project root: {root}")
    print(f"Ref configs: {len(ref_cfg)}")
    print(f"Dynamic configs: {len(dyn_cfg)}")
    print(f"Hybrid configs: {len(hyb_cfg)}")


if __name__ == "__main__":
    main()
