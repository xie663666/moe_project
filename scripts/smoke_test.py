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
from src.transfer import resolve_branch_fusion_weights, resolve_fixed_branch_weights, resolve_fixed_experts


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

    dyn_cfg_data = load_yaml(dyn_cfg[0])
    model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=dyn_cfg_data["model"]["feature_dim"],
        num_experts=dyn_cfg_data["model"]["moe"]["num_experts"],
        top_k=dyn_cfg_data["model"]["moe"]["top_k"],
        fixed_experts=[],
        hidden_dim=dyn_cfg_data["model"]["moe"]["expert_mlp_hidden_dim"],
        router_noise_std=float(dyn_cfg_data["model"]["moe"].get("router_noise_std", 0.0)),
        num_classes=dyn_cfg_data["data"]["num_classes"],
    )
    x = torch.randn(2, 3, 32, 32)
    logits, aux = model(x)
    assert logits.shape == (2, dyn_cfg_data["data"]["num_classes"])
    assert aux["selected_idx"].shape[1] == dyn_cfg_data["model"]["moe"]["top_k"]

    hyb_cfg_data = load_yaml(hyb_cfg[0])
    fixed_experts = resolve_fixed_experts(hyb_cfg_data)
    fixed_branch_weights = resolve_fixed_branch_weights(hyb_cfg_data, fixed_experts=fixed_experts)
    beta_fixed, beta_dynamic = resolve_branch_fusion_weights(hyb_cfg_data, fixed_experts=fixed_experts)
    scheme3_model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=hyb_cfg_data["model"]["feature_dim"],
        num_experts=hyb_cfg_data["model"]["moe"]["num_experts"],
        top_k=hyb_cfg_data["model"]["moe"]["top_k"],
        fixed_experts=fixed_experts,
        hidden_dim=hyb_cfg_data["model"]["moe"]["expert_mlp_hidden_dim"],
        router_noise_std=float(hyb_cfg_data["model"]["moe"].get("router_noise_std", 0.0)),
        num_classes=hyb_cfg_data["data"]["num_classes"],
        routing_mode="fixed_branch_dynamic_branch",
        fixed_branch_weights=fixed_branch_weights,
        beta_fixed=beta_fixed,
        beta_dynamic=beta_dynamic,
    )
    y = torch.randn(2, 3, 32, 32)
    scheme3_logits, scheme3_aux = scheme3_model(y)
    assert scheme3_logits.shape == (2, hyb_cfg_data["data"]["num_classes"])
    assert len(scheme3_aux["fixed_branch_weights"]) == len(fixed_experts)
    assert abs(float(scheme3_aux["beta_fixed"]) + float(scheme3_aux["beta_dynamic"]) - 1.0) < 1e-6
    assert "dynamic_selected_idx" in scheme3_aux

    print("Smoke test passed.")
    print(f"Project root: {root}")
    print(f"Ref configs: {len(ref_cfg)}")
    print(f"Dynamic configs: {len(dyn_cfg)}")
    print(f"Hybrid configs: {len(hyb_cfg)}")


if __name__ == "__main__":
    main()
