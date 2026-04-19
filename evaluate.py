from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_task_dataloaders
from src.model import LiteCNNMoEClassifier
from src.transfer import resolve_branch_fusion_weights, resolve_fixed_branch_weights, resolve_fixed_experts
from src.utils import macro_f1_score
from src.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})
    merged_cfg = ckpt_cfg if isinstance(ckpt_cfg, dict) and ckpt_cfg else cfg

    loaders, _ = build_task_dataloaders(merged_cfg)
    device = torch.device(args.device)

    transfer_cfg = merged_cfg.get("transfer", {})
    model_cfg = merged_cfg.get("model", {}).get("moe", {})
    transfer_scheme = transfer_cfg.get("transfer_scheme", "legacy_hybrid")
    routing_mode = model_cfg.get("routing_mode", "legacy_hybrid")
    fixed_experts = ckpt.get("fixed_experts")
    if fixed_experts is None:
        fixed_experts = resolve_fixed_experts(merged_cfg)

    fixed_branch_weights = None
    beta_fixed = None
    beta_dynamic = None
    if transfer_scheme == "scheme3":
        routing_mode = "fixed_branch_dynamic_branch"
        fixed_branch_weights = resolve_fixed_branch_weights(merged_cfg, fixed_experts=fixed_experts)
        beta_fixed, beta_dynamic = resolve_branch_fusion_weights(merged_cfg, fixed_experts=fixed_experts)

    model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=merged_cfg["model"]["feature_dim"],
        num_experts=merged_cfg["model"]["moe"]["num_experts"],
        top_k=merged_cfg["model"]["moe"]["top_k"],
        fixed_experts=fixed_experts,
        hidden_dim=merged_cfg["model"]["moe"]["expert_mlp_hidden_dim"],
        router_noise_std=float(merged_cfg["model"]["moe"].get("router_noise_std", 0.0)),
        num_classes=merged_cfg["data"]["num_classes"],
        routing_mode=routing_mode,
        fixed_branch_weights=fixed_branch_weights,
        beta_fixed=beta_fixed,
        beta_dynamic=beta_dynamic,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    preds_all = []
    targets_all = []

    for batch in loaders["test"]:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_n += images.size(0)
        preds_all.extend(logits.argmax(dim=1).cpu().tolist())
        targets_all.extend(labels.cpu().tolist())

    print({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_loss": total_loss / max(1, total_n),
        "test_acc": sum(int(p == t) for p, t in zip(preds_all, targets_all)) / max(1, len(targets_all)),
        "test_macro_f1": macro_f1_score(targets_all, preds_all),
    })


if __name__ == "__main__":
    main()
