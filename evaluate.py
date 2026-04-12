from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_task_dataloaders
from src.model import LiteCNNMoEClassifier
from src.utils import accuracy_from_logits, macro_f1_score
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

    loaders, _ = build_task_dataloaders(cfg)
    device = torch.device(args.device)

    model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=cfg["model"]["feature_dim"],
        num_experts=cfg["model"]["moe"]["num_experts"],
        top_k=cfg["model"]["moe"]["top_k"],
        fixed_experts=ckpt.get("fixed_experts", []),
        hidden_dim=cfg["model"]["moe"]["expert_mlp_hidden_dim"],
        router_noise_std=float(cfg["model"]["moe"].get("router_noise_std", 0.0)),
        num_classes=cfg["data"]["num_classes"],
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
        "test_acc": accuracy_from_logits(torch.tensor([[0.0]]), torch.tensor([0])) if False else sum(int(p == t) for p, t in zip(preds_all, targets_all)) / max(1, len(targets_all)),
        "test_macro_f1": macro_f1_score(targets_all, preds_all),
    })


if __name__ == "__main__":
    main()
