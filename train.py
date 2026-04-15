from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import load_yaml
from src.data import build_task_dataloaders
from src.model import LiteCNNMoEClassifier
from src.transfer import maybe_load_source_expert_weights, resolve_fixed_experts
from src.utils import (
    AverageMeter,
    accuracy_from_logits,
    ensure_dir,
    macro_f1_score,
    save_json,
    save_yaml,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, load_balance_coef: float = 0.0):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    f1_targets = []
    f1_preds = []
    router_entropy_values = []
    load_balance_values = []
    timing_forward = 0.0
    timing_backward = 0.0
    timing_optimizer = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        t_fwd0 = time.perf_counter()
        logits, aux = model(images, track_usage=True)
        ce_loss = criterion(logits, labels)
        loss = ce_loss + float(load_balance_coef) * aux["load_balance_loss"]
        t_fwd1 = time.perf_counter()
        loss.backward()
        t_bwd1 = time.perf_counter()
        optimizer.step()
        t_opt1 = time.perf_counter()
        timing_forward += t_fwd1 - t_fwd0
        timing_backward += t_bwd1 - t_fwd1
        timing_optimizer += t_opt1 - t_bwd1

        acc = accuracy_from_logits(logits, labels)
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        f1_preds.extend(preds)
        f1_targets.extend(labels.detach().cpu().tolist())
        router_entropy_values.append(aux["router_entropy"].item())
        load_balance_values.append(aux["load_balance_loss"].item())

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "macro_f1": macro_f1_score(f1_targets, f1_preds),
        "routing_entropy": sum(router_entropy_values) / max(1, len(router_entropy_values)),
        "load_balance_loss": sum(load_balance_values) / max(1, len(load_balance_values)),
        "timing_breakdown_sec": {
            "forward": timing_forward,
            "backward": timing_backward,
            "optimizer": timing_optimizer,
        },
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, stage_name="eval"):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    f1_targets = []
    f1_preds = []
    router_entropy_values = []
    load_balance_values = []

    for batch in tqdm(loader, desc=stage_name, leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits, aux = model(images, track_usage=False)
        loss = criterion(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        preds = logits.argmax(dim=1).detach().cpu().tolist()
        f1_preds.extend(preds)
        f1_targets.extend(labels.detach().cpu().tolist())
        router_entropy_values.append(aux["router_entropy"].item())
        load_balance_values.append(aux["load_balance_loss"].item())

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
        "macro_f1": macro_f1_score(f1_targets, f1_preds),
        "routing_entropy": sum(router_entropy_values) / max(1, len(router_entropy_values)),
        "load_balance_loss": sum(load_balance_values) / max(1, len(load_balance_values)),
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_start_ts = time.perf_counter()
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    project_root = Path(cfg["runtime"]["project_root"]).resolve()
    save_root = project_root / cfg["runtime"]["save_dir"]
    run_dir = save_root / cfg["experiment"]["run_id"]
    ckpt_dir = project_root / cfg["runtime"]["checkpoint_dir"] / cfg["experiment"]["run_id"]
    log_dir = project_root / cfg["runtime"]["log_dir"]
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    save_yaml(run_dir / "config_snapshot.yaml", cfg)

    loaders, dataset_meta = build_task_dataloaders(cfg)
    device = torch.device(args.device)

    fixed_experts = resolve_fixed_experts(cfg)
    model = LiteCNNMoEClassifier(
        in_channels=3,
        feature_dim=cfg["model"]["feature_dim"],
        num_experts=cfg["model"]["moe"]["num_experts"],
        top_k=cfg["model"]["moe"]["top_k"],
        fixed_experts=fixed_experts,
        hidden_dim=cfg["model"]["moe"]["expert_mlp_hidden_dim"],
        router_noise_std=float(cfg["model"]["moe"].get("router_noise_std", 0.0)),
        num_classes=cfg["data"]["num_classes"],
    ).to(device)
    source_ckpt_loaded = None
    source_copied_experts: List[int] = []
    loaded = maybe_load_source_expert_weights(model, cfg, fixed_experts, project_root)
    if loaded is not None:
        source_ckpt_loaded, source_copied_experts = loaded

    reuse_source = bool(cfg["transfer"].get("reuse_source_expert_weights", False))
    freeze_fixed = bool(cfg["transfer"].get("freeze_fixed_experts", False))
    accelerate_fixed = bool(cfg["transfer"].get("accelerate_fixed_experts", False))
    if freeze_fixed and fixed_experts and not reuse_source:
        raise ValueError("freeze_fixed_experts=True requires reuse_source_expert_weights=True for direct source reuse experiments")
    if accelerate_fixed and not freeze_fixed:
        accelerate_fixed = False

    if freeze_fixed and fixed_experts:
        for idx in fixed_experts:
            for param in model.moe.experts[idx].parameters():
                param.requires_grad = False
    model.moe.set_frozen_experts(
        fixed_experts if freeze_fixed else [],
        no_grad_mode=accelerate_fixed,
    )

    criterion = nn.CrossEntropyLoss()
    load_balance_coef = float(cfg["train"].get("load_balance_coef", 0.0))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["train"]["optimizer"]["lr"]),
        weight_decay=float(cfg["train"]["optimizer"]["weight_decay"]),
    )

    history: List[Dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = ckpt_dir / "best.pt"

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, criterion, device, load_balance_coef=load_balance_coef)
        t1 = time.perf_counter()
        val_metrics = evaluate(model, loaders["val"], criterion, device, stage_name="val")
        t2 = time.perf_counter()
        test_metrics = evaluate(model, loaders["test"], criterion, device, stage_name="test")
        t3 = time.perf_counter()
        expert_usage = model.consume_epoch_usage()
        selection_counts = expert_usage["moe_0"]["selection_counts"]
        ranked_selected = sorted(range(len(selection_counts)), key=lambda i: (-selection_counts[i], i))

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "timing_sec": {
                "train": t1 - t0,
                "val": t2 - t1,
                "test": t3 - t2,
                "epoch_total": t3 - t0,
            },
            "expert_usage": expert_usage,
            "top_selected_experts": ranked_selected[: min(8, len(ranked_selected))],
        }
        history.append(epoch_record)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": cfg,
                    "fixed_experts": fixed_experts,
                },
                best_ckpt_path,
            )

        print(json.dumps({
            "epoch": epoch,
            "train_acc": round(train_metrics["acc"], 4),
            "val_acc": round(val_metrics["acc"], 4),
            "test_acc": round(test_metrics["acc"], 4),
            "routing_entropy": round(test_metrics["routing_entropy"], 4),
            "load_balance": round(train_metrics["load_balance_loss"], 6),
        }, ensure_ascii=False))

    save_json(run_dir / "metrics_history.json", history)

    expert_stats_per_epoch = {
        "run_id": cfg["experiment"]["run_id"],
        "mode": cfg["experiment"]["mode"],
        "task_name": cfg["data"]["task_name"],
        "seed": seed,
        "num_experts": cfg["model"]["moe"]["num_experts"],
        "top_k": cfg["model"]["moe"]["top_k"],
        "fixed_k": cfg["transfer"]["fixed_k"],
        "dynamic_k": cfg["transfer"]["dynamic_k"],
        "epochs": [
            {
                "epoch": rec["epoch"],
                "layers": rec["expert_usage"],
            }
            for rec in history
        ],
    }
    save_json(run_dir / "expert_stats_per_epoch.json", expert_stats_per_epoch)

    if cfg["experiment"]["mode"] == "ref_dynamic":
        from scripts.collect_ref_stats import build_last3_stats
        save_json(
            run_dir / "expert_stats_last3.json",
            build_last3_stats(run_dir / "expert_stats_per_epoch.json", total_epochs=int(cfg["train"]["epochs"])),
        )

    best_test_record = next(rec for rec in history if rec["epoch"] == best_epoch)
    final_test_record = history[-1]
    summary = {
        "run_id": cfg["experiment"]["run_id"],
        "mode": cfg["experiment"]["mode"],
        "pair_group": cfg["experiment"].get("pair_group"),
        "pair_id": cfg["experiment"].get("pair_id"),
        "seed": seed,
        "source_task": cfg["transfer"].get("source_task"),
        "target_task": cfg["transfer"].get("target_task"),
        "task_name": cfg["data"]["task_name"],
        "num_classes": cfg["data"]["num_classes"],
        "num_experts": cfg["model"]["moe"]["num_experts"],
        "top_k": cfg["model"]["moe"]["top_k"],
        "fixed_k": cfg["transfer"]["fixed_k"],
        "dynamic_k": cfg["transfer"]["dynamic_k"],
        "fixed_ratio": cfg["transfer"]["fixed_k"] / max(1, cfg["model"]["moe"]["top_k"]),
        "router_noise_std": float(cfg["model"]["moe"].get("router_noise_std", 0.0)),
        "load_balance_coef": load_balance_coef,
        "source_ref_run_id": cfg["transfer"].get("source_ref_run_id"),
        "source_stats_path": cfg["transfer"].get("source_stats_path"),
        "fixed_selection_rule": cfg["transfer"].get("fixed_selection_rule"),
        "reuse_source_expert_weights": reuse_source,
        "freeze_fixed_experts": freeze_fixed,
        "accelerate_fixed_experts": accelerate_fixed,
        "source_checkpoint_path": source_ckpt_loaded,
        "source_copied_experts": source_copied_experts,
        "fixed_experts": fixed_experts,
        "frozen_expert_count": len(fixed_experts) if freeze_fixed else 0,
        "total_parameter_count": total_param_count,
        "trainable_parameter_count": trainable_param_count,
        "best_val_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_record["test"]["acc"],
        "best_test_macro_f1": best_test_record["test"]["macro_f1"],
        "best_test_loss": best_test_record["test"]["loss"],
        "best_test_routing_entropy": best_test_record["test"]["routing_entropy"],
        "final_test_acc": final_test_record["test"]["acc"],
        "final_test_macro_f1": final_test_record["test"]["macro_f1"],
        "final_test_loss": final_test_record["test"]["loss"],
        "final_test_routing_entropy": final_test_record["test"]["routing_entropy"],
        "train_size": dataset_meta["train_size"],
        "val_size": dataset_meta["val_size"],
        "test_size": dataset_meta["test_size"],
        "checkpoint_path": str(best_ckpt_path),
        "config_path": str(run_dir / "config_snapshot.yaml"),
        "run_dir": str(run_dir),
        "log_path": str(log_dir / f"{cfg['experiment']['run_id']}.log"),
        "total_runtime_sec": time.perf_counter() - run_start_ts,
    }
    save_json(run_dir / "run_summary.json", summary)

    resolved = {
        "pair_id": cfg["experiment"].get("pair_id"),
        "source_task": cfg["transfer"].get("source_task"),
        "target_task": cfg["transfer"].get("target_task"),
        "source_ref_run_id": cfg["transfer"].get("source_ref_run_id"),
        "source_stats_path": cfg["transfer"].get("source_stats_path"),
        "fixed_selection_rule": cfg["transfer"].get("fixed_selection_rule"),
        "reuse_source_expert_weights": reuse_source,
        "freeze_fixed_experts": freeze_fixed,
        "accelerate_fixed_experts": accelerate_fixed,
        "source_checkpoint_path": source_ckpt_loaded,
        "source_copied_experts": source_copied_experts,
        "num_experts": cfg["model"]["moe"]["num_experts"],
        "top_k": cfg["model"]["moe"]["top_k"],
        "fixed_k": cfg["transfer"]["fixed_k"],
        "dynamic_k": cfg["transfer"]["dynamic_k"],
        "layers": {"moe_0": {"fixed_experts": fixed_experts}},
    }
    save_json(run_dir / "resolved_fixed_experts.json", resolved)

    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
