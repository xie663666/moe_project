from __future__ import annotations

import random
from pathlib import Path
from typing import List

import torch

from src.utils import load_json


def resolve_fixed_experts(cfg) -> List[int]:
    fixed_k = int(cfg["transfer"]["fixed_k"])
    if fixed_k == 0:
        return []

    num_experts = int(cfg["model"]["moe"]["num_experts"])
    if fixed_k > num_experts:
        raise ValueError(f"fixed_k={fixed_k} cannot exceed num_experts={num_experts}")

    rule = cfg["transfer"].get("fixed_selection_rule", "source_topF_last3")
    if rule == "random":
        seed = int(cfg["experiment"]["seed"])
        rng = random.Random(seed + 17 * num_experts + 31 * fixed_k)
        return sorted(rng.sample(list(range(num_experts)), fixed_k))

    if rule == "manual":
        manual = list(cfg["transfer"].get("fixed_expert_ids", []))
        if len(manual) != fixed_k:
            raise ValueError(f"manual fixed_expert_ids length={len(manual)} must equal fixed_k={fixed_k}")
        if len(set(manual)) != len(manual):
            raise ValueError("manual fixed_expert_ids contains duplicates")
        if any(idx < 0 or idx >= num_experts for idx in manual):
            raise ValueError(f"manual fixed_expert_ids must be in [0, {num_experts - 1}]")
        return sorted(manual)

    if rule != "source_topF_last3":
        raise ValueError(f"Unsupported fixed_selection_rule: {rule}")

    stats_path = cfg["transfer"].get("source_stats_path", "")
    if not stats_path:
        raise ValueError("source_stats_path is required for source_topF_last3")
    stats = load_json(stats_path)
    layer_stats = stats["layers"]["moe_0"]
    lookup = layer_stats.get("top_f_lookup", {})
    if str(fixed_k) in lookup:
        return list(lookup[str(fixed_k)])
    ranked = layer_stats["ranked_experts"]
    return list(ranked[:fixed_k])


def resolve_fixed_branch_weights(cfg) -> List[float]:
    fixed_experts = resolve_fixed_experts(cfg)
    if not fixed_experts:
        return []

    stats_path = cfg["transfer"].get("source_stats_path", "")
    if not stats_path:
        raise ValueError("source_stats_path is required when fixed_k > 0 for scheme3")

    stats = load_json(stats_path)
    try:
        window_sum_counts = stats["layers"]["moe_0"]["window_sum_counts"]
    except KeyError as exc:
        raise KeyError("source stats missing layers.moe_0.window_sum_counts") from exc

    counts = [float(window_sum_counts[idx]) for idx in fixed_experts]
    total = sum(counts)
    if total <= 0:
        raise ValueError("sum of fixed experts window_sum_counts is 0; cannot normalize fixed_branch_weights")
    return [c / total for c in counts]


def maybe_load_source_expert_weights(model, cfg, fixed_experts: List[int], project_root: Path):
    transfer_cfg = cfg.get("transfer", {})
    if not transfer_cfg.get("reuse_source_expert_weights", False):
        return None
    if not fixed_experts:
        return None

    source_ckpt_path = transfer_cfg.get("source_checkpoint_path", "")
    if source_ckpt_path:
        ckpt_path = Path(source_ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = project_root / ckpt_path
    else:
        source_ref_run_id = transfer_cfg.get("source_ref_run_id")
        if not source_ref_run_id:
            raise ValueError("source_ref_run_id is required when source_checkpoint_path is not set")
        ckpt_path = project_root / cfg["runtime"]["checkpoint_dir"] / source_ref_run_id / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {ckpt_path}")

    source_ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = source_ckpt["model"]
    target_state = model.state_dict()
    copied = []
    for idx in fixed_experts:
        prefix = f"moe.experts.{idx}."
        keys = [k for k in target_state.keys() if k.startswith(prefix)]
        for key in keys:
            if key not in source_state:
                raise KeyError(f"{key} not found in source checkpoint: {ckpt_path}")
            target_state[key] = source_state[key]
        copied.append(idx)
    model.load_state_dict(target_state, strict=True)
    return str(ckpt_path), copied
