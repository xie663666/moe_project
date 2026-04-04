from __future__ import annotations

from pathlib import Path
from typing import List

from src.config import load_yaml
from src.utils import load_json


def resolve_fixed_experts(cfg) -> List[int]:
    fixed_k = int(cfg["transfer"]["fixed_k"])
    if fixed_k == 0:
        return []

    stats_path = cfg["transfer"]["source_stats_path"]
    if not stats_path:
        raise ValueError("source_stats_path is required when fixed_k > 0")

    stats = load_json(stats_path)
    layer_stats = stats["layers"]["moe_0"]

    lookup = layer_stats.get("top_f_lookup", {})
    if str(fixed_k) in lookup:
        return list(lookup[str(fixed_k)])

    ranked = layer_stats["ranked_experts"]
    return list(ranked[:fixed_k])
