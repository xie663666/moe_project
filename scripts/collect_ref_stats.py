from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Dict, List

from src.utils import load_json, save_json


def build_last3_stats(per_epoch_path: str | Path, total_epochs: int | None = None):
    data = load_json(per_epoch_path)
    epochs = data["epochs"]
    selected = epochs[-3:]
    epoch_ids = [item["epoch"] for item in selected]

    layer_name = "moe_0"
    counts_per_epoch = [item["layers"][layer_name]["selection_counts"] for item in selected]
    num_experts = len(counts_per_epoch[0])

    summed = [0 for _ in range(num_experts)]
    for counts in counts_per_epoch:
        for i, value in enumerate(counts):
            summed[i] += value

    meaned = [v / len(selected) for v in summed]
    total = sum(summed)
    normalized = [v / max(1, total) for v in summed]
    ranked = sorted(range(num_experts), key=lambda i: (-summed[i], i))

    top_f_lookup = {}
    for f in [1, 2, 3, 4, 6, 8]:
        if f <= num_experts:
            top_f_lookup[str(f)] = ranked[:f]

    return {
        "run_id": data["run_id"],
        "task_name": data["task_name"],
        "mode": data["mode"],
        "seed": None,
        "num_experts": data["num_experts"],
        "top_k": data["top_k"],
        "fixed_k": data["fixed_k"],
        "stats_window_epochs": 3,
        "total_epochs": total_epochs or len(epochs),
        "window_epoch_indices": epoch_ids,
        "layers": {
            layer_name: {
                "window_sum_counts": summed,
                "window_mean_counts": meaned,
                "normalized_freq": normalized,
                "ranked_experts": ranked,
                "top_f_lookup": top_f_lookup,
            }
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    src = run_dir / "expert_stats_per_epoch.json"
    dst = run_dir / "expert_stats_last3.json"
    save_json(dst, build_last3_stats(src))
    print(f"saved: {dst}")


if __name__ == "__main__":
    main()
