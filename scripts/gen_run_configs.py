from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import load_yaml, save_yaml

LEGAL_COMBOS = [
    (8, 4, [1, 2, 3, 4]),
    (16, 4, [1, 2, 3, 4]),
    (16, 6, [1, 2, 4, 6]),
    (16, 8, [1, 2, 4, 6, 8]),
    (32, 4, [1, 2, 3, 4]),
    (32, 6, [1, 2, 4, 6]),
    (32, 8, [1, 2, 4, 6, 8]),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=str, default="main")
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    return parser.parse_args()


def build_base_runtime(project_root="."):
    return {
        "project_root": project_root,
        "save_dir": "results/runs",
        "checkpoint_dir": "checkpoints",
        "log_dir": "logs",
        "num_workers": 2,
    }


def build_train_cfg():
    return {
        "epochs": 10,
        "batch_size": 128,
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
    }


def build_model_cfg(E: int, K: int):
    return {
        "feature_dim": 256,
        "moe": {
            "num_experts": E,
            "top_k": K,
            "expert_mlp_hidden_dim": 512,
        },
    }


def build_data_cfg(task_name: str):
    return {
        "dataset_name": "cifar100_superclass_task",
        "root": "./datasets",
        "task_name": task_name,
        "num_classes": 5,
        "val_ratio": 0.1,
    }


def unique_ref_pairs(directed_pairs):
    refs = {}
    for item in directed_pairs:
        refs[item["source_task"]] = item["pair_group"]
    return refs


def write_config(path: Path, cfg: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(path, cfg)


def main():
    args = parse_args()
    pairs = load_yaml(args.pairs)["directed_pairs"]
    out_root = Path(args.output)

    ref_dir = out_root / "ref"
    dyn_dir = out_root / "dynamic"
    hyb_dir = out_root / "hybrid"

    for p in [ref_dir, dyn_dir, hyb_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Reference runs for each source task and each (E,K)
    source_tasks = sorted({item["source_task"] for item in pairs})
    for task_name in source_tasks:
        for seed in args.seeds:
            for E, K, _f_list in LEGAL_COMBOS:
                run_id = f"ref_{task_name}_E{E}_K{K}_s{seed}"
                cfg = {
                    "experiment": {
                        "mode": "ref_dynamic",
                        "run_id": run_id,
                        "pair_group": "reference",
                        "pair_id": None,
                        "seed": seed,
                    },
                    "data": build_data_cfg(task_name),
                    "transfer": {
                        "enabled": False,
                        "source_task": task_name,
                        "target_task": task_name,
                        "source_ref_run_id": run_id,
                        "source_stats_path": "",
                        "fixed_selection_rule": "none",
                        "fixed_k": 0,
                        "dynamic_k": K,
                    },
                    "model": build_model_cfg(E, K),
                    "train": build_train_cfg(),
                    "runtime": build_base_runtime("."),
                }
                write_config(ref_dir / f"{run_id}.yaml", cfg)

    for pair in pairs:
        pair_id = pair["pair_id"]
        source_task = pair["source_task"]
        target_task = pair["target_task"]
        pair_group = pair["pair_group"]
        for seed in args.seeds:
            for E, K, f_list in LEGAL_COMBOS:
                dyn_run_id = f"dyn_{pair_id}_E{E}_K{K}_s{seed}"
                dyn_cfg = {
                    "experiment": {
                        "mode": "target_dynamic",
                        "run_id": dyn_run_id,
                        "pair_group": pair_group,
                        "pair_id": pair_id,
                        "seed": seed,
                    },
                    "data": build_data_cfg(target_task),
                    "transfer": {
                        "enabled": False,
                        "source_task": source_task,
                        "target_task": target_task,
                        "source_ref_run_id": f"ref_{source_task}_E{E}_K{K}_s{seed}",
                        "source_stats_path": "",
                        "fixed_selection_rule": "none",
                        "fixed_k": 0,
                        "dynamic_k": K,
                    },
                    "model": build_model_cfg(E, K),
                    "train": build_train_cfg(),
                    "runtime": build_base_runtime("."),
                }
                write_config(dyn_dir / f"{dyn_run_id}.yaml", dyn_cfg)

                for F in f_list:
                    mode = "hybrid_transfer" if F < K else "hybrid_transfer"
                    prefix = "hyb" if F < K else "fix"
                    run_id = f"{prefix}_{pair_id}_E{E}_K{K}_F{F}_s{seed}"
                    hyb_cfg = {
                        "experiment": {
                            "mode": mode,
                            "run_id": run_id,
                            "pair_group": pair_group,
                            "pair_id": pair_id,
                            "seed": seed,
                        },
                        "data": build_data_cfg(target_task),
                        "transfer": {
                            "enabled": True,
                            "source_task": source_task,
                            "target_task": target_task,
                            "source_ref_run_id": f"ref_{source_task}_E{E}_K{K}_s{seed}",
                            "source_stats_path": f"./results/runs/ref_{source_task}_E{E}_K{K}_s{seed}/expert_stats_last3.json",
                            "fixed_selection_rule": "source_topF_last3",
                            "fixed_k": F,
                            "dynamic_k": K - F,
                        },
                        "model": build_model_cfg(E, K),
                        "train": build_train_cfg(),
                        "runtime": build_base_runtime("."),
                    }
                    write_config(hyb_dir / f"{run_id}.yaml", hyb_cfg)

    print(f"generated configs under: {out_root}")


if __name__ == "__main__":
    main()
