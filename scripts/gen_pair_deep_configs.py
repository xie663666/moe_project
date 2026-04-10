from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from typing import Dict, List

from src.config import deep_merge, load_yaml, save_yaml

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
    parser.add_argument("--source-task", type=str, default="vehicles1")
    parser.add_argument("--target-task", type=str, default="vehicles2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tasks", type=str, default="configs/tasks/cifar100_superclass_tasks.yaml")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    return parser.parse_args()


def load_bases(project_root: Path):
    cfg_root = project_root / "configs"
    dataset_base = load_yaml(cfg_root / "base" / "dataset.yaml")
    model_base = load_yaml(cfg_root / "base" / "model.yaml")
    train_base = load_yaml(cfg_root / "base" / "train.yaml")
    runtime_base = load_yaml(cfg_root / "base" / "runtime.yaml")
    template_ref = load_yaml(cfg_root / "templates" / "ref_dynamic.yaml")
    template_dyn = load_yaml(cfg_root / "templates" / "target_dynamic.yaml")
    template_hyb = load_yaml(cfg_root / "templates" / "hybrid_transfer.yaml")
    return {
        "dataset": dataset_base,
        "model": model_base,
        "train": train_base,
        "runtime": runtime_base,
        "ref": template_ref,
        "dyn": template_dyn,
        "hyb": template_hyb,
    }


def load_tasks(tasks_path: str | Path) -> Dict[str, Dict]:
    data = load_yaml(tasks_path)
    tasks = data.get("tasks", data)
    if not isinstance(tasks, dict):
        raise ValueError("tasks yaml must contain a dict under key 'tasks' or at top level")
    return tasks


def build_model_cfg(model_base: Dict, num_experts: int, top_k: int) -> Dict:
    model_cfg = deep_merge(model_base, {})
    model_cfg.setdefault("moe", {})
    model_cfg["moe"]["num_experts"] = num_experts
    model_cfg["moe"]["top_k"] = top_k
    if "classifier" in model_cfg:
        model_cfg["classifier"]["in_dim"] = model_cfg.get("feature_dim", 256)
    return model_cfg


def build_data_cfg(dataset_base: Dict, task_spec: Dict, task_name: str) -> Dict:
    data_cfg = deep_merge(dataset_base, {})
    data_cfg["task_name"] = task_name
    data_cfg["num_classes"] = int(task_spec.get("num_classes", 5))
    data_cfg["fine_classes"] = list(task_spec.get("fine_classes", []))
    return data_cfg


def build_runtime_cfg(runtime_base: Dict, project_root: str = ".") -> Dict:
    runtime_cfg = deep_merge(runtime_base, {})
    runtime_cfg["project_root"] = project_root
    return runtime_cfg


def build_train_cfg(train_base: Dict, epochs: int) -> Dict:
    train_cfg = deep_merge(train_base, {})
    train_cfg["epochs"] = int(epochs)
    return train_cfg


def write_config(path: Path, cfg: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(path, cfg)


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks)
    if args.source_task not in tasks:
        raise KeyError(f"unknown source task: {args.source_task}")
    if args.target_task not in tasks:
        raise KeyError(f"unknown target task: {args.target_task}")
    if args.source_task == args.target_task:
        raise ValueError("source-task and target-task must be different")

    bases = load_bases(PROJECT_ROOT)
    source_spec = tasks[args.source_task]
    target_spec = tasks[args.target_task]
    epoch_tag = f"ep{int(args.epochs)}"
    pair_id = f"{args.source_task}_to_{args.target_task}"
    pair_group = f"deep_{pair_id}_{epoch_tag}"

    if args.output is None:
        out_root = PROJECT_ROOT / "configs" / "generated" / pair_group
    else:
        out_root = Path(args.output)
        if not out_root.is_absolute():
            out_root = PROJECT_ROOT / out_root

    ref_dir = out_root / "ref"
    dyn_dir = out_root / "dynamic"
    hyb_dir = out_root / "hybrid"
    for p in [ref_dir, dyn_dir, hyb_dir]:
        p.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for num_experts, top_k, f_list in LEGAL_COMBOS:
            ref_run_id = f"ref_{args.source_task}_E{num_experts}_K{top_k}_{epoch_tag}_s{seed}"
            ref_cfg = {
                "experiment": {
                    "mode": "ref_dynamic",
                    "run_id": ref_run_id,
                    "pair_group": pair_group,
                    "pair_id": pair_id,
                    "seed": seed,
                },
                "data": build_data_cfg(bases["dataset"], source_spec, args.source_task),
                "transfer": {
                    "enabled": False,
                    "source_task": args.source_task,
                    "target_task": args.source_task,
                    "source_ref_run_id": ref_run_id,
                    "source_stats_path": "",
                    "fixed_selection_rule": "none",
                    "fixed_k": 0,
                    "dynamic_k": top_k,
                },
                "model": build_model_cfg(bases["model"], num_experts, top_k),
                "train": build_train_cfg(bases["train"], args.epochs),
                "runtime": build_runtime_cfg(bases["runtime"], "."),
            }
            ref_cfg = deep_merge(ref_cfg, bases["ref"])
            write_config(ref_dir / f"{ref_run_id}.yaml", ref_cfg)

            dyn_run_id = f"dyn_{pair_id}_E{num_experts}_K{top_k}_{epoch_tag}_s{seed}"
            dyn_cfg = {
                "experiment": {
                    "mode": "target_dynamic",
                    "run_id": dyn_run_id,
                    "pair_group": pair_group,
                    "pair_id": pair_id,
                    "seed": seed,
                },
                "data": build_data_cfg(bases["dataset"], target_spec, args.target_task),
                "transfer": {
                    "enabled": False,
                    "source_task": args.source_task,
                    "target_task": args.target_task,
                    "source_ref_run_id": ref_run_id,
                    "source_stats_path": "",
                    "fixed_selection_rule": "none",
                    "fixed_k": 0,
                    "dynamic_k": top_k,
                },
                "model": build_model_cfg(bases["model"], num_experts, top_k),
                "train": build_train_cfg(bases["train"], args.epochs),
                "runtime": build_runtime_cfg(bases["runtime"], "."),
            }
            dyn_cfg = deep_merge(dyn_cfg, bases["dyn"])
            write_config(dyn_dir / f"{dyn_run_id}.yaml", dyn_cfg)

            for fixed_k in f_list:
                prefix = "hyb" if fixed_k < top_k else "fix"
                run_id = f"{prefix}_{pair_id}_E{num_experts}_K{top_k}_F{fixed_k}_{epoch_tag}_s{seed}"
                hyb_cfg = {
                    "experiment": {
                        "mode": "hybrid_transfer",
                        "run_id": run_id,
                        "pair_group": pair_group,
                        "pair_id": pair_id,
                        "seed": seed,
                    },
                    "data": build_data_cfg(bases["dataset"], target_spec, args.target_task),
                    "transfer": {
                        "enabled": True,
                        "source_task": args.source_task,
                        "target_task": args.target_task,
                        "source_ref_run_id": ref_run_id,
                        "source_stats_path": f"./results/runs/{ref_run_id}/expert_stats_last3.json",
                        "fixed_selection_rule": "source_topF_last3",
                        "fixed_k": fixed_k,
                        "dynamic_k": top_k - fixed_k,
                    },
                    "model": build_model_cfg(bases["model"], num_experts, top_k),
                    "train": build_train_cfg(bases["train"], args.epochs),
                    "runtime": build_runtime_cfg(bases["runtime"], "."),
                }
                hyb_cfg = deep_merge(hyb_cfg, bases["hyb"])
                hyb_cfg["transfer"]["fixed_k"] = fixed_k
                hyb_cfg["transfer"]["dynamic_k"] = top_k - fixed_k
                write_config(hyb_dir / f"{run_id}.yaml", hyb_cfg)

    print(f"generated focused deep configs under: {out_root}")


if __name__ == "__main__":
    main()
