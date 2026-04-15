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
    parser.add_argument("--round", type=str, default="main")
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
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


def build_model_cfg(model_base: Dict, E: int, K: int) -> Dict:
    model_cfg = deep_merge(model_base, {})
    model_cfg.setdefault("moe", {})
    model_cfg["moe"]["num_experts"] = E
    model_cfg["moe"]["top_k"] = K
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


def build_train_cfg(train_base: Dict) -> Dict:
    return deep_merge(train_base, {})


def write_config(path: Path, cfg: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(path, cfg)


def main():
    args = parse_args()
    pairs = load_yaml(args.pairs)["directed_pairs"]
    tasks = load_tasks(args.tasks)
    bases = load_bases(PROJECT_ROOT)
    out_root = Path(args.output)

    ref_dir = out_root / "ref"
    dyn_dir = out_root / "dynamic"
    hyb_dir = out_root / "hybrid"
    for p in [ref_dir, dyn_dir, hyb_dir]:
        p.mkdir(parents=True, exist_ok=True)

    source_tasks = sorted({item["source_task"] for item in pairs})
    for task_name in source_tasks:
        task_spec = tasks[task_name]
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
                    "data": build_data_cfg(bases["dataset"], task_spec, task_name),
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
                    "model": build_model_cfg(bases["model"], E, K),
                    "train": build_train_cfg(bases["train"]),
                    "runtime": build_runtime_cfg(bases["runtime"], "."),
                }
                cfg = deep_merge(cfg, bases["ref"])
                write_config(ref_dir / f"{run_id}.yaml", cfg)

    for pair in pairs:
        pair_id = pair["pair_id"]
        source_task = pair["source_task"]
        target_task = pair["target_task"]
        pair_group = pair["pair_group"]
        target_spec = tasks[target_task]
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
                    "data": build_data_cfg(bases["dataset"], target_spec, target_task),
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
                    "model": build_model_cfg(bases["model"], E, K),
                    "train": build_train_cfg(bases["train"]),
                    "runtime": build_runtime_cfg(bases["runtime"], "."),
                }
                dyn_cfg = deep_merge(dyn_cfg, bases["dyn"])
                write_config(dyn_dir / f"{dyn_run_id}.yaml", dyn_cfg)

                for F in f_list:
                    prefix = "hyb" if F < K else "fix"
                    run_id = f"{prefix}_{pair_id}_E{E}_K{K}_F{F}_s{seed}"
                    hyb_cfg = {
                        "experiment": {
                            "mode": "hybrid_transfer",
                            "run_id": run_id,
                            "pair_group": pair_group,
                            "pair_id": pair_id,
                            "seed": seed,
                        },
                        "data": build_data_cfg(bases["dataset"], target_spec, target_task),
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
                        "model": build_model_cfg(bases["model"], E, K),
                        "train": build_train_cfg(bases["train"]),
                        "runtime": build_runtime_cfg(bases["runtime"], "."),
                    }
                    hyb_cfg = deep_merge(hyb_cfg, bases["hyb"])
                    hyb_cfg["transfer"]["fixed_k"] = F
                    hyb_cfg["transfer"]["dynamic_k"] = K - F
                    write_config(hyb_dir / f"{run_id}.yaml", hyb_cfg)

                    rand_prefix = "rand_hyb" if F < K else "rand_fix"
                    rand_run_id = f"{rand_prefix}_{pair_id}_E{E}_K{K}_F{F}_s{seed}"
                    rand_cfg = deep_merge(hyb_cfg, {})
                    rand_cfg["experiment"]["run_id"] = rand_run_id
                    rand_cfg["experiment"]["mode"] = "hybrid_random_control"
                    rand_cfg["transfer"]["fixed_selection_rule"] = "random"
                    rand_cfg["transfer"]["source_stats_path"] = ""
                    rand_cfg["transfer"]["reuse_source_expert_weights"] = False
                    rand_cfg["transfer"]["source_checkpoint_path"] = ""
                    write_config(hyb_dir / f"{rand_run_id}.yaml", rand_cfg)

    print(f"generated configs under: {out_root}")


if __name__ == "__main__":
    main()
