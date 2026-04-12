from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.config import load_yaml

LEGAL_COMBOS = {
    (8, 4): {1, 2, 3, 4},
    (16, 4): {1, 2, 3, 4},
    (16, 6): {1, 2, 4, 6},
    (16, 8): {1, 2, 4, 6, 8},
    (32, 4): {1, 2, 3, 4},
    (32, 6): {1, 2, 4, 6},
    (32, 8): {1, 2, 4, 6, 8},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--stage", type=str, default="prep", choices=["prep", "refs_ready", "hybrid_ready"])
    return parser.parse_args()


def load_generated_configs(root: Path) -> Dict[str, List[Path]]:
    gen_root = root / "configs" / "generated" / "main_round"
    return {
        "ref": sorted((gen_root / "ref").glob("*.yaml")),
        "dynamic": sorted((gen_root / "dynamic").glob("*.yaml")),
        "hybrid": sorted((gen_root / "hybrid").glob("*.yaml")),
    }


def validate_generated_counts(cfgs: Dict[str, List[Path]], directed_pairs_count: int, unique_source_tasks_count: int) -> List[str]:
    errors = []
    expected_ref = 7 * unique_source_tasks_count
    if cfgs["ref"] and len(cfgs["ref"]) != expected_ref:
        errors.append(f"unexpected ref config count: got {len(cfgs['ref'])}, expected {expected_ref}")
    expected_dyn = directed_pairs_count * 7
    expected_hyb = directed_pairs_count * 60
    if cfgs["dynamic"] and len(cfgs["dynamic"]) != expected_dyn:
        errors.append(f"unexpected dynamic config count: got {len(cfgs['dynamic'])}, expected {expected_dyn}")
    if cfgs["hybrid"] and len(cfgs["hybrid"]) != expected_hyb:
        errors.append(f"unexpected hybrid config count: got {len(cfgs['hybrid'])}, expected {expected_hyb}")
    return errors


def validate_run_ids(all_cfgs: Iterable[Path]) -> List[str]:
    run_ids = []
    for path in all_cfgs:
        cfg = load_yaml(path)
        run_ids.append(cfg["experiment"]["run_id"])
    counter = Counter(run_ids)
    duplicates = [run_id for run_id, count in counter.items() if count > 1]
    return [f"duplicate run_id found: {run_id}" for run_id in duplicates]


def validate_hybrid_combo(path: Path) -> List[str]:
    cfg = load_yaml(path)
    e = int(cfg["model"]["moe"]["num_experts"])
    k = int(cfg["model"]["moe"]["top_k"])
    f = int(cfg["transfer"]["fixed_k"])
    d = int(cfg["transfer"]["dynamic_k"])
    errors = []
    if (e, k) not in LEGAL_COMBOS:
        errors.append(f"illegal (E,K)=({e},{k}) in {path.name}")
    else:
        if f not in LEGAL_COMBOS[(e, k)]:
            errors.append(f"illegal F={f} for (E,K)=({e},{k}) in {path.name}")
    if f + d != k:
        errors.append(f"fixed_k + dynamic_k != top_k in {path.name}")
    return errors


def validate_reference_stats_exist(root: Path, hybrid_cfgs: Iterable[Path]) -> List[str]:
    errors = []
    for path in hybrid_cfgs:
        cfg = load_yaml(path)
        rule = cfg["transfer"].get("fixed_selection_rule", "source_topF_last3")
        if int(cfg["transfer"]["fixed_k"]) == 0:
            continue
        if rule != "source_topF_last3":
            continue
        stats_path = Path(cfg["transfer"]["source_stats_path"])
        if not stats_path.is_absolute():
            stats_path = root / stats_path
        if not stats_path.exists():
            errors.append(f"missing source stats for {path.name}: {stats_path}")
    return errors


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    required = [
        root / "configs" / "base",
        root / "configs" / "pairs",
        root / "configs" / "tasks",
        root / "configs" / "templates",
        root / "scripts",
        root / "src",
    ]
    errors = []
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path}")

    directed_pairs_path = root / "configs" / "pairs" / "directed_main_pairs.yaml"
    if not directed_pairs_path.exists():
        errors.append(f"missing directed pairs: {directed_pairs_path}")
        directed_pairs = []
    else:
        directed_pairs = load_yaml(directed_pairs_path).get("directed_pairs", [])
        for item in directed_pairs:
            if item["source_task"] == item["target_task"]:
                errors.append(f"source_task equals target_task in {item['pair_id']}")

    cfgs = load_generated_configs(root)
    all_cfg_paths = cfgs["ref"] + cfgs["dynamic"] + cfgs["hybrid"]
    if args.stage in {"prep", "refs_ready", "hybrid_ready"}:
        if not cfgs["ref"] or not cfgs["dynamic"] or not cfgs["hybrid"]:
            errors.append("generated configs missing; run expand_pairs.py and gen_run_configs.py first")
        else:
            errors.extend(validate_generated_counts(cfgs, len(directed_pairs), len({item["source_task"] for item in directed_pairs})))
            errors.extend(validate_run_ids(all_cfg_paths))
            for path in cfgs["hybrid"]:
                errors.extend(validate_hybrid_combo(path))

    if args.stage in {"refs_ready", "hybrid_ready"}:
        for ref_cfg_path in cfgs["ref"]:
            cfg = load_yaml(ref_cfg_path)
            run_id = cfg["experiment"]["run_id"]
            stats_path = root / "results" / "runs" / run_id / "expert_stats_last3.json"
            if not stats_path.exists():
                errors.append(f"missing ref stats: {stats_path}")

    if args.stage == "hybrid_ready":
        errors.extend(validate_reference_stats_exist(root, cfgs["hybrid"]))

    dataset_root = root / "datasets"
    if not dataset_root.exists():
        errors.append(f"dataset root missing: {dataset_root}")

    if errors:
        print("Preflight FAILED")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("Preflight passed")
    print(f"project_root: {root}")
    print(f"stage: {args.stage}")
    print(f"directed_pairs: {len(directed_pairs)}")
    print(f"ref_configs: {len(cfgs['ref'])}")
    print(f"dynamic_configs: {len(cfgs['dynamic'])}")
    print(f"hybrid_configs: {len(cfgs['hybrid'])}")


if __name__ == "__main__":
    main()
