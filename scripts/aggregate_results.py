from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="./results/runs")
    parser.add_argument("--out-dir", type=str, default="./results/summaries")
    return parser.parse_args()


def _stringify_list(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return value


def collect_run_rows(results_root: Path) -> List[Dict]:
    rows = []
    for summary_file in sorted(results_root.glob("*/run_summary.json")):
        row = load_json(summary_file)
        row["fixed_experts_layer0"] = _stringify_list(row.get("fixed_experts", []))
        row.setdefault("resolved_fixed_path", str(summary_file.parent / "resolved_fixed_experts.json"))
        rows.append(row)
    return rows


def collect_expert_usage_rows(results_root: Path) -> List[Dict]:
    rows = []
    for run_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        per_epoch_path = run_dir / "expert_stats_per_epoch.json"
        summary_path = run_dir / "run_summary.json"
        resolved_path = run_dir / "resolved_fixed_experts.json"
        metrics_path = run_dir / "metrics_history.json"
        if not per_epoch_path.exists() or not summary_path.exists():
            continue
        stats = load_json(per_epoch_path)
        summary = load_json(summary_path)
        resolved = load_json(resolved_path) if resolved_path.exists() else {"layers": {"moe_0": {"fixed_experts": []}}}
        metrics_hist = load_json(metrics_path) if metrics_path.exists() else []

        epochs = stats.get("epochs", [])
        if not epochs:
            continue
        total_counts = None
        total_fixed = None
        total_dynamic = None
        for rec in epochs:
            layer = rec["layers"]["moe_0"]
            sel = layer["selection_counts"]
            fix = layer.get("fixed_selection_counts", [0 for _ in sel])
            dyn = layer.get("dynamic_selection_counts", [0 for _ in sel])
            if total_counts is None:
                total_counts = [0 for _ in sel]
                total_fixed = [0 for _ in sel]
                total_dynamic = [0 for _ in sel]
            for i, value in enumerate(sel):
                total_counts[i] += value
                total_fixed[i] += fix[i]
                total_dynamic[i] += dyn[i]
        total = sum(total_counts)
        normalized = [value / max(1, total) for value in total_counts]
        entropy_mean = 0.0
        if metrics_hist:
            entropy_mean = sum(rec["test"]["routing_entropy"] for rec in metrics_hist) / len(metrics_hist)
        rows.append({
            "run_id": summary["run_id"],
            "layer": "moe_0",
            "mode": summary["mode"],
            "pair_id": summary.get("pair_id"),
            "pair_group": summary.get("pair_group"),
            "seed": summary.get("seed"),
            "source_task": summary.get("source_task"),
            "target_task": summary.get("target_task"),
            "num_experts": summary.get("num_experts"),
            "top_k": summary.get("top_k"),
            "fixed_k": summary.get("fixed_k"),
            "dynamic_k": summary.get("dynamic_k"),
            "fixed_ratio": summary.get("fixed_ratio"),
            "fixed_experts": resolved.get("layers", {}).get("moe_0", {}).get("fixed_experts", []),
            "dynamic_selection_counts": total_dynamic,
            "fixed_selection_counts": total_fixed,
            "total_selection_counts": total_counts,
            "normalized_total_freq": normalized,
            "router_entropy_mean": entropy_mean,
        })
    return rows


def build_agg_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "pair_id", "source_task", "target_task", "num_experts", "top_k",
        "fixed_k", "dynamic_k", "fixed_ratio", "mode",
    ]
    numeric_cols = [
        "best_val_acc", "best_test_acc", "best_test_macro_f1", "best_test_loss",
        "best_test_routing_entropy", "final_test_acc", "final_test_macro_f1",
        "final_test_loss", "final_test_routing_entropy",
    ]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(part) for part in col if part]).rstrip("_") for col in agg.columns.to_flat_index()]
    if "best_test_acc_mean" in agg.columns:
        agg = agg.rename(columns={
            "best_test_acc_mean": "mean_acc",
            "best_test_acc_std": "std_acc",
            "best_test_macro_f1_mean": "mean_macro_f1",
            "best_test_macro_f1_std": "std_macro_f1",
            "best_test_loss_mean": "mean_loss",
            "best_test_loss_std": "std_loss",
            "best_test_routing_entropy_mean": "mean_entropy",
            "best_test_routing_entropy_std": "std_entropy",
            "best_test_acc_count": "num_seeds",
        })
    return agg


def build_delta_vs_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["pair_id", "source_task", "target_task", "seed", "num_experts", "top_k"]
    dyn = df[df["fixed_k"] == 0][base_cols + ["best_test_acc", "best_test_macro_f1", "best_test_loss", "best_test_routing_entropy"]].copy()
    dyn = dyn.rename(columns={
        "best_test_acc": "dynamic_best_test_acc",
        "best_test_macro_f1": "dynamic_best_test_macro_f1",
        "best_test_loss": "dynamic_best_test_loss",
        "best_test_routing_entropy": "dynamic_best_test_routing_entropy",
    })
    merged = df.merge(dyn, on=base_cols, how="left")
    merged["delta_acc_vs_target_dynamic"] = merged["best_test_acc"] - merged["dynamic_best_test_acc"]
    merged["delta_macro_f1_vs_target_dynamic"] = merged["best_test_macro_f1"] - merged["dynamic_best_test_macro_f1"]
    merged["delta_loss_vs_target_dynamic"] = merged["best_test_loss"] - merged["dynamic_best_test_loss"]
    merged["delta_entropy_vs_target_dynamic"] = merged["best_test_routing_entropy"] - merged["dynamic_best_test_routing_entropy"]

    fixed = df[df["fixed_k"] == df["top_k"]][base_cols + ["best_test_acc"]].copy()
    fixed = fixed.rename(columns={"best_test_acc": "pure_fixed_best_test_acc"})
    merged = merged.merge(fixed, on=base_cols, how="left")
    merged["delta_acc_vs_pure_fixed"] = merged["best_test_acc"] - merged["pure_fixed_best_test_acc"]
    return merged


def build_best_f_tables(agg_df: pd.DataFrame):
    candidate = agg_df.dropna(subset=["pair_id"]).copy()
    candidate = candidate[candidate["mode"] != "ref_dynamic"]
    sort_cols = ["pair_id", "num_experts", "top_k", "mean_acc", "fixed_k"]
    candidate = candidate.sort_values(sort_cols, ascending=[True, True, True, False, True])
    best_rows = candidate.groupby(["pair_id", "source_task", "target_task", "num_experts", "top_k"], dropna=False).head(1).copy()
    best_f = best_rows[["pair_id", "source_task", "target_task", "num_experts", "top_k", "fixed_k", "fixed_ratio", "mode", "mean_acc", "std_acc", "num_seeds"]].copy()
    best_f = best_f.rename(columns={
        "fixed_k": "best_fixed_k",
        "fixed_ratio": "best_fixed_ratio",
        "mode": "best_mode",
        "mean_acc": "best_mean_acc",
        "std_acc": "best_std_acc",
    })
    best_ratio = best_f[["pair_id", "source_task", "target_task", "num_experts", "top_k", "best_fixed_k", "best_fixed_ratio", "best_mode", "best_mean_acc", "best_std_acc", "num_seeds"]].copy()
    return best_f, best_ratio


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_run_rows(results_root)
    if not rows:
        print("No run_summary.json found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["mode", "pair_id", "num_experts", "top_k", "fixed_k", "seed"], na_position="last")
    df.to_csv(out_dir / "run_summary.csv", index=False)

    expert_rows = collect_expert_usage_rows(results_root)
    with open(out_dir / "expert_usage.jsonl", "w", encoding="utf-8") as f:
        for row in expert_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    agg = build_agg_summary(df)
    agg.to_csv(out_dir / "agg_summary.csv", index=False)

    delta = build_delta_vs_dynamic(df)
    delta.to_csv(out_dir / "pairwise_delta_vs_dynamic.csv", index=False)

    best_f, best_ratio = build_best_f_tables(agg)
    best_f.to_csv(out_dir / "best_F_per_pair.csv", index=False)
    best_ratio.to_csv(out_dir / "best_fixed_ratio_per_pair.csv", index=False)

    print(f"saved: {out_dir / 'run_summary.csv'}")
    print(f"saved: {out_dir / 'agg_summary.csv'}")
    print(f"saved: {out_dir / 'pairwise_delta_vs_dynamic.csv'}")
    print(f"saved: {out_dir / 'best_F_per_pair.csv'}")
    print(f"saved: {out_dir / 'best_fixed_ratio_per_pair.csv'}")
    print(f"saved: {out_dir / 'expert_usage.jsonl'}")


if __name__ == "__main__":
    main()
