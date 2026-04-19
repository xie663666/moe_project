from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries-dir", type=str, default="./results/summaries")
    parser.add_argument("--out-dir", type=str, default="./results/analysis/tables")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries_dir = Path(args.summaries_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = summaries_dir / "run_summary.csv"
    delta_path = summaries_dir / "pairwise_delta_vs_dynamic.csv"
    agg_path = summaries_dir / "agg_summary.csv"

    if not run_summary_path.exists() or not delta_path.exists() or not agg_path.exists():
        raise FileNotFoundError("run aggregate_results.py first")

    delta_df = pd.read_csv(delta_path)
    agg_df = pd.read_csv(agg_path)

    delta_transfer = delta_df[delta_df["fixed_k"] > 0].copy()
    win_rate = delta_transfer.groupby(["num_experts", "top_k", "fixed_k", "fixed_ratio"], dropna=False)["delta_acc_vs_target_dynamic"].agg(
        mean_delta="mean",
        std_delta="std",
        win_rate=lambda s: (s > 0).mean(),
        num_runs="count",
    ).reset_index()
    win_rate.to_csv(out_dir / "win_rate_over_dynamic.csv", index=False)

    trend = agg_df[agg_df["mode"] != "ref_dynamic"].copy()
    trend = trend.groupby(["num_experts", "top_k", "fixed_k", "fixed_ratio", "mode"], dropna=False).agg(
        mean_acc=("mean_acc", "mean"),
        std_acc=("mean_acc", "std"),
        mean_macro_f1=("mean_macro_f1", "mean"),
        mean_entropy=("mean_entropy", "mean"),
        num_groups=("pair_id", "count"),
    ).reset_index()
    trend.to_csv(out_dir / "trend_summary_by_EK.csv", index=False)

    directional = delta_df[delta_df["pair_id"].notna()].copy()
    directional["pair_family"] = directional["pair_id"].str.extract(r"^(p\d+)_", expand=False)
    directional_summary = directional.groupby(["pair_family", "pair_id", "source_task", "target_task", "num_experts", "top_k", "fixed_k"], dropna=False).agg(
        mean_acc=("best_test_acc", "mean"),
        std_acc=("best_test_acc", "std"),
        mean_delta_vs_dynamic=("delta_acc_vs_target_dynamic", "mean"),
        num_runs=("run_id", "count"),
    ).reset_index()
    directional_summary.to_csv(out_dir / "directional_pair_summary.csv", index=False)

    best_f = pd.read_csv(summaries_dir / "best_F_per_pair.csv")
    best_f_dist = best_f.groupby(["num_experts", "top_k", "best_fixed_k", "best_fixed_ratio", "best_mode"], dropna=False).size().reset_index(name="count")
    best_f_dist.to_csv(out_dir / "best_F_distribution.csv", index=False)

    print(f"saved: {out_dir / 'win_rate_over_dynamic.csv'}")
    print(f"saved: {out_dir / 'trend_summary_by_EK.csv'}")
    print(f"saved: {out_dir / 'directional_pair_summary.csv'}")
    print(f"saved: {out_dir / 'best_F_distribution.csv'}")


if __name__ == "__main__":
    main()
