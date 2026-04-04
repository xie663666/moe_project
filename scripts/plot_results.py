from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries-dir", type=str, default="./results/summaries")
    parser.add_argument("--tables-dir", type=str, default="./results/analysis/tables")
    parser.add_argument("--out-dir", type=str, default="./results/analysis/plots")
    return parser.parse_args()


def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, out_path: Path):
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for mode, group in df.groupby("mode", dropna=False):
        group = group.sort_values(x_col)
        plt.plot(group[x_col], group[y_col], marker="o", label=str(mode))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_single_series(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, out_path: Path):
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(x_col)
    plt.figure(figsize=(8, 5))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    summaries_dir = Path(args.summaries_dir)
    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = summaries_dir / "agg_summary.csv"
    best_path = summaries_dir / "best_F_per_pair.csv"
    win_rate_path = tables_dir / "win_rate_over_dynamic.csv"
    trend_path = tables_dir / "trend_summary_by_EK.csv"

    if not agg_path.exists() or not best_path.exists():
        raise FileNotFoundError("run aggregate_results.py and make_analysis_tables.py first")

    agg_df = pd.read_csv(agg_path)
    best_df = pd.read_csv(best_path)

    pair_plot_dir = out_dir / "pair_curves"
    for (pair_id, e, k), group in agg_df[agg_df["pair_id"].notna()].groupby(["pair_id", "num_experts", "top_k"], dropna=False):
        title = f"{pair_id} | E={e}, K={k} | accuracy vs F"
        save_line_plot(
            group,
            x_col="fixed_k",
            y_col="mean_acc",
            title=title,
            xlabel="fixed_k (F)",
            ylabel="mean accuracy",
            out_path=pair_plot_dir / f"{pair_id}_E{e}_K{k}.png",
        )

    by_ek_dir = out_dir / "aggregate_by_EK"
    for (e, k), group in agg_df[agg_df["mode"] != "ref_dynamic"].groupby(["num_experts", "top_k"], dropna=False):
        save_line_plot(
            group,
            x_col="fixed_k",
            y_col="mean_acc",
            title=f"Aggregate accuracy vs F | E={e}, K={k}",
            xlabel="fixed_k (F)",
            ylabel="mean accuracy",
            out_path=by_ek_dir / f"accuracy_vs_F_E{e}_K{k}.png",
        )
        save_line_plot(
            group,
            x_col="fixed_ratio",
            y_col="mean_acc",
            title=f"Aggregate accuracy vs F/K | E={e}, K={k}",
            xlabel="fixed_ratio (F/K)",
            ylabel="mean accuracy",
            out_path=by_ek_dir / f"accuracy_vs_ratio_E{e}_K{k}.png",
        )

    best_dist_dir = out_dir / "best_F"
    best_counts = best_df.groupby(["num_experts", "top_k", "best_fixed_k"], dropna=False).size().reset_index(name="count")
    for (e, k), group in best_counts.groupby(["num_experts", "top_k"], dropna=False):
        save_single_series(
            group,
            x_col="best_fixed_k",
            y_col="count",
            title=f"Best F distribution | E={e}, K={k}",
            xlabel="best fixed_k",
            ylabel="count",
            out_path=best_dist_dir / f"best_F_dist_E{e}_K{k}.png",
        )

    if win_rate_path.exists():
        win_df = pd.read_csv(win_rate_path)
        win_dir = out_dir / "win_rate"
        for (e, k), group in win_df.groupby(["num_experts", "top_k"], dropna=False):
            save_single_series(
                group,
                x_col="fixed_k",
                y_col="win_rate",
                title=f"Win rate over dynamic | E={e}, K={k}",
                xlabel="fixed_k (F)",
                ylabel="win rate",
                out_path=win_dir / f"win_rate_E{e}_K{k}.png",
            )

    if trend_path.exists():
        trend_df = pd.read_csv(trend_path)
        trend_dir = out_dir / "trend_summary"
        for e, group in trend_df.groupby("num_experts", dropna=False):
            save_line_plot(
                group,
                x_col="fixed_ratio",
                y_col="mean_acc",
                title=f"Trend summary by fixed ratio | E={e}",
                xlabel="fixed_ratio (F/K)",
                ylabel="mean accuracy",
                out_path=trend_dir / f"trend_ratio_E{e}.png",
            )

    print(f"saved plots under: {out_dir}")


if __name__ == "__main__":
    main()
