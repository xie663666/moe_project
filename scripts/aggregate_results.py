from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import pandas as pd

from src.utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="./results/runs")
    parser.add_argument("--out-dir", type=str, default="./results/summaries")
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary_file in results_root.glob("*/run_summary.json"):
        rows.append(load_json(summary_file))

    if not rows:
        print("No run_summary.json found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "run_summary.csv", index=False)

    group_cols = ["pair_id", "source_task", "target_task", "num_experts", "top_k", "fixed_k", "dynamic_k", "mode"]
    numeric_cols = ["best_val_acc", "best_test_acc", "best_test_macro_f1", "best_test_loss", "best_test_routing_entropy"]
    agg = df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    agg.to_csv(out_dir / "agg_summary.csv", index=False)
    print(f"saved: {out_dir / 'run_summary.csv'}")
    print(f"saved: {out_dir / 'agg_summary.csv'}")


if __name__ == "__main__":
    main()
