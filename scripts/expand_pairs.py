from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

from src.config import load_yaml, save_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pair_cfg = load_yaml(args.pairs)
    directed = {
        "pair_group": pair_cfg["pair_group"],
        "directed_pairs": [],
    }

    for idx, pair in enumerate(pair_cfg["undirected_pairs"], start=1):
        a, b = pair
        pid = f"p{idx:02d}"
        directed["directed_pairs"].append({
            "pair_index": idx,
            "pair_id": f"{pid}_{a}_to_{b}",
            "source_task": a,
            "target_task": b,
            "pair_group": pair_cfg["pair_group"],
        })
        directed["directed_pairs"].append({
            "pair_index": idx,
            "pair_id": f"{pid}_{b}_to_{a}",
            "source_task": b,
            "target_task": a,
            "pair_group": pair_cfg["pair_group"],
        })

    save_yaml(args.out, directed)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
