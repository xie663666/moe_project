from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--failed-list", type=str, default="./logs/failed_runs.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = Path(args.config_dir)
    failed_list = Path(args.failed_list)
    failed_list.parent.mkdir(parents=True, exist_ok=True)

    cfg_paths = sorted(config_dir.glob("*.yaml"))
    if args.max_runs is not None:
        cfg_paths = cfg_paths[: args.max_runs]

    failures = []
    ran = 0
    skipped = 0

    for cfg_path in cfg_paths:
        cfg = load_yaml(cfg_path)
        run_id = cfg["experiment"]["run_id"]
        run_summary = PROJECT_ROOT / "results" / "runs" / run_id / "run_summary.json"
        log_path = PROJECT_ROOT / "logs" / f"{run_id}.log"
        if args.skip_existing and run_summary.exists():
            skipped += 1
            print(f"[SKIP] {run_id}")
            continue

        cmd = [sys.executable, "train.py", "--config", str(cfg_path), "--device", args.device]
        print(f"[RUN] {run_id}")
        with open(log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log_f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            failures.append(run_id)
            print(f"[FAIL] {run_id} -> {log_path}")
        else:
            ran += 1
            print(f"[OK] {run_id}")

    if failures:
        failed_list.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"failed runs written to: {failed_list}")
        raise SystemExit(1)

    print(f"stage complete | ran={ran} | skipped={skipped} | failures=0")


if __name__ == "__main__":
    main()
