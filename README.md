# six_moe

Lightweight MoE transfer experiment project for verifying:

> For target task B, if some experts are fixed from a semantically similar source task A and the remaining experts are dynamically selected by B, how does changing the number of fixed experts affect B's accuracy?

## What is included

- `configs/`: base configs, task definitions, pair lists, templates, generated configs
- `scripts/`: pair expansion, config generation, preflight checks, stats extraction, aggregation, table generation, plotting, smoke tests, batch runners
- `src/`: dataset, model, config utilities, transfer utilities, misc helpers
- `train.py`: training entry
- `evaluate.py`: checkpoint evaluation entry
- `bootstrap_local_windows.bat`: local environment bootstrap + config generation + preflight + smoke test
- `bootstrap_server.sh`: server bootstrap + config generation + preflight + smoke test
- `run_all_main_local_windows.bat`: full local pipeline
- `run_all_main_server.sh`: full server pipeline
- `run_analysis_local_windows.bat`: re-run analysis only on local
- `run_analysis_server.sh`: re-run analysis only on server

## Default model

- Stem: 3-layer lightweight CNN
- Feature dim: 256
- MoE: single layer
- Expert: MLP `256 -> 512 -> 256`
- Router: linear gating on 256-d features
- Classifier: linear `256 -> num_classes`

## Main modes

- `ref_dynamic`: A-source reference run for expert statistics
- `target_dynamic`: B-target pure dynamic baseline
- `hybrid_transfer`: B-target transfer run with fixed experts from A
- `hybrid_random_control`: B-target control run with fixed experts sampled randomly (same `E/K/F`)

## Hybrid routing details

For B-task runs with top-k routing:
- total active experts per sample: `K`
- fixed experts: `F`
- dynamic experts: `D = K - F`
- dynamic experts are selected from non-fixed experts to avoid duplicates

In this repo, transfer configs now support:
- `fixed_selection_rule: source_topF_last3 | random | manual`
- optional direct reuse of source-A fixed expert weights:
  - `reuse_source_expert_weights: true/false`
  - `freeze_fixed_experts: true/false`
  - `accelerate_fixed_experts: true/false` (when frozen, run fixed experts in no-grad mode for faster training)
  - `source_checkpoint_path` (optional override)
  - safety rule in training: `freeze_fixed_experts=true` requires `reuse_source_expert_weights=true`
  - note: this setup reduces backward/optimizer cost for frozen experts; forward still computes `K` selected experts (algorithm remains unchanged)
- runtime controls:
  - `model.moe.router_noise_std` (set `0.0` for no routing noise)
  - `train.load_balance_coef` (set `0.0` to disable load-balancing regularization)
  - per-epoch timing and top selected experts are recorded in `metrics_history.json`

Recommended robust setting (default in current generators/base config):
- multiple seeds (`--seeds 1 2 3`)
- `train.epochs: 100`
- disable routing noise for ablation by setting `model.moe.router_noise_std: 0.0`

## One-click usage

### Windows local

Setup and verify only:

```bat
bootstrap_local_windows.bat
```

Full pipeline:

```bat
run_all_main_local_windows.bat cuda
```

Analysis only:

```bat
run_analysis_local_windows.bat
```

### Linux / AutoDL

Setup and verify only:

```bash
bash bootstrap_server.sh
```

Full pipeline:

```bash
bash run_all_main_server.sh cuda
```

Analysis only:

```bash
bash run_analysis_server.sh
```

## Output files after training

Generated summaries are saved under:

- `results/summaries/run_summary.csv`
- `results/summaries/agg_summary.csv`
- `results/summaries/pairwise_delta_vs_dynamic.csv`
- `results/summaries/best_F_per_pair.csv`
- `results/summaries/best_fixed_ratio_per_pair.csv`
- `results/summaries/expert_usage.jsonl`

Generated analysis tables are saved under:

- `results/analysis/tables/win_rate_over_dynamic.csv`
- `results/analysis/tables/trend_summary_by_EK.csv`
- `results/analysis/tables/directional_pair_summary.csv`
- `results/analysis/tables/best_F_distribution.csv`

Generated plots are saved under:

- `results/analysis/plots/pair_curves/*.png`
- `results/analysis/plots/aggregate_by_EK/*.png`
- `results/analysis/plots/best_F/*.png`
- `results/analysis/plots/win_rate/*.png`
- `results/analysis/plots/trend_summary/*.png`
