# six_moe

Lightweight MoE transfer experiment project for verifying:

> For target task B, if some experts are fixed from a semantically similar source task A and the remaining experts are dynamically selected by B, how does changing the number of fixed experts affect B's accuracy?

## Project structure

- `configs/`: base configs, task definitions, pair lists, templates, generated configs
- `scripts/`: pair expansion, config generation, statistics extraction, aggregation, smoke tests, batch runners
- `src/`: dataset, model, config utilities, transfer utilities, misc helpers
- `train.py`: training entry
- `evaluate.py`: checkpoint evaluation entry
- `bootstrap_local_windows.bat`: one-click local environment setup, config generation, and smoke test
- `bootstrap_server.sh`: one-click Linux/server setup, config generation, and smoke test

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

## Quick start

### Windows
```bat
bootstrap_local_windows.bat
```

### Linux / AutoDL
```bash
bash bootstrap_server.sh
```

### Generate configs only
```bash
python scripts/expand_pairs.py --pairs configs/pairs/main_pairs.yaml --out configs/pairs/directed_main_pairs.yaml
python scripts/gen_run_configs.py --round main --tasks configs/tasks/cifar100_superclass_tasks.yaml --pairs configs/pairs/directed_main_pairs.yaml --output configs/generated/main_round
```

### Run batches
```bash
bash scripts/run_ref_batch.sh
bash scripts/run_target_dynamic_batch.sh
bash scripts/run_hybrid_batch.sh
```
