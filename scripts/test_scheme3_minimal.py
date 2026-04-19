from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import LiteCNNMoEClassifier
from src.transfer import resolve_branch_fusion_weights, resolve_fixed_expert_internal_weights


def run_case(case_name: str, num_experts: int, top_k: int, fixed_experts: list[int], counts: list[float]):
    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "stats.json"
        stats = {
            "layers": {
                "moe_0": {
                    "window_sum_counts": counts,
                    "top_f_lookup": {str(len(fixed_experts)): fixed_experts},
                    "ranked_experts": list(range(num_experts)),
                }
            }
        }
        stats_path.write_text(json.dumps(stats), encoding="utf-8")
        cfg = {
            "transfer": {
                "fixed_k": len(fixed_experts),
                "fixed_selection_rule": "source_topF_last3",
                "source_stats_path": str(stats_path),
            },
            "model": {"moe": {"num_experts": num_experts, "top_k": top_k}},
            "experiment": {"seed": 1},
        }
        fixed_expert_internal_weights = resolve_fixed_expert_internal_weights(cfg)
        branch_fusion_weight_fixed, branch_fusion_weight_dynamic = resolve_branch_fusion_weights(cfg)

        model = LiteCNNMoEClassifier(
            in_channels=3,
            feature_dim=16,
            num_experts=num_experts,
            top_k=top_k,
            fixed_experts=fixed_experts,
            hidden_dim=32,
            router_noise_std=0.0,
            num_classes=5,
            routing_mode="fixed_branch_dynamic_branch",
            fixed_expert_internal_weights=fixed_expert_internal_weights,
            branch_fusion_weight_fixed=branch_fusion_weight_fixed,
            branch_fusion_weight_dynamic=branch_fusion_weight_dynamic,
        )
        for idx in fixed_experts:
            for p in model.moe.experts[idx].parameters():
                p.requires_grad = False

        x = torch.randn(4, 3, 32, 32)
        logits, aux = model(x, track_usage=True)
        assert logits.shape == (4, 5)

        dyn = aux["dynamic_selected_idx"].detach().cpu().tolist()
        if top_k - len(fixed_experts) > 0:
            assert all(v not in set(fixed_experts) for row in dyn for v in row)

        print(f"[{case_name}] fixed_experts={aux['fixed_experts']}")
        print(f"[{case_name}] fixed_expert_internal_weights={aux['fixed_expert_internal_weights']}")
        print(f"[{case_name}] dynamic_selected_idx={dyn}")
        print(
            f"[{case_name}] branch_fusion_weight_fixed={aux['branch_fusion_weight_fixed']}, "
            f"branch_fusion_weight_dynamic={aux['branch_fusion_weight_dynamic']}"
        )


if __name__ == "__main__":
    run_case(
        case_name="F0",
        num_experts=6,
        top_k=3,
        fixed_experts=[],
        counts=[10, 9, 8, 7, 6, 5],
    )
    run_case(
        case_name="FK",
        num_experts=6,
        top_k=3,
        fixed_experts=[0, 2, 4],
        counts=[30, 1, 20, 1, 10, 1],
    )
    run_case(
        case_name="FMIX",
        num_experts=6,
        top_k=4,
        fixed_experts=[3, 1],
        counts=[1, 40, 1, 20, 1, 1],
    )
