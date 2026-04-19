from __future__ import annotations

from typing import Dict, List

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteCNNStem(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return x.flatten(1)


class MLPExpert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class SingleLayerMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        fixed_experts: List[int] | None = None,
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.fixed_experts = list(fixed_experts or [])
        self.router_noise_std = float(router_noise_std)
        self.frozen_experts = set()
        self.frozen_experts_no_grad = False
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([MLPExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.reset_epoch_usage()

    def set_frozen_experts(self, frozen_experts: List[int], no_grad_mode: bool):
        self.frozen_experts = set(int(i) for i in frozen_experts)
        self.frozen_experts_no_grad = bool(no_grad_mode)

    def reset_epoch_usage(self):
        self.epoch_usage = {
            "moe_0": {
                "selection_counts": [0 for _ in range(self.num_experts)],
                "fixed_selection_counts": [0 for _ in range(self.num_experts)],
                "dynamic_selection_counts": [0 for _ in range(self.num_experts)],
                "grad_tracked_selection_counts": [0 for _ in range(self.num_experts)],
            }
        }

    def consume_epoch_usage(self):
        usage = self.epoch_usage
        self.reset_epoch_usage()
        return usage

    def _update_usage(self, selected_idx: torch.Tensor, dynamic_idx: torch.Tensor | None, grad_tracked_idx: torch.Tensor | None):
        flat_selected = selected_idx.detach().cpu().reshape(-1).tolist()
        for idx in flat_selected:
            self.epoch_usage["moe_0"]["selection_counts"][idx] += 1
        for idx in self.fixed_experts:
            self.epoch_usage["moe_0"]["fixed_selection_counts"][idx] += selected_idx.size(0)
        if dynamic_idx is not None and dynamic_idx.numel() > 0:
            for idx in dynamic_idx.detach().cpu().reshape(-1).tolist():
                self.epoch_usage["moe_0"]["dynamic_selection_counts"][idx] += 1
        if grad_tracked_idx is not None and grad_tracked_idx.numel() > 0:
            for idx in grad_tracked_idx.detach().cpu().reshape(-1).tolist():
                self.epoch_usage["moe_0"]["grad_tracked_selection_counts"][idx] += 1

    def forward(self, x, track_usage: bool = True):
        logits = self.router(x)
        routing_logits = logits
        if self.training and self.router_noise_std > 0:
            routing_logits = routing_logits + torch.randn_like(routing_logits) * self.router_noise_std
        batch_size = x.size(0)
        fixed = self.fixed_experts
        fixed_k = len(fixed)
        dynamic_k = self.top_k - fixed_k
        if dynamic_k < 0:
            raise ValueError(f"fixed_k={fixed_k} cannot exceed top_k={self.top_k}")

        if dynamic_k > 0:
            masked_logits = routing_logits.clone()
            if fixed_k > 0:
                masked_logits[:, fixed] = float("-inf")
            _, dynamic_idx = torch.topk(masked_logits, k=dynamic_k, dim=1)
        else:
            dynamic_idx = x.new_zeros((batch_size, 0), dtype=torch.long)

        if fixed_k > 0:
            fixed_idx = torch.tensor(fixed, device=x.device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
            selected_idx = torch.cat([fixed_idx, dynamic_idx], dim=1)
        else:
            selected_idx = dynamic_idx

        selected_logits = routing_logits.gather(1, selected_idx)
        gates = F.softmax(selected_logits, dim=1)

        mixed = torch.zeros_like(x)
        active_experts = torch.unique(selected_idx)
        grad_tracked_idx_list: List[int] = []
        for expert_idx in active_experts.tolist():
            expert_mask = selected_idx.eq(expert_idx)
            token_ids, slot_ids = expert_mask.nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            if self.frozen_experts_no_grad and expert_idx in self.frozen_experts:
                with torch.no_grad():
                    expert_out = self.experts[expert_idx](x[token_ids].detach())
                expert_out = expert_out.detach()
            else:
                expert_out = self.experts[expert_idx](x[token_ids])
                grad_tracked_idx_list.extend([expert_idx] * token_ids.numel())
            weighted = expert_out * gates[token_ids, slot_ids].unsqueeze(-1)
            mixed.index_add_(0, token_ids, weighted)

        probs = F.softmax(routing_logits, dim=1)
        router_entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()
        importance = probs.mean(dim=0)
        uniform = torch.full_like(importance, 1.0 / self.num_experts)
        load_balance_loss = ((importance - uniform) ** 2).mean()

        if track_usage:
            grad_tracked_idx = torch.tensor(grad_tracked_idx_list, device=x.device, dtype=torch.long) if grad_tracked_idx_list else None
            self._update_usage(selected_idx, dynamic_idx, grad_tracked_idx)
        aux = {
            "selected_idx": selected_idx,
            "router_entropy": router_entropy,
            "load_balance_loss": load_balance_loss,
            "fixed_k": fixed_k,
            "dynamic_k": dynamic_k,
            "timing_router_score_sec": 0.0,
            "timing_topk_sec": 0.0,
            "timing_dynamic_softmax_sec": 0.0,
            "timing_moe_block_sec": 0.0,
        }
        return mixed, aux


class SingleLayerMoEScheme3(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        hidden_dim: int,
        fixed_experts: List[int] | None = None,
        fixed_branch_weights: List[float] | None = None,
        beta_fixed: float | None = None,
        beta_dynamic: float | None = None,
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.fixed_experts = list(fixed_experts or [])
        self.fixed_set = set(self.fixed_experts)
        self.non_fixed_experts = [i for i in range(num_experts) if i not in self.fixed_set]
        self.fixed_k = len(self.fixed_experts)
        self.dynamic_k = self.top_k - self.fixed_k
        if self.dynamic_k < 0:
            raise ValueError(f"fixed_k={self.fixed_k} cannot exceed top_k={self.top_k}")
        if self.dynamic_k > len(self.non_fixed_experts):
            raise ValueError(
                f"dynamic_k={self.dynamic_k} cannot exceed non_fixed_experts={len(self.non_fixed_experts)}"
            )
        if self.fixed_k > 0:
            if fixed_branch_weights is None or len(fixed_branch_weights) != self.fixed_k:
                raise ValueError("fixed_branch_weights must be provided and match fixed_k for scheme3")
            fixed_tensor = torch.tensor(fixed_branch_weights, dtype=torch.float32)
        else:
            fixed_tensor = torch.zeros(0, dtype=torch.float32)
        self.register_buffer("fixed_branch_weights", fixed_tensor)
        self.beta_fixed = float(beta_fixed) if beta_fixed is not None else (float(self.fixed_k) / float(self.top_k))
        self.beta_dynamic = float(beta_dynamic) if beta_dynamic is not None else (float(self.dynamic_k) / float(self.top_k))

        self.router_noise_std = float(router_noise_std)
        self.dynamic_candidate_count = len(self.non_fixed_experts)
        self.router = nn.Linear(dim, self.dynamic_candidate_count)
        self.experts = nn.ModuleList([MLPExpert(dim, hidden_dim) for _ in range(num_experts)])
        if self.dynamic_candidate_count > 0:
            self.register_buffer("non_fixed_lookup", torch.tensor(self.non_fixed_experts, dtype=torch.long))
        else:
            self.register_buffer("non_fixed_lookup", torch.zeros(0, dtype=torch.long))
        self.reset_epoch_usage()

    def set_frozen_experts(self, frozen_experts: List[int], no_grad_mode: bool):
        # scheme3 禁止使用 no_grad 切断 fixed branch 梯度到 stem
        if no_grad_mode:
            raise ValueError("scheme3 does not support no_grad_mode for fixed experts")
        return None

    def reset_epoch_usage(self):
        self.epoch_usage = {
            "moe_0": {
                "selection_counts": [0 for _ in range(self.num_experts)],
                "fixed_selection_counts": [0 for _ in range(self.num_experts)],
                "dynamic_selection_counts": [0 for _ in range(self.num_experts)],
                "selection_counts_note": "scheme3: selection_counts equals dynamic branch selections only",
            }
        }

    def consume_epoch_usage(self):
        usage = self.epoch_usage
        self.reset_epoch_usage()
        return usage

    def _update_dynamic_usage(self, dynamic_selected_idx: torch.Tensor):
        if dynamic_selected_idx.numel() == 0:
            return
        for idx in dynamic_selected_idx.detach().cpu().reshape(-1).tolist():
            self.epoch_usage["moe_0"]["selection_counts"][idx] += 1
            self.epoch_usage["moe_0"]["dynamic_selection_counts"][idx] += 1

    def forward(self, x, track_usage: bool = True):
        moe_t0 = time.perf_counter()
        batch_size = x.size(0)
        router_score_sec = 0.0
        topk_sec = 0.0
        dynamic_softmax_sec = 0.0

        # fixed branch: independent from router
        if self.fixed_k > 0:
            fixed_out = torch.zeros_like(x)
            for slot, expert_idx in enumerate(self.fixed_experts):
                w = self.fixed_branch_weights[slot]
                fixed_out = fixed_out + w * self.experts[expert_idx](x)
        else:
            fixed_out = torch.zeros_like(x)

        # dynamic branch: router only on non-fixed pool (no fixed logits are computed)
        if self.dynamic_k > 0:
            t_router0 = time.perf_counter()
            dynamic_logits_pool = self.router(x)
            if self.training and self.router_noise_std > 0:
                dynamic_logits_pool = dynamic_logits_pool + torch.randn_like(dynamic_logits_pool) * self.router_noise_std
            router_score_sec = time.perf_counter() - t_router0

            t_topk0 = time.perf_counter()
            dynamic_topk_logits, dynamic_topk_pool_idx = torch.topk(dynamic_logits_pool, k=self.dynamic_k, dim=1)
            topk_sec = time.perf_counter() - t_topk0

            t_softmax0 = time.perf_counter()
            dynamic_gates = F.softmax(dynamic_topk_logits, dim=1)
            dynamic_softmax_sec = time.perf_counter() - t_softmax0

            dynamic_selected_idx = self.non_fixed_lookup.to(x.device)[dynamic_topk_pool_idx]

            dynamic_out = torch.zeros_like(x)
            for slot in range(self.dynamic_k):
                idx = dynamic_selected_idx[:, slot]
                gate = dynamic_gates[:, slot].unsqueeze(-1)
                for expert_idx in idx.unique().tolist():
                    token_mask = idx.eq(expert_idx)
                    token_ids = token_mask.nonzero(as_tuple=False).squeeze(1)
                    expert_out = self.experts[expert_idx](x[token_ids])
                    dynamic_out.index_add_(0, token_ids, expert_out * gate[token_ids])

            dynamic_importance = F.softmax(dynamic_logits_pool, dim=1).mean(dim=0)
            uniform = torch.full_like(dynamic_importance, 1.0 / dynamic_importance.numel())
            load_balance_loss = ((dynamic_importance - uniform) ** 2).mean()
            router_entropy = -(dynamic_importance * (dynamic_importance.clamp_min(1e-8).log())).sum()
        else:
            dynamic_selected_idx = x.new_zeros((batch_size, 0), dtype=torch.long)
            dynamic_out = torch.zeros_like(x)
            load_balance_loss = x.new_tensor(0.0)
            router_entropy = x.new_tensor(0.0)

        mixed = self.beta_fixed * fixed_out + self.beta_dynamic * dynamic_out

        if track_usage:
            self._update_dynamic_usage(dynamic_selected_idx)
            for idx in self.fixed_experts:
                self.epoch_usage["moe_0"]["fixed_selection_counts"][idx] += batch_size

        moe_block_sec = time.perf_counter() - moe_t0
        aux = {
            "fixed_experts": list(self.fixed_experts),
            "fixed_branch_weights": self.fixed_branch_weights.detach().cpu().tolist(),
            "fixed_expert_weight_pairs": list(zip(self.fixed_experts, self.fixed_branch_weights.detach().cpu().tolist())),
            "dynamic_selected_idx": dynamic_selected_idx,
            "fixed_k": self.fixed_k,
            "dynamic_k": self.dynamic_k,
            "beta_fixed": self.beta_fixed,
            "beta_dynamic": self.beta_dynamic,
            "branch_fusion_source": "source_task_frequency",
            "router_entropy": router_entropy,
            "load_balance_loss": load_balance_loss,
            "timing_router_score_sec": router_score_sec,
            "timing_topk_sec": topk_sec,
            "timing_dynamic_softmax_sec": dynamic_softmax_sec,
            "timing_moe_block_sec": moe_block_sec,
        }
        return mixed, aux


class LiteCNNMoEClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_dim: int,
        num_experts: int,
        top_k: int,
        fixed_experts: List[int],
        hidden_dim: int,
        router_noise_std: float,
        num_classes: int,
        routing_mode: str = "legacy_hybrid",
        fixed_branch_weights: List[float] | None = None,
        beta_fixed: float | None = None,
        beta_dynamic: float | None = None,
    ):
        super().__init__()
        self.stem = LiteCNNStem(in_channels=in_channels, feature_dim=feature_dim)
        if routing_mode == "fixed_branch_dynamic_branch":
            self.moe = SingleLayerMoEScheme3(
                dim=feature_dim,
                num_experts=num_experts,
                top_k=top_k,
                hidden_dim=hidden_dim,
                fixed_experts=fixed_experts,
                fixed_branch_weights=fixed_branch_weights,
                beta_fixed=beta_fixed,
                beta_dynamic=beta_dynamic,
                router_noise_std=router_noise_std,
            )
        else:
            self.moe = SingleLayerMoE(
                dim=feature_dim,
                num_experts=num_experts,
                top_k=top_k,
                hidden_dim=hidden_dim,
                fixed_experts=fixed_experts,
                router_noise_std=router_noise_std,
            )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def consume_epoch_usage(self):
        return self.moe.consume_epoch_usage()

    def forward(self, x, track_usage: bool = True):
        feats = self.stem(x)
        moe_out, aux = self.moe(feats, track_usage=track_usage)
        logits = self.classifier(moe_out)
        return logits, aux
