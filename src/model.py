from __future__ import annotations

from typing import Dict, List, Tuple

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
        self.fixed_experts = sorted(fixed_experts or [])
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
            }
        }

    def consume_epoch_usage(self):
        usage = self.epoch_usage
        self.reset_epoch_usage()
        return usage

    def _update_usage(self, selected_idx: torch.Tensor, dynamic_idx: torch.Tensor | None):
        flat_selected = selected_idx.detach().cpu().reshape(-1).tolist()
        for idx in flat_selected:
            self.epoch_usage["moe_0"]["selection_counts"][idx] += 1
        for idx in self.fixed_experts:
            self.epoch_usage["moe_0"]["fixed_selection_counts"][idx] += selected_idx.size(0)
        if dynamic_idx is not None and dynamic_idx.numel() > 0:
            for idx in dynamic_idx.detach().cpu().reshape(-1).tolist():
                self.epoch_usage["moe_0"]["dynamic_selection_counts"][idx] += 1

    def forward(self, x, track_usage: bool = True):
        logits = self.router(x)  # [B, E]
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
            weighted = expert_out * gates[token_ids, slot_ids].unsqueeze(-1)
            mixed.index_add_(0, token_ids, weighted)

        probs = F.softmax(routing_logits, dim=1)
        router_entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean()
        importance = probs.mean(dim=0)
        uniform = torch.full_like(importance, 1.0 / self.num_experts)
        load_balance_loss = ((importance - uniform) ** 2).mean()

        if track_usage:
            self._update_usage(selected_idx, dynamic_idx)
        aux = {
            "selected_idx": selected_idx,
            "router_entropy": router_entropy,
            "load_balance_loss": load_balance_loss,
            "fixed_k": fixed_k,
            "dynamic_k": dynamic_k,
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
    ):
        super().__init__()
        self.stem = LiteCNNStem(in_channels=in_channels, feature_dim=feature_dim)
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
