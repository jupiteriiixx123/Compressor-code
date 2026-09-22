from __future__ import annotations

import torch


class GRPOAdvantage:
    """Weighted reward sum, then group normalization."""

    def __init__(self, weights, eps: float = 1e-6):
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.eps = eps

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        rewards: [batch, group, num_rewards]
        return:  [batch, group]
        """
        weights = self.weights.to(rewards.device, rewards.dtype)
        total = (rewards * weights).sum(dim=-1)

        mean = total.mean(dim=1, keepdim=True)
        std = total.std(dim=1, keepdim=True, unbiased=True)
        return (total - mean) / (std + self.eps)


class GDPOAdvantage:
    """Per-reward group normalization, weighted sum, then batch normalization."""

    def __init__(self, weights, eps: float = 1e-6):
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.eps = eps

    def __call__(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        rewards: [batch, group, num_rewards]
        return:  [batch, group]
        """
        weights = self.weights.to(rewards.device, rewards.dtype)

        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True, unbiased=True)
        per_reward_adv = (rewards - mean) / (std + self.eps)

        advantage = (per_reward_adv * weights).sum(dim=-1)

        batch_mean = advantage.mean()
        batch_std = advantage.std(unbiased=True)
        return (advantage - batch_mean) / (batch_std + self.eps)
