"""
Custom MLP policy network for QueryMind PPO agent.

Architecture:
    Input (128 dims) → [256 → 128 → 64] shared MLP with ReLU
    ├── Actor head  → logits over Discrete(64) action space
    └── Critic head → scalar value estimate

Uses Stable-Baselines3's ActorCriticPolicy with custom network architecture.
"""

from __future__ import annotations

from typing import Any

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class QueryMindNetwork(nn.Module):
    """Custom shared feature extractor for the QueryMind agent.

    Three-layer MLP with LayerNorm for training stability:
        128 → 256 (ReLU, LayerNorm)
        256 → 128 (ReLU, LayerNorm)
        128 → 64  (ReLU)

    The output feeds into separate actor and critic heads
    managed by SB3's ActorCriticPolicy.
    """

    def __init__(self, feature_dim: int, last_layer_dim: int = 64) -> None:
        super().__init__()

        self.latent_dim_pi = last_layer_dim  # actor output dim
        self.latent_dim_vf = last_layer_dim  # critic output dim

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, last_layer_dim),
            nn.ReLU(),
        )

        # Separate policy and value heads (after shared trunk)
        self.policy_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(last_layer_dim, last_layer_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through shared network, then split for actor/critic.

        Args:
            observations: Batch of observation vectors, shape (B, feature_dim).

        Returns:
            (policy_features, value_features) — both shape (B, last_layer_dim).
        """
        shared_features = self.shared_net(observations)
        return self.policy_net(shared_features), self.value_net(shared_features)

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor only."""
        shared_features = self.shared_net(observations)
        return self.policy_net(shared_features)

    def forward_critic(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic only."""
        shared_features = self.shared_net(observations)
        return self.value_net(shared_features)


class QueryMindPolicy(ActorCriticPolicy):
    """Custom ActorCriticPolicy using the QueryMind network architecture.

    Drop-in replacement for SB3's MlpPolicy with domain-specific
    network sizing and LayerNorm for stable training.

    Usage:
        model = PPO(QueryMindPolicy, env, ...)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Override net_arch to use our custom network
        kwargs["net_arch"] = []  # We handle architecture in QueryMindNetwork
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Override to use QueryMindNetwork instead of default MLP."""
        self.mlp_extractor = QueryMindNetwork(feature_dim=self.features_dim)
