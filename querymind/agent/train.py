"""
PPO Training Loop for QueryMind.

Trains a Proximal Policy Optimization agent to optimize PostgreSQL
query execution plans on TPC-H workload.

Training config (from architecture spec):
    n_steps:          2048
    batch_size:       64
    learning_rate:    3e-4 with linear decay
    total_timesteps:  500,000
    ent_coef:         0.01 (encourage exploration)
    gamma:            1.0 (single-step episodes)
    n_epochs:         10
    clip_range:       0.2

Usage:
    # From CLI:
    querymind-train --db-url postgresql://user:pass@localhost/tpch

    # From Python:
    from querymind.agent.train import train
    model = train(db_url="postgresql://...", total_timesteps=500_000)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import typer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from querymind.agent.callbacks import QueryMindCallback, make_eval_callback
from querymind.agent.policy import QueryMindPolicy
from querymind.benchmark.tpch_loader import TPCHLoader
from querymind.env.query_plan_env import QueryPlanEnv

logger = logging.getLogger(__name__)

# ── Default training hyperparameters ────────────────────────────────────────
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 1.0,  # single-step episodes → no discounting
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # entropy bonus for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}


def _linear_schedule(initial_lr: float) -> Any:
    """Linear learning rate decay schedule.

    Returns a function that takes progress_remaining ∈ [1, 0]
    and returns the current learning rate.
    """

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr

    return schedule


def train(
    db_url: str,
    total_timesteps: int = 500_000,
    output_dir: str = "checkpoints",
    log_dir: str = "logs",
    use_wandb: bool = True,
    seed: int = 42,
    config_overrides: dict[str, Any] | None = None,
) -> PPO:
    """Train a PPO agent to optimize PostgreSQL query plans.

    Args:
        db_url: PostgreSQL connection string (e.g., postgresql://user:pass@host/tpch).
        total_timesteps: Total training steps.
        output_dir: Directory to save model checkpoints.
        log_dir: Directory for training logs.
        use_wandb: Whether to log to Weights & Biases.
        seed: Random seed for reproducibility.
        config_overrides: Override default hyperparameters.

    Returns:
        Trained PPO model.
    """
    # ── Setup directories ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Load TPC-H queries ──────────────────────────────────────────────────
    loader = TPCHLoader()
    train_queries, train_ids = loader.get_train_queries()
    eval_queries, eval_ids = loader.get_test_queries()

    logger.info(f"Loaded {len(train_queries)} train / {len(eval_queries)} eval queries")

    # ── Create environments ─────────────────────────────────────────────────
    train_env = QueryPlanEnv(
        db_url=db_url,
        queries=train_queries,
        query_ids=train_ids,
        seed=seed,
    )
    eval_env = QueryPlanEnv(
        db_url=db_url,
        queries=eval_queries,
        query_ids=eval_ids,
        seed=seed + 1000,
    )

    # ── Build config ────────────────────────────────────────────────────────
    config = DEFAULT_CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)

    # Apply learning rate schedule
    initial_lr = config.pop("learning_rate", 3e-4)
    config["learning_rate"] = _linear_schedule(initial_lr)

    # ── Create PPO model ────────────────────────────────────────────────────
    model = PPO(
        policy=QueryMindPolicy,
        env=train_env,
        seed=seed,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        **config,
    )

    logger.info(f"PPO model created: {model.policy}")
    logger.info(f"Training for {total_timesteps} timesteps")

    # ── Setup callbacks ─────────────────────────────────────────────────────
    callbacks = CallbackList(
        [
            QueryMindCallback(
                log_dir=log_dir,
                log_freq=50,
                use_wandb=use_wandb,
            ),
            make_eval_callback(
                eval_env=eval_env,
                log_dir=log_dir,
                eval_freq=5000,
                n_eval_episodes=len(eval_queries),
            ),
        ]
    )

    # ── Train ───────────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # ── Save final model ────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "querymind_final")
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    train_env.close()
    eval_env.close()

    return model


# ── CLI Entry Point ─────────────────────────────────────────────────────────
app = typer.Typer(
    name="querymind-train",
    help="Train the QueryMind RL agent for SQL query optimization.",
)


@app.command()
def main(
    db_url: str = typer.Option(
        "postgresql://querymind:querymind@localhost:5434/tpch",
        help="PostgreSQL connection string",
    ),
    total_timesteps: int = typer.Option(
        500_000,
        help="Total training timesteps",
    ),
    output_dir: str = typer.Option(
        "checkpoints",
        help="Directory for model checkpoints",
    ),
    log_dir: str = typer.Option(
        "logs",
        help="Directory for training logs",
    ),
    use_wandb: bool = typer.Option(
        True,
        help="Log to Weights & Biases",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed",
    ),
) -> None:
    """Train the QueryMind PPO agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    train(
        db_url=db_url,
        total_timesteps=total_timesteps,
        output_dir=output_dir,
        log_dir=log_dir,
        use_wandb=use_wandb,
        seed=seed,
    )


if __name__ == "__main__":
    app()
