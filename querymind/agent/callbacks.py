"""
Training callbacks for QueryMind — W&B logging, evaluation, and checkpointing.

Integrates with Weights & Biases for experiment tracking and provides
periodic evaluation against the test query set.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

logger = logging.getLogger(__name__)


class QueryMindCallback(BaseCallback):
    """Custom callback for logging QueryMind training metrics.

    Tracks per-episode:
    - Latency ratio (agent vs baseline)
    - Action distribution (which knob configs are chosen)
    - Query-level performance breakdown

    Optionally logs to Weights & Biases if wandb is available and initialized.

    Args:
        log_dir: Directory to save CSV logs.
        log_freq: Log aggregated metrics every N episodes.
        use_wandb: Whether to log to Weights & Biases.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_freq: int = 50,
        use_wandb: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self._log_dir = Path(log_dir)
        self._log_freq = log_freq
        self._use_wandb = use_wandb

        # Episode tracking
        self._episode_rewards: list[float] = []
        self._episode_ratios: list[float] = []
        self._episode_actions: list[int] = []
        self._query_performance: dict[str, list[float]] = {}
        self._episode_count = 0

        # W&B import (lazy)
        self._wandb: Any = None

    def _init_callback(self) -> bool:
        """Initialize logging directory and W&B."""
        self._log_dir.mkdir(parents=True, exist_ok=True)

        if self._use_wandb:
            try:
                import wandb

                self._wandb = wandb
                if not wandb.run:
                    wandb.init(
                        project="querymind",
                        config={
                            "algorithm": "PPO",
                            "policy": "QueryMindPolicy",
                            "total_timesteps": self.model.num_timesteps if self.model else 0,
                        },
                        dir=str(self._log_dir),
                    )
                logger.info("W&B logging initialized")
            except ImportError:
                logger.warning("wandb not installed — skipping W&B logging")
                self._use_wandb = False
            except Exception as e:
                logger.warning(f"W&B init failed: {e} — skipping")
                self._use_wandb = False

        return True

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if episode completed (infos available)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward" in info:
                self._episode_count += 1
                self._episode_rewards.append(info["reward"])

                ratio = info.get("latency_ratio", 0.0)
                self._episode_ratios.append(ratio)

                action = info.get("action", -1)
                self._episode_actions.append(action)

                # Track per-query performance
                query_id = info.get("query_id", "unknown")
                if query_id not in self._query_performance:
                    self._query_performance[query_id] = []
                self._query_performance[query_id].append(ratio)

                # Periodic logging
                if self._episode_count % self._log_freq == 0:
                    self._log_metrics()

        return True

    def _log_metrics(self) -> None:
        """Compute and log aggregated metrics."""
        if not self._episode_rewards:
            return

        recent_rewards = self._episode_rewards[-self._log_freq :]
        recent_ratios = self._episode_ratios[-self._log_freq :]
        recent_actions = self._episode_actions[-self._log_freq :]

        metrics = {
            "episode": self._episode_count,
            "reward/mean": float(np.mean(recent_rewards)),
            "reward/std": float(np.std(recent_rewards)),
            "reward/max": float(np.max(recent_rewards)),
            "reward/min": float(np.min(recent_rewards)),
            "latency_ratio/mean": float(np.mean(recent_ratios)),
            "latency_ratio/median": float(np.median(recent_ratios)),
            "latency_ratio/pct_win": float(np.mean(np.array(recent_ratios) > 1.0)),
            "action/unique_count": len(set(recent_actions)),
            "action/most_common": int(max(set(recent_actions), key=recent_actions.count)),
        }

        if self.verbose >= 1:
            logger.info(
                f"[Episode {self._episode_count}] "
                f"reward={metrics['reward/mean']:.3f} "
                f"ratio={metrics['latency_ratio/mean']:.3f} "
                f"win_rate={metrics['latency_ratio/pct_win']:.1%}"
            )

        # Log to W&B
        if self._use_wandb and self._wandb and self._wandb.run:
            self._wandb.log(metrics, step=self._episode_count)

    def _on_training_end(self) -> None:
        """Log final summary metrics."""
        if self._episode_ratios:
            final_metrics = {
                "final/mean_ratio": float(np.mean(self._episode_ratios)),
                "final/median_ratio": float(np.median(self._episode_ratios)),
                "final/win_rate": float(
                    np.mean(np.array(self._episode_ratios) > 1.0)
                ),
                "final/total_episodes": self._episode_count,
            }
            logger.info(f"Training complete: {final_metrics}")

            if self._use_wandb and self._wandb and self._wandb.run:
                self._wandb.log(final_metrics)
                self._wandb.finish()

        # Save per-query performance summary
        self._save_query_report()

    def _save_query_report(self) -> None:
        """Save per-query performance breakdown to CSV."""
        report_path = self._log_dir / "query_performance.csv"
        try:
            with open(report_path, "w") as f:
                f.write("query_id,mean_ratio,median_ratio,n_episodes,win_rate\n")
                for qid, ratios in sorted(self._query_performance.items()):
                    arr = np.array(ratios)
                    f.write(
                        f"{qid},"
                        f"{np.mean(arr):.4f},"
                        f"{np.median(arr):.4f},"
                        f"{len(arr)},"
                        f"{np.mean(arr > 1.0):.4f}\n"
                    )
            logger.info(f"Query performance report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save query report: {e}")


def make_eval_callback(
    eval_env: Any,
    log_dir: str = "logs",
    eval_freq: int = 5000,
    n_eval_episodes: int = 20,
) -> EvalCallback:
    """Create an SB3 EvalCallback for periodic evaluation.

    Args:
        eval_env: Gymnasium environment for evaluation.
        log_dir: Directory for evaluation logs.
        eval_freq: Evaluate every N timesteps.
        n_eval_episodes: Number of episodes per evaluation round.

    Returns:
        Configured EvalCallback instance.
    """
    eval_log_dir = os.path.join(log_dir, "eval")
    os.makedirs(eval_log_dir, exist_ok=True)

    return EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=eval_log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
    )
