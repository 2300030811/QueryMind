"""
Benchmark Runner — Evaluates QueryMind agent against baselines on TPC-H.

Baselines:
    1. PostgreSQL default planner (all knobs ON)
    2. PostgreSQL with GEQO disabled (exhaustive search)
    3. Random action baseline (sanity check)

Metrics reported:
    - Geometric mean of latency ratio across test queries
    - % queries where agent beats PostgreSQL default
    - % queries where agent beats GEQO-disabled PostgreSQL
    - Worst-case overhead (max slowdown)
    - Per-query latency breakdown

Usage:
    querymind-bench --db-url postgresql://... --model-path checkpoints/querymind_final
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine, text

from querymind.benchmark.tpch_loader import TPCHLoader
from querymind.env.hint_builder import HintBuilder
from querymind.featurizer.encoder import QueryFeatureEncoder

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class QueryResult:
    """Result of running a single query under a specific configuration."""

    query_id: str
    latency_ms: float
    config_name: str
    action: int | None = None
    success: bool = True


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results across all queries and baselines."""

    query_results: list[QueryResult] = field(default_factory=list)

    def get_latencies(self, config_name: str) -> dict[str, float]:
        """Get latency map for a specific config."""
        return {
            r.query_id: r.latency_ms
            for r in self.query_results
            if r.config_name == config_name and r.success
        }

    def compute_ratios(
        self, baseline: str = "pg_default", agent: str = "querymind"
    ) -> dict[str, float]:
        """Compute latency ratios (baseline / agent) per query."""
        bl = self.get_latencies(baseline)
        ag = self.get_latencies(agent)
        ratios = {}
        for qid in bl:
            if qid in ag and ag[qid] > 0:
                ratios[qid] = bl[qid] / ag[qid]
        return ratios

    def geometric_mean_ratio(
        self, baseline: str = "pg_default", agent: str = "querymind"
    ) -> float:
        """Geometric mean of latency ratios."""
        ratios = self.compute_ratios(baseline, agent)
        if not ratios:
            return 0.0
        return float(np.exp(np.mean(np.log(list(ratios.values())))))

    def win_rate(
        self, baseline: str = "pg_default", agent: str = "querymind"
    ) -> float:
        """Fraction of queries where agent beats baseline."""
        ratios = self.compute_ratios(baseline, agent)
        if not ratios:
            return 0.0
        return float(np.mean([r > 1.0 for r in ratios.values()]))

    def worst_case_overhead(
        self, baseline: str = "pg_default", agent: str = "querymind"
    ) -> float:
        """Maximum slowdown (ratio < 1 means agent is slower)."""
        ratios = self.compute_ratios(baseline, agent)
        if not ratios:
            return 0.0
        return float(min(ratios.values()))


class BenchmarkRunner:
    """Runs QueryMind agent against baselines on TPC-H queries.

    Executes each query multiple times (with warm-up) under different
    planner configurations and compares latencies.

    Args:
        db_url: PostgreSQL connection string.
        model_path: Path to trained PPO model checkpoint.
        n_runs: Number of times to execute each query (for stable timing).
        warmup_runs: Number of warmup executions (results discarded).
    """

    def __init__(
        self,
        db_url: str,
        model_path: str | None = None,
        n_runs: int = 5,
        warmup_runs: int = 2,
    ) -> None:
        self._db_url = db_url
        self._engine = create_engine(db_url, pool_size=1, max_overflow=0)
        self._hint_builder = HintBuilder()
        self._encoder = QueryFeatureEncoder(db_url=db_url)
        self._model_path = model_path
        self._model: Any = None
        self._n_runs = n_runs
        self._warmup_runs = warmup_runs
        self._loader = TPCHLoader()

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load trained PPO model from checkpoint."""
        try:
            from stable_baselines3 import PPO

            self._model = PPO.load(path)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def run(self, query_ids: list[str] | None = None) -> BenchmarkResults:
        """Run the full benchmark suite.

        Args:
            query_ids: Specific query IDs to benchmark. If None, uses test set.

        Returns:
            BenchmarkResults with all query results.
        """
        if query_ids is None:
            queries, query_ids = self._loader.get_test_queries()
        else:
            queries, query_ids = self._loader.get_queries(query_ids)

        results = BenchmarkResults()
        console.print(f"\n[bold cyan]Running benchmark on {len(queries)} queries[/bold cyan]\n")

        for sql, qid in zip(queries, query_ids):
            console.print(f"[yellow]Benchmarking {qid}...[/yellow]")

            # ── Baseline 1: PostgreSQL default ──────────────────────────────
            default_latency = self._run_query_timed(sql, "pg_default")
            results.query_results.append(
                QueryResult(qid, default_latency, "pg_default")
            )

            # ── Baseline 2: GEQO disabled ──────────────────────────────────
            geqo_latency = self._run_with_geqo_disabled(sql)
            results.query_results.append(
                QueryResult(qid, geqo_latency, "pg_no_geqo")
            )

            # ── Baseline 3: Random action ───────────────────────────────────
            random_action = int(np.random.choice(self._hint_builder.valid_actions))
            random_latency = self._run_with_action(sql, random_action)
            results.query_results.append(
                QueryResult(qid, random_latency, "random", action=random_action)
            )

            # ── QueryMind agent ─────────────────────────────────────────────
            if self._model is not None:
                obs = self._encoder.encode(sql)
                action, _ = self._model.predict(obs, deterministic=True)
                agent_latency = self._run_with_action(sql, int(action))
                results.query_results.append(
                    QueryResult(qid, agent_latency, "querymind", action=int(action))
                )
            else:
                logger.warning("No model loaded — skipping agent evaluation")

        return results

    def _run_query_timed(self, sql: str, config_name: str) -> float:
        """Execute a query with default planner and return median latency."""
        latencies = []
        try:
            with self._engine.connect() as conn:
                # Reset to defaults
                for stmt in self._hint_builder.get_reset_statements():
                    conn.execute(text(stmt))

                # Warmup
                for _ in range(self._warmup_runs):
                    conn.execute(text(sql))

                # Timed runs
                for _ in range(self._n_runs):
                    start = time.perf_counter()
                    conn.execute(text(sql))
                    latencies.append((time.perf_counter() - start) * 1000.0)

                conn.commit()
        except Exception as e:
            logger.error(f"Query execution failed ({config_name}): {e}")
            return float("inf")

        return float(np.median(latencies))

    def _run_with_geqo_disabled(self, sql: str) -> float:
        """Execute with GEQO disabled (exhaustive optimizer search)."""
        try:
            with self._engine.connect() as conn:
                for stmt in self._hint_builder.get_reset_statements():
                    conn.execute(text(stmt))
                conn.execute(text("SET geqo = OFF;"))

                # Warmup
                for _ in range(self._warmup_runs):
                    conn.execute(text(sql))

                latencies = []
                for _ in range(self._n_runs):
                    start = time.perf_counter()
                    conn.execute(text(sql))
                    latencies.append((time.perf_counter() - start) * 1000.0)

                conn.execute(text("SET geqo = ON;"))
                conn.commit()
                return float(np.median(latencies))
        except Exception as e:
            logger.error(f"GEQO-disabled execution failed: {e}")
            return float("inf")

    def _run_with_action(self, sql: str, action: int) -> float:
        """Execute query with a specific planner knob configuration."""
        config = self._hint_builder.decode_action(action)
        try:
            with self._engine.connect() as conn:
                for stmt in config.to_set_statements():
                    conn.execute(text(stmt))

                # Warmup
                for _ in range(self._warmup_runs):
                    conn.execute(text(sql))

                latencies = []
                for _ in range(self._n_runs):
                    start = time.perf_counter()
                    conn.execute(text(sql))
                    latencies.append((time.perf_counter() - start) * 1000.0)

                # Reset
                for stmt in self._hint_builder.get_reset_statements():
                    conn.execute(text(stmt))
                conn.commit()
                return float(np.median(latencies))
        except Exception as e:
            logger.error(f"Action {action} execution failed: {e}")
            return float("inf")

    def close(self) -> None:
        """Dispose the database engine."""
        self._engine.dispose()


def print_results(results: BenchmarkResults) -> None:
    """Print a rich-formatted results table to the console."""
    table = Table(title="QueryMind Benchmark Results", show_lines=True)
    table.add_column("Query", style="cyan", justify="center")
    table.add_column("PG Default (ms)", justify="right")
    table.add_column("PG No-GEQO (ms)", justify="right")
    table.add_column("Random (ms)", justify="right")
    table.add_column("QueryMind (ms)", justify="right", style="bold green")
    table.add_column("Speedup", justify="right", style="bold yellow")

    pg_latencies = results.get_latencies("pg_default")
    geqo_latencies = results.get_latencies("pg_no_geqo")
    random_latencies = results.get_latencies("random")
    agent_latencies = results.get_latencies("querymind")

    all_query_ids = sorted(pg_latencies.keys())

    for qid in all_query_ids:
        pg = pg_latencies.get(qid, float("inf"))
        geqo = geqo_latencies.get(qid, float("inf"))
        rand = random_latencies.get(qid, float("inf"))
        agent = agent_latencies.get(qid, float("inf"))
        speedup = pg / agent if agent > 0 else 0

        speedup_str = f"{speedup:.2f}x"
        if speedup > 1.0:
            speedup_str = f"[green]{speedup_str}[/green]"
        elif speedup < 1.0:
            speedup_str = f"[red]{speedup_str}[/red]"

        table.add_row(
            qid,
            f"{pg:.1f}",
            f"{geqo:.1f}",
            f"{rand:.1f}",
            f"{agent:.1f}" if agent != float("inf") else "N/A",
            speedup_str,
        )

    # Summary row
    gmean = results.geometric_mean_ratio()
    win = results.win_rate()
    worst = results.worst_case_overhead()

    console.print(table)
    console.print(f"\n[bold]Geometric Mean Speedup:[/bold] [green]{gmean:.3f}x[/green]")
    console.print(f"[bold]Win Rate:[/bold] [green]{win:.1%}[/green]")
    console.print(f"[bold]Worst-Case Ratio:[/bold] [yellow]{worst:.3f}x[/yellow]\n")


# ── CLI Entry Point ─────────────────────────────────────────────────────────
app = typer.Typer(
    name="querymind-bench",
    help="Benchmark QueryMind agent against PostgreSQL baselines.",
)


@app.command()
def main(
    db_url: str = typer.Option(
        "postgresql://querymind:querymind@localhost:5434/tpch",
        help="PostgreSQL connection string",
    ),
    model_path: str = typer.Option(
        "checkpoints/querymind_final",
        help="Path to trained PPO model",
    ),
    n_runs: int = typer.Option(5, help="Number of timed runs per query"),
    warmup: int = typer.Option(2, help="Number of warmup runs"),
    query_ids: str | None = typer.Option(
        None,
        help="Comma-separated query IDs (e.g., Q3,Q5,Q10). Defaults to test set.",
    ),
) -> None:
    """Run the QueryMind benchmark suite."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    runner = BenchmarkRunner(
        db_url=db_url,
        model_path=model_path,
        n_runs=n_runs,
        warmup_runs=warmup,
    )

    ids = query_ids.split(",") if query_ids else None
    results = runner.run(query_ids=ids)
    print_results(results)
    runner.close()


if __name__ == "__main__":
    app()
