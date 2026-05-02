"""
Benchmark Report Generator — Produces formatted metrics tables and charts.

Generates:
    - Markdown report with per-query latency breakdown
    - CSV export for further analysis
    - Summary statistics (geometric mean, win rate, worst-case)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from querymind.benchmark.runner import BenchmarkResults

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates benchmark reports from BenchmarkResults.

    Args:
        output_dir: Directory to save report files.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(
        self,
        results: BenchmarkResults,
        title: str = "QueryMind Benchmark Report",
    ) -> str:
        """Generate a markdown report.

        Args:
            results: BenchmarkResults from a benchmark run.
            title: Report title.

        Returns:
            Markdown string.
        """
        ratios = results.compute_ratios()
        gmean = results.geometric_mean_ratio()
        win_rate = results.win_rate()
        worst = results.worst_case_overhead()

        pg_lat = results.get_latencies("pg_default")
        agent_lat = results.get_latencies("querymind")
        geqo_lat = results.get_latencies("pg_no_geqo")

        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Geometric Mean Speedup | **{gmean:.3f}x** |",
            f"| Win Rate (vs PG Default) | **{win_rate:.1%}** |",
            f"| Worst-Case Ratio | {worst:.3f}x |",
            f"| Queries Evaluated | {len(ratios)} |",
            "",
            "## Per-Query Results",
            "",
            "| Query | PG Default (ms) | PG No-GEQO (ms) | QueryMind (ms) | Speedup |",
            "|-------|-----------------|------------------|----------------|---------|",
        ]

        for qid in sorted(pg_lat.keys()):
            pg = pg_lat.get(qid, 0)
            geqo = geqo_lat.get(qid, 0)
            agent = agent_lat.get(qid, 0)
            ratio = ratios.get(qid, 0)

            emoji = "✅" if ratio > 1.0 else "❌" if ratio < 1.0 else "➖"
            lines.append(
                f"| {qid} | {pg:.1f} | {geqo:.1f} | {agent:.1f} | {emoji} {ratio:.2f}x |"
            )

        lines.extend([
            "",
            "## Methodology",
            "",
            "- **Database**: PostgreSQL 17 with pg_hint_plan 1.7.1",
            "- **Dataset**: TPC-H SF=1 (1GB)",
            "- **Agent**: PPO (Stable-Baselines3), Discrete(64) action space",
            "- **Reward**: baseline_latency / agent_latency, clipped to [-2, 5]",
            f"- **Timing**: Median of 5 runs after 2 warmup executions",
            "",
            "## References",
            "",
            "1. Bao: Learning to Steer Query Optimizers (Marcus et al., 2021)",
            "2. DQ: Deep Reinforcement Learning for Join Order Selection (2019)",
            "3. Neo: A Learned Query Optimizer (2021)",
            "4. Marcus et al., Towards a Learned Query Optimizer (CIDR 2019)",
        ])

        report = "\n".join(lines)

        # Save to file
        report_path = self._output_dir / "benchmark_report.md"
        report_path.write_text(report)
        logger.info(f"Markdown report saved to {report_path}")

        return report

    def generate_csv(self, results: BenchmarkResults) -> str:
        """Generate CSV export of benchmark results.

        Returns:
            Path to the saved CSV file.
        """
        csv_path = self._output_dir / "benchmark_results.csv"

        with open(csv_path, "w") as f:
            f.write("query_id,config,latency_ms,action,success\n")
            for r in results.query_results:
                action_str = str(r.action) if r.action is not None else ""
                f.write(
                    f"{r.query_id},{r.config_name},{r.latency_ms:.2f},"
                    f"{action_str},{r.success}\n"
                )

        logger.info(f"CSV export saved to {csv_path}")
        return str(csv_path)
