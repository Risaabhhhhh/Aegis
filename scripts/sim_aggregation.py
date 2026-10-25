#!/usr/bin/env python3
"""Monte Carlo simulation comparing three aggregation strategies under independent engine failures.

Three dummy engines:
- Engine 1: true score 0.9, weight 40.0
- Engine 2: true score 0.7, weight 35.0
- Engine 3: true score 0.8, weight 25.0

No-failure aggregate:
S* = (0.9 * 40 + 0.7 * 35 + 0.8 * 25) / (40 + 35 + 25) = 80.5 / 100 = 0.805

Aggregators compared against S*:
(a) AMEVA renormalization (renormalizes weights across surviving engines; 0.0 if all fail)
(b) Failed engine imputed as 0 (full weight denominator, failed engine score = 0)
(c) AMEVA aggregate multiplied by coverage (AMEVA * coverage)

Settings:
p in {0.1, 0.2, 0.3, 0.5}, 10,000 trials per setting.
Results saved to results/sim_aggregation.json and results/sim_aggregation.csv.
"""

import os
import sys
import json
import csv
import random
import argparse
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base import ActionPlan, EngineResult
from ameva.aggregator import calculate_aggregate
from ameva.coverage import calculate_coverage

DEFAULT_SEED = 42
DEFAULT_TRIALS = 10000
FAILURE_PROBABILITIES = [0.1, 0.2, 0.3, 0.5]

ENGINES = [
    {"name": "engine_1", "true_score": 0.9, "weight": 40.0},
    {"name": "engine_2", "true_score": 0.7, "weight": 35.0},
    {"name": "engine_3", "true_score": 0.8, "weight": 25.0},
]


def run_simulation(
    num_trials: int = DEFAULT_TRIALS,
    failure_probs: list[float] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], float]:
    if failure_probs is None:
        failure_probs = FAILURE_PROBABILITIES

    # Seed RNGs for full reproducibility
    random.seed(seed)
    np.random.seed(seed)

    engine_names = [e["name"] for e in ENGINES]
    true_scores = np.array([e["true_score"] for e in ENGINES], dtype=float)
    weights = np.array([e["weight"] for e in ENGINES], dtype=float)
    total_weight = float(np.sum(weights))

    # True aggregate without failures
    true_aggregate = float(np.sum(true_scores * weights) / total_weight)

    plan = ActionPlan(
        request_id="monte-carlo-sim",
        selected_engines=engine_names,
        weights={e["name"]: e["weight"] for e in ENGINES},
    )

    simulation_rows = []

    for p in failure_probs:
        abs_errors_a = []
        abs_errors_b = []
        abs_errors_c = []

        # Generate all failure indicators for this setting upfront: shape (num_trials, 3)
        # 1 means failed, 0 means succeeded
        failures_matrix = (np.random.rand(num_trials, len(ENGINES)) < p)

        for trial_idx in range(num_trials):
            failures = failures_matrix[trial_idx]

            # Construct EngineResult dictionary
            results: dict[str, EngineResult] = {}
            for name, score, failed in zip(engine_names, true_scores, failures):
                if failed:
                    results[name] = EngineResult(
                        status="FAILED",
                        score=None,
                        findings=[],
                        evidence=[],
                        metadata={},
                        logs=[],
                        failure_reason="internal_error",
                    )
                else:
                    results[name] = EngineResult(
                        status="SUCCESS",
                        score=float(score),
                        findings=[],
                        evidence=[],
                        metadata={},
                        logs=[],
                    )

            # (a) AMEVA renormalization
            agg_a, _ = calculate_aggregate(plan, results)
            score_a = agg_a if agg_a is not None else 0.0

            # (b) Failed engine imputed as 0
            # Denominator is total weight across all selected engines (100.0)
            imputed_scores = np.where(failures, 0.0, true_scores)
            score_b = float(np.sum(imputed_scores * weights) / total_weight)

            # (c) AMEVA aggregate multiplied by coverage
            cov = calculate_coverage(plan, results)
            score_c = (agg_a if agg_a is not None else 0.0) * cov

            abs_errors_a.append(abs(score_a - true_aggregate))
            abs_errors_b.append(abs(score_b - true_aggregate))
            abs_errors_c.append(abs(score_c - true_aggregate))

        mae_a = float(np.mean(abs_errors_a))
        mae_b = float(np.mean(abs_errors_b))
        mae_c = float(np.mean(abs_errors_c))

        simulation_rows.append({
            "p": float(p),
            "trials": num_trials,
            "true_aggregate": true_aggregate,
            "ameva_renormalization_mae": mae_a,
            "imputed_zero_mae": mae_b,
            "ameva_x_coverage_mae": mae_c,
            "ameva_renormalization": mae_a,
            "imputed_zero": mae_b,
            "ameva_x_coverage": mae_c,
        })

    return simulation_rows, true_aggregate


def print_table(rows: list[dict], true_aggregate: float) -> None:
    print("\n" + "=" * 78)
    print("  Monte Carlo Simulation: Aggregator Comparison Under Independent Failures")
    print("=" * 78)
    print(f"True Aggregate (no failure): {true_aggregate:.4f}")
    print(f"Dummy Engines: E1 (score=0.9, w=40), E2 (score=0.7, w=35), E3 (score=0.8, w=25)")
    print("-" * 78)
    header = f"{'p':<8} | {'(a) AMEVA Renorm MAE':<22} | {'(b) Imputed 0 MAE':<18} | {'(c) AMEVA x Cov MAE':<20}"
    print(header)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['p']:<8.1f} | "
            f"{r['ameva_renormalization_mae']:<22.6f} | "
            f"{r['imputed_zero_mae']:<18.6f} | "
            f"{r['ameva_x_coverage_mae']:<20.6f}"
        )
    print("=" * 78 + "\n")


def save_results(rows: list[dict], results_dir: str) -> tuple[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "sim_aggregation.json")
    csv_path = os.path.join(results_dir, "sim_aggregation.csv")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Save CSV
    csv_fieldnames = ["p", "trials", "ameva_renormalization_mae", "imputed_zero_mae", "ameva_x_coverage_mae"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in csv_fieldnames})

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulation of AMEVA aggregators.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of trials per p setting")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "results"), help="Directory for output files")
    args = parser.parse_args()

    rows, true_aggregate = run_simulation(num_trials=args.trials, seed=args.seed)
    print_table(rows, true_aggregate)
    json_path, csv_path = save_results(rows, args.output_dir)
    print(f"Saved results to:\n - JSON: {json_path}\n - CSV:  {csv_path}\n")


if __name__ == "__main__":
    main()
