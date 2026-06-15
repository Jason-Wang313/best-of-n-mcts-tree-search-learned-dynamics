from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from search_concentration_audit.cliff_benchmark import (
    TabularCliffModel,
    evaluate_selected_sequence,
    mcts_plan,
    static_rollout_plan,
)


RESULTS = ROOT / "results" / "cliffwalking_benchmark"
FIGURES = ROOT / "paper" / "figures"
BUDGETS = (32, 64, 128, 256)
HORIZON = 16


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(4401)
    boot = rng.choice(arr, size=(1200, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": float(arr.size),
    }


def _planner_rows(seed: int, budget: int) -> list[dict[str, Any]]:
    learned = TabularCliffModel(biased=True)
    true_model = TabularCliffModel(biased=False)
    planners = []

    seq, info = static_rollout_plan(learned, budget=budget, horizon=HORIZON, seed=seed, proposal="behavioral")
    planners.append(("static_rollout_pool", seq, info, learned, 0.0))

    seq, info = static_rollout_plan(learned, budget=budget, horizon=HORIZON, seed=40_000 + seed, proposal="uniform")
    planners.append(("uniform_static_pool", seq, info, learned, 0.0))

    seq, info = mcts_plan(learned, budget=budget, horizon=HORIZON, seed=10_000 + seed)
    planners.append(("uct_mcts", seq, info, learned, 0.0))

    seq, info = mcts_plan(
        learned,
        budget=budget,
        horizon=HORIZON,
        seed=20_000 + seed,
        uncertainty_penalty=6.00,
    )
    planners.append(("uncertainty_mcts", seq, info, learned, 6.00))

    seq, info = mcts_plan(true_model, budget=budget, horizon=HORIZON, seed=30_000 + seed)
    planners.append(("true_model_uct", seq, info, true_model, 0.0))

    rows: list[dict[str, Any]] = []
    for planner, sequence, info, model, penalty in planners:
        row = evaluate_selected_sequence(model, sequence, penalty)
        row.update(info)
        row.update(
            {
                "benchmark": "CliffWalking-v1",
                "seed": int(seed),
                "budget": int(budget),
                "horizon": int(HORIZON),
                "planner": planner,
                "action_sequence": " ".join(str(int(a)) for a in sequence),
            }
        )
        rows.append(row)
    return rows


def _plot(metrics: pd.DataFrame, output: Path) -> None:
    means = metrics.groupby(["planner", "budget"], as_index=False).mean(numeric_only=True)
    colors = {
        "static_rollout_pool": "#777777",
        "uniform_static_pool": "#c47f2d",
        "uct_mcts": "#b23b3b",
        "uncertainty_mcts": "#1a8f5a",
        "true_model_uct": "#111111",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for planner, color in colors.items():
        sub = means[means["planner"] == planner].sort_values("budget")
        axes[0].plot(sub["budget"], sub["selected_return_gap"], marker="o", color=color, label=planner)
        axes[1].plot(sub["budget"], sub["true_return"], marker="o", color=color)
        axes[2].plot(sub["budget"], sub["model_cliff_bias"], marker="o", color=color)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.28)
        ax.set_xlabel("forward-model budget")
    axes[0].set_ylabel("selected-return optimism gap")
    axes[1].set_ylabel("true executed return")
    axes[2].set_ylabel("learned shortcut exposure")
    axes[0].set_title("Model optimism")
    axes[1].set_title("True CliffWalking return")
    axes[2].set_title("Cliff shortcut usage")
    axes[0].legend(frameon=False, fontsize=6.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_cliffwalking_benchmark(
    *,
    seeds: list[int] | None = None,
    output: Path = RESULTS,
) -> dict[str, Any]:
    seeds = list(range(40, 60)) if seeds is None else [int(seed) for seed in seeds]
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for budget in BUDGETS:
            rows.extend(_planner_rows(seed, budget))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)

    max_budget = max(BUDGETS)
    max_rows = metrics[metrics["budget"] == max_budget]
    pivot = max_rows.pivot_table(index="seed", columns="planner", values=["selected_return_gap", "true_return", "model_cliff_bias"])
    effects = pd.DataFrame(
        {
            "seed": seeds,
            "uct_minus_static_gap": pivot["selected_return_gap"]["uct_mcts"] - pivot["selected_return_gap"]["static_rollout_pool"],
            "uniform_static_minus_static_gap": pivot["selected_return_gap"]["uniform_static_pool"]
            - pivot["selected_return_gap"]["static_rollout_pool"],
            "uncertainty_minus_uct_gap": pivot["selected_return_gap"]["uncertainty_mcts"] - pivot["selected_return_gap"]["uct_mcts"],
            "true_model_uct_gap": pivot["selected_return_gap"]["true_model_uct"],
            "uncertainty_minus_uct_return": pivot["true_return"]["uncertainty_mcts"] - pivot["true_return"]["uct_mcts"],
            "uct_minus_static_cliff_bias": pivot["model_cliff_bias"]["uct_mcts"] - pivot["model_cliff_bias"]["static_rollout_pool"],
            "uniform_static_minus_static_cliff_bias": pivot["model_cliff_bias"]["uniform_static_pool"]
            - pivot["model_cliff_bias"]["static_rollout_pool"],
            "uncertainty_minus_uct_cliff_bias": pivot["model_cliff_bias"]["uncertainty_mcts"] - pivot["model_cliff_bias"]["uct_mcts"],
        }
    )
    effects.to_csv(output / "effects.csv", index=False)

    figure = output / "cliffwalking_benchmark.png"
    _plot(metrics, figure)
    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, FIGURES / figure.name)

    summary = {
        "uct_minus_static_gap_ci": _ci(effects["uct_minus_static_gap"].astype(float).tolist()),
        "uniform_static_minus_static_gap_ci": _ci(effects["uniform_static_minus_static_gap"].astype(float).tolist()),
        "uncertainty_minus_uct_gap_ci": _ci(effects["uncertainty_minus_uct_gap"].astype(float).tolist()),
        "uncertainty_minus_uct_return_ci": _ci(effects["uncertainty_minus_uct_return"].astype(float).tolist()),
        "uct_minus_static_cliff_bias_ci": _ci(effects["uct_minus_static_cliff_bias"].astype(float).tolist()),
        "uniform_static_minus_static_cliff_bias_ci": _ci(
            effects["uniform_static_minus_static_cliff_bias"].astype(float).tolist()
        ),
        "uncertainty_minus_uct_cliff_bias_ci": _ci(effects["uncertainty_minus_uct_cliff_bias"].astype(float).tolist()),
        "true_model_uct_gap_ci": _ci(effects["true_model_uct_gap"].astype(float).tolist()),
    }
    claims = {
        "cliff_uct_gap_exceeds_static": summary["uct_minus_static_gap_ci"]["lo"] > 50.0,
        "cliff_uncertainty_reduces_gap": summary["uncertainty_minus_uct_gap_ci"]["hi"] < -100.0,
        "cliff_uncertainty_improves_true_return": summary["uncertainty_minus_uct_return_ci"]["lo"] > 100.0,
        "cliff_uct_concentrates_on_shortcut": summary["uct_minus_static_cliff_bias_ci"]["lo"] > 2.0,
        "cliff_uncertainty_avoids_shortcut": summary["uncertainty_minus_uct_cliff_bias_ci"]["hi"] < -2.0,
        "cliff_true_model_has_low_gap": abs(summary["true_model_uct_gap_ci"]["mean"]) < 1.0,
    }
    payload = {
        "benchmark": "CliffWalking-v1",
        "seeds": seeds,
        "budgets": list(BUDGETS),
        "horizon": HORIZON,
        "metrics_rows": int(len(metrics)),
        "effect_rows": int(len(effects)),
        "summary": summary,
        "claims": claims,
        "all_passed": all(bool(v) for v in claims.values()),
    }
    with (output / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps({"all_passed": payload["all_passed"], **summary}, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    run_cliffwalking_benchmark(seeds=seeds, output=args.output)


if __name__ == "__main__":
    main()
