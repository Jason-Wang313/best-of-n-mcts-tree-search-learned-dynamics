from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bonmcts.diagnostics import closed_loop_episode, evaluate_plan, write_json
from bonmcts.envs import PointMassWorld, make_action_set
from bonmcts.models import BiasPocketConfig, BiasedLearnedModel
from bonmcts.planners import BestOfNPlanner, MCTSPlanner, OpenLoopRandomPlanner, PlannerConfig


def make_planners(action_set: np.ndarray, budget: int, horizon: int, gamma: float):
    return [
        OpenLoopRandomPlanner(
            action_set,
            PlannerConfig(name="random_open_loop", horizon=horizon, budget=budget, gamma=gamma),
        ),
        BestOfNPlanner(
            action_set,
            PlannerConfig(name="best_of_n", horizon=horizon, budget=budget, gamma=gamma),
        ),
        MCTSPlanner(
            action_set,
            PlannerConfig(name="uct_mcts", horizon=horizon, budget=budget, gamma=gamma, exploration=1.25),
        ),
        MCTSPlanner(
            action_set,
            PlannerConfig(
                name="value_mcts",
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.05,
                value_guided=True,
            ),
        ),
        MCTSPlanner(
            action_set,
            PlannerConfig(
                name="uncertainty_mcts",
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=0.85,
            ),
        ),
        MCTSPlanner(
            action_set,
            PlannerConfig(
                name="conservative_mcts",
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=2.4,
                conservative_backup=True,
            ),
        ),
    ]


def depth_profile(world: PointMassWorld, model: BiasedLearnedModel, state: np.ndarray, actions: np.ndarray, gamma: float):
    rows = []
    true_state = state.copy()
    model_state = state.copy()
    discount = 1.0
    for depth, action in enumerate(actions, start=1):
        pred = model.predict(model_state, action)
        true_next, true_reward, _, true_info = world.step(true_state, action)
        rows.append(
            {
                "depth": depth,
                "discount": discount,
                "model_reward": pred.reward,
                "true_reward": true_reward,
                "reward_bias": pred.reward - true_reward,
                "uncertainty": pred.uncertainty,
                "transition_error": float(np.linalg.norm(pred.next_state - true_next)),
                "model_pocket_score": pred.pocket_score,
                "true_pocket_score": float(true_info["pocket_score"]),
            }
        )
        model_state = pred.next_state
        true_state = true_next
        discount *= gamma
    return rows


def run(mode: str, output: Path) -> dict[str, Path | float | str]:
    horizon = 18
    gamma = 0.98
    action_set = make_action_set()
    budgets = [64, 128, 256] if mode == "smoke" else [64, 128, 256, 512, 1024]
    seeds = list(range(4)) if mode == "smoke" else list(range(20))
    episode_steps = 4 if mode == "smoke" else 5
    episode_seed_count = len(seeds) if mode == "smoke" else min(5, len(seeds))

    output.mkdir(parents=True, exist_ok=True)
    figure_dir = ROOT / "paper" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    world = PointMassWorld()
    model = BiasedLearnedModel(world, BiasPocketConfig())

    plan_rows: list[dict[str, float | str]] = []
    episode_rows: list[dict[str, float | str]] = []
    depth_rows: list[dict[str, float | str]] = []
    root_payload: dict[str, object] = {}

    for budget in budgets:
        for seed in seeds:
            for planner in make_planners(action_set, budget, horizon, gamma):
                state = world.reset()
                result = planner.plan(model, state, seed=seed + 1000 * budget)
                metrics = evaluate_plan(world, model, state, result, gamma=gamma)
                metrics.update({"seed": seed, "budget": budget})
                plan_rows.append(metrics)

                if budget == budgets[-1] and seed < episode_seed_count:
                    episode = closed_loop_episode(world, model, planner, steps=episode_steps, seed=seed, gamma=gamma)
                    episode.update({"planner": planner.name, "seed": seed, "budget": budget})
                    episode_rows.append(episode)

                if budget == budgets[-1] and seed == seeds[0] and planner.name in {
                    "best_of_n",
                    "uct_mcts",
                    "uncertainty_mcts",
                    "conservative_mcts",
                }:
                    for row in depth_profile(world, model, state, result.action_sequence, gamma=gamma):
                        row.update({"planner": planner.name, "seed": seed, "budget": budget})
                        depth_rows.append(row)
                    root_payload[f"{planner.name}_budget_{budget}_seed_{seed}"] = result.root_stats

    plan_df = pd.DataFrame(plan_rows)
    episode_df = pd.DataFrame(episode_rows)
    depth_df = pd.DataFrame(depth_rows)
    if episode_df.empty:
        merged = plan_df.copy()
    else:
        merged = plan_df.merge(
            episode_df[["planner", "seed", "budget", "episode_true_return", "episode_mean_plan_gap"]],
            on=["planner", "seed", "budget"],
            how="left",
        )
    summary = merged.groupby(["planner", "budget"], as_index=False).mean(numeric_only=True)

    plan_path = output / "plan_metrics.csv"
    episode_path = output / "episode_metrics.csv"
    depth_path = output / "depth_profile.csv"
    summary_path = output / "summary.csv"
    root_path = output / "root_stats.json"
    plan_df.to_csv(plan_path, index=False)
    episode_df.to_csv(episode_path, index=False)
    depth_df.to_csv(depth_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_json(root_path, root_payload)

    make_figures(plan_df, episode_df, depth_df, figure_dir)

    mechanism_rows = summary[summary["planner"].isin(["uct_mcts", "value_mcts"])].copy()
    strongest = mechanism_rows.sort_values("selected_return_gap", ascending=False).iloc[0]
    best_budget = int(strongest["budget"])
    mechanism_planner = str(strongest["planner"])
    best_slice = summary[summary["budget"] == best_budget].copy()
    mechanism_gap = float(strongest["selected_return_gap"])
    repair_slice = best_slice[best_slice["planner"].isin(["uncertainty_mcts", "conservative_mcts"])].copy()
    best_repair = repair_slice.sort_values("selected_return_gap", ascending=True).iloc[0]
    repair_planner = str(best_repair["planner"])
    repair_gap = float(best_repair["selected_return_gap"])
    best_of_n_gap = float(best_slice.loc[best_slice["planner"] == "best_of_n", "selected_return_gap"].iloc[0])
    repair_delta = mechanism_gap - repair_gap
    amplification_delta = mechanism_gap - best_of_n_gap

    manifest = {
        "mode": mode,
        "plan_metrics": str(plan_path),
        "episode_metrics": str(episode_path),
        "depth_profile": str(depth_path),
        "summary": str(summary_path),
        "root_stats": str(root_path),
        "figures": [
            str(figure_dir / "compute_scaling.png"),
            str(figure_dir / "bias_amplification.png"),
            str(figure_dir / "depth_bias_profile.png"),
        ],
        "strongest_result": (
            f"At budget {best_budget}, mean selected-return optimism gap was "
            f"{mechanism_gap:.3f} for {mechanism_planner}, {best_of_n_gap:.3f} for best-of-N, "
            f"and {repair_gap:.3f} for the best calibrated repair ({repair_planner})."
        ),
        "repair_delta": repair_delta,
        "amplification_delta": amplification_delta,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def make_figures(plan_df: pd.DataFrame, episode_df: pd.DataFrame, depth_df: pd.DataFrame, figure_dir: Path) -> None:
    palette = {
        "random_open_loop": "#6f6f6f",
        "best_of_n": "#2f6fdd",
        "uct_mcts": "#c2410c",
        "value_mcts": "#8b5cf6",
        "uncertainty_mcts": "#0f766e",
        "conservative_mcts": "#166534",
    }

    summary = plan_df.groupby(["planner", "budget"], as_index=False).mean(numeric_only=True)
    order = ["best_of_n", "uct_mcts", "value_mcts", "uncertainty_mcts", "conservative_mcts"]

    plt.figure(figsize=(7.0, 4.2))
    for planner in order:
        sub = summary[summary["planner"] == planner]
        if sub.empty:
            continue
        plt.plot(
            sub["budget"],
            sub["selected_return_gap"],
            marker="o",
            linewidth=2.0,
            label=planner,
            color=palette.get(planner),
        )
    plt.xscale("log", base=2)
    plt.xlabel("Forward-model budget")
    plt.ylabel("Selected model return - true return")
    plt.title("Optimism gap under static and adaptive search")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "compute_scaling.png", dpi=220)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    for planner in ["uct_mcts", "value_mcts", "uncertainty_mcts", "conservative_mcts"]:
        sub = summary[summary["planner"] == planner]
        if sub.empty:
            continue
        ax1.plot(
            sub["budget"],
            sub["search_concentration"],
            marker="s",
            linewidth=2.0,
            label=planner,
            color=palette.get(planner),
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Forward-model budget")
    ax1.set_ylabel("Root visit concentration")
    ax1.grid(alpha=0.25)
    ax1.set_title("Adaptive search concentration on preferred branches")
    ax1.legend(frameon=False, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(figure_dir / "bias_amplification.png", dpi=220)
    plt.close(fig)

    if not depth_df.empty:
        depth_summary = depth_df.groupby(["planner", "depth"], as_index=False).mean(numeric_only=True)
        plt.figure(figsize=(7.0, 4.2))
        for planner in ["best_of_n", "uct_mcts", "uncertainty_mcts", "conservative_mcts"]:
            sub = depth_summary[depth_summary["planner"] == planner]
            if sub.empty:
                continue
            plt.plot(
                sub["depth"],
                sub["reward_bias"],
                marker="o",
                linewidth=1.8,
                label=planner,
                color=palette.get(planner),
            )
        plt.xlabel("Depth along selected plan")
        plt.ylabel("Per-step model reward bias")
        plt.title("Depth-wise bias in selected trajectories")
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / "depth_bias_profile.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or (ROOT / "results" / args.mode)
    manifest = run(args.mode, output)
    print(manifest["strongest_result"])
    print(f"Manifest: {manifest['manifest']}")


if __name__ == "__main__":
    main()
