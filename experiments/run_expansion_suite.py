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

from search_concentration_audit.diagnostics import evaluate_plan, write_json
from search_concentration_audit.envs import PointMassConfig, PointMassWorld, make_action_set
from search_concentration_audit.models import BiasPocketConfig, BiasedLearnedModel
from search_concentration_audit.planners import MCTSPlanner, PlannerConfig, StaticRolloutPlanner


DISPLAY_NAMES = {
    "static_rollout_pool": "Static pool",
    "uct_mcts": "UCT MCTS",
    "uncertainty_mcts_240": "Unc. MCTS 2.40",
    "conservative_mcts": "Conservative MCTS",
}


def action_library(name: str) -> np.ndarray:
    if name == "cardinal4":
        return np.array([(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)], dtype=float)
    if name == "axial5":
        return np.vstack([np.zeros((1, 2), dtype=float), action_library("cardinal4")])
    if name == "default9":
        return make_action_set()
    if name == "dense17":
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        moves = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return np.vstack([np.zeros((1, 2), dtype=float), moves])
    if name == "longstep9":
        return make_action_set(step_scale=1.35)
    raise ValueError(f"unknown action library: {name}")


def make_planner(
    name: str,
    actions: np.ndarray,
    *,
    budget: int,
    horizon: int,
    gamma: float,
    exploration: float = 1.25,
) -> StaticRolloutPlanner | MCTSPlanner:
    if name == "static_rollout_pool":
        return StaticRolloutPlanner(
            actions,
            PlannerConfig(name=name, budget=budget, horizon=horizon, gamma=gamma),
        )
    if name == "uct_mcts":
        return MCTSPlanner(
            actions,
            PlannerConfig(name=name, budget=budget, horizon=horizon, gamma=gamma, exploration=exploration),
        )
    if name == "uncertainty_mcts_240":
        return MCTSPlanner(
            actions,
            PlannerConfig(
                name=name,
                budget=budget,
                horizon=horizon,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=2.40,
            ),
        )
    if name == "conservative_mcts":
        return MCTSPlanner(
            actions,
            PlannerConfig(
                name=name,
                budget=budget,
                horizon=horizon,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=2.40,
                conservative_backup=True,
            ),
        )
    raise ValueError(f"unknown planner: {name}")


def run_plan(
    *,
    dataset: str,
    output_tag: str,
    seed: int,
    world: PointMassWorld,
    model: BiasedLearnedModel,
    action_name: str,
    planner_name: str,
    budget: int,
    horizon: int,
    gamma: float,
    exploration: float = 1.25,
    start_label: str = "base",
) -> dict[str, float | int | str]:
    actions = action_library(action_name)
    planner = make_planner(
        planner_name,
        actions,
        budget=budget,
        horizon=horizon,
        gamma=gamma,
        exploration=exploration,
    )
    state = world.reset()
    result = planner.plan(model, state, seed=seed + 1000 * budget + 17 * horizon)
    metrics = evaluate_plan(world, model, state, result, gamma=gamma)
    metrics.update(
        {
            "dataset": dataset,
            "output_tag": output_tag,
            "seed": int(seed),
            "budget": int(budget),
            "horizon": int(horizon),
            "gamma": float(gamma),
            "planner": planner_name,
            "action_library": action_name,
            "action_count": int(len(actions)),
            "exploration": float(exploration),
            "reward_bias": float(model.config.reward_bias),
            "dynamics_bias": float(model.config.dynamics_bias),
            "uncertainty_pocket": float(model.config.uncertainty_pocket),
            "start_label": start_label,
            "episode_true_return": np.nan,
            "episode_max_pocket_score": np.nan,
        }
    )
    return metrics


def run_episode(
    *,
    dataset: str,
    output_tag: str,
    seed: int,
    world: PointMassWorld,
    model: BiasedLearnedModel,
    action_name: str,
    planner_name: str,
    budget: int,
    horizon: int,
    gamma: float,
    steps: int,
) -> dict[str, float | int | str]:
    state = world.reset()
    total = 0.0
    discount = 1.0
    gaps: list[float] = []
    max_pocket = 0.0
    model_steps = 0
    actions = action_library(action_name)
    planner = make_planner(planner_name, actions, budget=budget, horizon=horizon, gamma=gamma)
    for step in range(steps):
        result = planner.plan(model, state, seed=seed * 10_000 + step)
        diag = evaluate_plan(world, model, state, result, gamma=gamma)
        gaps.append(float(diag["selected_return_gap"]))
        state, reward, done, info = world.step(state, result.action)
        total += discount * reward
        discount *= gamma
        model_steps += result.model_steps
        max_pocket = max(max_pocket, float(info["pocket_score"]))
        if done:
            break
    return {
        "dataset": dataset,
        "output_tag": output_tag,
        "seed": int(seed),
        "budget": int(budget),
        "horizon": int(horizon),
        "gamma": float(gamma),
        "planner": planner_name,
        "action_library": action_name,
        "action_count": int(len(actions)),
        "exploration": np.nan,
        "reward_bias": float(model.config.reward_bias),
        "dynamics_bias": float(model.config.dynamics_bias),
        "uncertainty_pocket": float(model.config.uncertainty_pocket),
        "start_label": "base",
        "selected_return_gap": float(np.mean(gaps)) if gaps else 0.0,
        "episode_true_return": float(total),
        "episode_max_pocket_score": float(max_pocket),
        "episode_model_steps": float(model_steps),
        "episode_steps": int(step + 1 if "step" in locals() else 0),
        "model_steps": float(model_steps),
        "search_concentration": np.nan,
        "max_pocket_score": float(max_pocket),
        "model_uncertainty": np.nan,
        "transition_error": np.nan,
        "model_bias": np.nan,
        "true_max_pocket_score": float(max_pocket),
        "min_goal_distance": np.nan,
        "predicted_return": np.nan,
        "sequence_model_return": np.nan,
        "sequence_true_return": np.nan,
    }


def run_expansion_suite(
    *,
    output: Path,
    write_figures: bool = True,
    exploration_seeds: range = range(4),
    grid_seeds: range = range(3),
    calibration_seeds: range = range(4),
    drift_seeds: range = range(3),
    episode_seeds: range = range(2),
) -> dict[str, Path | str]:
    output.mkdir(parents=True, exist_ok=True)
    artifact_root = ROOT / "paper" if write_figures else output
    figure_dir = artifact_root / "figures"
    table_dir = artifact_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    gamma = 0.98
    rows: list[dict[str, float | int | str]] = []

    base_world = PointMassWorld()
    base_model = BiasedLearnedModel(base_world, BiasPocketConfig())
    for exploration in [0.35, 0.75, 1.25, 2.0, 3.0]:
        tag = f"exploration_{exploration:.2f}"
        for seed in exploration_seeds:
            for planner in ["static_rollout_pool", "uct_mcts"]:
                rows.append(
                    run_plan(
                        dataset="exploration_sweep",
                        output_tag=tag,
                        seed=seed,
                        world=base_world,
                        model=base_model,
                        action_name="default9",
                        planner_name=planner,
                        budget=768,
                        horizon=18,
                        gamma=gamma,
                        exploration=exploration,
                    )
                )

    for horizon in [8, 18, 24]:
        for budget in [256, 768]:
            tag = f"h{horizon}_b{budget}"
            for seed in grid_seeds:
                for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
                    rows.append(
                        run_plan(
                            dataset="horizon_budget_sweep",
                            output_tag=tag,
                            seed=seed,
                            world=base_world,
                            model=base_model,
                            action_name="default9",
                            planner_name=planner,
                            budget=budget,
                            horizon=horizon,
                            gamma=gamma,
                        )
                    )

    for action_name in ["cardinal4", "default9", "dense17"]:
        for seed in grid_seeds:
            for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
                rows.append(
                    run_plan(
                        dataset="action_library_sweep",
                        output_tag=action_name,
                        seed=seed,
                        world=base_world,
                        model=base_model,
                        action_name=action_name,
                        planner_name=planner,
                        budget=768,
                        horizon=18,
                        gamma=gamma,
                    )
                )

    for uncertainty_pocket in [0.0, 0.15, 0.55, 1.10]:
        world = PointMassWorld()
        model = BiasedLearnedModel(world, BiasPocketConfig(uncertainty_pocket=uncertainty_pocket))
        tag = f"uncertainty_pocket_{uncertainty_pocket:.2f}"
        for seed in calibration_seeds:
            for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240", "conservative_mcts"]:
                rows.append(
                    run_plan(
                        dataset="uncertainty_calibration_stress",
                        output_tag=tag,
                        seed=seed,
                        world=world,
                        model=model,
                        action_name="default9",
                        planner_name=planner,
                        budget=768,
                        horizon=18,
                        gamma=gamma,
                    )
                )

    for dynamics_bias in [0.0, 0.22, 0.36, 0.50]:
        world = PointMassWorld()
        model = BiasedLearnedModel(world, BiasPocketConfig(dynamics_bias=dynamics_bias))
        tag = f"dynamics_bias_{dynamics_bias:.2f}"
        for seed in drift_seeds:
            for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
                rows.append(
                    run_plan(
                        dataset="dynamics_drift_stress",
                        output_tag=tag,
                        seed=seed,
                        world=world,
                        model=model,
                        action_name="default9",
                        planner_name=planner,
                        budget=768,
                        horizon=18,
                        gamma=gamma,
                    )
                )

    start_cases = {
        "base": (-0.85, -0.55),
        "lower_start": (-0.85, -0.82),
        "pocket_adjacent": (-0.52, 0.20),
        "wide_left": (-1.00, -0.55),
    }
    for label, start in start_cases.items():
        world = PointMassWorld(PointMassConfig(start=start))
        model = BiasedLearnedModel(world, BiasPocketConfig())
        for seed in grid_seeds:
            for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
                rows.append(
                    run_plan(
                        dataset="start_state_stress",
                        output_tag=label,
                        seed=seed,
                        world=world,
                        model=model,
                        action_name="default9",
                        planner_name=planner,
                        budget=768,
                        horizon=18,
                        gamma=gamma,
                        start_label=label,
                    )
                )

    for budget in [512]:
        for seed in episode_seeds:
            for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240", "conservative_mcts"]:
                rows.append(
                    run_episode(
                        dataset="closed_loop_replay",
                        output_tag=f"budget_{budget}",
                        seed=seed,
                        world=base_world,
                        model=base_model,
                        action_name="default9",
                        planner_name=planner,
                        budget=budget,
                        horizon=18,
                        gamma=gamma,
                        steps=3,
                    )
                )

    df = pd.DataFrame(rows)
    metrics_path = output / "metrics.csv"
    summary_path = output / "summary.csv"
    paired_path = output / "paired_deltas.csv"
    claims_path = output / "claim_audit.json"
    manifest_path = output / "manifest.json"
    df.to_csv(metrics_path, index=False)
    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    paired = paired_deltas(df)
    paired.to_csv(paired_path, index=False)
    claims = audit_expansion_suite(df, summary, paired)
    write_json(claims_path, claims)
    write_tables(summary, paired, table_dir)
    figure_paths = [
        figure_dir / "exploration_constant_sweep.png",
        figure_dir / "horizon_budget_sweep.png",
        figure_dir / "action_library_sweep.png",
        figure_dir / "uncertainty_miscalibration.png",
        figure_dir / "dynamics_drift_stress.png",
        figure_dir / "start_state_stress.png",
        figure_dir / "closed_loop_replay.png",
    ]
    if write_figures:
        plot_exploration(summary, figure_paths[0])
        plot_horizon(summary, figure_paths[1])
        plot_action_library(summary, figure_paths[2])
        plot_uncertainty(summary, figure_paths[3])
        plot_dynamics(summary, figure_paths[4])
        plot_start_states(summary, figure_paths[5])
        plot_closed_loop(summary, figure_paths[6])
    manifest = {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "paired_deltas": str(paired_path),
        "claim_audit": str(claims_path),
        "figures": [str(path) for path in figure_paths] if write_figures else [],
        "tables": [
            str(table_dir / "expansion_summary.csv"),
            str(table_dir / "expansion_paired_deltas.csv"),
        ],
        "strongest_result": claims["summary"],
    }
    write_json(manifest_path, manifest)
    return manifest


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset",
        "output_tag",
        "planner",
        "budget",
        "horizon",
        "action_library",
        "action_count",
        "exploration",
        "reward_bias",
        "dynamics_bias",
        "uncertainty_pocket",
        "start_label",
    ]
    grouped = df.groupby(group_cols, dropna=False)["selected_return_gap"]
    summary = grouped.agg(mean_gap="mean", median_gap="median", max_gap="max", n="count").reset_index()
    summary["q90_gap"] = grouped.quantile(0.9).to_numpy()
    extra = (
        df.groupby(group_cols, dropna=False)[["episode_true_return", "episode_max_pocket_score", "search_concentration"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary.merge(extra, on=group_cols, how="left")


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    key_cols = [
        "dataset",
        "output_tag",
        "budget",
        "horizon",
        "action_library",
        "reward_bias",
        "dynamics_bias",
        "uncertainty_pocket",
        "start_label",
    ]
    for key, sub in df.groupby(key_cols, dropna=False):
        wide = sub.pivot_table(index="seed", columns="planner", values="selected_return_gap", aggfunc="mean")
        if "static_rollout_pool" not in wide:
            continue
        for planner in sorted(set(wide.columns) - {"static_rollout_pool"}):
            delta = (wide[planner] - wide["static_rollout_pool"]).dropna().to_numpy(dtype=float)
            if delta.size == 0:
                continue
            row = dict(zip(key_cols, key))
            row.update(
                {
                    "planner": planner,
                    "baseline": "static_rollout_pool",
                    "n": int(delta.size),
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "positive_fraction": float(np.mean(delta > 0.0)),
                    "max_delta": float(np.max(delta)),
                    "min_delta": float(np.min(delta)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _claim(status: bool, value: float, threshold: float, description: str) -> dict[str, float | str]:
    return {
        "status": "pass" if status else "fail",
        "value": float(value),
        "threshold": float(threshold),
        "description": description,
    }


def _summary_row(summary: pd.DataFrame, *, dataset: str, planner: str, output_tag: str | None = None, **kwargs) -> pd.Series:
    sub = summary[(summary["dataset"] == dataset) & (summary["planner"] == planner)]
    if output_tag is not None:
        sub = sub[sub["output_tag"] == output_tag]
    for key, value in kwargs.items():
        sub = sub[np.isclose(sub[key], value) if isinstance(value, float) else sub[key] == value]
    if sub.empty:
        raise ValueError(f"missing summary row for {dataset=} {planner=} {output_tag=} {kwargs=}")
    return sub.sort_values("max_gap", ascending=False).iloc[0]


def audit_expansion_suite(df: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> dict:
    exploration = summary[(summary["dataset"] == "exploration_sweep") & (summary["planner"] == "uct_mcts")]
    horizon_rows = summary[(summary["dataset"] == "horizon_budget_sweep") & (summary["planner"] == "uct_mcts")]
    action_rows = summary[(summary["dataset"] == "action_library_sweep") & (summary["planner"] == "uct_mcts")]
    aligned_uct = _summary_row(
        summary,
        dataset="uncertainty_calibration_stress",
        planner="uct_mcts",
        uncertainty_pocket=0.55,
    )
    aligned_unc = _summary_row(
        summary,
        dataset="uncertainty_calibration_stress",
        planner="uncertainty_mcts_240",
        uncertainty_pocket=0.55,
    )
    no_signal_unc = _summary_row(
        summary,
        dataset="uncertainty_calibration_stress",
        planner="uncertainty_mcts_240",
        uncertainty_pocket=0.0,
    )
    uncertainty_rows = summary[
        (summary["dataset"] == "uncertainty_calibration_stress")
        & (summary["planner"] == "uncertainty_mcts_240")
    ]
    drift_rows = summary[(summary["dataset"] == "dynamics_drift_stress") & (summary["planner"] == "uct_mcts")]
    start_rows = summary[(summary["dataset"] == "start_state_stress") & (summary["planner"] == "uct_mcts")]
    closed_uct = _summary_row(
        summary, dataset="closed_loop_replay", planner="uct_mcts", output_tag="budget_512"
    )
    closed_unc = _summary_row(
        summary, dataset="closed_loop_replay", planner="uncertainty_mcts_240", output_tag="budget_512"
    )

    exploration_range = float(exploration["max_gap"].max() - exploration["max_gap"].min())
    horizon_range = float(horizon_rows["max_gap"].max() - horizon_rows["max_gap"].min())
    action_range = float(action_rows["max_gap"].max() - action_rows["max_gap"].min())
    drift_range = float(drift_rows["max_gap"].max() - drift_rows["max_gap"].min())
    start_range = float(start_rows["max_gap"].max() - start_rows["max_gap"].min())
    uncertainty_strength_range = float(uncertainty_rows["max_gap"].max() - uncertainty_rows["max_gap"].min())
    claims = {
        "exploration_constant_changes_tail_size": _claim(
            exploration_range > 0.5,
            exploration_range,
            0.5,
            "UCT tail size is sensitive to the exploration constant rather than being a single fixed-parameter accident.",
        ),
        "horizon_budget_changes_tail_size": _claim(
            horizon_range > 0.25,
            horizon_range,
            0.25,
            "Lookahead and budget materially change UCT branch-capture tail exposure.",
        ),
        "action_library_changes_tail_size": _claim(
            action_range > 0.10,
            action_range,
            0.10,
            "Changing the action library materially changes adaptive-search tail exposure.",
        ),
        "calibration_sweep_finds_repair_backfire_case": _claim(
            float(aligned_unc["max_gap"] - aligned_uct["max_gap"]) > 1.0,
            float(aligned_unc["max_gap"] - aligned_uct["max_gap"]),
            1.0,
            "The expansion sweep finds a reduced-budget counterexample to the broad claim that uncertainty penalties always help.",
        ),
        "uncertainty_strength_alone_does_not_fix_backfire": _claim(
            uncertainty_strength_range <= 0.25,
            uncertainty_strength_range,
            0.25,
            "Changing only the uncertainty amplitude does not remove the reduced-budget penalty-induced capture case.",
        ),
        "dynamics_drift_changes_tail_size": _claim(
            drift_range > 0.25,
            drift_range,
            0.25,
            "Changing transition drift toward the goal materially changes UCT tail optimism.",
        ),
        "start_state_changes_tail_exposure": _claim(
            start_range > 0.25,
            start_range,
            0.25,
            "Tail exposure depends on the geometric relationship between the start state and the bias pocket.",
        ),
        "closed_loop_uncertainty_reduces_mean_plan_gap": _claim(
            float(closed_uct["mean_gap"] - closed_unc["mean_gap"]) > 0.05,
            float(closed_uct["mean_gap"] - closed_unc["mean_gap"]),
            0.05,
            "In receding-horizon replay, uncertainty-aware search lowers mean per-step planning optimism.",
        ),
    }
    return {
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
        "summary": (
            f"Expansion suite: exploration max-gap range {exploration_range:.3f}; "
            f"horizon/budget max-gap range {horizon_range:.3f}; "
            f"calibration backfire gap {float(aligned_unc['max_gap'] - aligned_uct['max_gap']):.3f}; "
            f"closed-loop mean-gap reduction {float(closed_uct['mean_gap'] - closed_unc['mean_gap']):.3f}."
        ),
    }


def write_tables(summary: pd.DataFrame, paired: pd.DataFrame, table_dir: Path) -> None:
    summary.to_csv(table_dir / "expansion_summary.csv", index=False)
    paired.to_csv(table_dir / "expansion_paired_deltas.csv", index=False)


def plot_exploration(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[(summary["dataset"] == "exploration_sweep") & (summary["planner"] == "uct_mcts")].copy()
    sub = sub.sort_values("exploration")
    fig, ax = plt.subplots(figsize=(6.7, 4.0))
    ax.plot(sub["exploration"], sub["mean_gap"], marker="o", label="mean")
    ax.plot(sub["exploration"], sub["q90_gap"], marker="s", label="90th pct.")
    ax.plot(sub["exploration"], sub["max_gap"], marker="^", label="max")
    ax.set_xlabel("UCT exploration constant")
    ax.set_ylabel("selected optimism gap")
    ax.set_title("Exploration constant controls tail exposure")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_horizon(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[(summary["dataset"] == "horizon_budget_sweep") & (summary["planner"] == "uct_mcts")]
    fig, ax = plt.subplots(figsize=(6.9, 4.1))
    for budget in sorted(sub["budget"].unique()):
        curve = sub[sub["budget"] == budget].sort_values("horizon")
        ax.plot(curve["horizon"], curve["max_gap"], marker="o", label=f"budget {budget}")
    ax.set_xlabel("planning horizon")
    ax.set_ylabel("UCT max optimism gap")
    ax.set_title("Lookahead length and budget change branch-capture tails")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_action_library(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["dataset"] == "action_library_sweep"].copy()
    order = ["cardinal4", "default9", "dense17"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.24
    x = np.arange(len(order))
    for offset, planner in [(-width, "static_rollout_pool"), (0.0, "uct_mcts"), (width, "uncertainty_mcts_240")]:
        curve = sub[sub["planner"] == planner].set_index("action_library").loc[order]
        ax.bar(x + offset, curve["max_gap"], width=width, label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("max optimism gap")
    ax.set_title("Action-library stress")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_uncertainty(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["dataset"] == "uncertainty_calibration_stress"].copy()
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for planner in ["uct_mcts", "uncertainty_mcts_240", "conservative_mcts"]:
        curve = sub[sub["planner"] == planner].sort_values("uncertainty_pocket")
        ax.plot(curve["uncertainty_pocket"], curve["max_gap"], marker="o", label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xlabel("localized uncertainty signal strength")
    ax.set_ylabel("max optimism gap")
    ax.set_title("Repair depends on uncertainty calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_dynamics(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["dataset"] == "dynamics_drift_stress"].copy()
    fig, ax = plt.subplots(figsize=(6.9, 4.0))
    for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
        curve = sub[sub["planner"] == planner].sort_values("dynamics_bias")
        ax.plot(curve["dynamics_bias"], curve["max_gap"], marker="o", label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xlabel("transition drift strength")
    ax.set_ylabel("max optimism gap")
    ax.set_title("Transition optimism amplifies branch capture")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_start_states(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["dataset"] == "start_state_stress"].copy()
    order = ["base", "lower_start", "pocket_adjacent", "wide_left"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(order))
    width = 0.26
    for offset, planner in [(-width, "static_rollout_pool"), (0.0, "uct_mcts"), (width, "uncertainty_mcts_240")]:
        curve = sub[sub["planner"] == planner].set_index("start_label").loc[order]
        ax.bar(x + offset, curve["max_gap"], width=width, label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("max optimism gap")
    ax.set_title("Geometry stress across start states")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_closed_loop(summary: pd.DataFrame, path: Path) -> None:
    sub = summary[summary["dataset"] == "closed_loop_replay"].copy()
    order = ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240", "conservative_mcts"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8))
    for ax, metric, title in [
        (axes[0], "mean_gap", "Mean planning optimism"),
        (axes[1], "episode_true_return", "Episode true return"),
    ]:
        x = np.arange(len(order))
        curve = sub[sub["output_tag"] == "budget_512"].set_index("planner").loc[order]
        ax.bar(x, curve[metric], width=0.55, label="budget 512")
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY_NAMES.get(item, item) for item in order], rotation=22, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("gap")
    axes[1].set_ylabel("true return")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "expansion")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    manifest = run_expansion_suite(output=args.output, write_figures=not args.no_figures)
    print(manifest["strongest_result"])
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
