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
from search_concentration_audit.envs import PointMassWorld, make_action_set
from search_concentration_audit.models import BiasPocketConfig, BiasedLearnedModel
from search_concentration_audit.planners import MCTSPlanner, PlannerConfig, StaticRolloutPlanner


DISPLAY_NAMES = {
    "static_rollout_pool": "Static pool",
    "uct_mcts": "UCT MCTS",
    "uncertainty_mcts_085": "Unc. MCTS 0.85",
    "uncertainty_mcts_240": "Unc. MCTS 2.40",
    "conservative_mcts": "Conservative MCTS",
}


def _planner(name: str, action_set: np.ndarray, *, budget: int, horizon: int, gamma: float):
    if name == "static_rollout_pool":
        return StaticRolloutPlanner(action_set, PlannerConfig(name=name, horizon=horizon, budget=budget, gamma=gamma))
    if name == "uct_mcts":
        return MCTSPlanner(
            action_set,
            PlannerConfig(name=name, horizon=horizon, budget=budget, gamma=gamma, exploration=1.25),
        )
    if name == "uncertainty_mcts_085":
        return MCTSPlanner(
            action_set,
            PlannerConfig(
                name=name,
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=0.85,
            ),
        )
    if name == "uncertainty_mcts_240":
        return MCTSPlanner(
            action_set,
            PlannerConfig(
                name=name,
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=2.40,
            ),
        )
    if name == "conservative_mcts":
        return MCTSPlanner(
            action_set,
            PlannerConfig(
                name=name,
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=2.40,
                conservative_backup=True,
            ),
        )
    if name.startswith("uncertainty_penalty_"):
        penalty = float(name.removeprefix("uncertainty_penalty_").replace("_", "."))
        return MCTSPlanner(
            action_set,
            PlannerConfig(
                name=name,
                horizon=horizon,
                budget=budget,
                gamma=gamma,
                exploration=1.1,
                uncertainty_penalty=penalty,
            ),
        )
    raise ValueError(f"unknown planner name: {name}")


def _run_plan(
    *,
    dataset: str,
    seed: int,
    budget: int,
    horizon: int,
    gamma: float,
    model: BiasedLearnedModel,
    planner_name: str,
    reward_bias: float,
    output_tag: str,
) -> dict[str, float | int | str]:
    world = model.world
    planner = _planner(planner_name, make_action_set(), budget=budget, horizon=horizon, gamma=gamma)
    result = planner.plan(model, world.reset(), seed=seed + 1000 * budget)
    metrics = evaluate_plan(world, model, world.reset(), result, gamma=gamma)
    metrics.update(
        {
            "dataset": dataset,
            "seed": int(seed),
            "budget": int(budget),
            "horizon": int(horizon),
            "planner": planner_name,
            "reward_bias": float(reward_bias),
            "output_tag": output_tag,
        }
    )
    return metrics


def run_tail_stress(
    *,
    output: Path,
    budget: int = 1024,
    horizon: int = 18,
    gamma: float = 0.98,
    replay_seeds: range = range(20),
    penalty_seeds: range = range(20),
    bias_seeds: range = range(12),
    write_figures: bool = True,
) -> dict[str, Path | str]:
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = ROOT / "paper" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    world = PointMassWorld()
    default_model = BiasedLearnedModel(world, BiasPocketConfig())

    replay_planners = [
        "static_rollout_pool",
        "uct_mcts",
        "uncertainty_mcts_085",
        "uncertainty_mcts_240",
        "conservative_mcts",
    ]
    for seed in replay_seeds:
        for planner_name in replay_planners:
            rows.append(
                _run_plan(
                    dataset="capture_replay",
                    seed=seed,
                    budget=budget,
                    horizon=horizon,
                    gamma=gamma,
                    model=default_model,
                    planner_name=planner_name,
                    reward_bias=default_model.config.reward_bias,
                    output_tag="default",
                )
            )

    penalties = [0.0, 0.25, 0.50, 0.85, 1.25, 1.75, 2.40, 3.20]
    for penalty in penalties:
        planner_name = f"uncertainty_penalty_{penalty:.2f}".replace(".", "_")
        for seed in penalty_seeds:
            rows.append(
                _run_plan(
                    dataset="penalty_sweep",
                    seed=seed,
                    budget=budget,
                    horizon=horizon,
                    gamma=gamma,
                    model=default_model,
                    planner_name=planner_name,
                    reward_bias=default_model.config.reward_bias,
                    output_tag=f"penalty_{penalty:.2f}",
                )
            )

    for reward_bias in [0.0, 0.60, 1.15, 1.80]:
        model = BiasedLearnedModel(world, BiasPocketConfig(reward_bias=reward_bias))
        for seed in bias_seeds:
            for planner_name in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
                rows.append(
                    _run_plan(
                        dataset="bias_strength_sweep",
                        seed=seed,
                        budget=budget,
                        horizon=horizon,
                        gamma=gamma,
                        model=model,
                        planner_name=planner_name,
                        reward_bias=reward_bias,
                        output_tag=f"reward_bias_{reward_bias:.2f}",
                    )
                )

    df = pd.DataFrame(rows)
    metrics_path = output / "metrics.csv"
    summary_path = output / "tail_summary.csv"
    deltas_path = output / "paired_deltas.csv"
    claims_path = output / "claim_audit.json"
    df.to_csv(metrics_path, index=False)
    tail_summary(df).to_csv(summary_path, index=False)
    paired_deltas(df).to_csv(deltas_path, index=False)
    claims = audit_tail_stress(df)
    write_json(claims_path, claims)
    figure_paths = [
        figure_dir / "tail_capture_replay.png",
        figure_dir / "uncertainty_penalty_sweep.png",
        figure_dir / "bias_strength_tail_stress.png",
    ]
    if write_figures:
        plot_capture_replay(df, figure_paths[0])
        plot_penalty_sweep(df, figure_paths[1])
        plot_bias_strength(df, figure_paths[2])
    manifest = {
        "metrics": str(metrics_path),
        "tail_summary": str(summary_path),
        "paired_deltas": str(deltas_path),
        "claim_audit": str(claims_path),
        "figures": [str(path) for path in figure_paths] if write_figures else [],
        "strongest_result": claims["summary"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def tail_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "output_tag", "reward_bias", "planner", "budget"])["selected_return_gap"]
    summary = grouped.agg(mean_gap="mean", median_gap="median", max_gap="max").reset_index()
    q = grouped.quantile([0.9, 0.95]).unstack(level=-1).reset_index().rename(columns={0.9: "q90_gap", 0.95: "q95_gap"})
    return summary.merge(q, on=["dataset", "output_tag", "reward_bias", "planner", "budget"], how="left")


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (dataset, output_tag, reward_bias, budget), sub in df.groupby(["dataset", "output_tag", "reward_bias", "budget"]):
        wide = sub.pivot(index="seed", columns="planner", values="selected_return_gap")
        if "static_rollout_pool" not in wide:
            continue
        for planner in sorted(set(wide.columns) - {"static_rollout_pool"}):
            delta = (wide[planner] - wide["static_rollout_pool"]).dropna().to_numpy(dtype=float)
            if delta.size == 0:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "output_tag": output_tag,
                    "reward_bias": float(reward_bias),
                    "budget": int(budget),
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
    return pd.DataFrame(rows)


def _stats(df: pd.DataFrame, *, dataset: str, planner: str, output_tag: str | None = None, reward_bias: float | None = None) -> pd.Series:
    sub = df[(df["dataset"] == dataset) & (df["planner"] == planner)]
    if output_tag is not None:
        sub = sub[sub["output_tag"] == output_tag]
    if reward_bias is not None:
        sub = sub[np.isclose(sub["reward_bias"], reward_bias)]
    if sub.empty:
        raise ValueError(f"missing stats for {dataset=} {planner=} {output_tag=} {reward_bias=}")
    values = sub["selected_return_gap"].astype(float)
    return pd.Series(
        {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q90": float(values.quantile(0.9)),
            "max": float(values.max()),
        }
    )


def audit_tail_stress(df: pd.DataFrame) -> dict:
    replay_uct = _stats(df, dataset="capture_replay", planner="uct_mcts")
    replay_static = _stats(df, dataset="capture_replay", planner="static_rollout_pool")
    replay_unc240 = _stats(df, dataset="capture_replay", planner="uncertainty_mcts_240")
    deltas = paired_deltas(df)
    replay_delta = deltas[(deltas["dataset"] == "capture_replay") & (deltas["planner"] == "uct_mcts")].iloc[0]
    pen240 = _stats(df, dataset="penalty_sweep", planner="uncertainty_penalty_2_40", output_tag="penalty_2.40")
    pen085 = _stats(df, dataset="penalty_sweep", planner="uncertainty_penalty_0_85", output_tag="penalty_0.85")
    rb0 = _stats(df, dataset="bias_strength_sweep", planner="uct_mcts", reward_bias=0.0)
    rb18 = _stats(df, dataset="bias_strength_sweep", planner="uct_mcts", reward_bias=1.8)
    rb18_unc = _stats(df, dataset="bias_strength_sweep", planner="uncertainty_mcts_240", reward_bias=1.8)

    claims = {
        "uct_tail_is_rare_not_dominance": {
            "status": "pass" if replay_delta["positive_fraction"] <= 0.25 and replay_delta["max_delta"] > 5.0 else "fail",
            "value": float(replay_delta["max_delta"]),
            "threshold": 5.0,
            "description": "UCT branch capture is a rare large-tail event, not paired-seed dominance.",
        },
        "strong_uncertainty_penalty_reduces_capture_tail": {
            "status": "pass" if replay_uct["max"] - replay_unc240["max"] > 6.0 else "fail",
            "value": float(replay_uct["max"] - replay_unc240["max"]),
            "threshold": 6.0,
            "description": "A stronger uncertainty penalty collapses the largest UCT capture event.",
        },
        "penalty_sweep_finds_low_tail_region": {
            "status": "pass" if pen240["max"] < 0.50 and pen240["q90"] < pen085["q90"] else "fail",
            "value": float(pen240["max"]),
            "threshold": 0.50,
            "description": "The penalty sweep finds a low-tail region rather than relying on the original repair knob.",
        },
        "bias_strength_increases_uct_tail": {
            "status": "pass" if rb18["max"] - rb0["max"] > 5.0 else "fail",
            "value": float(rb18["max"] - rb0["max"]),
            "threshold": 5.0,
            "description": "Increasing reward optimism in the pocket increases the UCT tail event size.",
        },
        "strong_uncertainty_controls_high_bias_tail": {
            "status": "pass" if rb18["max"] - rb18_unc["max"] > 6.0 else "fail",
            "value": float(rb18["max"] - rb18_unc["max"]),
            "threshold": 6.0,
            "description": "The strong uncertainty penalty controls the high-bias tail stress case.",
        },
    }
    return {
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
        "summary": (
            f"UCT max capture gap {replay_uct['max']:.3f} versus strong uncertainty "
            f"{replay_unc240['max']:.3f}; penalty 2.40 q90 {pen240['q90']:.3f}; "
            f"high-bias UCT max {rb18['max']:.3f}."
        ),
        "reference_values": {
            "capture_replay_uct": replay_uct.to_dict(),
            "capture_replay_static": replay_static.to_dict(),
            "capture_replay_uncertainty_240": replay_unc240.to_dict(),
            "penalty_085": pen085.to_dict(),
            "penalty_240": pen240.to_dict(),
            "bias_0_uct": rb0.to_dict(),
            "bias_1_8_uct": rb18.to_dict(),
            "bias_1_8_uncertainty_240": rb18_unc.to_dict(),
        },
    }


def plot_capture_replay(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["dataset"] == "capture_replay"]
    keep = ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_085", "uncertainty_mcts_240"]
    pivot = sub[sub["planner"].isin(keep)].pivot(index="seed", columns="planner", values="selected_return_gap")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for planner in keep:
        ax.plot(pivot.index, pivot[planner], marker="o", label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xlabel("seed")
    ax.set_ylabel("selected optimism gap")
    ax.set_title("Branch-capture replay at budget 1024")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_penalty_sweep(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["dataset"] == "penalty_sweep"].copy()
    sub["penalty"] = sub["output_tag"].str.removeprefix("penalty_").astype(float)
    grouped = sub.groupby("penalty")["selected_return_gap"]
    curve = grouped.agg(mean="mean", max="max").reset_index()
    curve["q90"] = grouped.quantile(0.9).to_numpy()
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(curve["penalty"], curve["mean"], marker="o", label="mean")
    ax.plot(curve["penalty"], curve["q90"], marker="s", label="90th pct.")
    ax.plot(curve["penalty"], curve["max"], marker="^", label="max")
    ax.set_xlabel("uncertainty penalty")
    ax.set_ylabel("selected optimism gap")
    ax.set_title("Uncertainty penalty sweep")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_bias_strength(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["dataset"] == "bias_strength_sweep"]
    grouped = sub.groupby(["reward_bias", "planner"])["selected_return_gap"]
    summary = grouped.agg(mean="mean", max="max").reset_index()
    summary["q90"] = grouped.quantile(0.9).to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for planner in ["static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"]:
        curve = summary[summary["planner"] == planner]
        ax.plot(curve["reward_bias"], curve["max"], marker="o", label=DISPLAY_NAMES.get(planner, planner))
    ax.set_xlabel("reward optimism in pocket")
    ax.set_ylabel("max selected optimism gap")
    ax.set_title("Bias strength controls branch-capture severity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tail_stress")
    args = parser.parse_args()
    manifest = run_tail_stress(output=args.output)
    print(manifest["strongest_result"])
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
