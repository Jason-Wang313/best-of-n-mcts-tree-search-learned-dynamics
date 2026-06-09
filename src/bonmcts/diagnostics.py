from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bonmcts.envs import Array, PointMassWorld
from bonmcts.models import BiasedLearnedModel
from bonmcts.planners import PlanResult


def evaluate_plan(
    world: PointMassWorld,
    model: BiasedLearnedModel,
    state: Array,
    result: PlanResult,
    gamma: float = 0.98,
) -> dict[str, float]:
    true_rollout = world.rollout(state, result.action_sequence, gamma=gamma)
    model_rollout = model.rollout(state, result.action_sequence, gamma=gamma)
    gap = float(model_rollout["return"]) - float(true_rollout["return"])
    stats = {
        "planner": result.planner,
        "predicted_return": float(result.predicted_return),
        "sequence_model_return": float(model_rollout["return"]),
        "sequence_true_return": float(true_rollout["return"]),
        "selected_return_gap": gap,
        "model_bias": float(model_rollout["bias"]),
        "model_uncertainty": float(model_rollout["uncertainty"]),
        "transition_error": float(model_rollout["transition_error"]),
        "max_pocket_score": float(model_rollout["max_pocket_score"]),
        "true_max_pocket_score": float(true_rollout["max_pocket_score"]),
        "min_goal_distance": float(true_rollout["min_goal_distance"]),
        "model_steps": float(result.model_steps),
    }
    stats.update({k: float(v) for k, v in result.diagnostics.items() if isinstance(v, (int, float))})
    return stats


def closed_loop_episode(
    world: PointMassWorld,
    model: BiasedLearnedModel,
    planner,
    steps: int,
    seed: int,
    gamma: float = 0.98,
) -> dict[str, float]:
    state = world.reset()
    total = 0.0
    discount = 1.0
    model_steps = 0
    max_pocket = 0.0
    gaps = []
    for t in range(steps):
        result = planner.plan(model, state, seed=seed * 10_000 + t)
        diag = evaluate_plan(world, model, state, result, gamma=gamma)
        gaps.append(diag["selected_return_gap"])
        state, reward, done, info = world.step(state, result.action)
        total += discount * reward
        discount *= gamma
        model_steps += result.model_steps
        max_pocket = max(max_pocket, float(info["pocket_score"]))
        if done:
            break
    return {
        "episode_true_return": float(total),
        "episode_model_steps": float(model_steps),
        "episode_max_pocket_score": float(max_pocket),
        "episode_mean_plan_gap": float(np.mean(gaps)) if gaps else 0.0,
    }


def summarize_by_planner(rows: Iterable[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    numeric = frame.select_dtypes(include=[float, int]).columns
    return frame.groupby(["planner", "budget"], as_index=False)[numeric].mean()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

