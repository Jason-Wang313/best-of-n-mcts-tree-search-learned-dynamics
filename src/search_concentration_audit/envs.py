from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class PointMassConfig:
    """A small deterministic continuous-control world.

    The state is a 2D position. The robot receives shaped reward for reaching a
    goal, pays a small action cost, and receives a mild penalty in the off-route
    bias-pocket region. The penalty makes model optimism there empirically
    visible without making the environment adversarial or stochastic.
    """

    start: tuple[float, float] = (-0.85, -0.55)
    goal: tuple[float, float] = (0.82, 0.55)
    step_size: float = 0.16
    action_cost: float = 0.025
    goal_bonus: float = 1.5
    goal_radius: float = 0.12
    bounds: tuple[float, float] = (-1.05, 1.05)
    pocket_center: tuple[float, float] = (-0.15, 0.82)
    pocket_radius: float = 0.22
    pocket_true_penalty: float = 0.35


def make_action_set(step_scale: float = 1.0, include_zero: bool = True) -> Array:
    """Return a compact 2D discrete action set for tree search."""

    moves = [
        (-1.0, 0.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (1.0, 1.0),
    ]
    actions = []
    if include_zero:
        actions.append((0.0, 0.0))
    for x, y in moves:
        vec = np.array([x, y], dtype=float)
        norm = np.linalg.norm(vec)
        actions.append(tuple(step_scale * vec / max(norm, 1.0)))
    return np.array(actions, dtype=float)


class PointMassWorld:
    """Deterministic point robot used for controlled planning diagnostics."""

    def __init__(self, config: PointMassConfig | None = None):
        self.config = config or PointMassConfig()
        self.start = np.array(self.config.start, dtype=float)
        self.goal = np.array(self.config.goal, dtype=float)
        self.pocket_center = np.array(self.config.pocket_center, dtype=float)

    def reset(self) -> Array:
        return self.start.copy()

    def transition(self, state: Array, action: Array) -> Array:
        low, high = self.config.bounds
        next_state = np.asarray(state, dtype=float) + self.config.step_size * np.asarray(action, dtype=float)
        return np.clip(next_state, low, high)

    def reward(self, state: Array, action: Array | None = None) -> float:
        state = np.asarray(state, dtype=float)
        action_vec = np.zeros(2, dtype=float) if action is None else np.asarray(action, dtype=float)
        distance = float(np.linalg.norm(state - self.goal))
        reward = -distance - self.config.action_cost * float(np.dot(action_vec, action_vec))
        if distance <= self.config.goal_radius:
            reward += self.config.goal_bonus
        reward -= self.config.pocket_true_penalty * self.pocket_score(state)
        return float(reward)

    def step(self, state: Array, action: Array) -> tuple[Array, float, bool, dict[str, float]]:
        next_state = self.transition(state, action)
        reward = self.reward(next_state, action)
        done = bool(np.linalg.norm(next_state - self.goal) <= self.config.goal_radius)
        info = {
            "goal_distance": float(np.linalg.norm(next_state - self.goal)),
            "pocket_score": self.pocket_score(next_state),
        }
        return next_state, reward, done, info

    def pocket_score(self, state: Array) -> float:
        radius = self.config.pocket_radius
        dist2 = float(np.sum((np.asarray(state, dtype=float) - self.pocket_center) ** 2))
        return float(np.exp(-dist2 / (2.0 * radius * radius)))

    def rollout(self, state: Array, actions: Array, gamma: float = 0.98) -> dict[str, float | Array]:
        state_now = np.asarray(state, dtype=float).copy()
        total = 0.0
        discount = 1.0
        max_pocket = 0.0
        min_goal_distance = float(np.linalg.norm(state_now - self.goal))
        for action in np.asarray(actions, dtype=float):
            state_now, reward, done, info = self.step(state_now, action)
            total += discount * reward
            discount *= gamma
            max_pocket = max(max_pocket, float(info["pocket_score"]))
            min_goal_distance = min(min_goal_distance, float(info["goal_distance"]))
            if done:
                break
        return {
            "return": float(total),
            "final_state": state_now.copy(),
            "max_pocket_score": float(max_pocket),
            "min_goal_distance": float(min_goal_distance),
        }

