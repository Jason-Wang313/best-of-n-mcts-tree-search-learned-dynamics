from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt

import gymnasium as gym
import numpy as np


ROWS = 4
COLS = 12
START_STATE = 36
GOAL_STATE = 47
CLIFF_STATES = set(range(37, 47))
ACTION_COUNT = 4


def _build_true_table() -> tuple[tuple[tuple[int, float, bool], ...], ...]:
    env = gym.make("CliffWalking-v1")
    try:
        return tuple(
            tuple(
                (
                    int(env.unwrapped.P[state][action][0][1]),
                    float(env.unwrapped.P[state][action][0][2]),
                    bool(env.unwrapped.P[state][action][0][3]),
                )
                for action in range(ACTION_COUNT)
            )
            for state in range(ROWS * COLS)
        )
    finally:
        env.close()


TRUE_TABLE = _build_true_table()


def row_col(state: int) -> tuple[int, int]:
    return divmod(int(state), COLS)


def encode(row: int, col: int) -> int:
    return int(max(0, min(ROWS - 1, row)) * COLS + max(0, min(COLS - 1, col)))


def intended_next(state: int, action: int) -> int:
    row, col = row_col(state)
    if action == 0:
        row -= 1
    elif action == 1:
        col += 1
    elif action == 2:
        row += 1
    elif action == 3:
        col -= 1
    else:
        raise ValueError(action)
    return encode(row, col)


@dataclass(frozen=True)
class DiscreteStep:
    next_state: int
    reward: float
    done: bool
    uncertainty: float
    cliff_bias: float


@dataclass(frozen=True)
class TabularCliffModel:
    """True or biased learned dynamics for CliffWalking-v1.

    The biased model treats the cliff row as a safe shortcut with high reward.
    Uncertainty is intentionally high on that shortcut so an uncertainty-aware
    planner has a fair public signal to use.
    """

    biased: bool = True
    shortcut_reward: float = 2.8
    goal_bonus: float = 10.0
    cliff_uncertainty: float = 2.2
    normal_uncertainty: float = 0.04

    def step(self, state: int, action: int) -> DiscreteStep:
        if not self.biased:
            next_state, reward, done = TRUE_TABLE[int(state)][int(action)]
            return DiscreteStep(next_state, reward, done, self.normal_uncertainty, 0.0)

        proposed = intended_next(int(state), int(action))
        if int(state) in CLIFF_STATES or proposed in CLIFF_STATES:
            done = proposed == GOAL_STATE
            reward = self.goal_bonus if done else self.shortcut_reward
            return DiscreteStep(proposed, reward, done, self.cliff_uncertainty, 1.0)
        if proposed == GOAL_STATE:
            return DiscreteStep(proposed, self.goal_bonus, True, self.normal_uncertainty, 0.0)
        next_state, reward, done = TRUE_TABLE[int(state)][int(action)]
        return DiscreteStep(next_state, reward, done, self.normal_uncertainty, 0.0)


def rollout_model(
    model: TabularCliffModel,
    actions: np.ndarray,
    *,
    gamma: float = 0.99,
    uncertainty_penalty: float = 0.0,
    start_state: int = START_STATE,
) -> dict[str, float]:
    state = int(start_state)
    total = 0.0
    discount = 1.0
    uncertainty = 0.0
    cliff_bias = 0.0
    reached_goal = 0.0
    for action in np.asarray(actions, dtype=int).reshape(-1):
        step = model.step(state, int(action))
        total += discount * (step.reward - uncertainty_penalty * step.uncertainty)
        uncertainty += step.uncertainty
        cliff_bias += step.cliff_bias
        state = step.next_state
        discount *= gamma
        if step.done:
            reached_goal = 1.0
            break
    return {
        "return": float(total),
        "uncertainty": float(uncertainty),
        "cliff_bias": float(cliff_bias),
        "reached_goal": float(reached_goal),
    }


def rollout_true(actions: np.ndarray, *, gamma: float = 0.99, start_state: int = START_STATE) -> dict[str, float]:
    state = int(start_state)
    total = 0.0
    discount = 1.0
    cliff_resets = 0.0
    reached_goal = 0.0
    for action in np.asarray(actions, dtype=int).reshape(-1):
        next_state, reward, done = TRUE_TABLE[state][int(action)]
        total += discount * reward
        if reward <= -100.0:
            cliff_resets += 1.0
        state = next_state
        discount *= gamma
        if done:
            reached_goal = 1.0
            break
    return {"return": float(total), "cliff_resets": float(cliff_resets), "reached_goal": float(reached_goal)}


def sample_sequences(rng: np.random.Generator, count: int, horizon: int) -> np.ndarray:
    return rng.integers(0, ACTION_COUNT, size=(int(count), int(horizon)), dtype=np.int64)


def sample_behavior_sequences(rng: np.random.Generator, count: int, horizon: int) -> np.ndarray:
    sequences = np.zeros((int(count), int(horizon)), dtype=np.int64)
    for i in range(int(count)):
        state = START_STATE
        for t in range(int(horizon)):
            row, col = row_col(state)
            if row == 3 and col < COLS - 1:
                probs = np.array([0.94, 0.005, 0.005, 0.05], dtype=float)
            elif row == 2 and col < COLS - 1:
                probs = np.array([0.03, 0.92, 0.001, 0.049], dtype=float)
            elif row == 2 and col == COLS - 1:
                probs = np.array([0.04, 0.05, 0.86, 0.05], dtype=float)
            else:
                probs = np.array([0.10, 0.50, 0.25, 0.15], dtype=float)
            action = int(rng.choice(ACTION_COUNT, p=probs / probs.sum()))
            sequences[i, t] = action
            next_state, _, done = TRUE_TABLE[state][action]
            state = next_state
            if done:
                break
    return sequences


def static_rollout_plan(
    model: TabularCliffModel,
    *,
    budget: int,
    horizon: int,
    seed: int,
    uncertainty_penalty: float = 0.0,
    proposal: str = "behavioral",
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    count = max(1, int(budget) // int(horizon))
    if proposal == "behavioral":
        sequences = sample_behavior_sequences(rng, count, horizon)
    elif proposal == "uniform":
        sequences = sample_sequences(rng, count, horizon)
    else:
        raise ValueError(proposal)
    best_idx = 0
    best_score = -np.inf
    best_info: dict[str, float] = {}
    for idx, sequence in enumerate(sequences):
        info = rollout_model(model, sequence, uncertainty_penalty=uncertainty_penalty)
        if info["return"] > best_score:
            best_score = info["return"]
            best_idx = int(idx)
            best_info = info
    best_info = dict(best_info)
    best_info["candidate_count"] = float(count)
    return sequences[best_idx].copy(), best_info


@dataclass
class Edge:
    action: int
    reward: float
    uncertainty: float
    cliff_bias: float
    child: "Node"
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass
class Node:
    state: int
    depth: int
    terminal: bool = False
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, Edge] = field(default_factory=dict)


def mcts_plan(
    model: TabularCliffModel,
    *,
    budget: int,
    horizon: int,
    seed: int,
    exploration: float = 1.25,
    uncertainty_penalty: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    root = Node(START_STATE, 0, False)
    model_steps = 0
    simulations = 0
    while model_steps < int(budget):
        node = root
        path: list[Edge] = []
        while node.depth < int(horizon) and not node.terminal and model_steps < int(budget):
            if len(node.children) < ACTION_COUNT:
                action = int(rng.choice([a for a in range(ACTION_COUNT) if a not in node.children]))
                step = model.step(node.state, action)
                reward = step.reward - uncertainty_penalty * step.uncertainty
                child = Node(step.next_state, node.depth + 1, step.done)
                edge = Edge(action, reward, step.uncertainty, step.cliff_bias, child)
                node.children[action] = edge
                path.append(edge)
                node = child
                model_steps += 1
                break
            edge = _select_edge(node, exploration)
            path.append(edge)
            node = edge.child

        leaf_value = 0.0
        discount = 1.0
        while node.depth < int(horizon) and not node.terminal and model_steps < int(budget):
            action = int(rng.integers(0, ACTION_COUNT))
            step = model.step(node.state, action)
            leaf_value += discount * (step.reward - uncertainty_penalty * step.uncertainty)
            discount *= 0.99
            node = Node(step.next_state, node.depth + 1, step.done)
            model_steps += 1

        value = float(leaf_value)
        for edge in reversed(path):
            value = edge.reward + 0.99 * value
            edge.visits += 1
            edge.value_sum += value
        root.visits += 1
        root.value_sum += value
        simulations += 1

    sequence = _greedy_sequence(root, horizon)
    root_visits = np.array([edge.visits for edge in root.children.values()], dtype=float)
    total_visits = float(root_visits.sum()) if root_visits.size else 0.0
    model_info = rollout_model(model, sequence, uncertainty_penalty=uncertainty_penalty)
    model_info.update(
        {
            "simulations": float(simulations),
            "model_steps": float(model_steps),
            "root_visit_concentration": float(root_visits.max() / total_visits) if total_visits else 0.0,
        }
    )
    return sequence, model_info


def _select_edge(node: Node, exploration: float) -> Edge:
    parent_visits = max(sum(edge.visits for edge in node.children.values()), 1)
    best_score = -np.inf
    best_edge: Edge | None = None
    for edge in node.children.values():
        if edge.visits == 0:
            score = np.inf
        else:
            score = edge.mean + exploration * sqrt(log(parent_visits + 1.0) / edge.visits)
        if score > best_score:
            best_score = score
            best_edge = edge
    assert best_edge is not None
    return best_edge


def _greedy_sequence(root: Node, horizon: int) -> np.ndarray:
    actions: list[int] = []
    node = root
    while node.children and len(actions) < int(horizon):
        edge = max(node.children.values(), key=lambda e: (e.mean, e.visits))
        actions.append(edge.action)
        node = edge.child
        if node.terminal:
            break
    while len(actions) < int(horizon):
        actions.append(0)
    return np.asarray(actions, dtype=np.int64)


def evaluate_selected_sequence(model: TabularCliffModel, sequence: np.ndarray, penalty: float = 0.0) -> dict[str, float]:
    predicted = rollout_model(model, sequence, uncertainty_penalty=penalty)
    true = rollout_true(sequence)
    return {
        "predicted_return": predicted["return"],
        "true_return": true["return"],
        "selected_return_gap": predicted["return"] - true["return"],
        "model_uncertainty": predicted["uncertainty"],
        "model_cliff_bias": predicted["cliff_bias"],
        "true_cliff_resets": true["cliff_resets"],
        "true_reached_goal": true["reached_goal"],
    }
