from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Protocol

import numpy as np

from search_concentration_audit.envs import Array
from search_concentration_audit.models import BiasedLearnedModel


@dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 18
    budget: int = 256
    gamma: float = 0.98
    exploration: float = 1.2
    uncertainty_penalty: float = 0.0
    conservative_backup: bool = False
    value_guided: bool = False
    name: str = "planner"


@dataclass
class PlanResult:
    planner: str
    action: Array
    action_sequence: Array
    predicted_return: float
    model_steps: int
    diagnostics: dict[str, float] = field(default_factory=dict)
    root_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class Planner(Protocol):
    name: str

    def plan(self, model: BiasedLearnedModel, state: Array, seed: int = 0) -> PlanResult:
        ...


def _sample_sequences(rng: np.random.Generator, action_set: Array, n: int, horizon: int) -> Array:
    indices = rng.integers(0, len(action_set), size=(n, horizon))
    return action_set[indices]


class OpenLoopRandomPlanner:
    """Draw one random open-loop sequence and execute its first action."""

    def __init__(self, action_set: Array, config: PlannerConfig):
        self.action_set = np.asarray(action_set, dtype=float)
        self.config = config
        self.name = config.name

    def plan(self, model: BiasedLearnedModel, state: Array, seed: int = 0) -> PlanResult:
        rng = np.random.default_rng(seed)
        sequence = _sample_sequences(rng, self.action_set, 1, self.config.horizon)[0]
        model_rollout = model.rollout(state, sequence, gamma=self.config.gamma, penalty=self.config.uncertainty_penalty)
        return PlanResult(
            planner=self.name,
            action=sequence[0].copy(),
            action_sequence=sequence.copy(),
            predicted_return=float(model_rollout["return"]),
            model_steps=self.config.horizon,
            diagnostics={
                "candidate_count": 1.0,
                "static_search": 1.0,
                "model_bias": float(model_rollout["bias"]),
                "model_uncertainty": float(model_rollout["uncertainty"]),
            },
        )


class StaticRolloutPlanner:
    """Static rollout-pool selection under the learned model."""

    def __init__(self, action_set: Array, config: PlannerConfig):
        self.action_set = np.asarray(action_set, dtype=float)
        self.config = config
        self.name = config.name

    def plan(self, model: BiasedLearnedModel, state: Array, seed: int = 0) -> PlanResult:
        rng = np.random.default_rng(seed)
        n = max(1, self.config.budget // self.config.horizon)
        sequences = _sample_sequences(rng, self.action_set, n, self.config.horizon)
        best_idx = 0
        best_return = -np.inf
        best_diag: dict[str, float | Array] | None = None
        for idx, sequence in enumerate(sequences):
            rollout = model.rollout(state, sequence, gamma=self.config.gamma, penalty=self.config.uncertainty_penalty)
            score = float(rollout["return"])
            if score > best_return:
                best_idx = idx
                best_return = score
                best_diag = rollout
        assert best_diag is not None
        return PlanResult(
            planner=self.name,
            action=sequences[best_idx, 0].copy(),
            action_sequence=sequences[best_idx].copy(),
            predicted_return=float(best_return),
            model_steps=int(n * self.config.horizon),
            diagnostics={
                "candidate_count": float(n),
                "static_search": 1.0,
                "model_bias": float(best_diag["bias"]),
                "model_uncertainty": float(best_diag["uncertainty"]),
                "transition_error": float(best_diag["transition_error"]),
                "max_pocket_score": float(best_diag["max_pocket_score"]),
            },
        )


@dataclass
class Edge:
    action_index: int
    action: Array
    reward: float
    uncertainty: float
    child: "TreeNode"
    visits: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    raw_reward_bias: float = 0.0
    transition_error: float = 0.0
    pocket_score: float = 0.0

    @property
    def mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def std(self) -> float:
        if self.visits <= 1:
            return 0.0
        mean = self.mean
        var = max(0.0, self.value_sq_sum / self.visits - mean * mean)
        return sqrt(var)


@dataclass
class TreeNode:
    state: Array
    depth: int
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, Edge] = field(default_factory=dict)


class MCTSPlanner:
    """UCT-style tree search over a learned model.

    Variants are controlled by PlannerConfig:
    - value_guided adds a learned leaf value estimate.
    - uncertainty_penalty subtracts reported uncertainty from rewards.
    - conservative_backup applies the penalty during backup rather than only in
      rollout scoring, making the repair directly target optimistic backups.
    """

    def __init__(self, action_set: Array, config: PlannerConfig):
        self.action_set = np.asarray(action_set, dtype=float)
        self.config = config
        self.name = config.name

    def plan(self, model: BiasedLearnedModel, state: Array, seed: int = 0) -> PlanResult:
        rng = np.random.default_rng(seed)
        root = TreeNode(np.asarray(state, dtype=float).copy(), depth=0)
        model_steps = 0
        simulations = 0
        while model_steps < self.config.budget:
            path: list[Edge] = []
            nodes: list[TreeNode] = [root]
            node = root

            while node.depth < self.config.horizon and model_steps < self.config.budget:
                if len(node.children) < len(self.action_set):
                    edge = self._expand(model, node, rng)
                    model_steps += 1
                    path.append(edge)
                    node = edge.child
                    nodes.append(node)
                    break
                edge = self._select(node)
                path.append(edge)
                node = edge.child
                nodes.append(node)

            leaf_value = 0.0
            if node.depth < self.config.horizon and model_steps < self.config.budget:
                if self.config.value_guided:
                    remaining = self.config.horizon - node.depth
                    value, uncertainty = model.value(node.state, remaining)
                    leaf_value = value - self.config.uncertainty_penalty * uncertainty * max(remaining, 1)
                else:
                    leaf_value, used = self._rollout(model, node.state, node.depth, rng, model_steps)
                    model_steps += used

            total_return = self._backup(path, leaf_value)
            for visited in nodes:
                visited.visits += 1
                visited.value_sum += total_return
            simulations += 1

        if not root.children:
            action = self.action_set[0].copy()
            sequence = np.repeat(action[None, :], self.config.horizon, axis=0)
            return PlanResult(self.name, action, sequence, 0.0, model_steps, {"simulations": 0.0}, {})

        chosen = max(root.children.values(), key=lambda edge: (edge.mean, edge.visits))
        sequence = self._greedy_sequence(root)
        root_stats = self._root_stats(root)
        concentration = self._visit_concentration(root)
        return PlanResult(
            planner=self.name,
            action=chosen.action.copy(),
            action_sequence=sequence,
            predicted_return=float(chosen.mean),
            model_steps=int(model_steps),
            diagnostics={
                "simulations": float(simulations),
                "tree_nodes": float(self._count_nodes(root)),
                "search_concentration": concentration,
                "adaptive_search": 1.0,
                "root_value": float(root.value_sum / max(root.visits, 1)),
            },
            root_stats=root_stats,
        )

    def _expand(self, model: BiasedLearnedModel, node: TreeNode, rng: np.random.Generator) -> Edge:
        unexplored = [idx for idx in range(len(self.action_set)) if idx not in node.children]
        action_index = int(rng.choice(unexplored))
        action = self.action_set[action_index]
        pred = model.predict(node.state, action)
        child = TreeNode(pred.next_state.copy(), depth=node.depth + 1)
        edge = Edge(
            action_index=action_index,
            action=action.copy(),
            reward=pred.reward,
            uncertainty=pred.uncertainty,
            child=child,
            raw_reward_bias=pred.reward_bias,
            transition_error=pred.transition_error,
            pocket_score=pred.pocket_score,
        )
        node.children[action_index] = edge
        return edge

    def _select(self, node: TreeNode) -> Edge:
        parent_visits = max(node.visits, 1)
        best_score = -np.inf
        best_edge: Edge | None = None
        for edge in node.children.values():
            if edge.visits == 0:
                score = np.inf
            else:
                bonus = self.config.exploration * sqrt(log(parent_visits + 1.0) / edge.visits)
                score = edge.mean + bonus
            if score > best_score:
                best_score = score
                best_edge = edge
        assert best_edge is not None
        return best_edge

    def _rollout(
        self,
        model: BiasedLearnedModel,
        state: Array,
        depth: int,
        rng: np.random.Generator,
        model_steps_so_far: int,
    ) -> tuple[float, int]:
        state_now = np.asarray(state, dtype=float).copy()
        total = 0.0
        discount = 1.0
        used = 0
        while depth + used < self.config.horizon and model_steps_so_far + used < self.config.budget:
            action = self.action_set[int(rng.integers(0, len(self.action_set)))]
            pred = model.predict(state_now, action)
            reward = pred.reward - self.config.uncertainty_penalty * pred.uncertainty
            total += discount * reward
            discount *= self.config.gamma
            state_now = pred.next_state
            used += 1
        return float(total), used

    def _backup(self, path: list[Edge], leaf_value: float) -> float:
        value = float(leaf_value)
        for edge in reversed(path):
            reward = edge.reward
            if self.config.conservative_backup:
                reward -= self.config.uncertainty_penalty * edge.uncertainty
            value = reward + self.config.gamma * value
            edge.visits += 1
            edge.value_sum += value
            edge.value_sq_sum += value * value
        return float(value)

    def _greedy_sequence(self, root: TreeNode) -> Array:
        actions: list[Array] = []
        node = root
        while node.children and len(actions) < self.config.horizon:
            edge = max(node.children.values(), key=lambda child_edge: (child_edge.mean, child_edge.visits))
            actions.append(edge.action.copy())
            node = edge.child
        if not actions:
            actions.append(self.action_set[0].copy())
        while len(actions) < self.config.horizon:
            actions.append(actions[-1].copy())
        return np.asarray(actions, dtype=float)

    def _root_stats(self, root: TreeNode) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for idx, edge in sorted(root.children.items()):
            stats[str(idx)] = {
                "visits": float(edge.visits),
                "mean": float(edge.mean),
                "std": float(edge.std),
                "uncertainty": float(edge.uncertainty),
                "reward_bias": float(edge.raw_reward_bias),
                "transition_error": float(edge.transition_error),
                "pocket_score": float(edge.pocket_score),
                "action_x": float(edge.action[0]),
                "action_y": float(edge.action[1]),
            }
        return stats

    @staticmethod
    def _visit_concentration(root: TreeNode) -> float:
        visits = np.array([edge.visits for edge in root.children.values()], dtype=float)
        total = float(np.sum(visits))
        if total <= 0.0:
            return 0.0
        return float(np.max(visits) / total)

    @staticmethod
    def _count_nodes(root: TreeNode) -> int:
        count = 1
        stack = [root]
        while stack:
            node = stack.pop()
            for edge in node.children.values():
                count += 1
                stack.append(edge.child)
        return count
