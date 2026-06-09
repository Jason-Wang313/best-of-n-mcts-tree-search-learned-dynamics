from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bonmcts.envs import Array, PointMassWorld


@dataclass(frozen=True)
class BiasPocketConfig:
    """Parameters for a deliberately misspecified learned model."""

    reward_bias: float = 1.15
    dynamics_bias: float = 0.22
    uncertainty_base: float = 0.025
    uncertainty_pocket: float = 0.55
    value_bias: float = 1.0
    value_scale: float = 4.0


@dataclass(frozen=True)
class Prediction:
    next_state: Array
    reward: float
    uncertainty: float
    true_next_state: Array
    true_reward: float
    reward_bias: float
    transition_error: float
    pocket_score: float


class BiasedLearnedModel:
    """A learned dynamics/reward surrogate with a controllable optimism pocket.

    The model is accurate in most of the state space, but near an off-route
    pocket it predicts extra reward and a transition drift that makes the pocket
    appear closer to the goal than it really is. It also reports high epistemic
    uncertainty there, enabling uncertainty-aware repairs to be tested.
    """

    def __init__(self, world: PointMassWorld, config: BiasPocketConfig | None = None):
        self.world = world
        self.config = config or BiasPocketConfig()

    def predict(self, state: Array, action: Array) -> Prediction:
        true_next = self.world.transition(state, action)
        true_reward = self.world.reward(true_next, action)
        pocket = self.world.pocket_score(true_next)
        goal_direction = self.world.goal - true_next
        model_next = true_next + self.config.dynamics_bias * pocket * goal_direction
        low, high = self.world.config.bounds
        model_next = np.clip(model_next, low, high)

        transition_error = float(np.linalg.norm(model_next - true_next))
        reward_bias = self.config.reward_bias * pocket
        reward = self.world.reward(model_next, action) + reward_bias
        uncertainty = self.config.uncertainty_base + self.config.uncertainty_pocket * pocket + transition_error
        return Prediction(
            next_state=model_next.astype(float),
            reward=float(reward),
            uncertainty=float(uncertainty),
            true_next_state=true_next.astype(float),
            true_reward=float(true_reward),
            reward_bias=float(reward - true_reward),
            transition_error=transition_error,
            pocket_score=float(pocket),
        )

    def value(self, state: Array, horizon_remaining: int) -> tuple[float, float]:
        """Return a biased learned value and an uncertainty estimate."""

        state = np.asarray(state, dtype=float)
        dist = float(np.linalg.norm(state - self.world.goal))
        pocket = self.world.pocket_score(state)
        value = -self.config.value_scale * dist + self.config.value_bias * pocket * max(horizon_remaining, 1)
        uncertainty = self.config.uncertainty_base + self.config.uncertainty_pocket * pocket
        return float(value), float(uncertainty)

    def rollout(self, state: Array, actions: Array, gamma: float = 0.98, penalty: float = 0.0) -> dict[str, float | Array]:
        state_now = np.asarray(state, dtype=float).copy()
        total = 0.0
        discount = 1.0
        bias_total = 0.0
        uncertainty_total = 0.0
        transition_error_total = 0.0
        max_pocket = 0.0
        for action in np.asarray(actions, dtype=float):
            pred = self.predict(state_now, action)
            adjusted_reward = pred.reward - penalty * pred.uncertainty
            total += discount * adjusted_reward
            bias_total += discount * pred.reward_bias
            uncertainty_total += discount * pred.uncertainty
            transition_error_total += discount * pred.transition_error
            max_pocket = max(max_pocket, pred.pocket_score)
            discount *= gamma
            state_now = pred.next_state
        return {
            "return": float(total),
            "final_state": state_now.copy(),
            "bias": float(bias_total),
            "uncertainty": float(uncertainty_total),
            "transition_error": float(transition_error_total),
            "max_pocket_score": float(max_pocket),
        }

