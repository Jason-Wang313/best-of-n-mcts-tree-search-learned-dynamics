from __future__ import annotations

import numpy as np

from search_concentration_audit.cliff_benchmark import (
    TabularCliffModel,
    evaluate_selected_sequence,
    mcts_plan,
    rollout_true,
    static_rollout_plan,
)


def test_biased_cliff_model_overvalues_shortcut():
    shortcut = np.array([1, 1, 1, 1], dtype=int)
    learned = evaluate_selected_sequence(TabularCliffModel(biased=True), shortcut)
    true = rollout_true(shortcut)

    assert learned["selected_return_gap"] > 100.0
    assert true["cliff_resets"] >= 1.0


def test_uncertainty_penalty_reduces_cliff_shortcut_exposure():
    learned = TabularCliffModel(biased=True)
    uct_seq, _ = mcts_plan(learned, budget=128, horizon=16, seed=123)
    repair_seq, _ = mcts_plan(learned, budget=128, horizon=16, seed=123, uncertainty_penalty=6.0)
    uct = evaluate_selected_sequence(learned, uct_seq)
    repair = evaluate_selected_sequence(learned, repair_seq, penalty=6.0)

    assert repair["model_cliff_bias"] <= uct["model_cliff_bias"]
    assert repair["selected_return_gap"] < uct["selected_return_gap"]


def test_behavioral_static_pool_avoids_uniform_pool_failure_mode():
    learned = TabularCliffModel(biased=True)
    behavioral, _ = static_rollout_plan(learned, budget=256, horizon=16, seed=42, proposal="behavioral")
    uniform, _ = static_rollout_plan(learned, budget=256, horizon=16, seed=42, proposal="uniform")
    behavioral_eval = evaluate_selected_sequence(learned, behavioral)
    uniform_eval = evaluate_selected_sequence(learned, uniform)

    assert behavioral_eval["selected_return_gap"] < uniform_eval["selected_return_gap"]
