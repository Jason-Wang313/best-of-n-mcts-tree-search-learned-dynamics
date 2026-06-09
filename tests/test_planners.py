import numpy as np

from bonmcts.envs import PointMassWorld, make_action_set
from bonmcts.models import BiasedLearnedModel
from bonmcts.planners import BestOfNPlanner, Edge, MCTSPlanner, PlannerConfig, TreeNode


def test_best_of_n_budget_accounting():
    world = PointMassWorld()
    model = BiasedLearnedModel(world)
    planner = BestOfNPlanner(
        make_action_set(),
        PlannerConfig(name="best_of_n", horizon=5, budget=23),
    )

    result = planner.plan(model, world.reset(), seed=1)

    assert result.model_steps == 20
    assert result.model_steps <= 23
    assert result.action_sequence.shape == (5, 2)


def test_mcts_budget_accounting():
    world = PointMassWorld()
    model = BiasedLearnedModel(world)
    planner = MCTSPlanner(
        make_action_set(),
        PlannerConfig(name="uct", horizon=6, budget=31),
    )

    result = planner.plan(model, world.reset(), seed=2)

    assert result.model_steps <= 31
    assert result.model_steps > 0
    assert result.action_sequence.shape == (6, 2)


def test_mcts_backup_correctness():
    planner = MCTSPlanner(
        make_action_set(),
        PlannerConfig(name="uct", horizon=3, budget=3, gamma=0.5),
    )
    leaf = TreeNode(np.array([0.0, 0.0]), depth=2)
    edge2 = Edge(0, np.zeros(2), reward=2.0, uncertainty=0.0, child=leaf)
    edge1 = Edge(1, np.ones(2), reward=1.0, uncertainty=0.0, child=TreeNode(np.ones(2), depth=1))

    total = planner._backup([edge1, edge2], leaf_value=4.0)

    assert total == 1.0 + 0.5 * (2.0 + 0.5 * 4.0)
    assert edge1.visits == 1
    assert edge2.visits == 1
    assert edge1.mean == total
    assert edge2.mean == 4.0


def test_conservative_backup_penalizes_uncertainty():
    plain = MCTSPlanner(
        make_action_set(),
        PlannerConfig(name="plain", horizon=1, budget=1, gamma=1.0, uncertainty_penalty=0.5),
    )
    conservative = MCTSPlanner(
        make_action_set(),
        PlannerConfig(
            name="conservative",
            horizon=1,
            budget=1,
            gamma=1.0,
            uncertainty_penalty=0.5,
            conservative_backup=True,
        ),
    )
    plain_edge = Edge(0, np.zeros(2), reward=3.0, uncertainty=2.0, child=TreeNode(np.zeros(2), depth=1))
    conservative_edge = Edge(0, np.zeros(2), reward=3.0, uncertainty=2.0, child=TreeNode(np.zeros(2), depth=1))

    assert plain._backup([plain_edge], leaf_value=0.0) == 3.0
    assert conservative._backup([conservative_edge], leaf_value=0.0) == 2.0

