from bonmcts.diagnostics import evaluate_plan
from bonmcts.envs import PointMassWorld, make_action_set
from bonmcts.models import BiasedLearnedModel
from bonmcts.planners import BestOfNPlanner, PlannerConfig


def test_diagnostics_expose_known_bias_gap():
    world = PointMassWorld()
    model = BiasedLearnedModel(world)
    planner = BestOfNPlanner(
        make_action_set(),
        PlannerConfig(name="best_of_n", horizon=10, budget=100),
    )
    result = planner.plan(model, world.reset(), seed=42)

    metrics = evaluate_plan(world, model, world.reset(), result)

    assert "selected_return_gap" in metrics
    assert metrics["model_steps"] == result.model_steps
    assert metrics["model_uncertainty"] >= 0.0

