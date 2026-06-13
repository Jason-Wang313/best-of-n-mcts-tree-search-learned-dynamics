from search_concentration_audit.diagnostics import evaluate_plan
from search_concentration_audit.envs import PointMassWorld, make_action_set
from search_concentration_audit.models import BiasedLearnedModel
from search_concentration_audit.planners import StaticRolloutPlanner, PlannerConfig


def test_diagnostics_expose_known_bias_gap():
    world = PointMassWorld()
    model = BiasedLearnedModel(world)
    planner = StaticRolloutPlanner(
        make_action_set(),
        PlannerConfig(name="static_rollout_pool", horizon=10, budget=100),
    )
    result = planner.plan(model, world.reset(), seed=42)

    metrics = evaluate_plan(world, model, world.reset(), result)

    assert "selected_return_gap" in metrics
    assert metrics["model_steps"] == result.model_steps
    assert metrics["model_uncertainty"] >= 0.0

