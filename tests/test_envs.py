import numpy as np

from bonmcts.envs import PointMassWorld, make_action_set


def test_world_transition_is_deterministic():
    world = PointMassWorld()
    state = world.reset()
    action = np.array([1.0, 0.0])

    next_a, reward_a, done_a, info_a = world.step(state, action)
    next_b, reward_b, done_b, info_b = world.step(state, action)

    np.testing.assert_allclose(next_a, next_b)
    assert reward_a == reward_b
    assert done_a == done_b
    assert info_a == info_b


def test_action_set_has_unit_bounded_actions():
    actions = make_action_set()
    assert actions.shape[1] == 2
    assert np.max(np.linalg.norm(actions, axis=1)) <= 1.0 + 1e-9
    assert any(np.allclose(action, [0.0, 0.0]) for action in actions)

