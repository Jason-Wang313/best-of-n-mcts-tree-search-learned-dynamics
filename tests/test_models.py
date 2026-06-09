import numpy as np

from bonmcts.envs import PointMassWorld
from bonmcts.models import BiasedLearnedModel


def test_bias_pocket_reports_higher_bias_and_uncertainty():
    world = PointMassWorld()
    model = BiasedLearnedModel(world)
    pocket_state = world.pocket_center.copy()
    far_state = world.reset()
    zero = np.zeros(2)

    pocket = model.predict(pocket_state, zero)
    far = model.predict(far_state, zero)

    assert pocket.reward_bias > far.reward_bias
    assert pocket.uncertainty > far.uncertainty
    assert pocket.pocket_score > far.pocket_score

