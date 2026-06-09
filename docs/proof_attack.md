# Proof Attack

## Mechanism Sketch

Consider a deterministic search tree evaluated by a learned model \(\hat M\). Let one depth-\(d\) branch enter a bias pocket where the learned return is optimistic by \(\epsilon > 0\) relative to the true return. A backed-up value at the root receives approximately \(\gamma^d \epsilon\) additional value. If that bonus makes the branch exceed competitors by more than the UCB exploration term, repeated MCTS selection increases visits to the same optimistic branch. The learned error is therefore not just observed; it is adaptively re-sampled and propagated.

By contrast, static best-of-N samples \(N\) independent trajectories and selects the maximum learned return once. It can still pick a biased trajectory, but it does not use earlier biased evaluations to allocate additional evaluations toward the same region.

## Conservative Backup Sketch

Suppose the model supplies an uncertainty estimate \(u(s,a)\) satisfying an approximate optimism envelope:

\[
\hat r(s,a) - r(s,a) \leq \lambda u(s,a)
\]

along biased transitions. A lower-confidence backup

\[
Q(s,a) \leftarrow \hat r(s,a) - \lambda u(s,a) + \gamma V(\hat s')
\]

removes the one-step optimistic excess when the envelope is calibrated. Over a trajectory, the discounted correction accumulates and can prevent the optimistic pocket from dominating the root UCB scores.

The current implementation tests two practical variants:

- `uncertainty_mcts`: subtracts uncertainty during rollout/search scoring.
- `conservative_mcts`: subtracts uncertainty during tree backup.

## What Could Break This Argument

- The uncertainty estimate may be miscalibrated or anticorrelated with model bias.
- Optimism may arise from value extrapolation rather than transition/reward error; a one-step transition uncertainty penalty may be insufficient.
- In continuous action spaces, the discretization or expansion policy can dominate the observed effect.
- If the true reward has broad shaping that aligns with the biased pocket, the selected-return gap may shrink even though search concentrates there.
- UCB concentration depends on the relation between bias size, branching factor, rollout noise, and exploration constant.
- Best-of-N can look worse than MCTS when its independent samples happen to hit the bias pocket often; the comparison is distributional, not deterministic per seed.

## Tests Added To Pressure The Mechanism

- Deterministic transition checks ensure the true world is not injecting stochastic confounds.
- Planner budget tests ensure static and adaptive methods are compared by forward-model calls.
- Backup arithmetic tests verify discounted values are propagated as intended.
- Bias-pocket tests check that model optimism and uncertainty are both localized.
- Full repeated-seed results are saved in `results/full/`.

