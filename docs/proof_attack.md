# Proof Attack

## Mechanism Sketch

Consider a deterministic search tree evaluated by a learned model \(\hat M\). Let one depth-\(d\) branch enter a bias pocket where the learned return is optimistic by \(\epsilon > 0\) relative to the true return. A backed-up value at the root receives approximately \(\gamma^d \epsilon\) additional value. If that bonus makes the branch exceed competitors by more than the UCB exploration term, repeated MCTS selection increases visits to the same optimistic branch. The learned error is therefore not just observed; it is adaptively re-sampled and propagated.

By contrast, the static rollout pool scores independent trajectories and selects the maximum learned return once. It can still pick a biased trajectory, but it does not use earlier biased evaluations to allocate additional evaluations toward the same region.

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
- The static rollout pool can look worse than MCTS when its independent samples happen to hit the bias pocket often; the comparison is distributional, not deterministic per seed.
- In the current full run, UCT's paired median delta against the static pool is near zero. The proof sketch supports possible branch-capture amplification, not a guarantee that every paired seed will show larger optimism.

## Tests Added To Pressure The Mechanism

- Deterministic transition checks ensure the true world is not injecting stochastic confounds.
- Planner budget tests ensure static and adaptive methods are compared by forward-model calls.
- Backup arithmetic tests verify discounted values are propagated as intended.
- Bias-pocket tests check that model optimism and uncertainty are both localized.
- Full repeated-seed results are saved in `results/full/`.
- Tail summaries and paired deltas are saved to expose whether any headline mean is driven by rare events.
- `experiments/run_tail_stress.py` replays capture seeds, sweeps uncertainty penalties, and varies reward-bias strength.
- `experiments/run_expansion_suite.py` adds exploration, horizon/budget, action-library, calibration, dynamics-drift, start-state, and closed-loop stress tests.
- `experiments/run_claim_audit.py` generates pass/fail claim checks so the paper cannot silently drift into a dominance or benchmark claim.

## Final Attack Result

The strongest attack is that the main mean gap might be a fragile seed artifact. The tail-stress pass answers this by embracing the tail structure: UCT is worse on only a minority of seeds, but its maximum replayed branch-capture delta is large. A stronger uncertainty penalty collapses that maximum in the same seed set, and increasing the injected reward optimism increases the UCT maximum.

The second strongest attack is that the repair story is too optimistic. The expansion suite answers by preserving a reduced-budget backfire case: a strong uncertainty penalty can redirect search into a larger selected-return gap even when the replayed capture seeds are repaired. The result remains a controlled mechanism claim, not a theorem for all MCTS planners or all uncertainty estimates.
