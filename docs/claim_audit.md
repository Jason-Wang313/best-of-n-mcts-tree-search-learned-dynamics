# Claim Audit

## Supported By Current Code And Artifacts

- The repository implements a deterministic continuous-control toy world and a biased learned dynamics/reward surrogate.
- The planner suite includes random open-loop, static rollout-pool selection, UCT MCTS, value-guided MCTS, uncertainty-penalized MCTS, and conservative-backup MCTS.
- The diagnostics measure selected-return optimism, model reward bias, uncertainty exposure, transition error, depth-wise bias, search concentration, tail quantiles, and paired seed deltas.
- In `results/full/manifest.json`, UCT MCTS has a larger mean selected-return optimism gap than the static rollout pool at budget 1024 (`0.686` versus `0.301`).
- In `results/full/tail_metrics.csv`, UCT MCTS also has the larger worst optimism event at budget 1024 (`7.021` versus `3.301`).
- In `results/full/pairwise_deltas.csv`, the paired UCT-minus-static mean delta is positive (`0.386`) but the median delta is near zero and only 20% of paired seeds are positive. The supported result is therefore a tail-risk claim, not a dominance claim.
- In the same full run, the best calibrated repair (`uncertainty_mcts`) reduces the mean gap to `0.236` and the paired repair delta mean to `-0.065`.
- Unit tests pass for deterministic transitions, budget accounting, MCTS backup arithmetic, uncertainty behavior, and diagnostic output.

## Plausible But Not Yet Established

- The same effect will appear in high-dimensional image-based robotics tasks.
- The same uncertainty penalty will transfer across environments or model classes.
- Conservative backup is always preferable to uncertainty-penalized search. In the current artifacts, both are implemented, but the best full-run repair is `uncertainty_mcts`.
- Test-time compute scaling curves will remain monotone under larger budgets. The current curves are noisy because the task is intentionally small and seed-sensitive.
- UCT MCTS has uniformly larger optimism than the static rollout pool. The paired evidence contradicts this stronger claim.

## Claims To Avoid

- Do not claim state-of-the-art robotics performance.
- Do not claim a new MCTS convergence theorem beyond the proof sketch.
- Do not claim learned dynamics are trained from data in this version; the current surrogate is hand-designed to expose a mechanism.
- Do not claim uncertainty is calibrated in the statistical sense; it is a controlled signal aligned with the injected bias pocket.
- Do not claim a statistically significant paired mean difference unless a future run adds enough seeds and the interval excludes zero.

## Next Evidence Needed

- Replace the hand-designed model with a trained ensemble on biased/off-policy data.
- Add continuous-action expansion or CEM-guided action proposals.
- Run on at least one standard continuous-control benchmark with learned dynamics.
- Measure repair sensitivity to uncertainty scale and penalty coefficient.
- Add confidence intervals and statistical tests over larger seed counts.
