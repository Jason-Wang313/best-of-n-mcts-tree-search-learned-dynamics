# Novelty Map

## One-Sentence Claim

MCTS over learned dynamics can behave as a test-time optimizer that concentrates computation on localized model optimism, producing rare branch-capture tails that are visible against a static rollout-pool baseline and reducible by uncertainty-aware scoring in a controlled task.

## Positioning Against Closest Work

| Area | Representative work | What it establishes | Gap this repo targets |
| --- | --- | --- | --- |
| Learned latent model planning | PlaNet: Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (arXiv:1811.04551), https://arxiv.org/abs/1811.04551 | Planning in learned latent dynamics can solve visual continuous-control tasks. | We isolate how search itself can exploit learned-model optimism, independent of pixel representation. |
| Learned imagination/value learning | Dreamer: Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (arXiv:1912.01603), https://arxiv.org/abs/1912.01603 | Learned world models and latent value propagation support long-horizon behavior learning. | We stress-test test-time search/value backups when the model/value contains localized optimistic pockets. |
| Probabilistic dynamics and uncertainty | PETS: Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (arXiv:1805.12114), https://arxiv.org/abs/1805.12114 | Ensembles and trajectory sampling improve model-based RL under uncertainty. | We use explicit uncertainty to ask when search should distrust model returns, not only propagate uncertainty. |
| MCTS with imperfect transitions | Kohankhaki et al., "Monte Carlo Tree Search in the Presence of Transition Uncertainty", AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/29994 | Standard MCTS degrades with model transition errors; uncertainty-adapted MCTS improves robustness. | We focus on learned-dynamics optimism as a test-time compute failure mode and compare adaptive tree search to a non-adaptive rollout pool. |
| Continuous-system tree search | Riviere, Lathrop, Chung, "Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems" (arXiv:2412.11270), https://arxiv.org/abs/2412.11270 | Spectral expansion discretizes continuous dynamics for real-time tree search. | Our environment is deliberately small; the novelty is bias-amplification diagnostics, not a new continuous-action expansion rule. |
| Test-time compute scaling | Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (arXiv:2408.03314), https://arxiv.org/abs/2408.03314 | Search and verifier-based inference-time compute can outperform simple independent sampling under compute allocation strategies. | We translate the static-vs-adaptive test-time compute lens to learned-dynamics planning and study failure under biased simulators. |

## Outside-the-Box Angle

The project treats tree search as an optimizer over a learned simulator. This makes model error analogous to verifier/reward-model error in test-time compute systems: repeated search can amplify whatever the learned model overvalues. The repository therefore measures not only task return, but also:

- selected model return minus true return,
- cumulative learned reward bias on the selected plan,
- depth-wise model bias on backed-up trajectories,
- root visit concentration as adaptive compute concentrates,
- uncertainty exposure of the selected branch,
- tail quantiles and paired seed deltas against the static rollout pool.

## Current Empirical Anchor

In the committed full controlled run (`results/full/manifest.json`), the largest budget setting reports:

- UCT MCTS selected-return optimism gap: `0.686`,
- static rollout-pool selected-return optimism gap: `0.301`,
- best calibrated repair (`uncertainty_mcts`) optimism gap: `0.236`.

The paired UCT-minus-static delta has mean `0.386`, median near zero, positive fraction `0.20`, and max `7.021`. This supports a controlled branch-capture tail claim, not a claim that UCT is worse on most seeds. It does not yet establish the effect in high-dimensional robotics benchmarks.
