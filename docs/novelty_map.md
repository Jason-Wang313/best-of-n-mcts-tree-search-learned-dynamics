# Novelty Map

## One-Sentence Claim

MCTS over learned dynamics can behave as a test-time optimizer that adaptively amplifies localized model optimism through repeated expansion and backup, whereas best-of-N rollouts are a static sampling baseline; uncertainty-calibrated search/backups can reduce this amplification in controlled tasks.

## Positioning Against Closest Work

| Area | Representative work | What it establishes | Gap this repo targets |
| --- | --- | --- | --- |
| Learned latent model planning | PlaNet: Hafner et al., "Learning Latent Dynamics for Planning from Pixels" (arXiv:1811.04551), https://arxiv.org/abs/1811.04551 | Planning in learned latent dynamics can solve visual continuous-control tasks. | We isolate how search itself can exploit learned-model optimism, independent of pixel representation. |
| Learned imagination/value learning | Dreamer: Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination" (arXiv:1912.01603), https://arxiv.org/abs/1912.01603 | Learned world models and latent value propagation support long-horizon behavior learning. | We stress-test test-time search/value backups when the model/value contains localized optimistic pockets. |
| Probabilistic dynamics and uncertainty | PETS: Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (arXiv:1805.12114), https://arxiv.org/abs/1805.12114 | Ensembles and trajectory sampling improve model-based RL under uncertainty. | We use explicit uncertainty to ask when search should distrust model returns, not only propagate uncertainty. |
| MCTS with imperfect transitions | Kohankhaki et al., "Monte Carlo Tree Search in the Presence of Transition Uncertainty", AAAI 2024, https://ojs.aaai.org/index.php/AAAI/article/view/29994 | Standard MCTS degrades with model transition errors; uncertainty-adapted MCTS improves robustness. | We focus on learned-dynamics optimism as a test-time compute failure mode and compare adaptive tree search to static best-of-N. |
| Continuous-system tree search | Riviere, Lathrop, Chung, "Monte Carlo Tree Search with Spectral Expansion for Planning with Dynamical Systems" (arXiv:2412.11270), https://arxiv.org/abs/2412.11270 | Spectral expansion discretizes continuous dynamics for real-time tree search. | Our environment is deliberately small; the novelty is bias-amplification diagnostics, not a new continuous-action expansion rule. |
| Test-time compute scaling | Snell et al., "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (arXiv:2408.03314), https://arxiv.org/abs/2408.03314 | Search/verifier-based inference-time compute can outperform static best-of-N under compute allocation strategies. | We translate the static-vs-adaptive test-time compute lens to learned-dynamics planning and study failure under biased verifiers/models. |

## Outside-the-Box Angle

The project treats tree search as an optimizer over a learned simulator. This makes model error analogous to verifier/reward-model error in test-time compute systems: repeated search can amplify whatever the learned model overvalues. The repository therefore measures not only task return, but also:

- selected model return minus true return,
- cumulative learned reward bias on the selected plan,
- depth-wise model bias on backed-up trajectories,
- root visit concentration as adaptive compute concentrates,
- uncertainty exposure of the selected branch.

## Current Empirical Anchor

In the committed full controlled run (`results/full/manifest.json`), the largest budget setting reports:

- UCT MCTS selected-return optimism gap: `0.686`,
- static best-of-N selected-return optimism gap: `0.301`,
- best calibrated repair (`uncertainty_mcts`) optimism gap: `0.236`.

This supports the mechanism claim in a controlled world. It does not yet establish the effect in high-dimensional robotics benchmarks.

