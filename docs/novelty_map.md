# Novelty Map

## One-Sentence Claim

MCTS over learned dynamics can behave as a test-time optimizer that concentrates computation on localized model optimism, producing rare branch-capture tails that are visible against a static rollout-pool baseline; uncertainty-aware scoring can repair replayed capture events when calibrated, but can also backfire under reduced-budget stress.

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

## Tail-Stress And Expansion Extensions

The final pass adds `experiments/run_tail_stress.py` and `experiments/run_expansion_suite.py`, which make the tail claim much harder to attack:

- Capture replay keeps the same 20 full-run seeds and shows the severe UCT events are concentrated in a small number of seeds, especially seeds 4 and 5.
- Uncertainty-penalty sensitivity shows the original repair is not the only knob: penalty `2.40` reduces the max replayed capture gap from `7.021` to `0.430` and the 90th percentile to `0.036`.
- Reward-bias-strength stress shows UCT max tail severity grows as the injected pocket optimism grows (`2.566` at zero reward bias to `8.903` at high reward bias), while the strong uncertainty penalty controls the high-bias tail.
- The expansion suite adds exploration-constant, horizon/budget, action-library, uncertainty-calibration, dynamics-drift, start-state-geometry, and closed-loop stress tests.
- The calibration slice deliberately keeps a repair-backfire case: the strong uncertainty penalty can produce a `5.772` maximum gap at budget `768` while UCT is benign on the same seeds. This prevents the paper from claiming uncertainty penalties are universally protective.

This keeps the paper distinct from the other projects: the object of study is MCTS allocation feedback over a biased learned simulator, not generic independent candidate selection.
