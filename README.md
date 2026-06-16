# Search-Concentration Audits for Learned-Dynamics Tree Planning

This repository studies when adaptive tree search over a learned dynamics model concentrates computation on local model optimism instead of real return.

The core thesis is narrow and testable: a static rollout pool scores independent open-loop trajectories once, while MCTS adaptively reallocates expansion and backup pressure toward branches that look valuable under the learned model. If the learned model contains localized optimistic errors, adaptive allocation can create rare branch-capture events. Uncertainty-calibrated search is tested as a repair knob.

## What Is Included

- A deterministic 2D point-mass robotics-style world with an off-route bias pocket.
- A learned dynamics/reward surrogate with controllable transition error, reward optimism, and uncertainty estimates.
- Planners:
  - random open-loop rollout,
  - static rollout-pool selection,
  - vanilla UCT MCTS,
  - value-guided MCTS,
  - uncertainty-penalized MCTS,
  - conservative-backup MCTS.
- Diagnostics for selected-return optimism gap, reward-bias exposure, transition error, uncertainty exposure, search concentration, depth-wise bias, tail quantiles, and paired seed deltas.
- Reproducible smoke/full, tail-stress, and expansion-suite experiment scripts with generated CSV/JSON/PNG artifacts.
- An ICLR-style paper source in `paper/`.

## Quick Start

```powershell
python -m pip install -e .[dev]
python -m pytest
python experiments\run_mechanism.py --mode smoke --output results\smoke
python experiments\run_mechanism.py --mode full --output results\full
python experiments\run_tail_stress.py --output results\tail_stress
python experiments\run_expansion_suite.py --output results\expansion
python experiments\run_cliffwalking_benchmark.py --output results\cliffwalking_benchmark
python experiments\run_claim_audit.py
.\scripts\build_paper.ps1 -DesktopCopy "C:\Users\wangz\OneDrive\Desktop\best of n mcts tree search learned dynamics-v4.pdf"
```

The committed full run currently reports:

```text
At budget 1024, mean selected-return optimism gap was 0.686 for UCT MCTS,
0.301 for the static rollout pool, and 0.236 for the original calibrated repair
(Uncertainty MCTS). The paired UCT MCTS minus static delta had mean 0.386,
median -0.000, positive fraction 0.20, and max 7.021. The v3 tail-stress pass
replays the capture seeds and finds that a stronger uncertainty penalty reduces
the max capture gap from 7.021 to 0.430 while keeping the claim tail-specific.
The expansion suite adds exploration, horizon/budget, action-library,
uncertainty-calibration, dynamics-drift, start-state, and closed-loop stress
tests. It also finds a reduced-budget repair-backfire case, so the paper keeps
the uncertainty claim conditional rather than universal. The v4 standard
benchmark tier adds Gymnasium CliffWalking-v1: UCT amplifies the learned
cliff-shortcut optimism gap by 165.38 relative to a behavioral static rollout
pool, while uncertainty-penalized MCTS reduces the gap by 184.02 and improves
true return by 166.82.
```

## Repository Map

- `src/search_concentration_audit/envs.py`: deterministic point-mass world.
- `src/search_concentration_audit/models.py`: biased learned dynamics and value surrogate.
- `src/search_concentration_audit/planners.py`: static rollout and MCTS variants.
- `src/search_concentration_audit/diagnostics.py`: rollout evaluation and aggregate metrics.
- `experiments/run_mechanism.py`: smoke/full mechanism experiments and figure generation.
- `experiments/run_tail_stress.py`: v3 capture replay, penalty sweep, and bias-strength stress tests.
- `experiments/run_expansion_suite.py`: exploration, horizon/budget, action-library, calibration, dynamics, geometry, and closed-loop stress tests.
- `experiments/run_cliffwalking_benchmark.py`: Gymnasium CliffWalking-v1 standard tabular planning benchmark.
- `experiments/run_claim_audit.py`: generated pass/fail audit for manuscript claims.
- `tests/`: unit tests for dynamics, planner budgets, MCTS backup math, uncertainty behavior, and diagnostics.
- `docs/`: novelty map, proof attack, claim audit, and final audit.
- `paper/`: manuscript source, references, generated figures, and final v4 PDF location.

## Scope

This is controlled-mechanism evidence plus a standard tabular planning benchmark, not a claim of state-of-the-art robotics performance. The supported claim is tail-risk specific: adaptive search can create rare optimistic branch-capture events under a biased learned simulator. Uncertainty-aware scoring can reduce the observed tail when uncertainty and penalty strength are calibrated, but the committed expansion suite also includes a counterexample where a strong penalty backfires.
