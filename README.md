# Best-of-N MCTS Tree Search with Learned Dynamics

This repository is a first-pass research scaffold for studying when tree search over a learned dynamics model acts as a test-time optimizer for model error.

The core thesis is narrow and testable: best-of-N rollout selection samples independent open-loop trajectories, while MCTS adaptively reallocates expansion and backup pressure toward branches that look valuable under the learned model. If the learned model contains localized optimistic errors, those errors can be amplified by repeated expansion and backups. Uncertainty-calibrated search is a simple repair knob.

## What Is Included

- A deterministic 2D point-mass robotics-style world with an off-route bias pocket.
- A learned dynamics/reward surrogate with controllable transition error, reward optimism, and uncertainty estimates.
- Planners:
  - random open-loop rollout,
  - static best-of-N rollout selection,
  - vanilla UCT MCTS,
  - value-guided MCTS,
  - uncertainty-penalized MCTS,
  - conservative-backup MCTS.
- Diagnostics for selected-return optimism gap, reward-bias exposure, transition error, uncertainty exposure, search concentration, and depth-wise bias.
- Reproducible smoke/full experiment scripts and generated CSV/JSON/PNG artifacts.
- An ICLR-style paper source in `paper/`.

## Quick Start

```powershell
python -m pip install -e .[dev]
python -m pytest
python experiments\run_mechanism.py --mode smoke --output results\smoke
python experiments\run_mechanism.py --mode full --output results\full
.\scripts\build_paper.ps1
```

The committed full run currently reports:

```text
At budget 1024, mean selected-return optimism gap was 0.686 for uct_mcts,
0.301 for best-of-N, and 0.236 for the best calibrated repair
(uncertainty_mcts).
```

## Repository Map

- `src/bonmcts/envs.py`: deterministic point-mass world.
- `src/bonmcts/models.py`: biased learned dynamics and value surrogate.
- `src/bonmcts/planners.py`: best-of-N and MCTS variants.
- `src/bonmcts/diagnostics.py`: rollout evaluation and aggregate metrics.
- `experiments/run_mechanism.py`: smoke/full mechanism experiments and figure generation.
- `tests/`: unit tests for dynamics, planner budgets, MCTS backup math, uncertainty behavior, and diagnostics.
- `docs/`: novelty map, proof attack, claim audit, and final audit.
- `paper/`: manuscript source, references, generated figures, and final PDF location.

## Scope

This is controlled-mechanism evidence, not a claim of state-of-the-art robotics performance. The purpose is to isolate a failure mode and a repair direction before scaling to high-dimensional learned world models.

