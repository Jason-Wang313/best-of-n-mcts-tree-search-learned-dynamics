# Final Audit

## Status

Submission-oriented v2 repository prepared at:

`C:\Users\wangz\best of n mcts tree search learned dynamics`

## Verification Snapshot

- Unit tests: `python -m pytest`
- Smoke run: `python experiments\run_mechanism.py --mode smoke --output results\smoke`
- Full run: `python experiments\run_mechanism.py --mode full --output results\full`
- Paper build command: `.\scripts\build_paper.ps1`

The paper build copies the PDF to `paper/final/iclr_submission.pdf`, Downloads, and the versioned Desktop artifact.

## Current Full-Run Result

From `results/full/manifest.json`:

`At budget 1024, mean selected-return optimism gap was 0.686 for UCT MCTS, 0.301 for the static rollout pool, and 0.236 for the best calibrated repair (Uncertainty MCTS). The paired UCT MCTS minus static delta had mean 0.386, median -0.000, positive fraction 0.20, and max 7.021.`

This is now framed as a tail-risk result: adaptive UCT search has a larger mean gap and a larger worst branch-capture event under equal forward-model budget, but the paired evidence does not support a claim that UCT is worse on most seeds. Uncertainty-calibrated search lowers the mean gap and paired mean delta in this artifact.

## Generated Artifacts

- Full CSV/JSON outputs: `results/full/`
- Smoke CSV/JSON outputs: `results/smoke/`
- Tail and paired audit outputs:
  - `results/full/tail_metrics.csv`
  - `results/full/pairwise_deltas.csv`
- Paper figures:
  - `paper/figures/compute_scaling.png`
  - `paper/figures/bias_amplification.png`
  - `paper/figures/depth_bias_profile.png`
- Expected final PDF:
  - `paper/final/iclr_submission.pdf`
  - `C:\Users\wangz\Downloads\iclr_submission_bon_mcts_dynamics.pdf`
  - `C:\Users\wangz\OneDrive\Desktop\best of n mcts tree search learned dynamics-v2.pdf`

## Remaining Risks

- The empirical result is controlled-mechanism evidence from a hand-designed surrogate, not trained-model robotics evidence.
- The repair is calibration-dependent.
- The full-run result selects `uncertainty_mcts` as the strongest repair; conservative backup remains implemented but should be studied further before being foregrounded as uniformly superior.
- The paired bootstrap interval should be treated as weak at 20 seeds; the manuscript avoids statistical-significance language.
