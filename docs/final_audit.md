# Final Audit

## Status

Strong v1 repository created at:

`C:\Users\wangz\best of n mcts tree search learned dynamics`

## Verification Snapshot

- Unit tests: `python -m pytest`
- Smoke run: `python experiments\run_mechanism.py --mode smoke --output results\smoke`
- Full run: `python experiments\run_mechanism.py --mode full --output results\full`
- Paper build command: `.\scripts\build_paper.ps1`

The paper build completed successfully with MiKTeX and copied the PDF to both final paths.

## Current Full-Run Result

From `results/full/manifest.json`:

`At budget 1024, mean selected-return optimism gap was 0.686 for uct_mcts, 0.301 for best-of-N, and 0.236 for the best calibrated repair (uncertainty_mcts).`

This is the strongest clean empirical result in the current artifact set: adaptive UCT search shows a larger optimism gap than static best-of-N under equal forward-model budget, and uncertainty-calibrated search lowers the gap below both.

## Generated Artifacts

- Full CSV/JSON outputs: `results/full/`
- Smoke CSV/JSON outputs: `results/smoke/`
- Paper figures:
  - `paper/figures/compute_scaling.png`
  - `paper/figures/bias_amplification.png`
  - `paper/figures/depth_bias_profile.png`
- Expected final PDF:
  - `paper/final/iclr_submission.pdf`
  - `C:\Users\wangz\Downloads\iclr_submission_bon_mcts_dynamics.pdf`

## Remaining Risks

- The empirical result is controlled-mechanism evidence from a hand-designed surrogate, not trained-model robotics evidence.
- The repair is calibration-dependent.
- The full-run result selects `uncertainty_mcts` as the strongest repair; conservative backup remains implemented but should be studied further before being foregrounded as uniformly superior.
