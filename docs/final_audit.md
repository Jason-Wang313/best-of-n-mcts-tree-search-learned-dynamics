# Final Audit

## Final Artifact and Provenance

- Paper: `best of n mcts tree search learned dynamics-v4.pdf`
- Source folder: `C:\Users\wangz\best of n mcts tree search learned dynamics`
- GitHub remote: `https://github.com/Jason-Wang313/best-of-n-mcts-tree-search-learned-dynamics.git`
- Repository PDF: `paper/final/best of n mcts tree search learned dynamics-v4.pdf`
- Visible Desktop PDF: `C:\Users\wangz\OneDrive\Desktop\best of n mcts tree search learned dynamics-v4.pdf`
- SHA256: `FDCB633D734B7E24069D187137B7EBC1287BAE6B192DEF041423A7B4E0F425C7`
- Page count: 26
- Repo/Desktop hash match: yes
- Verified on: 2026-06-16

## Final Verification

```powershell
python -m compileall src experiments tests -q
python -m pytest -q
python experiments\run_claim_audit.py
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1 -DesktopCopy "C:\Users\wangz\OneDrive\Desktop\best of n mcts tree search learned dynamics-v4.pdf"
rg -n "undefined|Citation.*undefined|Reference.*undefined|Rerun to get|Overfull|LaTeX Warning|Package natbib Warning" "paper\main.log"
pdfinfo "paper\final\best of n mcts tree search learned dynamics-v4.pdf"
pdftoppm -png "paper\final\best of n mcts tree search learned dynamics-v4.pdf" "tmp\pdfs\mcts_v4\page"
```

Results:

- Compile check: passed.
- Unit tests: 13 passed.
- Claim audit: all claims passed.
- Final LaTeX log scan: no unresolved citations, unresolved references, rerun warnings, overfull boxes, or natbib warnings.
- PDF render: all 26 pages rendered.
- Visual QA: pages 1, 5, 17, 18, 19, and 26 inspected for title/abstract, expansion table, CliffWalking benchmark, references, claim ledger, appendix tail, clipping, and readability.

1. **Core thesis.** Adaptive tree search over learned dynamics can concentrate computation on local model optimism and create rare branch-capture events.

2. **What is new.** The paper is not a generic test-time-compute curve. It audits the allocation feedback loop in MCTS: optimistic branches receive more visits, backups, and model queries.

3. **V3 evidence added.** The tail-stress pass adds capture-seed replay, uncertainty-penalty sensitivity, and reward-bias-strength stress. The expansion suite adds exploration, horizon/budget, action-library, uncertainty-calibration, dynamics-drift, start-state, and closed-loop stress tests.

4. **V4 benchmark evidence.** Gymnasium CliffWalking-v1 is now included as a standard tabular planning benchmark. The biased learned model treats the cliff row as a shortcut; UCT concentrates on that learned shortcut, while uncertainty-penalized MCTS repairs the selected-return gap and true return.

5. **Strongest caution.** The paired median in the point-mass run is near zero and UCT is worse on only a minority of paired seeds. The claim must remain a branch-capture tail-risk claim, not a dominance claim.

6. **Main remaining limitation.** The learned models are hand-designed; the next version should train a dynamics ensemble on biased data and evaluate on continuous-control benchmarks.

7. **Final PDF location.** Expected repository path: `paper/final/best of n mcts tree search learned dynamics-v4.pdf`. The visible Desktop copy is produced with `scripts\build_paper.ps1 -DesktopCopy` after verification.

## Claim Status

- `uct_mean_gap_exceeds_static_at_1024`: pass (0.3859 vs 0.2500)
- `uct_tail_max_exceeds_static_at_1024`: pass (3.7208 vs 3.0000)
- `paired_result_is_tail_not_dominance`: pass (0.2000 vs 0.2500)
- `uncertainty_repair_reduces_mean_and_q90`: pass (0.4506 vs 0.3000)
- `bias_strength_increases_uct_tail`: pass (6.3371 vs 5.0000)
- `penalty_sweep_finds_low_tail_region`: pass (0.4305 vs 0.5000)
- `strong_uncertainty_controls_high_bias_tail`: pass (8.3129 vs 6.0000)
- `strong_uncertainty_penalty_reduces_capture_tail`: pass (6.5910 vs 6.0000)
- `uct_tail_is_rare_not_dominance`: pass (7.0215 vs 5.0000)
- `action_library_changes_tail_size`: pass (0.2313 vs 0.1000)
- `calibration_sweep_finds_repair_backfire_case`: pass (5.6192 vs 1.0000)
- `closed_loop_uncertainty_reduces_mean_plan_gap`: pass (1.0551 vs 0.0500)
- `dynamics_drift_changes_tail_size`: pass (3.1003 vs 0.2500)
- `exploration_constant_changes_tail_size`: pass (0.7312 vs 0.5000)
- `horizon_budget_changes_tail_size`: pass (1.0907 vs 0.2500)
- `start_state_changes_tail_exposure`: pass (11.5836 vs 0.2500)
- `uncertainty_strength_alone_does_not_fix_backfire`: pass (0.0000 vs 0.2500)
- `cliff_uct_gap_exceeds_static`: pass (165.3796 vs 50.0000)
- `cliff_uncertainty_reduces_gap`: pass (-184.0181 vs -100.0000)
- `cliff_uncertainty_improves_true_return`: pass (166.8224 vs 100.0000)
- `cliff_uct_concentrates_on_shortcut`: pass (3.5500 vs 2.0000)
- `cliff_uncertainty_avoids_shortcut`: pass (-3.6500 vs -2.0000)
- `cliff_true_model_has_low_gap`: pass (0.0000 vs 1.0000)
