import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_expansion_suite import run_expansion_suite


def test_expansion_suite_writes_expected_datasets(tmp_path):
    manifest = run_expansion_suite(
        output=tmp_path,
        write_figures=False,
        exploration_seeds=range(1),
        grid_seeds=range(1),
        calibration_seeds=range(1),
        drift_seeds=range(1),
        episode_seeds=range(1),
    )

    metrics = pd.read_csv(manifest["metrics"])
    assert {
        "action_library_sweep",
        "closed_loop_replay",
        "dynamics_drift_stress",
        "exploration_sweep",
        "horizon_budget_sweep",
        "start_state_stress",
        "uncertainty_calibration_stress",
    }.issubset(set(metrics["dataset"]))
    assert {"static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"}.issubset(set(metrics["planner"]))
    assert Path(manifest["claim_audit"]).exists()
