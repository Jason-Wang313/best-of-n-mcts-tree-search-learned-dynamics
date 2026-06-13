import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_tail_stress import run_tail_stress


def test_tail_stress_writes_expected_datasets(tmp_path):
    manifest = run_tail_stress(
        output=tmp_path,
        budget=64,
        horizon=6,
        replay_seeds=range(2),
        penalty_seeds=range(2),
        bias_seeds=range(2),
        write_figures=False,
    )

    metrics = pd.read_csv(manifest["metrics"])
    assert set(metrics["dataset"]) == {"capture_replay", "penalty_sweep", "bias_strength_sweep"}
    assert {"static_rollout_pool", "uct_mcts", "uncertainty_mcts_240"}.issubset(set(metrics["planner"]))
    assert (tmp_path / "claim_audit.json").exists()
