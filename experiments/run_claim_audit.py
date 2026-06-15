from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.run_mechanism import run as run_mechanism
from experiments.run_expansion_suite import run_expansion_suite
from experiments.run_cliffwalking_benchmark import run_cliffwalking_benchmark
from experiments.run_tail_stress import run_tail_stress
from search_concentration_audit.diagnostics import write_json


def _claim(status: bool, value: float, threshold: float, description: str) -> dict[str, float | str]:
    return {
        "status": "pass" if status else "fail",
        "value": float(value),
        "threshold": float(threshold),
        "description": description,
    }


def _tail_row(tail: pd.DataFrame, *, planner: str, budget: int = 1024) -> pd.Series:
    row = tail[(tail["planner"] == planner) & (tail["budget"] == budget)]
    if row.empty:
        raise ValueError(f"missing tail row for {planner=} {budget=}")
    return row.iloc[0]


def _paired_row(pairwise: pd.DataFrame, *, planner: str, budget: int = 1024) -> pd.Series:
    row = pairwise[(pairwise["planner"] == planner) & (pairwise["budget"] == budget)]
    if row.empty:
        raise ValueError(f"missing paired row for {planner=} {budget=}")
    return row.iloc[0]


def audit_claims(*, full_dir: Path, stress_dir: Path, expansion_dir: Path, output: Path) -> dict:
    cliff_dir = ROOT / "results" / "cliffwalking_benchmark"
    required_full = [full_dir / "tail_metrics.csv", full_dir / "pairwise_deltas.csv", full_dir / "manifest.json"]
    if any(not path.exists() for path in required_full):
        run_mechanism("full", full_dir)
    if not (stress_dir / "claim_audit.json").exists():
        run_tail_stress(output=stress_dir)
    if not (expansion_dir / "claim_audit.json").exists():
        run_expansion_suite(output=expansion_dir)
    if not (cliff_dir / "aggregate.json").exists():
        run_cliffwalking_benchmark(output=cliff_dir)

    tail = pd.read_csv(full_dir / "tail_metrics.csv")
    pairwise = pd.read_csv(full_dir / "pairwise_deltas.csv")
    stress = json.loads((stress_dir / "claim_audit.json").read_text(encoding="utf-8"))
    expansion = json.loads((expansion_dir / "claim_audit.json").read_text(encoding="utf-8"))
    cliff = json.loads((cliff_dir / "aggregate.json").read_text(encoding="utf-8"))

    static = _tail_row(tail, planner="static_rollout_pool")
    uct = _tail_row(tail, planner="uct_mcts")
    uncertainty = _tail_row(tail, planner="uncertainty_mcts")
    uct_pair = _paired_row(pairwise, planner="uct_mcts")

    claims = {
        "uct_mean_gap_exceeds_static_at_1024": _claim(
            float(uct["mean_gap"] - static["mean_gap"]) > 0.25,
            float(uct["mean_gap"] - static["mean_gap"]),
            0.25,
            "UCT MCTS has a larger mean selected-return optimism gap than the static pool at budget 1024.",
        ),
        "uct_tail_max_exceeds_static_at_1024": _claim(
            float(uct["max_gap"] - static["max_gap"]) > 3.0,
            float(uct["max_gap"] - static["max_gap"]),
            3.0,
            "UCT MCTS has a larger worst branch-capture event than the static pool.",
        ),
        "paired_result_is_tail_not_dominance": _claim(
            float(uct_pair["positive_fraction"]) <= 0.25 and abs(float(uct_pair["median_delta"])) < 0.001,
            float(uct_pair["positive_fraction"]),
            0.25,
            "The paired-seed result is a rare-tail claim, not a dominance claim.",
        ),
        "uncertainty_repair_reduces_mean_and_q90": _claim(
            float(uct["mean_gap"] - uncertainty["mean_gap"]) > 0.30
            and float(uct["q90_gap"] - uncertainty["q90_gap"]) > 0.50,
            float(uct["mean_gap"] - uncertainty["mean_gap"]),
            0.30,
            "Uncertainty MCTS reduces both mean and upper-tail optimism in the full run.",
        ),
    }
    claims.update(stress["claims"])
    claims.update(expansion["claims"])
    claims.update(
        {
            name: _claim(
                bool(cliff["claims"][name]),
                float(cliff["summary"][summary_key]["mean"]),
                threshold,
                description,
            )
            for name, summary_key, threshold, description in [
                (
                    "cliff_uct_gap_exceeds_static",
                    "uct_minus_static_gap_ci",
                    50.0,
                    "On Gymnasium CliffWalking-v1, UCT MCTS must amplify learned shortcut optimism relative to a behavioral static rollout pool.",
                ),
                (
                    "cliff_uncertainty_reduces_gap",
                    "uncertainty_minus_uct_gap_ci",
                    -100.0,
                    "Uncertainty-penalized MCTS must reduce the CliffWalking selected-return optimism gap.",
                ),
                (
                    "cliff_uncertainty_improves_true_return",
                    "uncertainty_minus_uct_return_ci",
                    100.0,
                    "Uncertainty-penalized MCTS must improve true CliffWalking return over UCT.",
                ),
                (
                    "cliff_uct_concentrates_on_shortcut",
                    "uct_minus_static_cliff_bias_ci",
                    2.0,
                    "UCT MCTS must expose the learned cliff-shortcut bias more than the behavioral static pool.",
                ),
                (
                    "cliff_uncertainty_avoids_shortcut",
                    "uncertainty_minus_uct_cliff_bias_ci",
                    -2.0,
                    "Uncertainty-penalized MCTS must reduce learned cliff-shortcut exposure.",
                ),
                (
                    "cliff_true_model_has_low_gap",
                    "true_model_uct_gap_ci",
                    1.0,
                    "MCTS with the true Gymnasium transition table must have near-zero selected-return optimism gap.",
                ),
            ]
        }
    )
    payload = {
        "full_dir": str(full_dir),
        "stress_dir": str(stress_dir),
        "expansion_dir": str(expansion_dir),
        "cliffwalking_dir": str(cliff_dir),
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
    }
    write_json(output, payload)
    _write_claim_markdown(payload)
    _write_final_audit(payload)
    return payload


def _write_claim_markdown(payload: dict) -> None:
    lines = [
        "# Claim Audit",
        "",
        "This file is generated by `python experiments/run_claim_audit.py`.",
        "The paper should only make claims whose checks pass here.",
        "",
        f"- Full results dir: `{payload['full_dir']}`",
        f"- Tail-stress dir: `{payload['stress_dir']}`",
        f"- Expansion dir: `{payload['expansion_dir']}`",
        f"- CliffWalking benchmark dir: `{payload['cliffwalking_dir']}`",
        f"- All claims passed: `{payload['all_passed']}`",
        "",
        "| Claim | Status | Value | Threshold | Meaning |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, item in payload["claims"].items():
        lines.append(
            f"| `{name}` | {item['status']} | {item['value']:.4f} | "
            f"{item['threshold']:.4f} | {item['description']} |"
        )
    lines.append("")
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "claim_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_audit(payload: dict) -> None:
    claims = payload["claims"]
    lines = [
        "# Final Audit",
        "",
        "1. **Core thesis.** Adaptive tree search over learned dynamics can concentrate computation on local model optimism and create rare branch-capture events.",
        "",
        "2. **What is new.** The paper is not a generic test-time-compute curve. It audits the allocation feedback loop in MCTS: optimistic branches receive more visits, backups, and model queries.",
        "",
        "3. **V3 evidence added.** The tail-stress pass adds capture-seed replay, uncertainty-penalty sensitivity, and reward-bias-strength stress. The expansion suite adds exploration, horizon/budget, action-library, uncertainty-calibration, dynamics-drift, start-state, and closed-loop stress tests.",
        "",
        "4. **V4 benchmark evidence.** Gymnasium CliffWalking-v1 is now included as a standard tabular planning benchmark. The biased learned model treats the cliff row as a shortcut; UCT concentrates on that learned shortcut, while uncertainty-penalized MCTS repairs the selected-return gap and true return.",
        "",
        "5. **Strongest caution.** The paired median in the point-mass run is near zero and UCT is worse on only a minority of paired seeds. The claim must remain a branch-capture tail-risk claim, not a dominance claim.",
        "",
        "6. **Main remaining limitation.** The learned models are hand-designed; the next version should train a dynamics ensemble on biased data and evaluate on continuous-control benchmarks.",
        "",
        "7. **Final PDF location.** Expected repository path: `paper/final/best of n mcts tree search learned dynamics-v4.pdf`. Desktop publication is a post-verification step only.",
        "",
        "## Claim Status",
        "",
    ]
    for name, item in claims.items():
        lines.append(f"- `{name}`: {item['status']} ({item['value']:.4f} vs {item['threshold']:.4f})")
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "final_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--stress-dir", type=Path, default=ROOT / "results" / "tail_stress")
    parser.add_argument("--expansion-dir", type=Path, default=ROOT / "results" / "expansion")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "claim_audit.json")
    args = parser.parse_args()
    payload = audit_claims(
        full_dir=args.full_dir,
        stress_dir=args.stress_dir,
        expansion_dir=args.expansion_dir,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
