"""Unit and regression tests for Phase 5 result analysis and statistics.

Validates that all tables, figures, statistical summaries, hypothesis test metrics,
and uncertainty bounds generated from verified_ab_v2 raw data match expected values.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_DIR = REPO_ROOT / "results" / "verified_ab_v2"
TABLES_DIR = VERIFIED_DIR / "tables"
FIGURES_DIR = VERIFIED_DIR / "figures"
DATA_DIR = VERIFIED_DIR / "data"


def test_phase5_artifact_existence():
    """Verify all required tables, figures, and summary files exist."""
    required_files = [
        TABLES_DIR / "statistical_uncertainty_table.md",
        TABLES_DIR / "statistical_uncertainty_table.csv",
        TABLES_DIR / "pairwise_comparisons_table.md",
        TABLES_DIR / "pairwise_comparisons_table.csv",
        TABLES_DIR / "resource_vs_quality_table.md",
        TABLES_DIR / "resource_vs_quality_table.csv",
        FIGURES_DIR / "figure1_cut_distribution_4cases.png",
        FIGURES_DIR / "figure2_solution_quality_and_uncertainty.png",
        FIGURES_DIR / "figure3_resource_vs_fidelity_tradeoff.png",
        DATA_DIR / "statistical_analysis_summary.json",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required Phase 5 artifact: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"


def test_phase5_statistical_summary_content():
    """Verify the numeric values in statistical_analysis_summary.json."""
    summary_path = DATA_DIR / "statistical_analysis_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["c_max"] == 6.0
    assert abs(meta["analytic_expected_cut"] - 5.04) < 0.01
    assert abs(meta["analytic_approx_ratio"] - 0.84) < 0.01
    assert meta["shots"] == 2048

    cases = {c["Key"]: c for c in data["case_summary"]}
    assert set(cases.keys()) == {
        "processor_a_ideal_reference",
        "processor_a_noisy_baseline",
        "processor_b_ideal_reference",
        "processor_b_noisy_baseline",
    }

    # Processor A ideal
    a_ideal = cases["processor_a_ideal_reference"]
    assert a_ideal["Best Cut"] == 6.0
    assert abs(a_ideal["Mean Cut <C>"] - 5.0742) < 1e-3
    assert abs(a_ideal["Sample Approx Ratio"] - 0.8457) < 1e-3
    assert a_ideal["Optimal Hits"] == 871
    assert a_ideal["Native 2Q Gates"] == 18
    assert a_ideal["Basis 2Q Gates"] == 12
    assert a_ideal["Routing 2Q Gates"] == 6
    assert a_ideal["Routing SWAPs"] == 2
    assert a_ideal["Circuit Depth"] == 66
    assert a_ideal["Active Physical Qubits"] == 5

    # Processor A noisy
    a_noisy = cases["processor_a_noisy_baseline"]
    assert a_noisy["Best Cut"] == 6.0
    assert abs(a_noisy["Mean Cut <C>"] - 4.9556) < 1e-3
    assert abs(a_noisy["Sample Approx Ratio"] - 0.8259) < 1e-3
    assert a_noisy["Optimal Hits"] == 797
    assert a_noisy["Native 2Q Gates"] == 18
    assert a_noisy["Circuit Depth"] == 66

    # Processor B ideal
    b_ideal = cases["processor_b_ideal_reference"]
    assert b_ideal["Best Cut"] == 6.0
    assert abs(b_ideal["Mean Cut <C>"] - 5.0337) < 1e-3
    assert abs(b_ideal["Sample Approx Ratio"] - 0.8389) < 1e-3
    assert b_ideal["Optimal Hits"] == 837
    assert b_ideal["Native 2Q Gates"] == 21
    assert b_ideal["Basis 2Q Gates"] == 12
    assert b_ideal["Routing 2Q Gates"] == 9
    assert b_ideal["Routing SWAPs"] == 3
    assert b_ideal["Circuit Depth"] == 81
    assert b_ideal["Active Physical Qubits"] == 5

    # Processor B noisy
    b_noisy = cases["processor_b_noisy_baseline"]
    assert b_noisy["Best Cut"] == 6.0
    assert abs(b_noisy["Mean Cut <C>"] - 4.9346) < 1e-3
    assert abs(b_noisy["Sample Approx Ratio"] - 0.8224) < 1e-3
    assert b_noisy["Optimal Hits"] == 757
    assert b_noisy["Native 2Q Gates"] == 21
    assert b_noisy["Circuit Depth"] == 81


def test_phase5_hypothesis_tests():
    """Verify pairwise hypothesis test p-values and conclusion language."""
    summary_path = DATA_DIR / "statistical_analysis_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = {t["Comparison"]: t for t in data["pairwise_hypothesis_tests"]}

    # Comparison 1: Processor A Ideal vs Noisy
    comp1 = tests["Comparison 1: Processor A Ideal vs. Noisy"]
    assert float(comp1["p-value (Mean Cut)"]) < 0.05
    assert comp1["Mean Cut Difference Detected?"] == "Yes (p < 0.05)"
    assert float(comp1["p-value (Proportion)"]) < 0.05
    assert comp1["Proportion Difference Detected?"] == "Yes (p < 0.05)"

    # Comparison 2: Processor B Ideal vs Noisy
    comp2 = tests["Comparison 2: Processor B Ideal vs. Noisy"]
    assert float(comp2["p-value (Mean Cut)"]) < 0.05
    assert comp2["Mean Cut Difference Detected?"] == "Yes (p < 0.05)"
    assert float(comp2["p-value (Proportion)"]) < 0.05
    assert comp2["Proportion Difference Detected?"] == "Yes (p < 0.05)"

    # Comparison 3a: Processor A vs B Ideal
    comp3a = tests["Comparison 3a: Processor A vs. B (Controlled Ideal)"]
    assert float(comp3a["p-value (Mean Cut)"]) > 0.05
    assert comp3a["Mean Cut Difference Detected?"] == "No statistically significant difference detected"
    assert "equivalent" not in comp3a["Mean Cut Difference Detected?"].lower()

    # Comparison 3b: Processor A vs B Noisy
    comp3b = tests["Comparison 3b: Processor A vs. B (Controlled Noisy)"]
    assert float(comp3b["p-value (Mean Cut)"]) > 0.05
    assert comp3b["Mean Cut Difference Detected?"] == "No statistically significant difference detected"
    assert "equivalent" not in comp3b["Mean Cut Difference Detected?"].lower()
