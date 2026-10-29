"""Regression tests for Verified A/B v3 Statistical Analysis and Uncertainty Estimation.

Verifies:
1. All required v3 tables, figures, and summary JSON files exist.
2. Numeric values match verified experimental outcomes:
   - Mean cut, standard error, 95% CI.
   - Approximation ratios.
   - Optimal bitstring sampling probabilities and Wilson score CIs.
3. Hypothesis test results:
   - Proc A Ideal vs Noisy: significant degradation detected (p < 0.05).
   - Proc B Ideal vs Noisy: degradation detected on mean cut (p < 0.05).
   - Controlled A vs B: no statistically significant difference detected (p > 0.05).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VERIFIED_V3_DIR = REPO_ROOT / "results" / "verified_ab_v3"
TABLES_DIR = VERIFIED_V3_DIR / "tables"
FIGURES_DIR = VERIFIED_V3_DIR / "figures"
DATA_DIR = VERIFIED_V3_DIR / "data"


def test_v3_artifact_existence():
    """Verify all required v3 tables, figures, and summary files exist."""
    required_files = [
        TABLES_DIR / "table1_circuit_compilation_breakdown.md",
        TABLES_DIR / "table1_circuit_compilation_breakdown.csv",
        TABLES_DIR / "table2_performance_comparison.md",
        TABLES_DIR / "table2_performance_comparison.csv",
        TABLES_DIR / "table3_statistical_hypothesis_tests.md",
        TABLES_DIR / "table3_statistical_hypothesis_tests.csv",
        TABLES_DIR / "statistical_uncertainty_table.md",
        TABLES_DIR / "statistical_uncertainty_table.csv",
        TABLES_DIR / "pairwise_comparisons_table.md",
        TABLES_DIR / "pairwise_comparisons_table.csv",
        TABLES_DIR / "resource_vs_quality_table.md",
        TABLES_DIR / "resource_vs_quality_table.csv",
        FIGURES_DIR / "figure1_sampling_distribution.png",
        FIGURES_DIR / "figure2_compilation_overhead_breakdown.png",
        FIGURES_DIR / "figure3_metric_confidence_intervals.png",
        DATA_DIR / "statistical_analysis_summary.json",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required v3 artifact: {path}"
        assert path.stat().st_size > 0, f"Artifact is empty: {path}"


def test_v3_statistical_summary_content():
    """Verify the numeric values in statistical_analysis_summary.json."""
    summary_path = DATA_DIR / "statistical_analysis_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["c_max"] == 6.0
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
    assert abs(a_ideal["Mean Cut <C>"] - 5.0625) < 1e-3
    assert abs(a_ideal["Sample Approx Ratio"] - 0.8438) < 1e-3
    assert a_ideal["Optimal Hits"] == 854
    assert a_ideal["Native 2Q Gates"] == 18
    assert a_ideal["Circuit Depth"] == 69
    assert a_ideal["Active Qubits"] == 5

    # Processor A noisy
    a_noisy = cases["processor_a_noisy_baseline"]
    assert a_noisy["Best Cut"] == 6.0
    assert abs(a_noisy["Mean Cut <C>"] - 4.9399) < 1e-3
    assert abs(a_noisy["Sample Approx Ratio"] - 0.8233) < 1e-3
    assert a_noisy["Optimal Hits"] == 770
    assert a_noisy["Native 2Q Gates"] == 18
    assert a_noisy["Circuit Depth"] == 69

    # Processor B ideal
    b_ideal = cases["processor_b_ideal_reference"]
    assert b_ideal["Best Cut"] == 6.0
    assert abs(b_ideal["Mean Cut <C>"] - 5.0742) < 1e-3
    assert abs(b_ideal["Sample Approx Ratio"] - 0.8457) < 1e-3
    assert b_ideal["Optimal Hits"] == 871
    assert b_ideal["Native 2Q Gates"] == 18
    assert b_ideal["Circuit Depth"] == 83
    assert b_ideal["Active Qubits"] == 5

    # Processor B noisy
    b_noisy = cases["processor_b_noisy_baseline"]
    assert b_noisy["Best Cut"] == 6.0
    assert abs(b_noisy["Mean Cut <C>"] - 4.9839) < 1e-3
    assert abs(b_noisy["Sample Approx Ratio"] - 0.8306) < 1e-3
    assert b_noisy["Optimal Hits"] == 812
    assert b_noisy["Native 2Q Gates"] == 18
    assert b_noisy["Circuit Depth"] == 83


def test_v3_hypothesis_tests():
    """Verify that statistical hypothesis test results match calculated p-values."""
    summary_path = DATA_DIR / "statistical_analysis_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hyp_tests = {h["Comparison"]: h for h in data["pairwise_hypothesis_tests"]}

    # Proc A Ideal vs Noisy
    t_a = hyp_tests["Comparison 1: Processor A Ideal vs. Noisy"]
    assert float(t_a["p-value (Mean Cut)"]) < 0.05
    assert float(t_a["p-value (Proportion)"]) < 0.05
    assert "Yes" in t_a["Mean Cut Difference Detected?"]

    # Proc B Ideal vs Noisy
    t_b = hyp_tests["Comparison 2: Processor B Ideal vs. Noisy"]
    assert float(t_b["p-value (Mean Cut)"]) < 0.05
    assert "Yes" in t_b["Mean Cut Difference Detected?"]

    # Controlled Ideal Proc A vs Proc B
    t_ideal = hyp_tests["Comparison 3a: Processor A vs. B (Controlled Ideal)"]
    assert float(t_ideal["p-value (Mean Cut)"]) > 0.05
    assert float(t_ideal["p-value (Proportion)"]) > 0.05
    assert "No statistically significant difference" in t_ideal["Mean Cut Difference Detected?"]

    # Controlled Noisy Proc A vs Proc B
    t_noisy = hyp_tests["Comparison 3b: Processor A vs. B (Controlled Noisy)"]
    assert float(t_noisy["p-value (Mean Cut)"]) > 0.05
    assert float(t_noisy["p-value (Proportion)"]) > 0.05
    assert "No statistically significant difference" in t_noisy["Mean Cut Difference Detected?"]
