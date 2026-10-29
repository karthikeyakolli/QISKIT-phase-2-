"""
tests/test_artifacts.py - Comprehensive tests for deliverables, schemas, figures, and provenance integrity.
"""
import os
import csv
import pytest

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "qff_phase2.py",
    "data/instance.json",
    "processors/processor_A.json",
    "processors/processor_B.json",
    "processors/processor_D.json",
    "results/tables/results.csv",
    "results/tables/bell.csv",
    "report.md",
    "results/figures/weighted_graph.png",
    "results/figures/processor_D_topology.png",
    "results/figures/quality_comparison.png",
    "results/figures/resource_comparison.png",
]


@pytest.mark.unit
def test_required_deliverables_exist_at_official_paths():
    """Verify that all required files and artifacts exist at their exact prescribed locations."""
    for file_path in REQUIRED_FILES:
        assert os.path.isfile(file_path), f"Required deliverable missing: {file_path}"
        assert os.path.getsize(file_path) > 0, f"File is empty: {file_path}"


@pytest.mark.unit
def test_results_csv_schema_and_column_integrity():
    """Verify Section 9 official schema, experiment rows, and non-executable reporting for Processor A."""
    expected_header = [
        "problem_id",
        "processor",
        "condition",
        "executable",
        "logical_qubits",
        "physical_qubits",
        "logical_cx",
        "cx",
        "depth",
        "two_q_depth",
        "swaps",
        "cx_overhead",
        "shots",
        "quality_metric",
        "quality_value",
        "p_opt",
        "classical_optimum",
        "random_baseline",
    ]
    with open("results/tables/results.csv", "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert list(reader[0].keys()) == expected_header
        assert len(reader) == 6
        
    for row in reader:
        assert row["problem_id"] == "PS-O1"
        assert row["classical_optimum"] == "22"
        assert row["random_baseline"] == "0.6136"
        assert row["quality_metric"] == "approx_ratio"
        
        if row["processor"] == "A":
            assert row["executable"] == "false"
            assert row["logical_qubits"] == "6"
            assert row["physical_qubits"] == "5"
            # Metrics must remain empty
            assert row["cx"] == ""
            assert row["depth"] == ""
            assert row["quality_value"] == ""
            assert row["p_opt"] == ""
        else:
            assert row["executable"] == "true"
            assert int(row["logical_cx"]) == 18
            assert int(row["shots"]) == 4096
            assert float(row["quality_value"]) > 0.75
            assert float(row["p_opt"]) > 0.10


@pytest.mark.unit
def test_figures_are_valid_png_and_high_res():
    """Verify generated figures are valid PNG files with proper PNG magic bytes."""
    figures = [
        "results/figures/weighted_graph.png",
        "results/figures/processor_D_topology.png",
        "results/figures/quality_comparison.png",
        "results/figures/resource_comparison.png",
    ]
    png_magic = b"\x89PNG\r\n\x1a\n"
    for fig in figures:
        assert os.path.isfile(fig)
        with open(fig, "rb") as f:
            header = f.read(8)
            assert header == png_magic, f"{fig} is not a valid PNG file"
        # Check that file size indicates non-trivial publication rendering (> 50 KB)
        assert os.path.getsize(fig) > 50_000, f"{fig} file size suspiciously small"


@pytest.mark.unit
def test_no_stale_or_placeholder_content():
    """Verify that no placeholder text (TODO, TBD, placeholder) remains in critical deliverables."""
    text_files = [
        "results/tables/results.csv",
        "results/tables/bell.csv",
        "report.md",
        "README.md",
    ]
    forbidden_tokens = ["TODO", "TBD", "FIXME", "PLACEHOLDER", "dummy", "synthetic_test"]
    for tf in text_files:
        with open(tf, "r", encoding="utf-8") as f:
            content = f.read()
        for token in forbidden_tokens:
            assert token not in content, f"Found forbidden token '{token}' in {tf}"


@pytest.mark.unit
def test_report_and_artifact_consistency():
    """Verify that numbers and metrics cited in report.md match the generated CSV tables."""
    with open("report.md", "r", encoding="utf-8") as f:
        report_text = f.read()
        
    # Core mathematical and hardware invariants
    assert "22" in report_text  # Classical optimum
    assert "0.6136" in report_text  # Random baseline
    assert "001011" in report_text and "110100" in report_text  # Optimal bitstrings
    assert "652e064ecfd10f405a4226bc274f941a36f45865326901030a898ba8d2f37c57" in report_text  # SHA-256
    assert "18" in report_text  # Logical CX
    assert "21" in report_text  # Mapped CX on B
