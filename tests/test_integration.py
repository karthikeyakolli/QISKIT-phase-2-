"""
tests/test_integration.py - Full end-to-end integration, notebook execution smoke-test, and robustness tests.
"""
import os
import tempfile
import pytest
import nbformat
from nbclient import NotebookClient

from src.formulation import exact_classical_optimum
from src.qaoa import run_multi_start_qaoa
from src.bell import run_bell_state_suite, write_bell_csv
from src.benchmarks import run_all_benchmarks, write_results_csv
from qff_phase2 import load_processor, validate_processor_d


@pytest.mark.integration
def test_end_to_end_pipeline_in_temporary_directory():
    """Verify that the full workflow executes cleanly from end-to-end writing into a fresh temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_csv = os.path.join(tmpdir, "results.csv")
        bell_csv = os.path.join(tmpdir, "bell.csv")
        
        # 1. Classical exact solver
        max_c, opts = exact_classical_optimum()
        assert max_c == 22
        assert len(opts) == 2
        
        # 2. QAOA multi-start parameter optimization (fast with maxiter=30 for integration)
        best_run, all_runs = run_multi_start_qaoa(seeds=[0, 1, 2, 3, 4], maxiter=30)
        assert best_run["final_r"] > 0.75
        
        # 3. Processor D validation
        D = load_processor("processors/processor_D.json")
        errs = validate_processor_d(D, required_qubits=6)
        assert len(errs) == 0
        
        # 4. Bell-state experiments
        bell_rows = run_bell_state_suite("processors/processor_D.json", shots=1024)
        assert len(bell_rows) == 4
        write_bell_csv(bell_rows, output_path=bell_csv)
        assert os.path.isfile(bell_csv)
        assert os.path.getsize(bell_csv) > 0
        
        # 5. Hardware benchmarks
        b_rows, _ = run_all_benchmarks(
            best_gamma=best_run["optimal_gamma"],
            best_beta=best_run["optimal_beta"],
            processor_d_path="processors/processor_D.json",
            shots=1024,
        )
        assert len(b_rows) == 6
        write_results_csv(b_rows, output_path=results_csv)
        assert os.path.isfile(results_csv)
        assert os.path.getsize(results_csv) > 0


@pytest.mark.integration
@pytest.mark.slow
def test_notebook_headless_execution():
    """Smoke-test main.ipynb by executing top-to-bottom from a clean kernel with nbclient."""
    notebook_path = "main.ipynb"
    assert os.path.isfile(notebook_path), "main.ipynb missing"
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()
    
    # Check that cells completed without error and produced outputs
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 10
    
    # Verify outputs exist
    outputs_found = sum(len(c.get("outputs", [])) for c in code_cells)
    assert outputs_found > 0


@pytest.mark.unit
def test_missing_processor_file_error_recovery():
    """Verify that referencing a non-existent processor file fails with clean FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_processor("processors/non_existent_processor.json")


@pytest.mark.unit
def test_output_directory_auto_creation():
    """Verify write_results_csv and write_bell_csv auto-create parent directories if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "deeply", "nested", "tables")
        target_csv = os.path.join(nested_dir, "test_results.csv")
        assert not os.path.exists(nested_dir)
        
        sample_rows = [{
            "problem_id": "PS-O1",
            "processor": "A",
            "condition": "ideal",
            "executable": "false",
            "logical_qubits": 6,
            "physical_qubits": 5,
            "logical_cx": "",
            "cx": "",
            "depth": "",
            "two_q_depth": "",
            "swaps": "",
            "cx_overhead": "",
            "shots": "",
            "quality_metric": "approx_ratio",
            "quality_value": "",
            "p_opt": "",
            "classical_optimum": 22,
            "random_baseline": 0.6136,
        }]
        write_results_csv(sample_rows, output_path=target_csv)
        assert os.path.isfile(target_csv)
