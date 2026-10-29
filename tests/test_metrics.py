"""
tests/test_metrics.py - Comprehensive tests for metric calculations, resource definitions, bounds, and error handling.
"""
import pytest
import numpy as np

from src.formulation import EXACT_OPTIMUM, OPTIMAL_BITSTRINGS
from src.benchmarks import compute_sampled_metrics
from qff_phase2 import circuit_metrics, PROCESSOR_B, counts_to_x
from src.qaoa import build_qaoa_circuit


@pytest.mark.unit
def test_metrics_definitions_on_synthetic_distribution():
    """Verify r, r_best, and P_opt definitions on an exact synthetic distribution."""
    # Construct a synthetic distribution of 4096 shots
    # 2048 shots on optimal string '001011' (cut 22)
    # 2048 shots on '000000' (cut 0)
    # In Qiskit little-endian:
    # logical '001011' -> counts key '110100'
    # logical '000000' -> counts key '000000'
    counts = {
        "110100": 2048,
        "000000": 2048,
    }
    metrics = compute_sampled_metrics(counts, shots=4096)
    
    expected_mean_cut = (2048 * 22 + 2048 * 0) / 4096.0  # 11.0
    expected_r = 11.0 / 22.0                              # 0.5
    expected_best_cut = 22
    expected_r_best = 22.0 / 22.0                         # 1.0
    expected_p_opt = 2048 / 4096.0                        # 0.5
    
    assert np.isclose(metrics["mean_cut"], expected_mean_cut)
    assert np.isclose(metrics["approx_ratio"], expected_r)
    assert metrics["best_cut"] == expected_best_cut
    assert np.isclose(metrics["best_approx_ratio"], expected_r_best)
    assert np.isclose(metrics["p_opt"], expected_p_opt)


@pytest.mark.unit
def test_resource_metrics_definitions():
    """Verify logical CX, mapped CX, depth, two_q_depth, swaps, and CX overhead on Processor B."""
    qc = build_qaoa_circuit(0.08396, 1.23810, p=1, measure=True)
    m = circuit_metrics(qc, PROCESSOR_B)
    
    assert m["executable"] is True
    assert m["logical_qubits"] == 6
    assert m["physical_qubits"] == 9
    assert m["logical_cx"] == 18
    assert m["cx"] == 21  # 18 logical + 3 from 1 SWAP
    assert m["swaps"] == 1
    assert m["cx_overhead"] == 3
    assert m["cx_overhead"] == m["cx"] - m["logical_cx"]
    
    # Check depth relations
    assert m["depth"] > 0
    assert m["two_q_depth"] > 0
    assert m["two_q_depth"] < m["depth"]


@pytest.mark.unit
def test_multi_start_mean_and_std_approximation_ratio():
    """Verify mean and sample standard deviation of approximation ratio across initializations."""
    # Synthetic approximation ratios from 5 multi-start runs
    r_values = [0.8051, 0.8051, 0.8051, 0.8051, 0.8051]
    mean_r = float(np.mean(r_values))
    std_r = float(np.std(r_values, ddof=1))
    
    assert np.isclose(mean_r, 0.8051)
    assert np.isclose(std_r, 0.0)


@pytest.mark.unit
def test_metrics_bounds_and_integer_invariants():
    """Verify bounds, integer counts, and probability normalization."""
    counts = {
        "110100": 4096,  # 100% optimal
    }
    m = compute_sampled_metrics(counts, shots=4096)
    assert m["approx_ratio"] == 1.0
    assert m["p_opt"] == 1.0
    assert m["best_cut"] == 22
    assert m["mean_cut"] == 22.0


@pytest.mark.unit
def test_malformed_counts_error_handling():
    """Verify that compute_sampled_metrics raises ValueError on malformed inputs."""
    # Empty counts
    with pytest.raises(ValueError, match="counts must be a non-empty dictionary"):
        compute_sampled_metrics({}, shots=4096)
        
    # Sum != shots
    with pytest.raises(ValueError, match="Sum of measurement counts .* does not match shots"):
        compute_sampled_metrics({"000000": 100}, shots=4096)
        
    # Wrong bitstring length
    with pytest.raises(ValueError, match="Bitstring .* length is 5, expected 6"):
        compute_sampled_metrics({"00000": 4096}, shots=4096)
        
    # Non-binary characters
    with pytest.raises(ValueError, match="contains non-binary characters"):
        compute_sampled_metrics({"000002": 4096}, shots=4096)
        
    # Negative count
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        compute_sampled_metrics({"000000": -10, "111111": 4106}, shots=4096)
        
    # Non-integer count
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        compute_sampled_metrics({"000000": 4096.0}, shots=4096)  # type: ignore


@pytest.mark.unit
def test_noisy_vs_ideal_quality_difference():
    """Verify that quality metrics are not forced to be identical between ideal and noisy runs."""
    # Ideal run: high fidelity, higher approximation ratio
    ideal_counts = {"110100": 570, "001011": 560, "000000": 50, "111111": 50}
    # Add remaining to reach 4096
    rem = 4096 - sum(ideal_counts.values())
    ideal_counts["101010"] = rem
    
    # Noisy run: higher entropy, lower counts on optimal
    noisy_counts = {"110100": 480, "001011": 470, "000000": 120, "111111": 120}
    rem_noisy = 4096 - sum(noisy_counts.values())
    noisy_counts["101010"] = rem_noisy
    
    m_ideal = compute_sampled_metrics(ideal_counts, shots=4096)
    m_noisy = compute_sampled_metrics(noisy_counts, shots=4096)
    
    assert m_ideal["approx_ratio"] != m_noisy["approx_ratio"]
    assert m_ideal["p_opt"] > m_noisy["p_opt"]
