"""
tests/test_qaoa.py - Comprehensive tests for QAOA circuits, angle validation, seeds, and optimization.
"""
import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.formulation import NUM_VERTICES, LOCKED_EDGES, EXACT_OPTIMUM
from src.qaoa import (
    build_qaoa_circuit,
    evaluate_statevector_metrics,
    optimize_qaoa_seed,
    run_multi_start_qaoa,
)
from qff_phase2 import simulator, counts_to_x, SHOTS
from src.benchmarks import compute_sampled_metrics


@pytest.mark.unit
def test_qaoa_circuit_structure_depth_p1():
    """Verify depth p=1 QAOA circuit composition, gate counts, and measurement toggling."""
    qc = build_qaoa_circuit(gamma=0.1, beta=0.2, p=1, measure=True)
    assert qc.num_qubits == NUM_VERTICES
    ops = qc.count_ops()
    assert ops["h"] == 6
    assert ops["cx"] == 18  # 9 edges * 2 CX per edge
    assert ops["rz"] == 9   # 9 edges
    assert ops["rx"] == 6   # 6 mixer gates
    assert ops["measure"] == 6

    # Test circuit without measurement
    qc_no_meas = build_qaoa_circuit(gamma=0.1, beta=0.2, p=1, measure=False)
    ops_no_meas = qc_no_meas.count_ops()
    assert "measure" not in ops_no_meas


@pytest.mark.unit
def test_qaoa_optional_p2_functionality():
    """Verify optional p=2 circuit builds correctly without replacing the p=1 standard."""
    gammas = [0.1, 0.2]
    betas = [0.3, 0.4]
    qc_p2 = build_qaoa_circuit(gamma=gammas, beta=betas, p=2, measure=True)
    ops = qc_p2.count_ops()
    assert ops["h"] == 6
    assert ops["cx"] == 36  # 18 * 2 layers
    assert ops["rz"] == 18  # 9 * 2 layers
    assert ops["rx"] == 12  # 6 * 2 layers
    assert ops["measure"] == 6


@pytest.mark.unit
def test_qaoa_invalid_parameters_handling():
    """Verify that invalid depths, angle lengths, or non-finite values fail cleanly."""
    # Invalid depth
    with pytest.raises(ValueError, match="QAOA depth p must be a positive integer"):
        build_qaoa_circuit(0.1, 0.2, p=0)
        
    with pytest.raises(ValueError, match="QAOA depth p must be a positive integer"):
        build_qaoa_circuit(0.1, 0.2, p=-1)

    # Angle length mismatch
    with pytest.raises(ValueError, match="Length of gamma .* must equal p=2"):
        build_qaoa_circuit(gamma=[0.1], beta=[0.2, 0.3], p=2)

    # Non-finite angles (NaN, Inf)
    with pytest.raises(ValueError, match="QAOA angles .* must be finite"):
        build_qaoa_circuit(gamma=float("nan"), beta=0.2, p=1)
        
    with pytest.raises(ValueError, match="QAOA angles .* must be finite"):
        build_qaoa_circuit(gamma=0.1, beta=float("inf"), p=1)


@pytest.mark.unit
def test_random_initialization_reproducibility():
    """Verify numpy.default_rng(seed) produces deterministic uniform angles in [0, pi]."""
    for seed in [0, 1, 2, 3, 4]:
        rng1 = np.random.default_rng(seed)
        g1 = float(rng1.uniform(0.0, np.pi))
        b1 = float(rng1.uniform(0.0, np.pi))
        
        rng2 = np.random.default_rng(seed)
        g2 = float(rng2.uniform(0.0, np.pi))
        b2 = float(rng2.uniform(0.0, np.pi))
        
        assert g1 == g2
        assert b1 == b2
        assert 0.0 <= g1 <= np.pi
        assert 0.0 <= b1 <= np.pi

    # Check distinct initializations across seeds 0-4
    inits = [
        (float(np.random.default_rng(s).uniform(0, np.pi)), float(np.random.default_rng(s).uniform(0, np.pi)))
        for s in range(5)
    ]
    assert len(set(inits)) == 5


@pytest.mark.unit
def test_qaoa_objective_agrees_with_analytical_single_edge():
    """Verify QAOA expectation agrees with exact analytical formula on 2-qubit single edge.
    
    For a single edge (0, 1, w=1) with p=1 QAOA circuit using CX-RZ(2*gamma)-CX:
    <C> = 0.5 * (1 - sin(4*beta) * sin(2*gamma))
    """
    gamma_val = 0.35
    beta_val = 0.42
    analytical_cut = 0.5 * (1.0 - np.sin(4.0 * beta_val) * np.sin(2.0 * gamma_val))
    
    metrics = evaluate_statevector_metrics(
        gamma=gamma_val,
        beta=beta_val,
        p=1,
        num_vertices=2,
        edges=[(0, 1, 1)],
    )
    assert np.isclose(metrics["expected_cut"], analytical_cut, atol=1e-5)



@pytest.mark.integration
def test_shot_based_vs_statevector_statistical_consistency():
    """Verify that 4096-shot simulation matches exact statevector expectation within 3-sigma."""
    gamma_test = 0.08396
    beta_test = 1.23810
    
    # Exact expectation
    sv_metrics = evaluate_statevector_metrics(gamma_test, beta_test)
    exact_exp_cut = sv_metrics["expected_cut"]
    
    # 4096-shot simulation
    qc = build_qaoa_circuit(gamma_test, beta_test, p=1, measure=True)
    sim = simulator(noisy=False)
    job = sim.run(qc, shots=4096)
    counts = job.result().get_counts()
    
    sampled = compute_sampled_metrics(counts, shots=4096)
    sampled_cut = sampled["mean_cut"]
    
    # Standard error for cut weight with std ~ 3.5: SE ~ 3.5 / sqrt(4096) ~ 0.055
    # 3-sigma bound ~ 0.25
    diff = abs(sampled_cut - exact_exp_cut)
    assert diff < 0.30, f"Sampled cut {sampled_cut} deviated from exact {exact_exp_cut} by {diff} (> 0.30)"


@pytest.mark.integration
def test_multi_start_optimization_honesty_and_provenance():
    """Verify 5-seed multi-start optimization provenance and honest convergence reporting."""
    best_run, all_runs = run_multi_start_qaoa(seeds=[0, 1, 2, 3, 4], maxiter=50)
    assert len(all_runs) == 5
    
    for run in all_runs:
        assert run["optimizer"] == "COBYLA"
        assert run["maxiter"] == 50
        assert run["nfev"] > 0
        assert np.isfinite(run["initial_cut"])
        assert np.isfinite(run["final_cut"])
        assert run["final_r"] > run["initial_r"]
        
    # Global best should be Seed 3
    assert best_run["seed"] == 3
    assert best_run["final_r"] > 0.80
    assert best_run["final_p_opt"] > 0.12
    assert np.isclose(best_run["optimal_gamma"], 0.08396, atol=1e-2)
    assert np.isclose(best_run["optimal_beta"], 1.23810, atol=1e-2)
