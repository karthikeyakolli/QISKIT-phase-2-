"""
tests/test_noise_and_simulation.py - Comprehensive tests for noise models, simulator settings, and reproducibility.
"""
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer.noise import NoiseModel

from qff_phase2 import (
    official_noise_model,
    simulator,
    BASIS_GATES,
    OPT_LEVEL,
    SEED,
    SHOTS,
    P1_DEPOL,
    P2_DEPOL,
    P_READOUT,
    PROCESSOR_B,
    load_processor,
    coupling_map_from_edges,
)
from src.qaoa import build_qaoa_circuit
from src.benchmarks import compute_sampled_metrics


@pytest.mark.unit
def test_official_noise_model_parameters():
    """Verify official synthetic noise model parameters and target basis gates."""
    nm = official_noise_model()
    assert isinstance(nm, NoiseModel)
    assert set(nm.basis_gates) == set(BASIS_GATES)
    
    # Check constants
    assert P1_DEPOL == 0.001
    assert P2_DEPOL == 0.005
    assert P_READOUT == 0.01


@pytest.mark.unit
def test_simulator_ideal_vs_noisy_separation():
    """Verify ideal and noisy simulators are strictly separated."""
    sim_ideal = simulator(noisy=False)
    # Ideal simulator has no noise model configured
    assert getattr(sim_ideal.options, "noise_model", None) is None
    
    sim_noisy = simulator(noisy=True)
    assert getattr(sim_noisy.options, "noise_model", None) is not None


@pytest.mark.integration
def test_simulator_reproducibility_with_fixed_seed():
    """Verify simulator with seed 42 produces identical counts on identical circuits."""
    qc = build_qaoa_circuit(0.08396, 1.23810, p=1, measure=True)
    tqc = transpile(qc, basis_gates=BASIS_GATES, optimization_level=OPT_LEVEL, seed_transpiler=SEED)
    
    sim1 = simulator(noisy=False)
    job1 = sim1.run(tqc, shots=4096)
    counts1 = job1.result().get_counts()
    
    sim2 = simulator(noisy=False)
    job2 = sim2.run(tqc, shots=4096)
    counts2 = job2.result().get_counts()
    
    assert counts1 == counts2


@pytest.mark.integration
def test_noisy_simulation_causes_fidelity_decay():
    """Verify that noisy simulation degrades ideal states due to depolarizing and readout noise."""
    # Test a simple 2-qubit Bell pair
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    D = load_processor("processors/processor_D.json")
    cm = coupling_map_from_edges(D["edges"], D["num_qubits"])
    tqc = transpile(qc, basis_gates=BASIS_GATES, coupling_map=cm, optimization_level=OPT_LEVEL, seed_transpiler=SEED)
    
    # 1. Ideal run: should produce only '00' and '11'
    sim_ideal = simulator(noisy=False)
    counts_ideal = sim_ideal.run(tqc, shots=4096).result().get_counts()
    assert set(counts_ideal.keys()).issubset({"00", "11"})
    
    # 2. Noisy run: noise produces error states '01' and '10'
    sim_noisy = simulator(noisy=True, processor=D)
    counts_noisy = sim_noisy.run(tqc, shots=4096).result().get_counts()
    assert "01" in counts_noisy or "10" in counts_noisy
    assert counts_noisy.get("01", 0) + counts_noisy.get("10", 0) > 0


@pytest.mark.unit
def test_invalid_noise_parameters_handling():
    """Verify that negative or out-of-range noise rates raise error."""
    with pytest.raises(Exception):
        official_noise_model(p1=-0.1)
        
    with pytest.raises(Exception):
        official_noise_model(p_ro=1.5)
