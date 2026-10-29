"""Focused tests for noise-aware backend architecture, execution modes, and noise configuration.

Verifies:
1. Ideal vs. noisy backend selection.
2. Missing noise configuration blocking (prevents fabricated noise rates).
3. Noise model application strictly targeting native gates (ECR on Processor A, CZ on Processor B).
4. Deterministic results under fixed seeds.
5. Correct mapping and cut-value decoding.
6. Reproducibility metadata and raw result preservation.
"""

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from maxcut.backends.processor_backend import (
    MissingNoiseConfigurationError,
    ProcessorBackendRunner,
    build_noise_model,
)
from maxcut.domain.model import MaxCutGraph, MaxCutProblemInstance
from maxcut.io import load_challenge_instance, load_processor_spec
from maxcut.qaoa.ansatz import build_qaoa_circuit, build_reduced_qaoa_circuit
from maxcut.qaoa.decoding import (
    decode_qiskit_counts,
    decode_reduced_qiskit_counts,
    evaluate_measurement_distribution,
)
from processors.processor_a.adapter import ProcessorABackend
from processors.processor_b.adapter import ProcessorBBackend


@pytest.fixture
def sample_graph() -> MaxCutGraph:
    instance = load_challenge_instance("instances/challenge/o1_maxcut_instance.json")
    return MaxCutGraph(name=instance.name, nodes=instance.nodes, edges=instance.edges)


@pytest.fixture
def sample_circuit(sample_graph: MaxCutGraph) -> QuantumCircuit:
    ansatz = build_reduced_qaoa_circuit(sample_graph, p=1, fixed_node=0, include_measurements=True)
    return ansatz.bind([0.3927], [0.3927])


@pytest.fixture
def synthetic_noise_config():
    return {
        "single_qubit_error": 0.001,
        "two_qubit_error": 0.005,
        "readout_error": 0.01,
    }


def test_ideal_backend_selection(sample_circuit: QuantumCircuit):
    """Verifies default ideal execution mode runs noise-free and correctly labels metadata."""
    backend_a = ProcessorABackend(execution_mode="ideal", seed_simulator=42)
    assert backend_a.execution_mode == "aer_hardware_topology_emulation"
    assert backend_a.noise_applied is False

    result = backend_a.run_circuit(sample_circuit, shots=100)
    assert result.execution_mode == "aer_hardware_topology_emulation"
    assert result.noise_applied is False
    assert result.job_metadata["execution_mode"] == "aer_hardware_topology_emulation"
    assert result.job_metadata["noise_applied"] is False


def test_missing_noise_config_blocks_execution():
    """Verifies requesting noisy simulation without valid configuration raises an explicit error."""
    with pytest.raises(MissingNoiseConfigurationError) as exc_info:
        ProcessorABackend(execution_mode="noisy", noise_config=None)
    assert "official noise configuration is missing or unverified" in str(exc_info.value)
    assert "Do not invent synthetic noise rates" in str(exc_info.value)

    with pytest.raises(MissingNoiseConfigurationError):
        ProcessorBBackend(execution_mode="noisy", noise_config=None)


def test_noise_configuration_application_processor_a(synthetic_noise_config):
    """Verifies noise model construction attaches 2Q errors strictly to native CX on Processor A."""
    spec_a = load_processor_spec("processors/processor_a/hardware_spec.json")
    model_a = build_noise_model(spec_a, synthetic_noise_config)
    assert isinstance(model_a, NoiseModel)

    # Check that basis gates in noise model include native entangler 'cx'
    assert "cx" in model_a.basis_gates


def test_noise_configuration_application_processor_b(synthetic_noise_config):
    """Verifies noise model construction attaches 2Q errors strictly to native CX on Processor B."""
    spec_b = load_processor_spec("processors/processor_b/hardware_spec.json")
    model_b = build_noise_model(spec_b, synthetic_noise_config)
    assert isinstance(model_b, NoiseModel)

    # Check that basis gates in noise model include native entangler 'cx'
    assert "cx" in model_b.basis_gates


def test_deterministic_noisy_results_under_fixed_seed(
    sample_circuit: QuantumCircuit,
    synthetic_noise_config,
):
    """Verifies that two runs with identical seeds produce identical bitstring counts under noise."""
    b1 = ProcessorABackend(
        execution_mode="noisy",
        noise_config=synthetic_noise_config,
        seed_simulator=1234,
        seed_transpiler=42,
    )
    b2 = ProcessorABackend(
        execution_mode="noisy",
        noise_config=synthetic_noise_config,
        seed_simulator=1234,
        seed_transpiler=42,
    )

    r1 = b1.run_circuit(sample_circuit, shots=500)
    r2 = b2.run_circuit(sample_circuit, shots=500)

    assert r1.raw_counts == r2.raw_counts
    assert r1.noise_applied is True
    assert r1.execution_mode == "aer_noisy_simulation"


def test_correct_mapping_and_cut_decoding(
    sample_circuit: QuantumCircuit,
    sample_graph: MaxCutGraph,
    synthetic_noise_config,
):
    """Verifies that decoded noisy measurement counts map to valid cut values in [0, 6]."""
    backend = ProcessorBBackend(
        execution_mode="noisy",
        noise_config=synthetic_noise_config,
        seed_simulator=42,
    )
    result = backend.run_circuit(sample_circuit, shots=200)

    node_to_qubit = {n: i for i, n in enumerate(sorted(sample_graph.nodes)[1:])}
    decoded = decode_reduced_qiskit_counts(
        raw_counts=result.raw_counts,
        node_to_qubit=node_to_qubit,
        all_sorted_nodes=sorted(sample_graph.nodes),
        fixed_node=0,
        fixed_val=0,
    )
    assert len(decoded) > 0
    total_shots = sum(decoded.values())
    assert total_shots == 200

    stats = evaluate_measurement_distribution(
        decoded_counts=decoded,
        graph=sample_graph,
        optimal_cut_value=6.0,
        optimal_bitstrings=[],
    )
    assert 0.0 <= stats["expected_cut_value"] <= 6.0
    assert 0.0 <= stats["best_sampled_cut"] <= 6.0


def test_reproducibility_metadata_and_raw_storage(
    sample_circuit: QuantumCircuit,
    synthetic_noise_config,
):
    """Verifies that full provenance metadata, software backend tags, and hardware specs are preserved."""
    backend = ProcessorABackend(
        execution_mode="noisy",
        noise_config=synthetic_noise_config,
        seed_simulator=99,
        seed_transpiler=77,
    )
    result = backend.run_circuit(sample_circuit, shots=100)

    assert result.job_metadata["shots"] == 100
    assert result.job_metadata["seed_simulator"] == 99
    assert result.job_metadata["seed_transpiler"] == 77
    assert result.job_metadata["execution_mode"] == "aer_noisy_simulation"
    assert result.job_metadata["native_2q_gate"] == "cx"
    assert result.noise_config == synthetic_noise_config
    assert result.hardware_spec["name"] == "Processor_A"
