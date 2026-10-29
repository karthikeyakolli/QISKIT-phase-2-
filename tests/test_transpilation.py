"""
tests/test_transpilation.py - Comprehensive tests for transpilation, routing stage SWAP extraction, and metrics.
"""
import pytest
from qiskit import QuantumCircuit, transpile

from qff_phase2 import (
    PROCESSOR_A,
    PROCESSOR_B,
    load_processor,
    circuit_metrics,
    transpile_for,
    coupling_map_from_edges,
    BASIS_GATES,
    OPT_LEVEL,
    SEED,
)
from src.qaoa import build_qaoa_circuit


@pytest.mark.unit
def test_transpile_direct_vs_nondirect_connections():
    """Verify that adjacent physical couplings transpile with 1 CX while non-adjacent couplings require routing."""
    D = load_processor("processors/processor_D.json")
    cm = coupling_map_from_edges(D["edges"], D["num_qubits"])
    
    # 1. Direct connection: (0, 2) has distance 1 in D
    qc_direct = QuantumCircuit(6)
    qc_direct.cx(0, 2)
    tqc_direct = transpile(
        qc_direct,
        basis_gates=BASIS_GATES,
        coupling_map=cm,
        initial_layout=[0, 1, 2, 3, 4, 5],
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED,
    )
    assert tqc_direct.count_ops().get("cx", 0) == 1
    
    # 2. Non-direct connection: (0, 1) has distance 2 in D
    qc_nondirect = QuantumCircuit(6)
    qc_nondirect.cx(0, 1)
    tqc_nondirect = transpile(
        qc_nondirect,
        basis_gates=BASIS_GATES,
        coupling_map=cm,
        initial_layout=[0, 1, 2, 3, 4, 5],
        optimization_level=OPT_LEVEL,
        seed_transpiler=SEED,
    )
    # Routing requires SWAP or multiple CXs (> 1 CX)
    assert tqc_nondirect.count_ops().get("cx", 0) > 1


@pytest.mark.unit
def test_logical_vs_mapped_cx_definitions():
    """Verify logical CX is measured unconstrained and CX overhead is exact difference."""
    qc = build_qaoa_circuit(0.08396, 1.23810, p=1, measure=True)
    m = circuit_metrics(qc, PROCESSOR_B)
    
    # Logical CX: 18 (9 edges * 2 CX)
    assert m["logical_cx"] == 18
    # Mapped CX: 21 (due to 1 SWAP adding 3 CX)
    assert m["cx"] == 21
    # CX overhead: mapped - logical = 3
    assert m["cx_overhead"] == m["cx"] - m["logical_cx"] == 3


@pytest.mark.unit
def test_swap_count_from_routing_stage():
    """Verify SWAP count is extracted before decomposition, not inferred heuristically."""
    qc = build_qaoa_circuit(0.08396, 1.23810, p=1, measure=True)
    
    m_b = circuit_metrics(qc, PROCESSOR_B)
    assert m_b["swaps"] == 1
    
    D = load_processor("processors/processor_D.json")
    m_d = circuit_metrics(qc, D)
    assert m_d["swaps"] == 0


@pytest.mark.unit
def test_depth_excludes_measurements_and_barriers():
    """Verify circuit depth metric strictly excludes measurements and barriers."""
    p = {"name": "TestProc", "num_qubits": 2, "edges": [(0, 1)]}
    
    # Circuit without barriers and measurements
    qc_clean = QuantumCircuit(2)
    qc_clean.h(0)
    qc_clean.cx(0, 1)
    m_clean = circuit_metrics(qc_clean, p)
    
    # Circuit with multiple barriers and measurements added
    qc_with_meas = QuantumCircuit(2)
    qc_with_meas.h(0)
    qc_with_meas.cx(0, 1)
    qc_with_meas.barrier()
    qc_with_meas.measure_all()
    m_with_meas = circuit_metrics(qc_with_meas, p)
    
    # Depth must be identical because measurements and barriers are filtered out
    assert m_clean["depth"] == m_with_meas["depth"]
    assert m_clean["two_q_depth"] == m_with_meas["two_q_depth"]
    # Decomposed H (depth 3: rz-sx-rz) + CX (depth 1) = depth 4
    assert m_clean["depth"] == 4
    assert m_clean["two_q_depth"] == 1



@pytest.mark.unit
def test_transpiler_seed_reproducibility():
    """Verify repeated transpilation with seed 42 produces identical circuits."""
    qc = build_qaoa_circuit(0.08396, 1.23810, p=1, measure=True)
    
    tqc1 = transpile_for(qc, PROCESSOR_B)
    tqc2 = transpile_for(qc, PROCESSOR_B)
    
    assert tqc1.count_ops() == tqc2.count_ops()
    assert tqc1.depth() == tqc2.depth()
    assert [inst.operation.name for inst in tqc1.data] == [inst.operation.name for inst in tqc2.data]


@pytest.mark.unit
def test_impossible_mapping_fails_cleanly():
    """Verify transpiling a circuit with more qubits than processor capacity raises ValueError."""
    qc_7q = QuantumCircuit(7)
    qc_7q.h(range(7))
    
    with pytest.raises(ValueError, match="has 5 qubits but the circuit needs 7"):
        transpile_for(qc_7q, PROCESSOR_A)
        
    D = load_processor("processors/processor_D.json")
    with pytest.raises(ValueError, match="has 6 qubits but the circuit needs 7"):
        transpile_for(qc_7q, D)


@pytest.mark.unit
def test_resource_metrics_not_fabricated_for_non_executable():
    """Verify circuit_metrics returns clean non-executable response without fabricated numbers."""
    qc = build_qaoa_circuit(0.1, 0.2, p=1, measure=True)
    m = circuit_metrics(qc, PROCESSOR_A)
    
    assert m["executable"] is False
    assert "reason" in m
    assert "needs 6 qubits" in m["reason"]
    # Ensure neither CX nor depth are reported
    assert "cx" not in m
    assert "depth" not in m
    assert "swaps" not in m
