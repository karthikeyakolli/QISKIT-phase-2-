"""
tests/test_processors.py - Comprehensive tests for Processor A, B, and D specifications, connectivity, validation, and frozen hash.
"""
import copy
import hashlib
import json
import pytest

from qff_phase2 import (
    PROCESSOR_A,
    PROCESSOR_B,
    load_processor,
    validate_processor_d,
    circuit_metrics,
    coupling_map_from_edges,
    BASIS_GATES,
    OPT_LEVEL,
    SEED,
    P1_DEPOL,
    P2_DEPOL,
    P_READOUT,
)
from src.qaoa import build_qaoa_circuit


@pytest.mark.unit
def test_processor_a_capacity_and_non_executable_rule():
    """Verify Processor A has 5 qubits and full 6-qubit PS-O1 is marked non-executable without fabrication."""
    assert PROCESSOR_A["name"] == "Processor A"
    assert PROCESSOR_A["num_qubits"] == 5
    assert len(PROCESSOR_A["edges"]) == 4
    
    # QAOA circuit has 6 qubits
    qc = build_qaoa_circuit(0.1, 0.2, p=1, measure=True)
    assert qc.num_qubits == 6
    
    m = circuit_metrics(qc, PROCESSOR_A)
    assert m["executable"] is False
    assert "needs 6 qubits" in m["reason"]
    # Ensure no fabricated quality or resource metrics exist in non-executable result
    assert "cx" not in m or m.get("cx") is None
    assert "depth" not in m or m.get("depth") is None


@pytest.mark.unit
def test_processor_b_exact_connectivity_and_routing():
    """Verify Processor B 9-qubit connectivity and routing support."""
    assert PROCESSOR_B["name"] == "Processor B"
    assert PROCESSOR_B["num_qubits"] == 9
    
    expected_b_edges = [
        (0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5),
        (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)
    ]
    assert PROCESSOR_B["edges"] == expected_b_edges
    assert len(PROCESSOR_B["edges"]) == 12
    
    # Check that transpiling onto Processor B respects basis gates and routing
    qc = build_qaoa_circuit(0.1, 0.2, p=1, measure=True)
    m = circuit_metrics(qc, PROCESSOR_B)
    assert m["executable"] is True
    assert m["logical_qubits"] == 6
    assert m["physical_qubits"] == 9
    assert m["swaps"] == 1
    assert m["cx_overhead"] == 3


@pytest.mark.unit
def test_processor_d_official_validation_and_freeze_hash():
    """Validate official Processor D JSON against validate_processor_d and verify frozen SHA-256."""
    D = load_processor("processors/processor_D.json")
    errs = validate_processor_d(D, required_qubits=6)
    assert len(errs) == 0, f"Processor D validation failed: {errs}"
    
    assert D["name"] == "Processor D"
    assert D["num_qubits"] == 6
    assert len(D["edges"]) == 9
    assert D["basis_gates"] == BASIS_GATES
    
    # Maximum degree check: max degree <= 4
    cm = coupling_map_from_edges(D["edges"], D["num_qubits"])
    for q in range(D["num_qubits"]):
        # cm is bidirectional, so neighbors count equals degree
        assert len(cm.neighbors(q)) <= 4
        
    # Verify exact frozen SHA-256 hash (normalized to LF line endings for cross-platform reproducibility)
    with open("processors/processor_D.json", "rb") as f:
        file_bytes = f.read().replace(b"\r\n", b"\n")
        file_hash = hashlib.sha256(file_bytes).hexdigest()
    assert file_hash == "652e064ecfd10f405a4226bc274f941a36f45865326901030a898ba8d2f37c57"


@pytest.mark.unit
def test_processor_d_zero_swap_overhead():
    """Verify Processor D achieves 0 SWAP gates and 0 CX routing overhead for PS-O1."""
    D = load_processor("processors/processor_D.json")
    qc = build_qaoa_circuit(0.1, 0.2, p=1, measure=True)
    m = circuit_metrics(qc, D)
    
    assert m["executable"] is True
    assert m["swaps"] == 0
    assert m["cx_overhead"] == 0
    assert m["cx"] == 18
    assert m["logical_cx"] == 18


@pytest.mark.unit
def test_processor_d_invalid_variants_rejected():
    """Verify validate_processor_d strictly catches and rejects invalid processor designs."""
    valid_d = load_processor("processors/processor_D.json")
    
    # 1. Insufficient qubits (< 6)
    p_too_few = copy.deepcopy(valid_d)
    p_too_few["num_qubits"] = 5
    errs = validate_processor_d(p_too_few, required_qubits=6)
    assert any("num_qubits must be an integer" in e for e in errs)
    
    # 2. Too many qubits (> 12)
    p_too_many = copy.deepcopy(valid_d)
    p_too_many["num_qubits"] = 13
    errs = validate_processor_d(p_too_many, required_qubits=6)
    assert any("num_qubits must be an integer" in e for e in errs)
    
    # 3. Disconnected graph (isolated qubit)
    p_disconn = copy.deepcopy(valid_d)
    # remove all edges incident to qubit 5: (1, 5) and (3, 5)
    p_disconn["edges"] = [e for e in p_disconn["edges"] if 5 not in e]
    errs = validate_processor_d(p_disconn, required_qubits=6)
    assert any("coupling graph must be connected" in e for e in errs)
    
    # 4. Excessive degree (> 4)
    p_high_deg = copy.deepcopy(valid_d)
    # Connect qubit 0 to 1, 2, 3, 4, 5 (degree 5)
    p_high_deg["edges"] = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (2, 3), (3, 4), (4, 5)
    ]
    errs = validate_processor_d(p_high_deg, required_qubits=6)
    assert any("exceed max degree 4" in e for e in errs)
    
    # 5. Invalid edge endpoints (out of range or self-loop)
    p_self_loop = copy.deepcopy(valid_d)
    p_self_loop["edges"].append((0, 0))
    errs = validate_processor_d(p_self_loop, required_qubits=6)
    assert any("invalid edge" in e for e in errs)
    
    p_out_of_range = copy.deepcopy(valid_d)
    p_out_of_range["edges"].append((0, 9))
    errs = validate_processor_d(p_out_of_range, required_qubits=6)
    assert any("invalid edge" in e for e in errs)
    
    # 6. Unsupported basis gates
    p_bad_gates = copy.deepcopy(valid_d)
    p_bad_gates["basis_gates"] = ["cx", "u3"]
    errs = validate_processor_d(p_bad_gates, required_qubits=6)
    assert any("basis_gates must be exactly" in e for e in errs)
    
    # 7. Unrealistic noise (lower/better than official model)
    p_cheat_noise = copy.deepcopy(valid_d)
    p_cheat_noise["noise"] = {"p1": 0.00001, "p2": 0.0001, "p_readout": 0.001}
    errs = validate_processor_d(p_cheat_noise, required_qubits=6)
    assert any("noise rates in the JSON may not be lower" in e for e in errs)
