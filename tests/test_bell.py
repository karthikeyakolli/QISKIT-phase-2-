"""
tests/test_bell.py - Comprehensive tests for Bell-state tomography, parity formulas, SE, and routing noise comparison.
"""
import csv
import pytest
import numpy as np
from qiskit.quantum_info import Statevector

from qff_phase2 import (
    bell_circuit,
    parity_expectation,
    bell_experiment,
    load_processor,
    coupling_map_from_edges,
    SHOTS,
)
from src.bell import find_bell_pairs, run_bell_state_suite


@pytest.mark.unit
def test_bell_circuit_bases_and_target_state():
    """Verify |Phi+> state preparation and tomography rotations for XX, YY, and ZZ."""
    # Circuit without basis change is |Phi+>
    sv_phi_plus = Statevector.from_label("00").evolve(bell_circuit("ZZ").remove_final_measurements(inplace=False))
    target = (Statevector.from_label("00") + Statevector.from_label("11")) / np.sqrt(2)
    assert sv_phi_plus.equiv(target)
    
    # Check XX circuit has H on both qubits
    qc_xx = bell_circuit("XX")
    ops_xx = qc_xx.count_ops()
    assert ops_xx["h"] >= 3  # init H + 2 measurement H
    
    # Check YY circuit has sdg and H on both qubits
    qc_yy = bell_circuit("YY")
    ops_yy = qc_yy.count_ops()
    assert ops_yy["sdg"] == 2
    
    # Check invalid basis raises ValueError
    with pytest.raises(ValueError, match="basis must be 'XX', 'YY' or 'ZZ'"):
        bell_circuit("XZ")


@pytest.mark.unit
def test_bell_pair_discovery_and_ordering():
    """Verify pair discovery correctly identifies distance 1 and maximum-shortest-path distance 2 on D."""
    D = load_processor("processors/processor_D.json")
    cm = coupling_map_from_edges(D["edges"], D["num_qubits"])
    
    adj_pair, far_pair = find_bell_pairs(D)
    
    assert cm.distance(*adj_pair) == 1
    assert cm.distance(*far_pair) == 2
    
    # Ensure they are not swapped
    assert cm.distance(*adj_pair) < cm.distance(*far_pair)


@pytest.mark.unit
def test_parity_expectation_and_se_formula():
    """Verify <P> = (N_even - N_odd)/N and binomial standard error."""
    # 1. Perfectly even counts
    counts_even = {"00": 2048, "11": 2048}
    e, se = parity_expectation(counts_even)
    assert np.isclose(e, 1.0)
    assert np.isclose(se, 0.0)
    
    # 2. Perfectly odd counts
    counts_odd = {"01": 2048, "10": 2048}
    e_odd, se_odd = parity_expectation(counts_odd)
    assert np.isclose(e_odd, -1.0)
    assert np.isclose(se_odd, 0.0)
    
    # 3. Equal split (unpolarized)
    counts_split = {"00": 1024, "01": 1024, "10": 1024, "11": 1024}
    e_split, se_split = parity_expectation(counts_split)
    assert np.isclose(e_split, 0.0)
    expected_se = np.sqrt(1.0 / 4096.0)  # ~ 0.015625
    assert np.isclose(se_split, expected_se)


@pytest.mark.unit
def test_three_basis_fidelity_vs_incorrect_shortcut():
    """Verify that P(00)+P(11) is an invalid fidelity metric under dephasing noise.
    
    For |Phi-> = (|00> - |11>)/sqrt(2):
    P(00) + P(11) = 1.0  (falsely indicates perfect state)
    True three-basis fidelity:
    <XX> = -1, <YY> = +1, <ZZ> = +1
    F = (1 + <XX> - <YY> + <ZZ>) / 4 = (1 - 1 - 1 + 1)/4 = 0.0 (orthogonal!)
    """
    # Create |Phi-> statevector
    phi_minus = (Statevector.from_label("00") - Statevector.from_label("11")) / np.sqrt(2)
    probs = phi_minus.probabilities_dict()
    naive_metric = probs.get("00", 0) + probs.get("11", 0)
    assert np.isclose(naive_metric, 1.0)
    
    # True Pauli expectations for |Phi->
    from qiskit.quantum_info import Pauli
    xx = phi_minus.expectation_value(Pauli("XX")).real  # -1.0
    yy = phi_minus.expectation_value(Pauli("YY")).real  # +1.0
    zz = phi_minus.expectation_value(Pauli("ZZ")).real  # +1.0
    
    true_fidelity = (1.0 + xx - yy + zz) / 4.0
    assert np.isclose(true_fidelity, 0.0)
    assert naive_metric != true_fidelity


@pytest.mark.integration
def test_bell_suite_ideal_and_noisy_fidelity():
    """Verify execution of all 12 Bell circuits with official 4096 shots."""
    rows = run_bell_state_suite("processors/processor_D.json", shots=4096)
    assert len(rows) == 4  # 2 pairs x 2 conditions (ideal, noisy)
    
    for r in rows:
        assert r["condition"] in ("ideal", "noisy")
        assert -1.0 <= r["xx"] <= 1.0
        assert -1.0 <= r["yy"] <= 1.0
        assert -1.0 <= r["zz"] <= 1.0
        assert 0.0 <= r["fidelity"] <= 1.0
        assert r["fidelity_se"] >= 0.0
        
    # Ideal checks
    ideal_rows = [r for r in rows if r["condition"] == "ideal"]
    for r in ideal_rows:
        assert r["fidelity"] == 1.0
        assert r["fidelity_se"] == 0.0
        assert r["xx"] == 1.0
        assert r["yy"] == -1.0
        assert r["zz"] == 1.0
        
    # Noisy distance comparison: dist 2 pair must have lower fidelity than dist 1 pair
    noisy_rows = [r for r in rows if r["condition"] == "noisy"]
    adj_noisy = [r for r in noisy_rows if r["pair_distance"] == 1][0]
    far_noisy = [r for r in noisy_rows if r["pair_distance"] == 2][0]
    
    assert adj_noisy["fidelity"] > far_noisy["fidelity"], (
        f"Adjacent fidelity {adj_noisy['fidelity']} should be higher than distance-2 {far_noisy['fidelity']}"
    )


@pytest.mark.unit
def test_bell_csv_schema_and_values():
    """Verify results/tables/bell.csv structure, columns, and numeric formats."""
    expected_fields = [
        "pair",
        "pair_distance",
        "condition",
        "transpiled_cx",
        "xx",
        "xx_se",
        "yy",
        "yy_se",
        "zz",
        "zz_se",
        "fidelity",
        "fidelity_se",
    ]
    with open("results/tables/bell.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_fields
        rows = list(reader)
        assert len(rows) == 4
        
        for row in rows:
            assert int(row["pair_distance"]) in (1, 2)
            assert row["condition"] in ("ideal", "noisy")
            assert int(row["transpiled_cx"]) >= 1
            assert -1.0 <= float(row["xx"]) <= 1.0
            assert -1.0 <= float(row["yy"]) <= 1.0
            assert -1.0 <= float(row["zz"]) <= 1.0
            assert 0.0 <= float(row["fidelity"]) <= 1.0
            assert float(row["fidelity_se"]) >= 0.0
