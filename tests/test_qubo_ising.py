"""
tests/test_qubo_ising.py - Comprehensive tests for QUBO formulation, Ising Hamiltonian, and energy mappings.
"""
import pytest
import numpy as np
from qiskit.quantum_info import Statevector
from src.formulation import (
    NUM_VERTICES,
    LOCKED_EDGES,
    TOTAL_WEIGHT,
    EXACT_OPTIMUM,
    OPTIMAL_BITSTRINGS,
    cut_weight,
    qubo_objective,
    qubo_matrix,
    cost_hamiltonian,
    enumerate_all_assignments,
    random_baseline,
)


@pytest.mark.unit
def test_qubo_cut_equivalence_and_rankings():
    """Verify QUBO objective Q(x) = -C(x) and strict inversion of ranking order."""
    scores = enumerate_all_assignments()
    sorted_cuts = sorted(scores.items(), key=lambda kv: kv[1])
    
    for bitstring, c in scores.items():
        q = qubo_objective(bitstring)
        assert q == -c, f"QUBO value mismatch: Q({bitstring}) = {q} != -C = {-c}"
        
    # Check that sorting by ascending Q(x) strictly matches descending C(x)
    sorted_by_c_desc = [b for b, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    sorted_by_q_asc = [b for b, _ in sorted(scores.items(), key=lambda kv: qubo_objective(kv[0]))]
    
    # Best states in both should be identical
    assert sorted_by_c_desc[:2] == sorted_by_q_asc[:2]
    assert set(sorted_by_q_asc[:2]) == OPTIMAL_BITSTRINGS


@pytest.mark.unit
def test_qubo_matrix_algebraic_equivalence():
    """Verify that x^T Q_mat x evaluates exactly to Q(x) = -C(x) for all 64 binary vectors."""
    Q = qubo_matrix()
    assert Q.shape == (NUM_VERTICES, NUM_VERTICES)
    assert np.allclose(Q, Q.T), "QUBO matrix is not symmetric"
    
    # Check diagonal and off-diagonal coefficients
    # Diagonal: - sum_{v} w_{uv} for all edges incident on u
    for u in range(NUM_VERTICES):
        deg_w = sum(w for i, j, w in LOCKED_EDGES if i == u or j == u)
        assert np.isclose(Q[u, u], -deg_w)
        
    # For every binary assignment x, x^T Q x must equal -cut_weight(x)
    scores = enumerate_all_assignments()
    for bitstring, c in scores.items():
        x_vec = np.array([int(b) for b in bitstring], dtype=float)
        matrix_eval = float(x_vec @ Q @ x_vec)
        assert np.isclose(matrix_eval, -c), f"Matrix Q eval {matrix_eval} != -cut {-c} for {bitstring}"


@pytest.mark.unit
def test_cost_hamiltonian_structure_and_terms():
    """Verify Cost Hamiltonian Pauli strings, weights, and identity offset."""
    H_C = cost_hamiltonian()
    pauli_table = H_C.to_list()
    
    # 1 Identity term + 9 ZZ terms = 10 terms
    assert len(pauli_table) == 10
    
    # Identity term has coefficient W/2 = 27/2 = 13.5
    id_terms = [(p, c.real) for p, c in pauli_table if p == "I" * NUM_VERTICES]
    assert len(id_terms) == 1
    assert np.isclose(id_terms[0][1], TOTAL_WEIGHT / 2.0)
    
    # Remaining 9 terms must each have 2 'Z's and coefficient -w/2
    zz_terms = [(p, c.real) for p, c in pauli_table if p != "I" * NUM_VERTICES]
    assert len(zz_terms) == 9
    for p, coeff in zz_terms:
        assert p.count("Z") == 2
        assert p.count("I") == NUM_VERTICES - 2
        assert coeff < 0  # -w/2


@pytest.mark.unit
def test_cost_hamiltonian_eigenvalues_and_ground_state():
    """Verify that <x|H_C|x> == C(x) for all 64 states and -H_C ground state is -22."""
    H_C = cost_hamiltonian()
    scores = enumerate_all_assignments()
    
    for bitstring, c in scores.items():
        # In Qiskit, qubit 0 is the rightmost character in Statevector label
        sv_label = bitstring[::-1]
        sv = Statevector.from_label(sv_label)
        exp_val = sv.expectation_value(H_C).real
        assert np.isclose(exp_val, c, atol=1e-5), f"<x|H_C|x> {exp_val} != C(x) {c} for {bitstring}"
        
    # Test -H_C ground state energy
    H_min = -H_C
    matrix = H_min.to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = float(np.min(eigenvalues))
    assert np.isclose(min_eig, -EXACT_OPTIMUM, atol=1e-5)
    
    # Ground state degeneracy must be exactly 2
    ground_count = np.sum(np.isclose(eigenvalues, -EXACT_OPTIMUM, atol=1e-5))
    assert ground_count == 2


@pytest.mark.unit
def test_hamiltonian_uniform_superposition_expectation():
    """Verify <+|H_C|+> equals analytical random baseline of 13.5."""
    H_C = cost_hamiltonian()
    plus_sv = Statevector.from_label("+" * NUM_VERTICES)
    exp_cut = plus_sv.expectation_value(H_C).real
    
    rand = random_baseline()
    assert np.isclose(exp_cut, rand["expected_cut"], atol=1e-5)
    assert np.isclose(exp_cut, 13.5, atol=1e-5)
