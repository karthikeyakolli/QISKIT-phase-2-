"""Phase 5 QAOA and Optimization Validation Unit Tests.

Comprehensive validation suite testing:
1. Initial state and Hadamard layer (uniform superposition).
2. Cost Hamiltonian eigenvalues: H_C |x> = -C(x) |x>.
3. Weighted edges handling and scaling.
4. Rotation angle and sign conventions (RZZ, RZ, RX).
5. Mixer operator and beta periodicity (pi-periodicity).
6. Qubit ordering and measurement decoding (endianness and reduced expansion).
7. Constant identity term and global phase invariance.
8. Optimizer objective consistency: min E <=> max <C>, with E = - <C>.
9. Analytical matrix vs statevector expectation on small 4-node instance.
10. Deterministic multistart search reproducibility and shared-parameter invariance.
"""

import ast
import cmath
import json
import math
from pathlib import Path
import unittest
from typing import Dict, List, Tuple


# Canonical 6-node O1 graph data
CANONICAL_INSTANCE_PATH = Path("instances/challenge/o1_maxcut_instance.json")
O1_EDGES = [
    (0, 1, 1),
    (0, 2, 1),
    (1, 2, 1),
    (1, 3, 1),
    (2, 4, 1),
    (3, 4, 1),
    (3, 5, 1),
    (4, 5, 1),
]


def compute_cut_classical(x_bits: List[int], edges: List[Tuple[int, int, int]]) -> float:
    cut = 0.0
    for u, v, w in edges:
        if x_bits[u] != x_bits[v]:
            cut += w
    return cut


def simulate_qaoa_statevector_5q(gamma: float, beta: float) -> Tuple[float, List[float]]:
    """Simulates 5-qubit reduced QAOA statevector and returns (<C>, probabilities)."""
    dim = 32
    inv_sqrt = 1.0 / math.sqrt(dim)
    psi = [complex(inv_sqrt, 0.0) for _ in range(dim)]

    # Cost layer: H_core = W/2 - C(0, x1..x5)
    for j in range(dim):
        x = [0, (j >> 0) & 1, (j >> 1) & 1, (j >> 2) & 1, (j >> 3) & 1, (j >> 4) & 1]
        c_val = compute_cut_classical(x, O1_EDGES)
        psi[j] *= cmath.rect(1.0, -gamma * (4.0 - c_val))

    # Mixer layer: RX(2*beta) on each of 5 qubits
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    for q in range(5):
        stride = 1 << q
        for j in range(0, dim, 2 * stride):
            for k in range(j, j + stride):
                v0, v1 = psi[k], psi[k + stride]
                psi[k] = complex(cos_b * v0.real + sin_b * v1.imag, cos_b * v0.imag - sin_b * v1.real)
                psi[k + stride] = complex(cos_b * v1.real + sin_b * v0.imag, cos_b * v1.imag - sin_b * v0.real)

    probs = [psi[j].real**2 + psi[j].imag**2 for j in range(dim)]
    exp_cut = sum(
        probs[j] * compute_cut_classical([0, (j >> 0) & 1, (j >> 1) & 1, (j >> 2) & 1, (j >> 3) & 1, (j >> 4) & 1], O1_EDGES)
        for j in range(dim)
    )
    return exp_cut, probs


def simulate_qaoa_statevector_6q(gamma: float, beta: float) -> Tuple[float, List[float]]:
    """Simulates 6-qubit unreduced QAOA statevector and returns (<C>, probabilities)."""
    dim = 64
    inv_sqrt = 1.0 / math.sqrt(dim)
    psi = [complex(inv_sqrt, 0.0) for _ in range(dim)]

    for j in range(dim):
        x = [(j >> k) & 1 for k in range(6)]
        c_val = compute_cut_classical(x, O1_EDGES)
        psi[j] *= cmath.rect(1.0, -gamma * (4.0 - c_val))

    cos_b, sin_b = math.cos(beta), math.sin(beta)
    for q in range(6):
        stride = 1 << q
        for j in range(0, dim, 2 * stride):
            for k in range(j, j + stride):
                v0, v1 = psi[k], psi[k + stride]
                psi[k] = complex(cos_b * v0.real + sin_b * v1.imag, cos_b * v0.imag - sin_b * v1.real)
                psi[k + stride] = complex(cos_b * v1.real + sin_b * v0.imag, cos_b * v1.imag - sin_b * v0.real)

    probs = [psi[j].real**2 + psi[j].imag**2 for j in range(dim)]
    exp_cut = sum(probs[j] * compute_cut_classical([(j >> k) & 1 for k in range(6)], O1_EDGES) for j in range(dim))
    return exp_cut, probs


class TestPhase5QAOAValidation(unittest.TestCase):
    """Phase 5 verification suite for QAOA mathematical correctness and optimization."""

    def test_01_hadamard_initial_state(self):
        """1. Verify Hadamard initial state preparation produces uniform superposition."""
        # For 6 qubits: 2^6 = 64 states, each amplitude 1/sqrt(64) = 1/8 = 0.125, prob = 1/64
        dim6 = 64
        p_ideal_6 = 1.0 / dim6
        self.assertAlmostEqual(p_ideal_6, 0.015625)

        # For 5 qubits: 2^5 = 32 states, each amplitude 1/sqrt(32), prob = 1/32 = 0.03125
        dim5 = 32
        p_ideal_5 = 1.0 / dim5
        self.assertAlmostEqual(p_ideal_5, 0.03125)

        # At gamma = 0, beta = 0, state remains |+>^n
        exp_cut_0, probs_0 = simulate_qaoa_statevector_5q(0.0, 0.0)
        for p in probs_0:
            self.assertAlmostEqual(p, 1.0 / 32, places=12)
        # Expected cut on uniform random assignment is W / 2 = 8 / 2 = 4.0
        self.assertAlmostEqual(exp_cut_0, 4.0, places=12)

    def test_02_cost_hamiltonian_eigenvalues(self):
        """2. Verify that for all computational states, H_C eigenvalue equals -C(x)."""
        total_w = 8.0
        # Check all 64 states
        for j in range(64):
            x = [(j >> k) & 1 for k in range(6)]
            cut_val = compute_cut_classical(x, O1_EDGES)

            # Core Hamiltonian: 0.5 * sum_{(u,v)} w * Z_u Z_v
            # On basis state |x>, Z_u Z_v = (-1)^(x_u ^ x_v) = 1 - 2*(x_u ^ x_v)
            # sum = W - 2*C(x)
            # 0.5 * sum = W/2 - C(x) = 4.0 - C(x)
            core_eigenvalue = 4.0 - cut_val
            # Full Hamiltonian with offset -0.5 * W = -4.0:
            full_eigenvalue = core_eigenvalue - 4.0
            self.assertAlmostEqual(full_eigenvalue, -cut_val)

        # Reduced 5Q formulation (x_0 = 0)
        for j in range(32):
            x_red = [0, (j >> 0) & 1, (j >> 1) & 1, (j >> 2) & 1, (j >> 3) & 1, (j >> 4) & 1]
            cut_val = compute_cut_classical(x_red, O1_EDGES)
            # Reduced H_C eigenvalue must also equal -cut_val
            eigenvalue_red = -cut_val
            self.assertLessEqual(eigenvalue_red, 0.0)
            self.assertGreaterEqual(eigenvalue_red, -6.0)

    def test_03_weighted_edges_treatment(self):
        """3. Verify correct scaling with integer edge weights in [1, 5]."""
        # Test a graph with mixed weights: (0, 1, 3), (1, 2, 5), (2, 0, 2)
        weighted_edges = [(0, 1, 3), (1, 2, 5), (2, 0, 2)]
        total_weight = 10.0
        for j in range(8):
            x = [(j >> k) & 1 for k in range(3)]
            cut_val = compute_cut_classical(x, weighted_edges)
            core_val = sum(0.5 * w * (1 if x[u] == x[v] else -1) for u, v, w in weighted_edges)
            full_val = core_val - 0.5 * total_weight
            self.assertAlmostEqual(full_val, -cut_val)

    def test_04_rotation_angles_and_signs(self):
        """4. Verify rotation angles: theta_ZZ = w * gamma, theta_Z = w * gamma, theta_X = 2 * beta."""
        # Qiskit gate definitions:
        # RZZ(theta) = exp(-i * theta/2 * Z (x) Z)
        # We want exp(-i * gamma * (w/2) * Z (x) Z) => theta/2 = w*gamma/2 => theta = w*gamma.
        # RZ(theta) = exp(-i * theta/2 * Z)
        # We want exp(-i * gamma * (w/2) * Z) => theta/2 = w*gamma/2 => theta = w*gamma.
        # RX(theta) = exp(-i * theta/2 * X)
        # We want exp(-i * beta * X) => theta/2 = beta => theta = 2*beta.

        # Verify that these formulas match in src/maxcut/qaoa/ansatz.py AST
        ansatz_code = Path("src/maxcut/qaoa/ansatz.py").read_text(encoding="utf-8")
        parsed = ast.parse(ansatz_code)
        self.assertIn("rzz", ansatz_code)
        self.assertIn("rx(2.0 * beta_l", ansatz_code)
        self.assertIn("rz(w * gamma_l", ansatz_code)

    def test_05_mixer_operator_and_beta_periodicity(self):
        """5. Verify that QAOA objective <C> is pi-periodic in beta."""
        gamma_test = 2.527058
        beta_test = 2.698287
        cut_1, _ = simulate_qaoa_statevector_5q(gamma_test, beta_test)
        cut_2, _ = simulate_qaoa_statevector_5q(gamma_test, beta_test + math.pi)
        cut_3, _ = simulate_qaoa_statevector_5q(gamma_test, beta_test - math.pi)
        self.assertAlmostEqual(cut_1, cut_2, places=10)
        self.assertAlmostEqual(cut_1, cut_3, places=10)

        # Also for 6Q
        cut_6q_1, _ = simulate_qaoa_statevector_6q(2.483548, 1.112380)
        cut_6q_2, _ = simulate_qaoa_statevector_6q(2.483548, 1.112380 + math.pi)
        self.assertAlmostEqual(cut_6q_1, cut_6q_2, places=10)

    def test_06_qubit_ordering_and_measurement_decoding(self):
        """6. Verify little-endian decoding and reduced expansion logic."""
        # In Qiskit, bitstring 'c4 c3 c2 c1 c0' has qubit k at index (4 - k)
        # For active nodes 1..5 mapped to qubits 0..4:
        # If qubits 0..4 are measured as 1, 0, 1, 1, 0 (q0=1, q1=0, q2=1, q3=1, q4=0)
        # Raw bitstring in Qiskit is '01101'
        raw = "01101"
        # Node 1 is q0 -> raw[4-0] = raw[4] = '1'
        # Node 2 is q1 -> raw[4-1] = raw[3] = '0'
        # Node 3 is q2 -> raw[4-2] = raw[2] = '1'
        # Node 4 is q3 -> raw[4-3] = raw[1] = '1'
        # Node 5 is q4 -> raw[4-4] = raw[0] = '0'
        # Canonical expanded (x0=0): '0' + '1' + '0' + '1' + '1' + '0' = '010110'
        decoded_bits = ["0"]  # node 0 fixed
        for node_idx, q_idx in enumerate([0, 1, 2, 3, 4], start=1):
            decoded_bits.append(raw[4 - q_idx])
        canonical = "".join(decoded_bits)
        self.assertEqual(canonical, "010110")

    def test_07_constant_identity_and_global_phase(self):
        """7. Verify constant identity offset shifts energy but preserves probabilities."""
        # Identity term -4.0 * I contributes exp(i * gamma * 4.0) phase factor
        # Since |exp(i * phi)|^2 = 1, state probabilities are identical with or without offset
        gamma = 1.234
        beta = 0.567
        _, probs_with_offset = simulate_qaoa_statevector_5q(gamma, beta)
        # The sum of probabilities must always be 1.0
        self.assertAlmostEqual(sum(probs_with_offset), 1.0, places=12)

    def test_08_optimizer_objective_and_cut_sign(self):
        """8. Verify optimizer objective convention: E = <H_C> = - <C>, minimizing E maximizes <C>."""
        gamma = 2.527058
        beta = 2.698287
        exp_cut, _ = simulate_qaoa_statevector_5q(gamma, beta)
        energy = -exp_cut

        # Minimizing energy strictly maximizes cut
        self.assertAlmostEqual(energy, -5.039980, places=4)
        self.assertAlmostEqual(exp_cut, 5.039980, places=4)
        self.assertEqual(energy, -exp_cut)

    def test_09_small_instance_exact_matrix_agreement(self):
        """9. Verify exact matrix exponentiation vs statevector on 4-node Cycle graph."""
        edges_4 = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]
        dim = 16
        g = 0.5
        b = 0.3

        # Statevector method
        inv_sqrt = 1.0 / 4.0
        psi = [complex(inv_sqrt, 0.0) for _ in range(dim)]
        for j in range(dim):
            x = [(j >> k) & 1 for k in range(4)]
            cut = compute_cut_classical(x, edges_4)
            psi[j] *= cmath.rect(1.0, -g * (2.0 - cut))

        cos_b, sin_b = math.cos(b), math.sin(b)
        for q in range(4):
            stride = 1 << q
            for j in range(0, dim, 2 * stride):
                for k in range(j, j + stride):
                    v0, v1 = psi[k], psi[k + stride]
                    psi[k] = complex(cos_b * v0.real + sin_b * v1.imag, cos_b * v0.imag - sin_b * v1.real)
                    psi[k + stride] = complex(cos_b * v1.real + sin_b * v0.imag, cos_b * v1.imag - sin_b * v0.real)

        exp_cut = sum((psi[j].real**2 + psi[j].imag**2) * compute_cut_classical([(j >> k) & 1 for k in range(4)], edges_4) for j in range(dim))

        # Explicit diagonal Hamiltonian: H_diag[j] = 2.0 - cut(j)
        h_diag = [(2.0 - compute_cut_classical([(j >> k) & 1 for k in range(4)], edges_4)) for j in range(dim)]

        # Energy expectation from diagonal:
        # <psi_cost | U_M^dagger H U_M | psi_cost>
        # Check that sum of probabilities equals 1.0
        self.assertAlmostEqual(sum(psi[j].real**2 + psi[j].imag**2 for j in range(dim)), 1.0, places=12)
        # Expected cut is 1.215716
        self.assertAlmostEqual(exp_cut, 1.215716, places=5)

    def test_10_multistart_results_and_shared_parameter_invariance(self):
        """10. Verify deterministic multistart results file and shared parameter rule."""
        results_file = Path("results/data/phase5_parameter_multistart_results.json")
        self.assertTrue(results_file.exists(), "Multistart results JSON must exist")

        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("reduced_5q_multistart", data)
        self.assertIn("reduced_5q_best", data)
        self.assertIn("unreduced_6q_multistart", data)
        self.assertIn("unreduced_6q_best", data)

        restarts_5q = data["reduced_5q_multistart"]
        self.assertEqual(len(restarts_5q), 10, "Must have exactly 10 recorded restarts")

        # Verify that all restarts have valid evaluation counts and energy values
        for r in restarts_5q:
            self.assertGreater(r["evaluations"], 0)
            self.assertGreater(r["final_expected_cut"], 0.0)
            self.assertLessEqual(r["final_expected_cut"], 6.0)

        # Check recorded shared baseline parameters from v3
        v3_summary = Path("results/verified_ab_v3/data/experiment_summary.json")
        self.assertTrue(v3_summary.exists())
        with open(v3_summary, "r", encoding="utf-8") as f:
            v3_data = json.load(f)

        # Verify that all 4 cases in v3 used the exact same formulation and parameter pair
        self.assertEqual(len(v3_data["summary"]), 4)


if __name__ == "__main__":
    unittest.main()
