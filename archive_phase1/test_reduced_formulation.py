"""Exhaustive test suite for 5-qubit symmetry-reduced Max-Cut formulation.

Verifies:
1. All 64 assignments of the original graph evaluated against reduced formulation.
2. Canonicalization of each assignment against its bitwise complement (Z_2 spin-flip symmetry).
3. Cut value invariance under bitwise complement: C(x) == C(1 - x).
4. All 32 reduced assignments cover the 32 partition equivalence classes.
5. Optimum remains 6.0000 for the official instance under the reduced formulation.
6. The 10 optimal bitstrings form exactly 5 complementary pairs, yielding 5 unique optimal reduced assignments.
7. Exact mathematical consistency across classical cut, reduced QUBO, reduced Ising, reduced Cost Hamiltonian, and decoding.
"""

import unittest

from qiskit.quantum_info import Statevector

from maxcut.classical.solver import ExactExhaustiveSolver
from maxcut.domain.model import MaxCutGraph
from maxcut.formulation.qubo import (
    build_cost_hamiltonian,
    build_maxcut_ising,
    build_maxcut_qubo,
    build_reduced_cost_hamiltonian,
    build_reduced_maxcut_ising,
    build_reduced_maxcut_qubo,
    canonicalize_cut_assignment,
    evaluate_reduced_representations,
    expand_reduced_assignment,
    reduce_canonical_assignment,
)
from maxcut.io import load_challenge_instance
from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
from maxcut.qaoa.decoding import decode_reduced_qiskit_counts


class TestReducedFormulationExhaustive(unittest.TestCase):
    """Exhaustive equivalence and validation tests for 5-qubit symmetry reduction."""

    def setUp(self):
        """Load the official challenge instance (6 vertices, 8 edges, unweighted)."""
        instance_data = load_challenge_instance("instances/challenge/o1_maxcut_instance.json")
        self.graph = MaxCutGraph.from_challenge_instance(instance_data)
        self.solver = ExactExhaustiveSolver()
        self.classical_res = self.solver.solve(self.graph)

    def test_01_evaluate_all_64_assignments(self):
        """Test 1: Evaluate all 64 assignments of the original graph."""
        self.assertEqual(self.graph.num_nodes, 6)
        all_cuts = {}
        for i in range(64):
            bs = f"{i:06b}"
            cut_val = self.graph.compute_cut(bs).cut_value
            all_cuts[bs] = cut_val

        self.assertEqual(len(all_cuts), 64)
        max_cut = max(all_cuts.values())
        min_cut = min(all_cuts.values())
        self.assertEqual(max_cut, 6.0)
        self.assertEqual(min_cut, 0.0)

    def test_02_canonicalize_each_assignment_against_complement(self):
        """Test 2: Canonicalize each assignment against its bitwise complement."""
        for i in range(64):
            bs = f"{i:06b}"
            complement = "".join("1" if b == "0" else "0" for b in bs)

            canon_bs = canonicalize_cut_assignment(bs, fixed_node=0)
            canon_comp = canonicalize_cut_assignment(complement, fixed_node=0)

            # Both an assignment and its complement must map to the IDENTICAL canonical bitstring
            self.assertEqual(
                canon_bs,
                canon_comp,
                f"Mismatch for bitstring {bs} vs complement {complement}",
            )
            # The canonical bitstring must strictly fix x_0 = 0
            self.assertEqual(canon_bs[0], "0")

    def test_03_verify_cut_value_is_unchanged(self):
        """Test 3: Verify that cut value is identical for assignment, complement, and canonical form."""
        for i in range(64):
            bs = f"{i:06b}"
            complement = "".join("1" if b == "0" else "0" for b in bs)
            canon_bs = canonicalize_cut_assignment(bs, fixed_node=0)

            cut_original = self.graph.compute_cut(bs).cut_value
            cut_complement = self.graph.compute_cut(complement).cut_value
            cut_canonical = self.graph.compute_cut(canon_bs).cut_value

            self.assertEqual(cut_original, cut_complement)
            self.assertEqual(cut_original, cut_canonical)

    def test_04_verify_32_reduced_assignments_cover_partition_classes(self):
        """Test 4: Verify that all 32 reduced assignments cover all 32 partition equivalence classes."""
        canonical_classes = set()
        for i in range(64):
            bs = f"{i:06b}"
            canon_bs = canonicalize_cut_assignment(bs, fixed_node=0)
            canonical_classes.add(canon_bs)

        # 64 states / 2 (complement symmetry) = exactly 32 equivalence classes
        self.assertEqual(len(canonical_classes), 32)

        # Generate all 32 reduced assignments (5 bits: x1 through x5)
        reduced_classes = set()
        for j in range(32):
            red_bs = f"{j:05b}"
            expanded = expand_reduced_assignment(
                red_bs, num_total_nodes=6, fixed_node=0, fixed_val=0
            )
            self.assertEqual(expanded[0], "0")
            self.assertEqual(expanded[1:], red_bs)
            reduced_classes.add(expanded)

        # The set of 32 expanded reduced assignments must exactly equal the 32 canonical classes
        self.assertEqual(canonical_classes, reduced_classes)

    def test_05_confirm_optimum_remains_6(self):
        """Test 5: Confirm the optimum remains 6.0 for the supplied official instance under reduction."""
        red_qubo = build_reduced_maxcut_qubo(self.graph, fixed_node=0)
        red_ising = build_reduced_maxcut_ising(self.graph, fixed_node=0)
        red_h = build_reduced_cost_hamiltonian(self.graph, fixed_node=0, include_offset=True)

        max_cut_reduced = -1.0
        min_qubo_val = float("inf")
        min_ising_val = float("inf")
        min_h_energy = float("inf")

        for j in range(32):
            red_bs = f"{j:05b}"
            full_bs = expand_reduced_assignment(
                red_bs, num_total_nodes=6, fixed_node=0, fixed_val=0
            )
            cut_val = self.graph.compute_cut(full_bs).cut_value
            q_val = red_qubo.evaluate(red_bs)
            is_val = red_ising.evaluate(red_bs)
            h_val = red_h.evaluate_assignment(red_bs)

            if cut_val > max_cut_reduced:
                max_cut_reduced = cut_val
            if q_val < min_qubo_val:
                min_qubo_val = q_val
            if is_val < min_ising_val:
                min_ising_val = is_val
            if h_val < min_h_energy:
                min_h_energy = h_val

        # Ground truth optimum must remain strictly 6.0
        self.assertEqual(max_cut_reduced, 6.0)
        self.assertAlmostEqual(min_qubo_val, -6.0, places=9)
        self.assertAlmostEqual(min_ising_val, -6.0, places=9)
        self.assertAlmostEqual(min_h_energy, -6.0, places=9)

    def test_06_verify_10_optimal_bitstrings_form_5_complementary_pairs(self):
        """Test 6: Verify the 10 reported optimal bitstrings form exactly 5 complementary pairs."""
        optimal_states = self.classical_res.optimal_bitstrings
        self.assertEqual(len(optimal_states), 10)

        # Verify that for every optimal state, its bitwise complement is also in the optimal set
        for bs in optimal_states:
            complement = "".join("1" if b == "0" else "0" for b in bs)
            self.assertIn(
                complement,
                optimal_states,
                f"Bitwise complement {complement} not in optimal set",
            )

        # When canonicalized to x_0 = 0, exactly 5 unique canonical bitstrings must exist
        canonical_optima = set(
            canonicalize_cut_assignment(bs, fixed_node=0) for bs in optimal_states
        )
        self.assertEqual(len(canonical_optima), 5)

        # Extract the 5 reduced 5-qubit bitstrings (x1 through x5)
        reduced_optima = set(
            reduce_canonical_assignment(bs, fixed_node=0) for bs in optimal_states
        )
        self.assertEqual(len(reduced_optima), 5)

        # Verify each of the 5 reduced optima achieves C = 6.0
        for red_bs in reduced_optima:
            full_bs = expand_reduced_assignment(
                red_bs, num_total_nodes=6, fixed_node=0, fixed_val=0
            )
            self.assertEqual(self.graph.compute_cut(full_bs).cut_value, 6.0)

    def test_07_qubo_ising_qaoa_objective_and_decoding_consistency(self):
        """Test 7: Check QUBO, Ising, QAOA objective and decoding consistency across all 32 assignments."""
        red_qubo = build_reduced_maxcut_qubo(self.graph, fixed_node=0)
        red_ising = build_reduced_maxcut_ising(self.graph, fixed_node=0)
        red_h = build_reduced_cost_hamiltonian(self.graph, fixed_node=0, include_offset=True)

        for j in range(32):
            red_bs = f"{j:05b}"
            eval_dict = evaluate_reduced_representations(
                self.graph, red_bs, fixed_node=0
            )
            self.assertTrue(eval_dict["is_consistent"])

            c_val = eval_dict["classical_cut"]
            q_val = eval_dict["qubo_cost"]
            is_val = eval_dict["ising_cost"]
            h_val = eval_dict["quantum_energy"]
            h_cut = eval_dict["quantum_cut_value"]

            # Exact numerical consistency across all four representations
            self.assertAlmostEqual(c_val, -q_val, places=9)
            self.assertAlmostEqual(c_val, -is_val, places=9)
            self.assertAlmostEqual(c_val, -h_val, places=9)
            self.assertAlmostEqual(c_val, h_cut, places=9)

        # Test QAOA circuit construction on 5 qubits
        ansatz = build_reduced_qaoa_circuit(self.graph, p=1, fixed_node=0)
        self.assertEqual(ansatz.num_qubits, 5)
        self.assertEqual(len(ansatz.gamma_parameters), 1)
        self.assertEqual(len(ansatz.beta_parameters), 1)

        # Test statevector energy calculation for reduced ansatz
        bound_qc = ansatz.bind(gamma_values=[2.4835], beta_values=[4.2540])
        sv = Statevector.from_instruction(bound_qc)
        expected_energy = sv.expectation_value(red_h.cost_operator).real
        expected_cut = -expected_energy
        self.assertGreater(expected_cut, 0.0)
        self.assertLessEqual(expected_cut, 6.0)

        # Test decoding function
        dummy_raw_counts = {"00000": 100, "11111": 200}
        active_nodes = [1, 2, 3, 4, 5]
        node_to_qubit = {node: i for i, node in enumerate(active_nodes)}
        decoded = decode_reduced_qiskit_counts(
            raw_counts=dummy_raw_counts,
            node_to_qubit=node_to_qubit,
            all_sorted_nodes=[0, 1, 2, 3, 4, 5],
            fixed_node=0,
            fixed_val=0,
        )
        # "00000" on active nodes -> "000000"
        # "11111" on active nodes -> "011111"
        self.assertIn("000000", decoded)
        self.assertIn("011111", decoded)
        self.assertEqual(decoded["000000"], 100)
        self.assertEqual(decoded["011111"], 200)


if __name__ == "__main__":
    unittest.main()
