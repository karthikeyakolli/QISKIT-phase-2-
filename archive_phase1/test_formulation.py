"""Unit tests for mathematical formulation: QUBO, Ising, and Quantum Cost Hamiltonian."""

import unittest

from qiskit.quantum_info import Statevector

from maxcut.classical import ExactExhaustiveSolver
from maxcut.domain import MaxCutGraph
from maxcut.formulation import (
    IsingFormulation,
    QUBOFormulation,
    QuantumCostHamiltonian,
    build_cost_hamiltonian,
    build_maxcut_ising,
    build_maxcut_qubo,
    evaluate_all_representations,
)


class TestMathematicalFormulation(unittest.TestCase):
    """Test suite verifying mathematical equivalence across classical, QUBO, Ising, and quantum operators."""

    def test_exhaustive_equivalence_tiny_triangle(self):
        """Verify for every assignment in K_3 that classical cut == -QUBO == -Ising == -Quantum energy."""
        # 3-node unweighted triangle: 8 total assignments
        graph = MaxCutGraph.from_edges([(0, 1), (1, 2), (2, 0)], name="K3_Triangle")
        qubo = build_maxcut_qubo(graph)
        ising = build_maxcut_ising(graph)
        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

        for i in range(1 << graph.num_nodes):  # 0 to 7
            bitstring = f"{i:03b}"
            cut_res = graph.compute_cut(bitstring)
            c_val = cut_res.cut_value

            q_val = qubo.evaluate(bitstring)
            is_val = ising.evaluate(bitstring)
            h_val = hamiltonian.evaluate_assignment(bitstring)

            # Strict mathematical assertions
            self.assertAlmostEqual(
                c_val,
                -q_val,
                places=9,
                msg=f"QUBO mismatch on bitstring {bitstring}: C={c_val}, QUBO={q_val}",
            )
            self.assertAlmostEqual(
                c_val,
                -is_val,
                places=9,
                msg=f"Ising mismatch on bitstring {bitstring}: C={c_val}, Ising={is_val}",
            )
            self.assertAlmostEqual(
                c_val,
                -h_val,
                places=9,
                msg=f"Quantum Hamiltonian mismatch on {bitstring}: C={c_val}, H_C={h_val}",
            )

    def test_exhaustive_equivalence_weighted_4cycle(self):
        """Verify for all 16 assignments in a weighted 4-cycle that all representations agree."""
        edges = [
            (0, 1, 2.5),
            (1, 2, 1.2),
            (2, 3, 3.4),
            (3, 0, 4.1),
        ]
        graph = MaxCutGraph.from_edges(edges, name="Weighted_C4")

        for i in range(1 << graph.num_nodes):
            bitstring = f"{i:04b}"
            eval_dict = evaluate_all_representations(graph, bitstring)
            self.assertTrue(
                eval_dict["is_consistent"],
                f"Inconsistency detected on assignment {bitstring}: {eval_dict}",
            )
            self.assertAlmostEqual(
                eval_dict["classical_cut"],
                -eval_dict["qubo_cost"],
                places=9,
            )
            self.assertAlmostEqual(
                eval_dict["classical_cut"],
                -eval_dict["ising_cost"],
                places=9,
            )
            self.assertAlmostEqual(
                eval_dict["classical_cut"],
                -eval_dict["quantum_energy"],
                places=9,
            )

    def test_ground_state_energy_matches_classical_optimum(self):
        """Verify that minimum eigenvalue of H_C equals -C_max from the classical solver."""
        edges = [(0, 1, 2.0), (1, 2, 3.0), (2, 3, 1.0), (3, 0, 5.0), (0, 2, 4.0)]
        graph = MaxCutGraph.from_edges(edges, name="Pentagon_Diagonal")

        # 1. Classical exact solver
        classical_solver = ExactExhaustiveSolver()
        classical_result = classical_solver.solve(graph)
        c_max = classical_result.optimal_cut_value

        # 2. Quantum Cost Hamiltonian
        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

        # Compute ground state energy by evaluating all basis states
        min_quantum_energy = float("inf")
        for i in range(1 << graph.num_nodes):
            bitstring = f"{i:04b}"
            energy = hamiltonian.evaluate_assignment(bitstring)
            if energy < min_quantum_energy:
                min_quantum_energy = energy

        self.assertAlmostEqual(
            min_quantum_energy,
            -c_max,
            places=9,
            msg=f"Ground state energy ({min_quantum_energy}) does not equal -C_max ({-c_max})",
        )

    def test_qubit_indexing_and_endianness_asymmetry(self):
        """Verify that asymmetric weights correctly differentiate qubit 0 vs qubit n-1 (no endianness inversion)."""
        # Linear path 0 - 1 - 2
        # Edge (0, 1) has weight 10.0
        # Edge (1, 2) has weight 1.0
        edges = [(0, 1, 10.0), (1, 2, 1.0)]
        graph = MaxCutGraph.from_edges(edges, name="AsymmetricPath")
        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

        # Assignment A: node 0 is 1, nodes 1 and 2 are 0 -> cut = 10.0 (edge 0-1 cut, edge 1-2 uncut)
        # Expected energy: -10.0
        res_a = graph.compute_cut("100")
        self.assertEqual(res_a.cut_value, 10.0)
        energy_a = hamiltonian.evaluate_assignment("100")
        self.assertAlmostEqual(energy_a, -10.0, places=9)

        # Assignment B: node 2 is 1, nodes 0 and 1 are 0 -> cut = 1.0 (edge 0-1 uncut, edge 1-2 cut)
        # Expected energy: -1.0
        res_b = graph.compute_cut("001")
        self.assertEqual(res_b.cut_value, 1.0)
        energy_b = hamiltonian.evaluate_assignment("001")
        self.assertAlmostEqual(energy_b, -1.0, places=9)

        # Direct verification with Qiskit Statevector label:
        # In Qiskit, label 'q2 q1 q0':
        # For Assignment A (q0=1, q1=0, q2=0): Qiskit label is '001'
        sv_a = Statevector.from_label("001")
        self.assertAlmostEqual(sv_a.expectation_value(hamiltonian.cost_operator).real, -10.0, places=9)

        # For Assignment B (q0=0, q1=0, q2=1): Qiskit label is '100'
        sv_b = Statevector.from_label("100")
        self.assertAlmostEqual(sv_b.expectation_value(hamiltonian.cost_operator).real, -1.0, places=9)

    def test_core_hamiltonian_and_offset(self):
        """Verify that H_C = H_core + offset * I."""
        edges = [(0, 1, 3.0), (1, 2, 4.0)]
        graph = MaxCutGraph.from_edges(edges, name="OffsetTest")

        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)
        hamiltonian_no_offset = build_cost_hamiltonian(graph, include_offset=False)

        expected_offset = -0.5 * (3.0 + 4.0)  # -3.5
        self.assertEqual(hamiltonian.offset, expected_offset)

        # Difference between full and core operator must be constant offset
        for i in range(1 << 3):
            bs = f"{i:03b}"
            e_full = hamiltonian.evaluate_assignment(bs)
            e_core = hamiltonian_no_offset.evaluate_assignment(bs)
            self.assertAlmostEqual(e_full - e_core, expected_offset, places=9)

    def test_single_edge_unweighted_and_weighted(self):
        """Test small known graph: single edge (2 nodes), unweighted and weighted.

        Verifies all 4 basis states against analytical truth table:
        |x>  | C(x) | QUBO | Ising | H_C (min) | H_cut (max)
        00   |  0   |  0   |   0   |    0      |    0
        01   |  w   | -w   |  -w   |   -w      |    w
        10   |  w   | -w   |  -w   |   -w      |    w
        11   |  0   |  0   |   0   |    0      |    0
        """
        for weight in [1.0, 3.5]:
            graph = MaxCutGraph.from_edges([(0, 1, weight)], name=f"SingleEdge_w{weight}")
            qubo = build_maxcut_qubo(graph)
            ising = build_maxcut_ising(graph)
            hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

            expected_table = {
                "00": 0.0,
                "01": weight,
                "10": weight,
                "11": 0.0,
            }

            for bitstring, exp_cut in expected_table.items():
                cut_res = graph.compute_cut(bitstring)
                self.assertAlmostEqual(cut_res.cut_value, exp_cut, places=9)
                self.assertAlmostEqual(qubo.evaluate(bitstring), -exp_cut, places=9)
                self.assertAlmostEqual(ising.evaluate(bitstring), -exp_cut, places=9)
                self.assertAlmostEqual(hamiltonian.evaluate_assignment(bitstring), -exp_cut, places=9)
                self.assertAlmostEqual(hamiltonian.evaluate_cut(bitstring), exp_cut, places=9)

    def test_disconnected_components_graph(self):
        """Test small known graph with disconnected components and isolated vertices.

        Case A: Two disconnected 2-node components: edge (0, 1) w=2.0 and edge (2, 3) w=3.0.
        Case B: Connected edge (0, 1) with an isolated vertex 2 (degree 0).
        """
        # Case A: Two disconnected edges
        edges_a = [(0, 1, 2.0), (2, 3, 3.0)]
        graph_a = MaxCutGraph(nodes=[0, 1, 2, 3], edges=edges_a, name="Disconnected_TwoEdges")
        qubo_a = build_maxcut_qubo(graph_a)
        ising_a = build_maxcut_ising(graph_a)
        ham_a = build_cost_hamiltonian(graph_a, include_offset=True)

        for i in range(1 << 4):
            bs = f"{i:04b}"
            b0, b1, b2, b3 = int(bs[0]), int(bs[1]), int(bs[2]), int(bs[3])
            expected_cut = 2.0 * (b0 ^ b1) + 3.0 * (b2 ^ b3)

            cut_res = graph_a.compute_cut(bs)
            self.assertAlmostEqual(cut_res.cut_value, expected_cut, places=9)
            self.assertAlmostEqual(qubo_a.evaluate(bs), -expected_cut, places=9)
            self.assertAlmostEqual(ising_a.evaluate(bs), -expected_cut, places=9)
            self.assertAlmostEqual(ham_a.evaluate_assignment(bs), -expected_cut, places=9)
            self.assertAlmostEqual(ham_a.evaluate_cut(bs), expected_cut, places=9)

        # Case B: Isolated vertex 2
        edges_b = [(0, 1, 4.0)]
        graph_b = MaxCutGraph(nodes=[0, 1, 2], edges=edges_b, name="IsolatedVertex")
        ham_b = build_cost_hamiltonian(graph_b, include_offset=True)

        for b0 in [0, 1]:
            for b1 in [0, 1]:
                # State of isolated node 2 must NOT affect cut or energy
                cut_node2_is_0 = graph_b.compute_cut(f"{b0}{b1}0").cut_value
                cut_node2_is_1 = graph_b.compute_cut(f"{b0}{b1}1").cut_value
                self.assertEqual(cut_node2_is_0, cut_node2_is_1)

                energy_node2_is_0 = ham_b.evaluate_assignment(f"{b0}{b1}0")
                energy_node2_is_1 = ham_b.evaluate_assignment(f"{b0}{b1}1")
                self.assertEqual(energy_node2_is_0, energy_node2_is_1)

    def test_complementary_assignments_symmetry(self):
        """Verify Z2 global spin-flip symmetry: C(x) == C(~x) and E(x) == E(~x) for all states."""
        edges = [(0, 1, 2.5), (1, 2, 1.5), (2, 3, 3.5), (3, 0, 4.0), (0, 2, 1.0)]
        graph = MaxCutGraph.from_edges(edges, name="SymmetryTestGraph")
        qubo = build_maxcut_qubo(graph)
        ising = build_maxcut_ising(graph)
        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

        for i in range(1 << graph.num_nodes):
            bs = f"{i:0{graph.num_nodes}b}"
            bs_comp = "".join("1" if b == "0" else "0" for b in bs)

            cut_orig = graph.compute_cut(bs).cut_value
            cut_comp = graph.compute_cut(bs_comp).cut_value
            self.assertAlmostEqual(cut_orig, cut_comp, places=9)

            qubo_orig = qubo.evaluate(bs)
            qubo_comp = qubo.evaluate(bs_comp)
            self.assertAlmostEqual(qubo_orig, qubo_comp, places=9)

            ising_orig = ising.evaluate(bs)
            ising_comp = ising.evaluate(bs_comp)
            self.assertAlmostEqual(ising_orig, ising_comp, places=9)

            ham_orig = hamiltonian.evaluate_assignment(bs)
            ham_comp = hamiltonian.evaluate_assignment(bs_comp)
            self.assertAlmostEqual(ham_orig, ham_comp, places=9)

            cut_op_orig = hamiltonian.evaluate_cut(bs)
            cut_op_comp = hamiltonian.evaluate_cut(bs_comp)
            self.assertAlmostEqual(cut_op_orig, cut_op_comp, places=9)

    def test_official_instance_64_assignments_exhaustive_equivalence(self):
        """Enumerate all 64 assignments for the official six-vertex challenge instance.

        Verifies state-by-state mathematical equivalence across classical cut, QUBO,
        Ising, and Quantum Cost / Cut Hamiltonian operators for the official benchmark.
        """
        from maxcut.io import load_challenge_instance
        instance = load_challenge_instance("instances/challenge/o1_maxcut_instance.json")
        graph = MaxCutGraph.from_challenge_instance(instance)

        self.assertEqual(graph.num_nodes, 6)
        self.assertEqual(graph.num_edges, 8)
        self.assertAlmostEqual(graph.total_weight, 8.0)

        qubo = build_maxcut_qubo(graph)
        ising = build_maxcut_ising(graph)
        hamiltonian = build_cost_hamiltonian(graph, include_offset=True)

        min_quantum_energy = float("inf")
        max_quantum_cut = -float("inf")
        max_classical_cut = -float("inf")
        optimal_states = []

        for i in range(1 << 6):  # 64 assignments
            bs = f"{i:06b}"
            cut_res = graph.compute_cut(bs)
            c_val = cut_res.cut_value

            q_val = qubo.evaluate(bs)
            is_val = ising.evaluate(bs)
            h_energy = hamiltonian.evaluate_assignment(bs)
            h_cut = hamiltonian.evaluate_cut(bs)

            # Mathematical equivalence assertions for each assignment
            self.assertAlmostEqual(c_val, -q_val, places=9)
            self.assertAlmostEqual(c_val, -is_val, places=9)
            self.assertAlmostEqual(c_val, -h_energy, places=9)
            self.assertAlmostEqual(c_val, h_cut, places=9)

            if c_val > max_classical_cut:
                max_classical_cut = c_val
                optimal_states = [bs]
            elif abs(c_val - max_classical_cut) < 1e-9:
                optimal_states.append(bs)

            if h_energy < min_quantum_energy:
                min_quantum_energy = h_energy
            if h_cut > max_quantum_cut:
                max_quantum_cut = h_cut

        # Official instance ground truth checks
        self.assertEqual(max_classical_cut, 6.0)
        self.assertEqual(len(optimal_states), 10)
        self.assertAlmostEqual(min_quantum_energy, -6.0, places=9)
        self.assertAlmostEqual(max_quantum_cut, 6.0, places=9)

    def test_classical_optimum_and_qaoa_objective_convention(self):
        """Verify that classical solver and QAOA optimizer use identical objective convention.

        Classical solver maximizes C(x).
        QAOA optimizer minimizes E = <H_C> = - <C>, which maximizes expected cut <C>.
        Expected cut is always <= C_max, and approximation ratio alpha = <C> / C_max in [0, 1].
        """
        from maxcut.io import load_challenge_instance
        from maxcut.qaoa.ansatz import build_qaoa_circuit
        from maxcut.qaoa.optimizer import QAOAOptimizer

        instance = load_challenge_instance("instances/challenge/o1_maxcut_instance.json")
        graph = MaxCutGraph.from_challenge_instance(instance)

        classical_solver = ExactExhaustiveSolver()
        classical_result = classical_solver.solve(graph)
        c_max = classical_result.optimal_cut_value

        self.assertEqual(c_max, 6.0)

        # Optimize QAOA at p=1
        optimizer = QAOAOptimizer(method="COBYLA", maxiter=50, tol=1e-4, seed=42)
        opt_res = optimizer.optimize(graph, p=1)

        # Verify optimizer energy minimizes -<C>
        self.assertAlmostEqual(opt_res.final_expected_cut, -opt_res.final_objective_value, places=9)
        self.assertGreater(opt_res.final_expected_cut, 0.0)
        self.assertLessEqual(opt_res.final_expected_cut, c_max)

        # Approximation ratio must be consistent
        alpha = opt_res.final_expected_cut / c_max
        self.assertGreater(alpha, 0.70)
        self.assertLessEqual(alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
