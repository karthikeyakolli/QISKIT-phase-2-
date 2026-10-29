"""Unit tests for QAOA circuit construction, parameter optimization, and decoding."""

from pathlib import Path
import tempfile
import unittest

from qiskit.quantum_info import Statevector

from maxcut.classical import ExactExhaustiveSolver
from maxcut.domain import MaxCutGraph
from maxcut.qaoa import (
    QAOAAnsatz,
    QAOAOptimizationResult,
    QAOAOptimizer,
    build_qaoa_circuit,
    decode_qiskit_counts,
    evaluate_measurement_distribution,
)
from maxcut.visualization import plot_qaoa_circuit


class TestQAOACore(unittest.TestCase):
    """Test suite for QAOA ansatz, parameter binding, decoding, and classical optimization."""

    def test_circuit_qubit_and_parameter_counts(self):
        """Verify correct qubit count (n) and parameter count (2p) across depths."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Cycle_4")

        # Test p = 1
        ansatz_p1 = build_qaoa_circuit(graph, p=1)
        self.assertEqual(ansatz_p1.num_qubits, 4)
        self.assertEqual(ansatz_p1.p, 1)
        self.assertEqual(len(ansatz_p1.gamma_parameters), 1)
        self.assertEqual(len(ansatz_p1.beta_parameters), 1)
        self.assertEqual(ansatz_p1.num_parameters, 2)
        self.assertEqual(ansatz_p1.circuit.num_qubits, 4)

        # Test p = 2
        ansatz_p2 = build_qaoa_circuit(graph, p=2)
        self.assertEqual(ansatz_p2.num_qubits, 4)
        self.assertEqual(ansatz_p2.p, 2)
        self.assertEqual(len(ansatz_p2.gamma_parameters), 2)
        self.assertEqual(len(ansatz_p2.beta_parameters), 2)
        self.assertEqual(ansatz_p2.num_parameters, 4)

        # Test p = 3
        ansatz_p3 = build_qaoa_circuit(graph, p=3)
        self.assertEqual(ansatz_p3.num_parameters, 6)

    def test_parameter_binding(self):
        """Verify that parameters bind cleanly without unassigned variables."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Triangle")
        ansatz = build_qaoa_circuit(graph, p=2)

        gamma_vals = [0.45, 0.85]
        beta_vals = [0.25, 0.65]

        bound_qc = ansatz.bind(gamma_vals, beta_vals)
        self.assertEqual(len(bound_qc.parameters), 0)

        # Statevector can be evaluated without unbound parameter errors
        sv = Statevector(bound_qc)
        self.assertAlmostEqual(sv.probabilities().sum(), 1.0, places=9)

    def test_measurement_mapping_and_decoding(self):
        """Verify decoding from Qiskit little-endian format to canonical node-ordered format."""
        # 3 nodes: 0, 1, 2
        node_to_qubit = {0: 0, 1: 1, 2: 2}
        sorted_nodes = [0, 1, 2]

        # Raw counts from Qiskit: key 'q2 q1 q0'
        # '001' means q0=1, q1=0, q2=0 -> node 0:1, node 1:0, node 2:0 -> canonical '100'
        # '110' means q0=0, q1=1, q2=1 -> node 0:0, node 1:1, node 2:1 -> canonical '011'
        raw_counts = {"001": 50, "110": 70}

        decoded = decode_qiskit_counts(
            raw_counts=raw_counts,
            node_to_qubit=node_to_qubit,
            sorted_nodes=sorted_nodes,
        )

        self.assertEqual(decoded["100"], 50)
        self.assertEqual(decoded["011"], 70)

    def test_distribution_evaluation(self):
        """Verify calculation of expectation value, best cut, and ground-state success probability."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Cycle_4")

        # Ground truth optimum is 4.0 with bitstrings '1010' and '0101'
        decoded_counts = {
            "1010": 40,  # cut = 4.0 (optimal)
            "0101": 30,  # cut = 4.0 (optimal)
            "0000": 10,  # cut = 0.0
            "1100": 20,  # cut = 2.0
        }

        stats = evaluate_measurement_distribution(
            decoded_counts=decoded_counts,
            graph=graph,
            optimal_cut_value=4.0,
            optimal_bitstrings=["1010", "0101"],
        )

        self.assertEqual(stats["total_shots"], 100)
        # Expected cut = (40*4 + 30*4 + 10*0 + 20*2) / 100 = (160 + 120 + 0 + 40) / 100 = 320 / 100 = 3.2
        self.assertAlmostEqual(stats["expected_cut_value"], 3.2)
        self.assertEqual(stats["best_sampled_cut"], 4.0)
        self.assertEqual(stats["best_sampled_bitstring"], "1010")
        self.assertEqual(stats["most_frequent_bitstring"], "1010")
        # Approx ratio = 3.2 / 4.0 = 0.8
        self.assertAlmostEqual(stats["approximation_ratio"], 0.8)
        # Success probability = (40 + 30) / 100 = 0.70
        self.assertAlmostEqual(stats["ground_state_success_probability"], 0.70)

    def test_qaoa_parameter_optimization_synthetic_graph(self):
        """Verify classical COBYLA parameter optimization on a synthetic 4-node graph."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Cycle_4")

        # Classical reference: C_max = 4.0
        classical_solver = ExactExhaustiveSolver()
        classical_result = classical_solver.solve(graph)
        c_max = classical_result.optimal_cut_value

        # Optimize QAOA with p = 1
        optimizer = QAOAOptimizer(method="COBYLA", maxiter=40, seed=42)
        opt_res = optimizer.optimize(graph, p=1)

        self.assertIsInstance(opt_res, QAOAOptimizationResult)
        self.assertEqual(opt_res.optimizer_name, "COBYLA")
        self.assertEqual(opt_res.p, 1)
        self.assertEqual(len(opt_res.optimal_gamma), 1)
        self.assertEqual(len(opt_res.optimal_beta), 1)
        self.assertGreater(opt_res.num_evaluations, 0)
        self.assertEqual(len(opt_res.history), opt_res.num_evaluations)

        # Verify mathematical sign convention: energy == - <C>
        self.assertAlmostEqual(
            opt_res.final_objective_value,
            -opt_res.final_expected_cut,
            places=9,
        )

        # Optimal expected cut must be positive and bounded by C_max
        self.assertGreater(opt_res.final_expected_cut, 0.0)
        self.assertLessEqual(opt_res.final_expected_cut, c_max + 1e-9)

        # Summary string check
        summary = opt_res.summary()
        self.assertIn("QAOA Parameter Optimization (p=1)", summary)
        self.assertIn("Optimizer: COBYLA", summary)

    def test_circuit_visualization(self):
        """Verify circuit drawing utility."""
        edges = [(0, 1), (1, 2)]
        graph = MaxCutGraph.from_edges(edges, name="Path_3")
        ansatz = build_qaoa_circuit(graph, p=1)

        # Text drawing
        ascii_text = ansatz.draw_ascii()
        self.assertIn("H", ascii_text)
        self.assertIn("ZZ", ascii_text)
        self.assertIn("Rx", ascii_text)

        # Export to text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            plot_qaoa_circuit(ansatz, save_path=temp_path, output="text")
            self.assertTrue(Path(temp_path).exists())
            self.assertGreater(Path(temp_path).stat().st_size, 0)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
