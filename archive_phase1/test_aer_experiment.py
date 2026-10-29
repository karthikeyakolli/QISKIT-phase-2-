"""Unit and integration tests for Qiskit Aer QAOA experiment pipeline."""

import json
from pathlib import Path
import tempfile
import unittest

from qiskit.circuit import QuantumCircuit

from maxcut.backends.aer_backend import AerBackendRunner
from maxcut.domain.model import MaxCutGraph
from maxcut.experiments.aer_experiment import QAOAAerExperiment, QAOAAerExperimentResult
from maxcut.visualization.plot_distribution import plot_measurement_distribution


class TestAerQAOAExperiment(unittest.TestCase):
    """Test suite for AerBackendRunner, QAOAAerExperiment, and visualization."""

    def setUp(self):
        # 3-node unweighted triangle graph: optimal cut is 2.0
        self.triangle_graph = MaxCutGraph(
            name="Test_Triangle",
            nodes=[0, 1, 2],
            edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        )

        # 4-node cycle graph: (0-1-2-3-0), optimal cut is 4.0
        self.c4_graph = MaxCutGraph(
            name="Test_C4",
            nodes=[0, 1, 2, 3],
            edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        )

    def test_aer_backend_runner(self):
        """Test basic AerBackendRunner execution, gate counting, and shot sum."""
        runner = AerBackendRunner(seed_simulator=42, seed_transpiler=42)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        res = runner.run_circuit(qc, shots=500, optimization_level=1)
        self.assertEqual(res.shots, 500)
        self.assertEqual(sum(res.raw_counts.values()), 500)
        self.assertGreaterEqual(res.transpiled_2q_gates, 1)
        self.assertGreaterEqual(res.transpiled_depth, 1)
        self.assertEqual(res.seed_simulator, 42)

    def test_aer_experiment_reproducibility(self):
        """Test that identical seeds yield identical parameters and deterministic simulation."""
        exp1 = QAOAAerExperiment(seed_global=123, seed_simulator=123, seed_transpiler=123)
        res1 = exp1.run(self.triangle_graph, depth_p=1, shots=1000, maxiter=30)

        exp2 = QAOAAerExperiment(seed_global=123, seed_simulator=123, seed_transpiler=123)
        res2 = exp2.run(self.triangle_graph, depth_p=1, shots=1000, maxiter=30)

        self.assertEqual(res1.optimal_parameters, res2.optimal_parameters)
        self.assertEqual(res1.raw_counts, res2.raw_counts)
        self.assertEqual(res1.decoded_counts, res2.decoded_counts)
        self.assertEqual(res1.best_measured_cut, res2.best_measured_cut)

    def test_configurable_depth_and_shots(self):
        """Test that depth p and shot counts are configurable and reflected in output."""
        exp = QAOAAerExperiment(seed_global=42, seed_simulator=42, seed_transpiler=42)
        res = exp.run(self.c4_graph, depth_p=2, shots=800, maxiter=25)

        self.assertEqual(res.depth_p, 2)
        self.assertEqual(res.shots, 800)
        self.assertEqual(len(res.optimal_gamma), 2)
        self.assertEqual(len(res.optimal_beta), 2)
        self.assertEqual(sum(res.decoded_counts.values()), 800)
        self.assertIn("seed_global", res.seeds)

    def test_classical_optimum_comparison_and_evaluation(self):
        """Test comparison against classical optimum and success probability calculation."""
        exp = QAOAAerExperiment(seed_global=42, seed_simulator=42, seed_transpiler=42)
        
        # Ground truth for C4: C_max = 4.0, optimal bitstrings '0101' and '1010'
        classical_ref = {
            "optimal_cut_value": 4.0,
            "optimal_bitstrings": ["0101", "1010"],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(classical_ref, f)
            ref_path = f.name

        try:
            res = exp.run(
                self.c4_graph,
                depth_p=1,
                shots=1000,
                maxiter=30,
                classical_reference=ref_path,
            )
            self.assertEqual(res.classical_optimal_cut, 4.0)
            self.assertIsNotNone(res.approximation_ratio)
            self.assertGreater(res.approximation_ratio, 0.5)
            self.assertIsNotNone(res.ground_state_success_probability)
            self.assertTrue(res.is_classical_optimum_achieved)
            self.assertEqual(res.best_measured_cut, 4.0)
            self.assertIn(res.best_measured_bitstring, ["0101", "1010"])
        finally:
            Path(ref_path).unlink(missing_ok=True)

    def test_json_serialization_and_summary(self):
        """Test that experiment results serialize cleanly to JSON and format a readable summary."""
        exp = QAOAAerExperiment(seed_global=42, seed_simulator=42, seed_transpiler=42)
        res = exp.run(self.triangle_graph, depth_p=1, shots=500, maxiter=20)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            res.to_json(temp_path)
            with open(temp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["problem_name"], "Test_Triangle")
            self.assertEqual(loaded["shots"], 500)
            self.assertIn("raw_counts", loaded)
            self.assertIn("optimal_parameters", loaded)
            self.assertIn("best_measured_cut", loaded)

            summary = res.summary()
            self.assertIn("End-to-End QAOA Aer Simulation", summary)
            self.assertIn("Best Measured Bitstring", summary)
            self.assertIn("NOTE ON BENCHMARKING", summary)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_distribution_plot_generation(self):
        """Test generation of the probability distribution plot."""
        exp = QAOAAerExperiment(seed_global=42, seed_simulator=42, seed_transpiler=42)
        res = exp.run(self.triangle_graph, depth_p=1, shots=400, maxiter=15)

        with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f:
            plot_path = f.name

        try:
            fig = plot_measurement_distribution(
                decoded_counts=res.decoded_counts,
                graph=self.triangle_graph,
                classical_optimal_cut=2.0,
                classical_optimal_bitstrings=["001", "110", "010", "101", "100", "011"],
                top_k=8,
                save_path=plot_path,
            )
            self.assertIsNotNone(fig)
            self.assertTrue(Path(plot_path).exists())
            self.assertGreater(Path(plot_path).stat().st_size, 1000)
        finally:
            Path(plot_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
