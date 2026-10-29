"""Integration and verification tests for Processor A and Processor B adapters."""

from pathlib import Path
import unittest

from qiskit.circuit import QuantumCircuit

from maxcut.backends import BaseBackendRunner, create_backend_runner
from maxcut.domain.model import MaxCutGraph
from maxcut.experiments.config import ExperimentConfig
from maxcut.experiments.runner import ExperimentRunner
from maxcut.io import load_challenge_instance
from maxcut.qaoa.ansatz import build_qaoa_circuit
from processors.processor_a import ProcessorABackend
from processors.processor_b import ProcessorBBackend


class TestProcessorsIntegration(unittest.TestCase):
    """Test suite for Processor A and Processor B integration, metadata, and circuit prep."""

    @classmethod
    def setUpClass(cls):
        # Load official Max-Cut challenge instance (6 nodes, 8 edges)
        instance_path = Path("instances/challenge/o1_maxcut_instance.json")
        instance = load_challenge_instance(instance_path)
        cls.official_graph = MaxCutGraph.from_challenge_instance(instance)

        cls.proc_a = ProcessorABackend()
        cls.proc_b = ProcessorBBackend()

    def test_processor_a_spec_and_metadata(self):
        """Verify Processor A architectural constraints and metadata completeness."""
        self.assertEqual(self.proc_a.backend_name, "Processor_A")
        self.assertEqual(self.proc_a.target_type, "hardware_processor")
        self.assertEqual(self.proc_a.num_qubits, 5)
        self.assertEqual(self.proc_a.spec.basis_gates, ["rz", "sx", "x", "cx"])
        self.assertEqual(len(self.proc_a.spec.coupling_map), 8)

        metadata = self.proc_a.get_metadata()
        self.assertIn("t1_us_mean", metadata["qubit_properties"])
        self.assertIn("t2_us_mean", metadata["qubit_properties"])
        self.assertEqual(metadata["native_2q_gate"], "cx")
        self.assertEqual(metadata["architecture_family"], "5Q-Linear-Chain")

    def test_processor_b_spec_and_metadata(self):
        """Verify Processor B architectural constraints and metadata completeness."""
        self.assertEqual(self.proc_b.backend_name, "Processor_B")
        self.assertEqual(self.proc_b.target_type, "hardware_processor")
        self.assertEqual(self.proc_b.num_qubits, 9)
        self.assertEqual(self.proc_b.spec.basis_gates, ["rz", "sx", "x", "cx"])
        self.assertEqual(len(self.proc_b.spec.coupling_map), 16)

        metadata = self.proc_b.get_metadata()
        self.assertIn("t1_us_mean", metadata["qubit_properties"])
        self.assertIn("t2_us_mean", metadata["qubit_properties"])
        self.assertEqual(metadata["native_2q_gate"], "cx")
        self.assertEqual(metadata["architecture_family"], "9Q-Sparse-Tree")

    def test_qaoa_circuit_preparation_processor_a(self):
        """Verify that the exact QAOA circuit for the official instance compiles onto Processor A."""
        from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
        ansatz = build_reduced_qaoa_circuit(self.official_graph, p=1, fixed_node=0, include_measurements=True)
        # Bind nominal parameters
        bound_qc = ansatz.bind(gamma_values=[2.4835], beta_values=[4.2540])

        verification = self.proc_a.verify_qaoa_circuit(bound_qc, optimization_level=1)
        self.assertTrue(verification["is_compilation_successful"])
        self.assertTrue(verification["is_basis_compliant"])
        self.assertEqual(len(verification["non_basis_gates"]), 0)
        self.assertGreater(verification["transpiled_depth"], 0)
        self.assertGreater(verification["transpiled_2q_gates"], 0)
        # Processor A uses native cx gates for 2Q entangling
        self.assertGreater(verification["cx_gate_count"], 0)

    def test_qaoa_circuit_preparation_processor_b(self):
        """Verify that the exact QAOA circuit for the official instance compiles onto Processor B."""
        from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
        ansatz = build_reduced_qaoa_circuit(self.official_graph, p=1, fixed_node=0, include_measurements=True)
        # Bind nominal parameters
        bound_qc = ansatz.bind(gamma_values=[2.4835], beta_values=[4.2540])

        verification = self.proc_b.verify_qaoa_circuit(bound_qc, optimization_level=1)
        self.assertTrue(verification["is_compilation_successful"])
        self.assertTrue(verification["is_basis_compliant"])
        self.assertEqual(len(verification["non_basis_gates"]), 0)
        self.assertGreater(verification["transpiled_depth"], 0)
        self.assertGreater(verification["transpiled_2q_gates"], 0)
        # Processor B uses native cx gates for 2Q entangling
        self.assertGreater(verification["cx_gate_count"], 0)

    def test_decoupled_backend_factory(self):
        """Verify that create_backend_runner decouples the application from concrete classes."""
        runner_a = create_backend_runner("processor_a")
        self.assertIsInstance(runner_a, BaseBackendRunner)
        self.assertEqual(runner_a.backend_name, "Processor_A")

        runner_b = create_backend_runner("processor_b")
        self.assertIsInstance(runner_b, BaseBackendRunner)
        self.assertEqual(runner_b.backend_name, "Processor_B")

    def test_experiment_runner_end_to_end_with_processors(self):
        """Verify that the common ExperimentRunner compiles circuits for both processors without code modification."""
        exp_runner = ExperimentRunner()

        # Run on Processor A
        cfg_a = ExperimentConfig(
            graph_name="O1_MaxCut_Challenge",
            backend_name="processor_a",
            depth_p=1,
            shots=256,
            optimization_settings={"max_iterations": 10},
        )
        res_a = exp_runner.run(config=cfg_a, graph=self.official_graph, backend=self.proc_a, save_result=False)
        self.assertEqual(res_a.backend_metadata["backend_name"], "Processor_A")
        self.assertEqual(res_a.backend_metadata["native_2q_gate"], "cx")
        self.assertGreater(res_a.circuit_metrics["transpiled_2q_gates"], 0)

        # Run on Processor B
        cfg_b = ExperimentConfig(
            graph_name="O1_MaxCut_Challenge",
            backend_name="processor_b",
            depth_p=1,
            shots=256,
            optimization_settings={"max_iterations": 10},
        )
        res_b = exp_runner.run(config=cfg_b, graph=self.official_graph, backend=self.proc_b, save_result=False)
        self.assertEqual(res_b.backend_metadata["backend_name"], "Processor_B")
        self.assertEqual(res_b.backend_metadata["native_2q_gate"], "cx")
        self.assertGreater(res_b.circuit_metrics["transpiled_2q_gates"], 0)


if __name__ == "__main__":
    unittest.main()
