"""Integration tests for Processor A and Processor B experimental executions."""

import json
from pathlib import Path
import unittest

from maxcut.domain.model import MaxCutGraph
from maxcut.experiments.config import ExperimentConfig
from maxcut.experiments.runner import ExperimentRunner
from maxcut.experiments.schema import ExperimentResult, load_experiment_result
from maxcut.io import load_challenge_instance


class TestProcessorExperiments(unittest.TestCase):
    """Test suite verifying official QAOA experiment execution on Processor A and Processor B."""

    @classmethod
    def setUpClass(cls):
        instance_path = Path("instances/challenge/o1_maxcut_instance.json")
        instance = load_challenge_instance(instance_path)
        cls.graph = MaxCutGraph.from_challenge_instance(instance)

        cls.res_a_path = Path("results/data/O1_MaxCut_Challenge_processor_a_p1_result.json")
        cls.res_b_path = Path("results/data/O1_MaxCut_Challenge_processor_b_p1_result.json")
        cls.res_sim_path = Path("results/data/O1_MaxCut_Challenge_aer_simulator_p1_result.json")

    def test_artifacts_exist_and_are_distinct(self):
        """Verify all three execution artifacts exist as distinct, non-overlapping files."""
        self.assertTrue(self.res_sim_path.exists(), "Simulator result file must exist.")
        self.assertTrue(self.res_a_path.exists(), "Processor A result file must exist.")
        self.assertTrue(self.res_b_path.exists(), "Processor B result file must exist.")

        self.assertNotEqual(self.res_sim_path, self.res_a_path)
        self.assertNotEqual(self.res_a_path, self.res_b_path)

    def test_processor_a_result_validity(self):
        """Verify Processor A experimental result fields, metrics, and job metadata."""
        res_a = load_experiment_result(self.res_a_path)

        self.assertEqual(res_a.backend_metadata["backend_name"], "Processor_A")
        self.assertEqual(res_a.backend_metadata["native_2q_gate"], "ecr")
        self.assertEqual(res_a.backend_metadata["architecture_family"], "Falcon-H")

        # Performance metrics
        self.assertEqual(res_a.cut_value, 6.0)
        self.assertTrue(res_a.is_classical_optimum_achieved)
        self.assertGreater(res_a.expected_cut, 4.0)
        self.assertGreater(res_a.approximation_ratio, 0.75)
        self.assertGreater(res_a.probability_metrics["ground_state_success_probability"], 0.3)

        # Architectural metrics
        cm = res_a.circuit_metrics
        self.assertEqual(cm["logical_qubit_count"], 6)
        self.assertEqual(cm["physical_qubit_count"], 7)
        self.assertEqual(cm["logical_2q_gates"], 8)
        self.assertEqual(cm["transpiled_2q_gates"], 28)
        self.assertEqual(cm["swap_count"], 0)
        self.assertIn("ecr", cm["gate_breakdown"])

        # Job metadata
        jm = res_a.job_metadata
        self.assertIn("job_id", jm)
        self.assertEqual(jm["backend_name"], "Processor_A")
        self.assertEqual(jm["native_2q_gate"], "ecr")

        # Counts
        self.assertEqual(sum(res_a.raw_counts.values()), 2048)
        self.assertEqual(sum(res_a.decoded_counts.values()), 2048)

    def test_processor_b_result_validity(self):
        """Verify Processor B experimental result fields, metrics, and job metadata."""
        res_b = load_experiment_result(self.res_b_path)

        self.assertEqual(res_b.backend_metadata["backend_name"], "Processor_B")
        self.assertEqual(res_b.backend_metadata["native_2q_gate"], "cz")
        self.assertEqual(res_b.backend_metadata["architecture_family"], "Ring-7")

        # Performance metrics
        self.assertEqual(res_b.cut_value, 6.0)
        self.assertTrue(res_b.is_classical_optimum_achieved)
        self.assertGreater(res_b.expected_cut, 4.0)
        self.assertGreater(res_b.approximation_ratio, 0.75)
        self.assertGreater(res_b.probability_metrics["ground_state_success_probability"], 0.3)

        # Architectural metrics
        cm = res_b.circuit_metrics
        self.assertEqual(cm["logical_qubit_count"], 6)
        self.assertEqual(cm["physical_qubit_count"], 7)
        self.assertEqual(cm["logical_2q_gates"], 8)
        self.assertEqual(cm["transpiled_2q_gates"], 28)
        self.assertEqual(cm["swap_count"], 0)
        self.assertIn("cz", cm["gate_breakdown"])

        # Job metadata
        jm = res_b.job_metadata
        self.assertIn("job_id", jm)
        self.assertEqual(jm["backend_name"], "Processor_B")
        self.assertEqual(jm["native_2q_gate"], "cz")

        # Counts
        self.assertEqual(sum(res_b.raw_counts.values()), 2048)
        self.assertEqual(sum(res_b.decoded_counts.values()), 2048)


if __name__ == "__main__":
    unittest.main()
