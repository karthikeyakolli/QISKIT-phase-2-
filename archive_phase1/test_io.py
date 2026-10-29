"""Unit and integration tests for Challenge Kit IO adapter."""

import json
from pathlib import Path
import tempfile
import unittest

from maxcut.io import (
    ChallengeKitValidationError,
    MaxCutProblemInstance,
    MissingChallengeKitArtifactError,
    ProcessorHardwareSpec,
    load_challenge_instance,
    load_processor_spec,
)


class TestChallengeKitIO(unittest.TestCase):
    """Test suite for organizer input parsing, validation, and error handling."""

    def test_missing_file_raises_custom_error(self):
        """Verify that an absent challenge file raises MissingChallengeKitArtifactError."""
        non_existent_path = Path("instances/challenge/non_existent_file.json")
        with self.assertRaises(MissingChallengeKitArtifactError):
            load_challenge_instance(non_existent_path)

    def test_load_valid_json_instance(self):
        """Verify parsing and validation of a canonical Max-Cut JSON instance."""
        payload = {
            "name": "O1_Sample_Challenge",
            "nodes": [0, 1, 2, 3],
            "edges": [
                [0, 1, 1.0],
                [1, 2, 2.0],
                [2, 3, 1.0],
                [3, 0, 1.0],
            ],
            "is_weighted": True,
            "reference_optimum": 5.0,
            "constraints": {"max_p": 2},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            temp_path = f.name

        try:
            instance = load_challenge_instance(temp_path)
            self.assertIsInstance(instance, MaxCutProblemInstance)
            self.assertEqual(instance.name, "O1_Sample_Challenge")
            self.assertEqual(instance.num_nodes, 4)
            self.assertEqual(instance.num_edges, 4)
            self.assertTrue(instance.is_weighted)
            self.assertEqual(instance.reference_optimum, 5.0)

            # Check networkx graph conversion
            graph = instance.to_networkx()
            self.assertEqual(graph.number_of_nodes(), 4)
            self.assertEqual(graph.number_of_edges(), 4)
            self.assertEqual(graph[1][2]["weight"], 2.0)

            # Check summary generation
            summary = instance.summary()
            self.assertIn("Nodes (|V|): 4", summary)
            self.assertIn("Reference Optimum (C_max): 5.0", summary)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_self_loop_rejection(self):
        """Verify that graph instances with self-loops are strictly rejected."""
        payload = {
            "name": "Invalid_Self_Loop",
            "nodes": [0, 1],
            "edges": [[0, 0, 1.0]],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            temp_path = f.name

        try:
            with self.assertRaises(ChallengeKitValidationError):
                load_challenge_instance(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_processor_spec(self):
        """Verify loading and validating processor hardware specifications."""
        spec_payload = {
            "name": "Processor_A",
            "num_qubits": 5,
            "coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4]],
            "basis_gates": ["rz", "sx", "x", "cx"],
            "qubit_properties": {"t1": 150.0},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec_payload, f)
            temp_path = f.name

        try:
            spec = load_processor_spec(temp_path)
            self.assertIsInstance(spec, ProcessorHardwareSpec)
            self.assertEqual(spec.name, "Processor_A")
            self.assertEqual(spec.num_qubits, 5)
            self.assertEqual(len(spec.coupling_map), 4)
            self.assertEqual(spec.basis_gates, ["rz", "sx", "x", "cx"])
            self.assertIn("Processor: Processor_A", spec.summary())
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
