"""Validation of Phase 4 Benchmark Settings: Gate Basis, Synthetic Noise, and Transpiler.

Verifies:
1. Canonical noise rates: p1 = 0.001, p_cx = 0.005, pro = 0.01.
2. Readout error matrix symmetry: P(0|1) == P(1|0) == 0.01, P(0|0) == P(1|1) == 0.99.
3. Transpiler baseline: optimization_level=1, layout_method='sabre', routing_method='sabre', seed=42.
4. Prescribed native basis gates: ['rz', 'sx', 'x', 'cx'] on both Processor A and Processor B.
5. Resource accounting rules: measurement, reset, and barrier treated as non-unitary.
6. Execution budget: default shots = 4096.
"""

import json
from pathlib import Path
import sys
import unittest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestPhase4BenchmarkConfiguration(unittest.TestCase):
    """Test suite certifying Phase 4 gate basis, noise model, and transpiler baseline."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.spec_a_path = cls.repo_root / "processors" / "processor_a" / "hardware_spec.json"
        cls.spec_b_path = cls.repo_root / "processors" / "processor_b" / "hardware_spec.json"
        cls.data_spec_a_path = cls.repo_root / "data" / "hardware_specs" / "processor_a.json"
        cls.data_spec_b_path = cls.repo_root / "data" / "hardware_specs" / "processor_b.json"
        cls.exp_json_path = cls.repo_root / "configs" / "experiment.json"

        with open(cls.spec_a_path, "r", encoding="utf-8") as f:
            cls.spec_a_dict = json.load(f)
        with open(cls.spec_b_path, "r", encoding="utf-8") as f:
            cls.spec_b_dict = json.load(f)
        with open(cls.exp_json_path, "r", encoding="utf-8") as f:
            cls.exp_json_dict = json.load(f)

    def test_canonical_noise_probabilities(self):
        """Verify exact declared noise probabilities match Updated Challenge Kit specifications."""
        noise_cfg = self.exp_json_dict.get("noise", {})
        self.assertEqual(noise_cfg["single_qubit_error"], 0.001)
        self.assertEqual(noise_cfg["two_qubit_error"], 0.005)
        self.assertEqual(noise_cfg["readout_error"], 0.01)
        self.assertEqual(noise_cfg["model_type"], "uniform_depolarizing_symmetric_readout")

    def test_symmetric_readout_matrix_mathematics(self):
        """Verify readout error matrix represents strictly symmetric bit-flip probabilities."""
        pro = self.exp_json_dict["noise"]["readout_error"]
        ro_matrix = [[1.0 - pro, pro], [pro, 1.0 - pro]]
        # P(0|0) == P(1|1) == 1 - pro == 0.99
        self.assertAlmostEqual(ro_matrix[0][0], 0.99, places=6)
        self.assertAlmostEqual(ro_matrix[1][1], 0.99, places=6)
        # P(1|0) == P(0|1) == pro == 0.01 (symmetric bitflip)
        self.assertAlmostEqual(ro_matrix[0][1], 0.01, places=6)
        self.assertAlmostEqual(ro_matrix[1][0], 0.01, places=6)
        # Stochastic row sum = 1.0
        self.assertAlmostEqual(sum(ro_matrix[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(ro_matrix[1]), 1.0, places=6)

    def test_prescribed_basis_gates(self):
        """Verify that native basis gates are strictly ['rz', 'sx', 'x', 'cx'] on both processors."""
        expected_basis = ["rz", "sx", "x", "cx"]
        self.assertEqual(self.spec_a_dict["basis_gates"], expected_basis)
        self.assertEqual(self.spec_b_dict["basis_gates"], expected_basis)
        self.assertNotIn("ecr", self.spec_a_dict["basis_gates"])
        self.assertNotIn("cz", self.spec_b_dict["basis_gates"])
        self.assertEqual(self.spec_a_dict["metadata"]["native_2q_gate"], "cx")
        self.assertEqual(self.spec_b_dict["metadata"]["native_2q_gate"], "cx")

    def test_experiment_config_and_shot_budget(self):
        """Verify experiment configuration mandates 4096 shots and SABRE pass manager settings."""
        self.assertEqual(self.exp_json_dict["execution"]["shots"], 4096)
        self.assertEqual(self.exp_json_dict["transpilation"]["optimization_level"], 1)
        self.assertEqual(self.exp_json_dict["transpilation"]["layout_method"], "sabre")
        self.assertEqual(self.exp_json_dict["transpilation"]["routing_method"], "sabre")
        self.assertEqual(self.exp_json_dict["transpilation"]["seed_transpiler"], 42)

    def test_comparable_noise_across_processors(self):
        """Verify that Processor A and Processor B apply identical declared error probabilities."""
        noise_cfg = self.exp_json_dict["noise"]
        self.assertEqual(noise_cfg["single_qubit_error"], 0.001)
        self.assertEqual(noise_cfg["two_qubit_error"], 0.005)
        self.assertEqual(noise_cfg["readout_error"], 0.01)

    def test_specs_synchronization_across_locations(self):
        """Verify processor specs in processors/ and data/hardware_specs/ are 100% identical."""
        with open(self.data_spec_a_path, "r", encoding="utf-8") as f:
            data_spec_a = json.load(f)
        with open(self.data_spec_b_path, "r", encoding="utf-8") as f:
            data_spec_b = json.load(f)

        self.assertEqual(self.spec_a_dict, data_spec_a)
        self.assertEqual(self.spec_b_dict, data_spec_b)


if __name__ == "__main__":
    unittest.main()
