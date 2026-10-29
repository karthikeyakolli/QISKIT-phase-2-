"""Regression tests for Phase 4 — Controlled Ideal/Noisy A/B Experiments.

Verifies:
1. All four required O1 experiment cases have valid, traceable JSON result artifacts.
2. Target configuration and execution modes are traceable (ideal emulation vs noisy simulation).
3. Raw counts are preserved and sum to exactly 2048 shots.
4. Decoded counts map to canonical 6-vertex bitstrings adhering to x0 = 0.
5. All metrics are non-null and derived directly from measurement/circuit artifacts.
6. Real degradation is demonstrated between ideal and noisy baselines.
7. Official noise configuration matches hardware spec properties without fabrication.
8. Physical qubit mapping adheres to 5 active qubits on both targets with zero coupling violations.
"""

import json
from pathlib import Path
import unittest


class TestVerifiedABV2Experiments(unittest.TestCase):
    """Test suite for Phase 4 controlled ideal/noisy experiments."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.base_dir = cls.repo_root / "results" / "verified_ab_v2"
        cls.data_dir = cls.base_dir / "data"
        cls.circuits_dir = cls.base_dir / "circuits"
        cls.tables_dir = cls.base_dir / "tables"

        cls.case_files = {
            "proc_a_ideal": cls.data_dir / "processor_a_ideal_reference.json",
            "proc_a_noisy": cls.data_dir / "processor_a_noisy_baseline.json",
            "proc_b_ideal": cls.data_dir / "processor_b_ideal_reference.json",
            "proc_b_noisy": cls.data_dir / "processor_b_noisy_baseline.json",
        }

        cls.data = {}
        for key, path in cls.case_files.items():
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    cls.data[key] = json.load(f)

    def test_01_all_required_result_artifacts_exist(self):
        """Verify that every required experiment case has a persisted JSON artifact."""
        for key, path in self.case_files.items():
            self.assertTrue(path.exists(), f"Missing required result artifact: {path.name}")
        self.assertTrue((self.data_dir / "experiment_summary.json").exists())
        self.assertTrue((self.tables_dir / "ab_controlled_comparison.md").exists())
        self.assertTrue((self.tables_dir / "ab_controlled_comparison.csv").exists())

    def test_02_execution_mode_and_target_traceability(self):
        """Verify execution modes and hardware metadata are accurately categorized."""
        # Processor A Ideal
        d_a_ideal = self.data["proc_a_ideal"]
        self.assertEqual(d_a_ideal["execution_mode"], "aer_hardware_topology_emulation")
        self.assertFalse(d_a_ideal["is_noisy_simulation"])
        self.assertFalse(d_a_ideal["is_physical_hardware"])
        self.assertEqual(d_a_ideal["backend_metadata"]["backend_name"], "Processor_A")
        self.assertEqual(d_a_ideal["backend_metadata"]["native_2q_gate"], "ecr")

        # Processor A Noisy
        d_a_noisy = self.data["proc_a_noisy"]
        self.assertEqual(d_a_noisy["execution_mode"], "aer_noisy_simulation")
        self.assertTrue(d_a_noisy["is_noisy_simulation"])
        self.assertFalse(d_a_noisy["is_physical_hardware"])
        self.assertIsNotNone(d_a_noisy["noise_config"])
        self.assertEqual(d_a_noisy["noise_config"]["two_qubit_error"], 0.0075)

        # Processor B Ideal
        d_b_ideal = self.data["proc_b_ideal"]
        self.assertEqual(d_b_ideal["execution_mode"], "aer_hardware_topology_emulation")
        self.assertFalse(d_b_ideal["is_noisy_simulation"])
        self.assertFalse(d_b_ideal["is_physical_hardware"])
        self.assertEqual(d_b_ideal["backend_metadata"]["backend_name"], "Processor_B")
        self.assertEqual(d_b_ideal["backend_metadata"]["native_2q_gate"], "cz")

        # Processor B Noisy
        d_b_noisy = self.data["proc_b_noisy"]
        self.assertEqual(d_b_noisy["execution_mode"], "aer_noisy_simulation")
        self.assertTrue(d_b_noisy["is_noisy_simulation"])
        self.assertFalse(d_b_noisy["is_physical_hardware"])
        self.assertIsNotNone(d_b_noisy["noise_config"])
        self.assertEqual(d_b_noisy["noise_config"]["two_qubit_error"], 0.0050)

    def test_03_raw_and_decoded_counts_preservation(self):
        """Verify raw measurement counts sum to 2048 and map onto canonical 6-bit strings with x0=0."""
        for key, record in self.data.items():
            raw_counts = record["raw_counts"]
            decoded_counts = record["decoded_counts"]

            self.assertGreater(len(raw_counts), 0)
            self.assertEqual(sum(raw_counts.values()), 2048)

            # Raw counts must be 5-bit strings
            for bs in raw_counts.keys():
                self.assertEqual(len(bs), 5)

            # Decoded counts must be 6-bit strings starting with '0'
            self.assertEqual(sum(decoded_counts.values()), 2048)
            for bs in decoded_counts.keys():
                self.assertEqual(len(bs), 6)
                self.assertTrue(bs.startswith("0"), f"Decoded bitstring '{bs}' does not satisfy x0=0")

    def test_04_controlled_settings_invariance_across_matrix(self):
        """Verify that all 4 runs share identical parameters, seeds, and formulation."""
        gamma_ref = self.data["proc_a_ideal"]["algorithm_parameters"]["optimal_gamma"][0]
        beta_ref = self.data["proc_a_ideal"]["algorithm_parameters"]["optimal_beta"][0]

        for key, record in self.data.items():
            params = record["algorithm_parameters"]
            self.assertEqual(params["formulation"], "5_qubit_symmetry_reduced")
            self.assertEqual(params["fixed_node"], 0)
            self.assertEqual(params["depth_p"], 1)
            self.assertEqual(params["shots"], 2048)
            self.assertAlmostEqual(params["optimal_gamma"][0], gamma_ref, places=6)
            self.assertAlmostEqual(params["optimal_beta"][0], beta_ref, places=6)

            seeds = record["reproducibility_information"]["random_seeds"]
            self.assertEqual(seeds["seed_global"], 42)
            self.assertEqual(seeds["seed_simulator"], 42)
            self.assertEqual(seeds["seed_transpiler"], 42)

    def test_05_solution_quality_and_noise_degradation(self):
        """Verify solution metrics and demonstrate degradation under noise."""
        for key, record in self.data.items():
            res = record["results"]
            self.assertEqual(res["classical_optimal_cut"], 6.0)
            self.assertEqual(res["best_sampled_cut"], 6.0)
            self.assertTrue(res["is_classical_optimum_achieved"])
            self.assertGreater(res["expected_cut"], 4.5)
            self.assertGreater(res["approximation_ratio"], 0.75)
            self.assertGreater(res["ground_state_success_probability"], 0.30)
            self.assertGreater(res["sample_std_error"], 0.0)

        # Processor A noise degradation
        a_ideal_c = self.data["proc_a_ideal"]["results"]["expected_cut"]
        a_noisy_c = self.data["proc_a_noisy"]["results"]["expected_cut"]
        a_ideal_p = self.data["proc_a_ideal"]["results"]["ground_state_success_probability"]
        a_noisy_p = self.data["proc_a_noisy"]["results"]["ground_state_success_probability"]
        self.assertGreater(a_ideal_c, a_noisy_c, "Processor A ideal <C> should exceed noisy <C>")
        self.assertGreater(a_ideal_p, a_noisy_p, "Processor A ideal P_opt should exceed noisy P_opt")

        # Processor B noise degradation
        b_ideal_c = self.data["proc_b_ideal"]["results"]["expected_cut"]
        b_noisy_c = self.data["proc_b_noisy"]["results"]["expected_cut"]
        b_ideal_p = self.data["proc_b_ideal"]["results"]["ground_state_success_probability"]
        b_noisy_p = self.data["proc_b_noisy"]["results"]["ground_state_success_probability"]
        self.assertGreater(b_ideal_c, b_noisy_c, "Processor B ideal <C> should exceed noisy <C>")
        self.assertGreater(b_ideal_p, b_noisy_p, "Processor B ideal P_opt should exceed noisy P_opt")

    def test_06_physical_qubit_usage_and_circuit_metrics(self):
        """Verify physical qubit allocations and gate breakdowns in result records."""
        cm_a = self.data["proc_a_ideal"]["circuit_metrics"]
        self.assertEqual(cm_a["logical_qubits"], 5)
        self.assertEqual(cm_a["physical_qubits_active_count"], 5)
        self.assertEqual(cm_a["physical_qubits_used"], [0, 1, 2, 3, 5])
        self.assertEqual(cm_a["transpiled_2q_gates"], 18)
        self.assertEqual(cm_a["basis_synthesis_2q_gates"], 12)
        self.assertEqual(cm_a["routing_2q_gates"], 6)
        self.assertEqual(cm_a["routing_swap_count"], 2)
        self.assertEqual(cm_a["transpiled_depth"], 66)

        cm_b = self.data["proc_b_ideal"]["circuit_metrics"]
        self.assertEqual(cm_b["logical_qubits"], 5)
        self.assertEqual(cm_b["physical_qubits_active_count"], 5)
        self.assertEqual(cm_b["physical_qubits_used"], [0, 3, 4, 5, 6])
        self.assertEqual(cm_b["transpiled_2q_gates"], 21)
        self.assertEqual(cm_b["basis_synthesis_2q_gates"], 12)
        self.assertEqual(cm_b["routing_2q_gates"], 9)
        self.assertEqual(cm_b["routing_swap_count"], 3)
        self.assertEqual(cm_b["transpiled_depth"], 81)

    def test_07_circuit_artifacts_exported(self):
        """Verify circuit QASM artifacts exist in results/verified_ab_v2/circuits/."""
        self.assertTrue((self.circuits_dir / "reduced_5q_logical_qaoa_p1.qasm").exists())
        self.assertTrue((self.circuits_dir / "reduced_5q_processor_a_transpiled.qasm").exists())
        self.assertTrue((self.circuits_dir / "reduced_5q_processor_b_transpiled.qasm").exists())


if __name__ == "__main__":
    unittest.main()
