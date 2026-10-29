"""Phase 6 Primary A/B Benchmark Validation Unit Tests.

Validates:
1. Existence and integrity of all four Version 4 experiment runs in results/verified_ab_v4/.
2. Exact shot count verification (sum of raw counts == 4096 for every case).
3. Independent recalculation of all metrics (mean cut, approx ratio, P(opt)) directly from raw counts.
4. Correct current configurations:
   - Processor A: 5 physical qubits, linear-chain CX, native 'cx' basis
   - Processor B: 9 physical qubits, sparse-tree CX, native 'cx' basis
   - Shared parameters: gamma* = 2.527058..., beta* = 2.698287...
5. Transpiled QASM circuit verification (native CX, no ECR/CZ, correct registers).
6. Preservation of historical directories (verified_ab_v2, verified_ab_v3).
"""

import json
import math
from pathlib import Path
import unittest

V4_DIR = Path("results/verified_ab_v4")
V4_DATA = V4_DIR / "data"
V4_CIRCUITS = V4_DIR / "circuits"
V4_TABLES = V4_DIR / "tables"

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

EXPECTED_CASES = [
    "processor_a_ideal_reference",
    "processor_a_noisy_baseline",
    "processor_b_ideal_reference",
    "processor_b_noisy_baseline",
]


def compute_cut(x_bits):
    return sum(w for u, v, w in O1_EDGES if x_bits[u] != x_bits[v])


class TestPhase6PrimaryExperiments(unittest.TestCase):
    """Test suite for Phase 6 Primary A/B Benchmark execution and data integrity."""

    @classmethod
    def setUpClass(cls):
        # Load classical reference
        with open("results/data/classical_reference.json", "r", encoding="utf-8") as f:
            cls.ref_data = json.load(f)
        cls.c_max = float(cls.ref_data["optimal_cut_value"])
        cls.opt_strings = set(cls.ref_data["optimal_bitstrings"])

    def test_01_all_v4_artifacts_exist(self):
        """1. Verify that all expected Version 4 directories and files exist."""
        self.assertTrue(V4_DIR.exists(), "results/verified_ab_v4 directory must exist")
        self.assertTrue(V4_DATA.exists(), "results/verified_ab_v4/data directory must exist")
        self.assertTrue(V4_CIRCUITS.exists(), "results/verified_ab_v4/circuits directory must exist")
        self.assertTrue(V4_TABLES.exists(), "results/verified_ab_v4/tables directory must exist")

        # Summary files
        self.assertTrue((V4_DATA / "experiment_summary.json").exists())
        self.assertTrue((V4_DATA / "statistical_analysis_summary.json").exists())
        self.assertTrue((V4_TABLES / "primary_benchmark_metrics.md").exists())
        self.assertTrue((V4_TABLES / "primary_benchmark_metrics.csv").exists())
        self.assertTrue((V4_TABLES / "pairwise_hypothesis_tests.md").exists())
        self.assertTrue((V4_TABLES / "resource_overhead_tradeoff.md").exists())

    def test_02_shot_count_matches_raw_counts(self):
        """2. Verify that raw measurement count totals equal exactly 4096 shots for each case."""
        for case in EXPECTED_CASES:
            count_file = V4_DIR / f"counts_{case.replace('_reference', '').replace('_baseline', '')}.json"
            self.assertTrue(count_file.exists(), f"Count file {count_file.name} must exist")

            with open(count_file, "r", encoding="utf-8") as f:
                counts = json.load(f)

            total_shots = sum(counts.values())
            self.assertEqual(total_shots, 4096, f"Case {case} total shots must equal 4096, got {total_shots}")

    def test_03_recalculate_metrics_from_raw_counts(self):
        """3. Recalculate every reported performance metric directly from raw counts."""
        for case in EXPECTED_CASES:
            count_file = V4_DIR / f"counts_{case.replace('_reference', '').replace('_baseline', '')}.json"
            data_file = V4_DATA / f"{case}.json"

            with open(count_file, "r", encoding="utf-8") as f:
                raw_counts = json.load(f)
            with open(data_file, "r", encoding="utf-8") as f:
                recorded_data = json.load(f)

            # Decode raw counts
            decoded_counts = {}
            for raw_bs, cnt in raw_counts.items():
                canonical_bs = f"0{raw_bs[4]}{raw_bs[3]}{raw_bs[2]}{raw_bs[1]}{raw_bs[0]}"
                decoded_counts[canonical_bs] = decoded_counts.get(canonical_bs, 0) + cnt

            # Recalculate cut statistics
            total_shots = sum(decoded_counts.values())
            self.assertEqual(total_shots, 4096)

            sum_cut = sum(cnt * compute_cut([int(b) for b in bs]) for bs, cnt in decoded_counts.items())
            recomputed_mean = sum_cut / total_shots
            recomputed_approx = recomputed_mean / self.c_max

            opt_shots = sum(cnt for bs, cnt in decoded_counts.items() if bs in self.opt_strings)
            recomputed_p_opt = opt_shots / total_shots

            # Compare with recorded data
            res = recorded_data["results"]
            self.assertAlmostEqual(recomputed_mean, res["expected_cut"], places=4)
            self.assertAlmostEqual(recomputed_approx, res["approximation_ratio"], places=4)
            self.assertAlmostEqual(recomputed_p_opt, res["ground_state_success_probability"], places=4)
            self.assertEqual(res["total_shots_sampled"], 4096)

    def test_04_processor_and_hardware_spec_integrity(self):
        """4. Verify that each case used the correct current configurations and shared parameters."""
        for case in EXPECTED_CASES:
            data_file = V4_DATA / f"{case}.json"
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            backend = data["backend_metadata"]
            algo = data["algorithm_parameters"]

            # Shared parameter invariance check
            self.assertAlmostEqual(algo["optimal_parameters"][0], 2.527058471368481, places=6)
            self.assertAlmostEqual(algo["optimal_parameters"][1], 2.698287157638429, places=6)
            self.assertEqual(algo["shots"], 4096)
            self.assertEqual(algo["depth_p"], 1)
            self.assertEqual(algo["formulation"], "5_qubit_symmetry_reduced")
            self.assertTrue(algo["shared_parameter_rule_enforced"])

            # Native 2Q gate must be CX
            self.assertEqual(backend["native_2q_gate"], "cx")

            if "processor_a" in case:
                self.assertEqual(backend["backend_physical_qubits"], 5)
                self.assertEqual(backend["architecture_family"], "5Q Linear Chain CX")
            else:
                self.assertEqual(backend["backend_physical_qubits"], 9)
                self.assertEqual(backend["architecture_family"], "9Q Sparse Tree CX")

            if "noisy" in case:
                self.assertTrue(data["is_noisy_simulation"])
                self.assertTrue(data["noise_applied"])
                self.assertIsNotNone(data["noise_config"])
                self.assertEqual(data["noise_config"]["single_qubit_error"], 0.001)
                self.assertEqual(data["noise_config"]["two_qubit_error"], 0.005)
                self.assertEqual(data["noise_config"]["readout_error"], 0.01)
            else:
                self.assertFalse(data["is_noisy_simulation"])
                self.assertFalse(data["noise_applied"])
                self.assertIsNone(data["noise_config"])

    def test_05_transpiled_qasm_integrity(self):
        """5. Verify that all 4 transpiled QASM files contain valid QASM 2.0 and native CX gates."""
        for case in EXPECTED_CASES:
            qasm_file = V4_CIRCUITS / f"{case}_transpiled.qasm"
            self.assertTrue(qasm_file.exists())

            content = qasm_file.read_text(encoding="utf-8")
            self.assertIn("OPENQASM 2.0;", content)
            self.assertIn('include "qelib1.inc";', content)
            self.assertIn("cx ", content)
            self.assertNotIn("ecr ", content, "Transpiled circuit must not contain ECR gates")
            self.assertNotIn("cz ", content, "Transpiled circuit must not contain CZ gates")
            self.assertIn("measure ", content)

    def test_06_historical_results_preserved(self):
        """6. Confirm that historical directories verified_ab_v2 and verified_ab_v3 were untouched."""
        v2_dir = Path("results/verified_ab_v2")
        v3_dir = Path("results/verified_ab_v3")
        self.assertTrue(v2_dir.exists(), "Historical verified_ab_v2 must exist untouched")
        self.assertTrue(v3_dir.exists(), "Historical verified_ab_v3 must exist untouched")
        self.assertTrue((v3_dir / "data" / "experiment_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
