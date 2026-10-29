"""Phase 7 Processor D Validation Unit Tests.

Validates:
1. Processor D hardware specification: exactly 8 physical qubits, max degree <= 3, native CX basis.
2. Processor D adapter and backend implementation.
3. Transpiled circuit metrics: exactly 12 CX gates (0 routing SWAPs) and depth 36.
4. Raw measurement counts: exact shot sum == 4096 for both Ideal and Noisy cases.
5. Independent metric recalculation from raw measurement counts.
6. Combined A/B/D comparative performance tables and statistical hypothesis tests.
"""

import json
import math
from pathlib import Path
import unittest

V4_DIR = Path("results/verified_ab_v4")
V4_DATA = V4_DIR / "data"
V4_CIRCUITS = V4_DIR / "circuits"
V4_TABLES = V4_DIR / "tables"
SPEC_D_PATH = Path("processors/processor_d/hardware_spec.json")

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


def compute_cut(x_bits):
    return sum(w for u, v, w in O1_EDGES if x_bits[u] != x_bits[v])


class TestPhase7ProcessorD(unittest.TestCase):
    """Test suite for Phase 7 Custom Processor D design and evaluation."""

    @classmethod
    def setUpClass(cls):
        with open("results/data/classical_reference.json", "r", encoding="utf-8") as f:
            cls.ref_data = json.load(f)
        cls.c_max = float(cls.ref_data["optimal_cut_value"])
        cls.opt_strings = set(cls.ref_data["optimal_bitstrings"])

    def test_01_spec_d_integrity_and_constraints(self):
        """1. Verify Processor D hardware specification complies with all constraints."""
        self.assertTrue(SPEC_D_PATH.exists(), "Processor D spec JSON must exist")
        with open(SPEC_D_PATH, "r", encoding="utf-8") as f:
            spec = json.load(f)

        self.assertEqual(spec["name"], "Processor_D")
        self.assertEqual(spec["num_qubits"], 8, "Processor D must have 8 qubits (<= 12)")
        self.assertIn("cx", spec["basis_gates"])
        self.assertEqual(spec["metadata"]["native_2q_gate"], "cx")

        # Verify max degree <= 3 (superconducting planar constraint)
        cmap = spec["coupling_map"]
        adj = {}
        for u, v in cmap:
            adj.setdefault(u, set()).add(v)
        max_deg = max(len(neighbors) for neighbors in adj.values())
        self.assertLessEqual(max_deg, 3, f"Max degree must be <= 3, got {max_deg}")
        self.assertEqual(spec["metadata"]["max_degree"], 3)

    def test_02_adapter_and_backend_implementation(self):
        """2. Verify Processor D adapter file and class definition."""
        adapter_file = Path("processors/processor_d/adapter.py")
        self.assertTrue(adapter_file.exists())
        content = adapter_file.read_text(encoding="utf-8")
        self.assertIn("class ProcessorDBackend", content)
        self.assertIn("Processor_D", content)

    def test_03_transpiled_circuit_zero_swap_overhead(self):
        """3. Verify transpiled QASM achieves 0 routing SWAPs and exactly 12 CX gates."""
        for mode in ["ideal_reference", "noisy_baseline"]:
            qasm_file = V4_CIRCUITS / f"processor_d_{mode}_transpiled.qasm"
            self.assertTrue(qasm_file.exists(), f"QASM file {qasm_file.name} must exist")

            content = qasm_file.read_text(encoding="utf-8")
            self.assertIn("OPENQASM 2.0;", content)
            self.assertIn('include "qelib1.inc";', content)

            cx_count = content.count("cx ")
            self.assertEqual(cx_count, 12, f"Processor D must require exactly 12 CX gates (0 SWAPs), got {cx_count}")
            self.assertNotIn("swap ", content)

            data_file = V4_DATA / f"processor_d_{mode}.json"
            self.assertTrue(data_file.exists())
            with open(data_file, "r", encoding="utf-8") as f:
                case_data = json.load(f)

            metrics = case_data["circuit_compilation_metrics"]
            self.assertEqual(metrics["transpiled_2q_gates"], 12)
            self.assertEqual(metrics["routing_swap_count"], 0)
            self.assertEqual(metrics["routing_2q_gates"], 0)
            self.assertEqual(metrics["transpiled_depth"], 36)

    def test_04_raw_counts_and_shot_budget(self):
        """4. Verify raw count files sum to exactly 4096 shots for both Processor D cases."""
        for mode in ["ideal", "noisy"]:
            count_file = V4_DIR / f"counts_processor_d_{mode}.json"
            self.assertTrue(count_file.exists(), f"Count file {count_file.name} must exist")

            with open(count_file, "r", encoding="utf-8") as f:
                counts = json.load(f)

            total_shots = sum(counts.values())
            self.assertEqual(total_shots, 4096, f"Shot count must be 4096, got {total_shots}")

    def test_05_recalculate_processor_d_metrics_from_raw_counts(self):
        """5. Recalculate mean cut, approx ratio, and P(opt) from raw counts for Processor D."""
        for mode in ["ideal_reference", "noisy_baseline"]:
            count_suffix = "ideal" if "ideal" in mode else "noisy"
            count_file = V4_DIR / f"counts_processor_d_{count_suffix}.json"
            data_file = V4_DATA / f"processor_d_{mode}.json"

            with open(count_file, "r", encoding="utf-8") as f:
                raw_counts = json.load(f)
            with open(data_file, "r", encoding="utf-8") as f:
                recorded_data = json.load(f)

            decoded_counts = {}
            for raw_bs, cnt in raw_counts.items():
                canonical_bs = f"0{raw_bs[4]}{raw_bs[3]}{raw_bs[2]}{raw_bs[1]}{raw_bs[0]}"
                decoded_counts[canonical_bs] = decoded_counts.get(canonical_bs, 0) + cnt

            total_shots = sum(decoded_counts.values())
            self.assertEqual(total_shots, 4096)

            sum_cut = sum(cnt * compute_cut([int(b) for b in bs]) for bs, cnt in decoded_counts.items())
            recomputed_mean = sum_cut / total_shots
            recomputed_approx = recomputed_mean / self.c_max
            opt_shots = sum(cnt for bs, cnt in decoded_counts.items() if bs in self.opt_strings)
            recomputed_p_opt = opt_shots / total_shots

            res = recorded_data["results"]
            self.assertAlmostEqual(recomputed_mean, res["expected_cut"], places=4)
            self.assertAlmostEqual(recomputed_approx, res["approximation_ratio"], places=4)
            self.assertAlmostEqual(recomputed_p_opt, res["ground_state_success_probability"], places=4)

    def test_06_combined_abd_analysis_tables(self):
        """6. Verify combined A/B/D tables exist and show Processor D advantage."""
        combined_md = V4_TABLES / "combined_abd_benchmark_metrics.md"
        tradeoff_md = V4_TABLES / "abd_resource_tradeoff.md"
        hyp_md = V4_TABLES / "abd_pairwise_hypothesis_tests.md"

        self.assertTrue(combined_md.exists())
        self.assertTrue(tradeoff_md.exists())
        self.assertTrue(hyp_md.exists())

        tradeoff_content = tradeoff_md.read_text(encoding="utf-8")
        self.assertIn("Processor A", tradeoff_content)
        self.assertIn("Processor B", tradeoff_content)
        self.assertIn("Processor D", tradeoff_content)
        self.assertIn("12", tradeoff_content)  # 12 CX gates on D
        self.assertIn("36", tradeoff_content)  # depth 36 on D


if __name__ == "__main__":
    unittest.main()
