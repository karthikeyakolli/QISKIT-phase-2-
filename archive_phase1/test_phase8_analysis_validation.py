"""Phase 8 Metrics, Statistics, Figures, and Master Analysis Validation Unit Tests.

Validates:
1. Master Phase 8 metrics table reconciliation with raw counts.
2. Wilson score confidence interval calculations and bounds.
3. Two-proportion z-tests and Welch t-tests.
4. All 8 required publication vector SVG figures existence and valid XML syntax.
5. Complete architecture resource reconciliation.
6. Historical record segregation (v2, v3, v4).
"""

import csv
import json
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

V4_DIR = Path("results/verified_ab_v4")
TABLES_DIR = V4_DIR / "tables"
FIGURES_DIR = V4_DIR / "figures"
DATA_DIR = V4_DIR / "data"

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
C_MAX = 6.0000


def compute_cut(x_bits):
    return sum(w for u, v, w in O1_EDGES if x_bits[u] != x_bits[v])


class TestPhase8AnalysisValidation(unittest.TestCase):
    """Test suite certifying Phase 8 metrics reconciliation, statistics, and figures."""

    @classmethod
    def setUpClass(cls):
        with open("results/data/classical_reference.json", "r", encoding="utf-8") as f:
            cls.ref_data = json.load(f)
        cls.opt_strings = set(cls.ref_data["optimal_bitstrings"])

        with open(TABLES_DIR / "master_phase8_metrics.json", "r", encoding="utf-8") as f:
            cls.master_metrics = json.load(f)

    def test_01_master_metrics_reconciliation_from_raw_counts(self):
        """1. Verify that all 6 experiment cases in master metrics reconcile with raw counts."""
        for case_id, row in self.master_metrics.items():
            count_suffix = "ideal" if "ideal" in case_id else "noisy"
            proc_prefix = "processor_a" if "A_" in case_id else ("processor_b" if "B_" in case_id else "processor_d")
            count_path = V4_DIR / f"counts_{proc_prefix}_{count_suffix}.json"

            self.assertTrue(count_path.exists(), f"Raw count file {count_path.name} must exist")
            with open(count_path, "r", encoding="utf-8") as f:
                raw_counts = json.load(f)

            decoded = {}
            for raw_bs, cnt in raw_counts.items():
                canon_bs = f"0{raw_bs[4]}{raw_bs[3]}{raw_bs[2]}{raw_bs[1]}{raw_bs[0]}"
                decoded[canon_bs] = decoded.get(canon_bs, 0) + cnt

            total_shots = sum(decoded.values())
            self.assertEqual(total_shots, 4096)
            self.assertEqual(row["total_shots"], 4096)

            all_cuts = []
            opt_cnt = 0
            for bs, cnt in decoded.items():
                c = compute_cut([int(b) for b in bs])
                all_cuts.extend([c] * cnt)
                if bs in self.opt_strings:
                    opt_cnt += cnt

            recomp_mean = sum(all_cuts) / 4096
            recomp_approx = recomp_mean / C_MAX
            recomp_popt = opt_cnt / 4096

            self.assertAlmostEqual(row["mean_cut"], recomp_mean, places=4)
            self.assertAlmostEqual(row["approx_ratio"], recomp_approx, places=4)
            self.assertAlmostEqual(row["p_opt"], recomp_popt, places=4)

    def test_02_wilson_confidence_intervals(self):
        """2. Verify Wilson score confidence intervals are mathematically valid."""
        for case_id, row in self.master_metrics.items():
            p = row["p_opt"]
            lower = row["wilson_lower_95"]
            upper = row["wilson_upper_95"]

            self.assertLessEqual(lower, p)
            self.assertGreaterEqual(upper, p)
            self.assertGreaterEqual(lower, 0.0)
            self.assertLessEqual(upper, 1.0)
            # Wilson interval for 4096 shots should have width ~ 0.03
            width = upper - lower
            self.assertTrue(0.025 <= width <= 0.035, f"Wilson width {width} unexpected")

    def test_03_statistical_hypothesis_tests(self):
        """3. Verify statistical hypothesis testing table exists and contains valid tests."""
        hyp_csv = TABLES_DIR / "statistical_hypothesis_tests_complete.csv"
        self.assertTrue(hyp_csv.exists())

        with open(hyp_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        self.assertEqual(len(reader), 6)
        comparisons = [r["comparison"] for r in reader]
        self.assertIn("Processor A: Ideal vs Noisy", comparisons)
        self.assertIn("Processor B: Ideal vs Noisy", comparisons)
        self.assertIn("Processor D: Ideal vs Noisy", comparisons)
        self.assertIn("Processor D vs Processor A (Noisy)", comparisons)
        self.assertIn("Processor D vs Processor B (Noisy)", comparisons)
        self.assertIn("Processor B vs Processor A (Noisy)", comparisons)

        for r in reader:
            p_val = float(r["p_val_cut"])
            self.assertTrue(0.0 <= p_val <= 1.0)
            if "Ideal vs Noisy" in r["comparison"]:
                self.assertEqual(r["significant_cut_p05"], "YES")
                self.assertLess(p_val, 1e-4)

    def test_04_all_8_svg_figures_exist_and_valid_xml(self):
        """4. Verify that all 8 required publication vector SVG figures exist and parse as valid XML."""
        expected_figures = [
            "figure1_maxcut_problem_graph.svg",
            "figure2_processor_a_coupling_map.svg",
            "figure3_processor_b_coupling_map.svg",
            "figure4_processor_d_coupling_map.svg",
            "figure5_logical_vs_transpiled_circuits.svg",
            "figure6_ideal_vs_noisy_distributions.svg",
            "figure7_architecture_resource_tradeoffs.svg",
            "figure8_combined_abd_performance_comparison.svg",
        ]

        for fig_name in expected_figures:
            fig_path = FIGURES_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} must exist")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure {fig_name} must be non-trivial")

            # Parse XML
            content = fig_path.read_text(encoding="utf-8")
            try:
                root = ET.fromstring(content)
                self.assertTrue(root.tag.endswith("svg"), f"Root tag must be svg, got {root.tag}")
            except ET.ParseError as e:
                self.fail(f"SVG figure {fig_name} failed XML parsing: {e}")

    def test_05_architecture_resource_breakdown(self):
        """5. Verify architecture resource breakdown CSV reconciles with physical specifications."""
        res_csv = TABLES_DIR / "architecture_resource_breakdown.csv"
        self.assertTrue(res_csv.exists())

        with open(res_csv, "r", encoding="utf-8") as f:
            reader = {r["processor"]: r for r in csv.DictReader(f)}

        # Processor A: 5Q, depth 58, 18 CX, 2 SWAPs
        self.assertEqual(int(reader["Processor A"]["backend_qubits"]), 5)
        self.assertEqual(int(reader["Processor A"]["transpiled_depth"]), 58)
        self.assertEqual(int(reader["Processor A"]["native_2q_cx"]), 18)
        self.assertEqual(int(reader["Processor A"]["routing_swaps"]), 2)

        # Processor B: 9Q, depth 52, 15 CX, 1 SWAP
        self.assertEqual(int(reader["Processor B"]["backend_qubits"]), 9)
        self.assertEqual(int(reader["Processor B"]["transpiled_depth"]), 52)
        self.assertEqual(int(reader["Processor B"]["native_2q_cx"]), 15)
        self.assertEqual(int(reader["Processor B"]["routing_swaps"]), 1)

        # Processor D: 8Q, depth 36, 12 CX, 0 SWAPs
        self.assertEqual(int(reader["Processor D"]["backend_qubits"]), 8)
        self.assertEqual(int(reader["Processor D"]["transpiled_depth"]), 36)
        self.assertEqual(int(reader["Processor D"]["native_2q_cx"]), 12)
        self.assertEqual(int(reader["Processor D"]["routing_swaps"]), 0)

    def test_06_historical_comparison_segregation(self):
        """6. Verify historical comparison documentation segregates v2, v3, and v4."""
        hist_md = TABLES_DIR / "historical_comparison_v2_v3_v4.md"
        self.assertTrue(hist_md.exists())
        content = hist_md.read_text(encoding="utf-8")
        self.assertIn("verified_ab_v2", content)
        self.assertIn("verified_ab_v3", content)
        self.assertIn("verified_ab_v4", content)
        self.assertIn("ECR", content)
        self.assertIn("CX", content)


if __name__ == "__main__":
    unittest.main()
