"""Statistical verification and finite-shot uncertainty analysis tests."""

import json
import math
from pathlib import Path
import unittest

import numpy as np
from scipy import stats

from maxcut.experiments.schema import load_experiment_result


class TestStatisticalAnalysis(unittest.TestCase):
    """Test suite verifying statistical consistency, uncertainty intervals, and A/B hypothesis testing."""

    @classmethod
    def setUpClass(cls):
        cls.sim_path = Path("results/data/O1_MaxCut_Challenge_aer_simulator_p1_result.json")
        cls.proc_a_path = Path("results/data/O1_MaxCut_Challenge_processor_a_p1_result.json")
        cls.proc_b_path = Path("results/data/O1_MaxCut_Challenge_processor_b_p1_result.json")

        cls.res_sim = load_experiment_result(cls.sim_path)
        cls.res_a = load_experiment_result(cls.proc_a_path)
        cls.res_b = load_experiment_result(cls.proc_b_path)

        # 10 certified optimal bitstrings for the O1 challenge instance (5 complement pairs with C(x)=6.0)
        cls.optimal_bitstrings = {
            "001101", "011001", "010011", "101100", "110011",
            "101101", "001100", "010010", "110010", "100110",
        }

    def _extract_samples(self, res):
        """Reconstructs cut samples from decoded counts."""
        cuts = []
        opt_count = 0
        total_shots = sum(res.decoded_counts.values())

        for bs, cnt in res.decoded_counts.items():
            val = res.cut_values_by_bitstring[bs]
            cuts.extend([val] * cnt)
            if bs in self.optimal_bitstrings:
                opt_count += cnt

        cuts = np.array(cuts)
        return cuts, opt_count, total_shots

    def test_recomputed_metrics_match_stored(self):
        """Verify that sample mean cut and cumulative optimal probability match stored results exactly."""
        for res in [self.res_sim, self.res_a, self.res_b]:
            cuts, opt_count, total_shots = self._extract_samples(res)
            recomputed_mean = float(np.mean(cuts))
            recomputed_p_opt = opt_count / total_shots

            self.assertEqual(total_shots, 2048)
            self.assertAlmostEqual(recomputed_mean, res.expected_cut, places=6)
            self.assertAlmostEqual(
                recomputed_p_opt,
                res.probability_metrics["ground_state_success_probability"],
                places=6,
            )

    def test_analytic_expectation_reconciliation(self):
        """Verify that analytic expectation (4.8671) falls strictly within the empirical 95% CI of all targets."""
        for res in [self.res_sim, self.res_a, self.res_b]:
            cuts, _, total_shots = self._extract_samples(res)
            mean_c = float(np.mean(cuts))
            std_c = float(np.std(cuts, ddof=1))
            se_c = std_c / math.sqrt(total_shots)

            ci_low = mean_c - 1.959964 * se_c
            ci_high = mean_c + 1.959964 * se_c

            analytic_c = res.algorithm_parameters["final_expected_cut_from_opt"]
            self.assertAlmostEqual(analytic_c, 4.867127, places=4)

            # Analytic value must fall inside the 95% confidence interval
            self.assertLessEqual(ci_low, analytic_c)
            self.assertGreaterEqual(ci_high, analytic_c)

    def test_optimal_solution_probability_definition(self):
        """Verify that optimal-solution probability represents the complete subspace of 10 degenerate optimal bitstrings."""
        for res in [self.res_sim, self.res_a, self.res_b]:
            # Sum counts exclusively across the 10 optimal bitstrings
            subspace_hits = sum(res.decoded_counts.get(bs, 0) for bs in self.optimal_bitstrings)
            p_subspace = subspace_hits / 2048

            self.assertAlmostEqual(
                p_subspace,
                res.probability_metrics["ground_state_success_probability"],
                places=6,
            )

            # Contrast with single bitstring probability: any single bitstring has p < 0.06
            for bs in self.optimal_bitstrings:
                single_p = res.decoded_counts.get(bs, 0) / 2048
                self.assertLess(single_p, 0.06)

    def test_processor_ab_hypothesis_testing(self):
        """Verify that A/B differences in cut value and optimal probability are not statistically significant at alpha=0.05."""
        cuts_a, opt_a, n_a = self._extract_samples(self.res_a)
        cuts_b, opt_b, n_b = self._extract_samples(self.res_b)

        # 1. Welch's t-test on expected cut
        t_stat, p_val_c = stats.ttest_ind(cuts_b, cuts_a, equal_var=False)
        self.assertGreater(p_val_c, 0.05, "Difference in mean cut must NOT be statistically significant at alpha=0.05")

        # 2. Two-proportion z-test on optimal probability
        p_a = opt_a / n_a
        p_b = opt_b / n_b
        se_diff_p = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        z_stat = (p_b - p_a) / se_diff_p
        p_val_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        self.assertGreater(p_val_p, 0.05, "Difference in success probability must NOT be statistically significant at alpha=0.05")

    def test_wilson_confidence_interval_bounds(self):
        """Verify that Wilson score intervals for success probability are mathematically valid and within [0, 1]."""
        for res in [self.res_sim, self.res_a, self.res_b]:
            _, opt_hits, total_shots = self._extract_samples(res)
            p_hat = opt_hits / total_shots
            z = 1.959964
            denom = 1 + z**2 / total_shots
            center = (p_hat + z**2 / (2 * total_shots)) / denom
            half = (z * math.sqrt(p_hat * (1 - p_hat) / total_shots + z**2 / (4 * total_shots**2))) / denom
            low, high = center - half, center + half

            self.assertGreater(low, 0.35)
            self.assertLess(high, 0.45)
            self.assertLess(low, p_hat)
            self.assertGreater(high, p_hat)


if __name__ == "__main__":
    unittest.main()
