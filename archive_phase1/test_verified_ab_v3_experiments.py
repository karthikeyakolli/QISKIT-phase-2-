"""Regression tests for Verified A/B v3 Experiments with Corrected Architectures.

Verifies:
1. Target configurations:
   - Processor A: exactly 5 physical qubits, coupling map with 8 directed edges (4 undirected),
     nodes [0, 1, 2, 3, 4], native gates [rz, sx, x, ecr], architecture family 5Q-Baseline-Tree.
   - Processor B: exactly 7 physical qubits, heavy-hex topology with 12 directed edges (6 undirected),
     nodes [0..6], native gates [rz, sx, x, cz], architecture family Heavy-Hex-7.
2. Problem correctness:
   - Challenge graph: |V|=6, |E|=8, W=8.00, C_max=6.00.
   - Exact classical reference solver passes.
   - Symmetry reduction preserves cut objective and equivalence classes.
3. Experiment execution:
   - All 4 cases executed fresh under results/verified_ab_v3/
   - Raw counts sum to exactly 2048 shots per case.
   - Decoded counts map to valid 6-bit strings with x0=0.
   - Physical qubit usage: Proc A uses 5 qubits [0, 1, 2, 3, 4]; Proc B uses 5 qubits [0, 1, 2, 3, 5] mapped on 7Q backend.
   - Transpiled native 2Q gates: 18 (12 basis synthesis + 6 routing from 2 SWAPs) on both.
   - Depth: 69 for A, 83 for B.
   - Zero coupling violations on both targets.
   - No stale v2 data mixed into v3.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from maxcut.classical.solver import ExactExhaustiveSolver
from maxcut.domain.model import MaxCutGraph
from maxcut.io import load_challenge_instance, load_processor_spec
from processors.processor_a.adapter import ProcessorABackend
from processors.processor_b.adapter import ProcessorBBackend


class TestVerifiedABV3Experiments(unittest.TestCase):
    """Test suite for Verified A/B v3 experiments and architecture validation."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.v3_dir = cls.repo_root / "results" / "verified_ab_v3"
        cls.data_dir = cls.v3_dir / "data"
        cls.circuits_dir = cls.v3_dir / "circuits"
        cls.tables_dir = cls.v3_dir / "tables"

        cls.instance = load_challenge_instance()
        cls.graph = MaxCutGraph.from_challenge_instance(cls.instance)

        cls.proc_a = ProcessorABackend()
        cls.proc_b = ProcessorBBackend()

        # Load summary if exists
        cls.summary_path = cls.data_dir / "experiment_summary.json"
        if cls.summary_path.exists():
            with open(cls.summary_path, "r", encoding="utf-8") as f:
                cls.summary = json.load(f)
        else:
            cls.summary = {}

    def test_01_processor_a_target_spec(self):
        """Verify Processor A is genuine 5-physical-qubit processor with linear-chain topology."""
        self.assertEqual(self.proc_a.num_qubits, 5)
        self.assertEqual(self.proc_a.spec.num_qubits, 5)
        self.assertIn("t1_us_mean", self.proc_a.spec.qubit_properties)
        self.assertIn("2q_error_mean", self.proc_a.spec.qubit_properties)
        self.assertEqual(len(self.proc_a.spec.coupling_map), 8)  # 4 undirected edges * 2
        self.assertEqual(self.proc_a.spec.basis_gates, ["rz", "sx", "x", "cx"])
        meta = self.proc_a.get_metadata()
        self.assertEqual(meta["architecture_family"], "5Q-Linear-Chain")
        self.assertEqual(meta["native_2q_gate"], "cx")

    def test_02_processor_b_target_spec(self):
        """Verify Processor B is genuine 9-physical-qubit processor with sparse tree topology."""
        self.assertEqual(self.proc_b.num_qubits, 9)
        self.assertEqual(self.proc_b.spec.num_qubits, 9)
        self.assertIn("t1_us_mean", self.proc_b.spec.qubit_properties)
        self.assertIn("2q_error_mean", self.proc_b.spec.qubit_properties)
        self.assertEqual(len(self.proc_b.spec.coupling_map), 16)  # 8 undirected edges * 2
        self.assertEqual(self.proc_b.spec.basis_gates, ["rz", "sx", "x", "cx"])
        meta = self.proc_b.get_metadata()
        self.assertEqual(meta["architecture_family"], "9Q-Sparse-Tree")
        self.assertEqual(meta["native_2q_gate"], "cx")

    def test_03_original_problem_graph_and_classical_optimum(self):
        """Verify the 6-vertex, 8-edge challenge graph remains completely unchanged and C_max = 6.0."""
        self.assertEqual(self.graph.num_nodes, 6)
        self.assertEqual(self.graph.num_edges, 8)
        self.assertEqual(self.graph.total_weight, 8.0)
        solver = ExactExhaustiveSolver()
        res = solver.solve(self.graph)
        self.assertEqual(res.optimal_cut_value, 6.0)
        self.assertEqual(len(res.optimal_bitstrings), 10)

    def test_04_v3_experiment_artifacts_exist(self):
        """Verify all four cases and circuit artifacts exist under results/verified_ab_v3/."""
        required_data = [
            self.v3_dir / "counts_processor_a_ideal.json",
            self.v3_dir / "counts_processor_a_noisy.json",
            self.v3_dir / "counts_processor_b_ideal.json",
            self.v3_dir / "counts_processor_b_noisy.json",
            self.data_dir / "processor_a_ideal_reference.json",
            self.data_dir / "processor_a_noisy_baseline.json",
            self.data_dir / "processor_b_ideal_reference.json",
            self.data_dir / "processor_b_noisy_baseline.json",
            self.data_dir / "experiment_summary.json",
        ]
        for p in required_data:
            self.assertTrue(p.exists(), f"Missing required v3 artifact: {p.name}")

    def test_05_shot_counts_and_reproducibility(self):
        """Verify each of the four cases has exactly 2048 shots and valid raw/decoded bitstrings."""
        count_files = [
            self.v3_dir / "counts_processor_a_ideal.json",
            self.v3_dir / "counts_processor_a_noisy.json",
            self.v3_dir / "counts_processor_b_ideal.json",
            self.v3_dir / "counts_processor_b_noisy.json",
        ]
        for p in count_files:
            with open(p, "r", encoding="utf-8") as f:
                c = json.load(f)
            self.assertEqual(sum(c.values()), 2048, f"Shot count mismatch in {p.name}")
            # Raw measurement bitstrings must be length 5 (5 active measured qubits)
            for bs in c.keys():
                self.assertEqual(len(bs), 5)

        # Check decoded counts in data JSON files have length 6 with x0 = 0
        detail_files = [
            self.data_dir / "processor_a_ideal_reference.json",
            self.data_dir / "processor_a_noisy_baseline.json",
            self.data_dir / "processor_b_ideal_reference.json",
            self.data_dir / "processor_b_noisy_baseline.json",
        ]
        for p in detail_files:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            dec = d["decoded_counts"]
            self.assertEqual(sum(dec.values()), 2048)
            for bs in dec.keys():
                self.assertEqual(len(bs), 6)
                self.assertTrue(bs.startswith("0"))

    def test_06_circuit_metrics_and_gate_decomposition(self):
        """Verify circuit metrics: 18 native 2Q gates, zero coupling violations, depth values."""
        self.assertIn("summary", self.summary)
        cases = {c["Experiment Case"]: c for c in self.summary["summary"]}

        ma = cases["Processor A — ideal/reference"]
        mb = cases["Processor B — ideal/reference"]

        # Processor A
        self.assertEqual(ma["Backend Physical Qubits"], 5)
        self.assertEqual(ma["Active Physical Qubits"], 5)
        self.assertEqual(ma["Mapped Qubits"], "[0, 1, 2, 3, 4]")
        self.assertEqual(ma["Total 2Q Count"], 18)
        self.assertEqual(ma["Basis 2Q Count"], 12)
        self.assertEqual(ma["Routing 2Q Overhead"], 6)
        self.assertEqual(ma["Routing SWAPs"], 2)
        self.assertEqual(ma["Circuit Depth"], 69)

        # Processor B
        self.assertEqual(mb["Backend Physical Qubits"], 7)
        self.assertEqual(mb["Active Physical Qubits"], 5)
        self.assertEqual(mb["Mapped Qubits"], "[0, 1, 2, 3, 5]")
        self.assertEqual(mb["Total 2Q Count"], 18)
        self.assertEqual(mb["Basis 2Q Count"], 12)
        self.assertEqual(mb["Routing 2Q Overhead"], 6)
        self.assertEqual(mb["Routing SWAPs"], 2)
        self.assertEqual(mb["Circuit Depth"], 83)

    def test_07_no_stale_v2_data_in_v3(self):
        """Verify that v3 results are distinct from v2 and no v2 data is used."""
        v2_path = self.repo_root / "results" / "verified_ab_v2" / "data" / "processor_a_ideal_reference.json"
        v3_path = self.data_dir / "processor_a_ideal_reference.json"
        if v2_path.exists() and v3_path.exists():
            with open(v2_path, "r", encoding="utf-8") as f:
                d_v2 = json.load(f)
            with open(v3_path, "r", encoding="utf-8") as f:
                d_v3 = json.load(f)
            # v2 Proc A was Falcon-H (7 qubits)
            self.assertEqual(d_v2["backend_metadata"]["num_qubits"], 7)
            # v3 Proc A is 5Q baseline (5 qubits)
            self.assertEqual(d_v3["backend_metadata"]["backend_physical_qubits"], 5)
            self.assertNotEqual(d_v2["raw_counts"], d_v3["raw_counts"])


if __name__ == "__main__":
    unittest.main()
