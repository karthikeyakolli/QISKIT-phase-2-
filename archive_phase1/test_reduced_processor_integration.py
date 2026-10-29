"""Regression tests for Phase 3 — Official Processor A/B Integration with 5-qubit reduced QAOA.

Verifies:
1. Target configurations match their verified source specs.
2. 5-qubit symmetry-reduced logical QAOA compiles successfully on both processors.
3. 100% connectivity compliance (zero coupling map violations).
4. 100% native basis compliance (zero unsupported gates).
5. Physical qubit mapping: 5 physical qubits on A ([0, 1, 2, 3, 5]) and valid 5-qubit subset on B ([0, 3, 4, 5, 6]).
6. Layout extraction and gate breakdown.
7. Two-qubit gate decomposition: total 2Q = basis synthesis + routing overhead.
8. Same logical objective used for both targets.
9. Exported circuit artifacts and serialized metrics integrity.
"""

import json
from pathlib import Path
import unittest

import qiskit
import qiskit.qasm2

from maxcut.domain.model import MaxCutGraph
from maxcut.io import load_challenge_instance
from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
from processors.processor_a.adapter import ProcessorABackend
from processors.processor_b.adapter import ProcessorBBackend


class TestReducedProcessorIntegration(unittest.TestCase):
    """Test suite for 5-qubit reduced QAOA compilation onto Processor A and Processor B."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.instance_path = cls.repo_root / "instances" / "challenge" / "o1_maxcut_instance.json"
        cls.instance = load_challenge_instance(cls.instance_path)
        cls.graph = MaxCutGraph.from_challenge_instance(cls.instance)

        cls.proc_a = ProcessorABackend(seed_transpiler=42)
        cls.proc_b = ProcessorBBackend(seed_transpiler=42)

        # Build 5-qubit reduced logical circuit
        cls.ansatz = build_reduced_qaoa_circuit(cls.graph, p=1, fixed_node=0, include_measurements=True)
        cls.bound_qc = cls.ansatz.bind(gamma_values=[2.4835], beta_values=[4.2540])

    def test_01_target_configurations_match_verified_source(self):
        """Verify Processor A and B match their authoritative hardware definitions."""
        # Processor A (5Q-Linear-Chain)
        self.assertEqual(self.proc_a.backend_name, "Processor_A")
        self.assertEqual(self.proc_a.num_qubits, 5)
        self.assertEqual(self.proc_a.spec.basis_gates, ["rz", "sx", "x", "cx"])
        self.assertEqual(len(self.proc_a.spec.coupling_map), 8)
        meta_a = self.proc_a.get_metadata()
        self.assertEqual(meta_a["native_2q_gate"], "cx")
        self.assertEqual(meta_a["architecture_family"], "5Q-Linear-Chain")

        # Processor B (9Q-Sparse-Tree)
        self.assertEqual(self.proc_b.backend_name, "Processor_B")
        self.assertEqual(self.proc_b.num_qubits, 9)
        self.assertEqual(self.proc_b.spec.basis_gates, ["rz", "sx", "x", "cx"])
        self.assertEqual(len(self.proc_b.spec.coupling_map), 16)
        meta_b = self.proc_b.get_metadata()
        self.assertEqual(meta_b["native_2q_gate"], "cx")
        self.assertEqual(meta_b["architecture_family"], "9Q-Sparse-Tree")

    def test_02_logical_circuit_architecture_and_objective_invariance(self):
        """Verify the logical 5-qubit reduced circuit is identical for both compilation targets."""
        self.assertEqual(self.bound_qc.num_qubits, 5)
        ops = self.bound_qc.count_ops()
        # 6 RZZ interactions from edges not incident to fixed vertex 0
        self.assertEqual(ops.get("rzz", 0), 6)
        # 2 RZ interactions from edges incident to fixed vertex 0 ((0,1) and (0,2))
        self.assertEqual(ops.get("rz", 0), 2)
        # 5 RX mixer rotations
        self.assertEqual(ops.get("rx", 0), 5)
        # 5 Hadamard initializations
        self.assertEqual(ops.get("h", 0), 5)
        # 5 Measurements
        self.assertEqual(ops.get("measure", 0), 5)
        self.assertEqual(self.bound_qc.depth(), 9)

    def test_03_compilation_success_both_processors(self):
        """Verify that the 5-qubit circuit compiles successfully on Processor A and Processor B."""
        qc_a, metrics_a = self.proc_a.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_a["is_compilation_successful"])
        self.assertGreater(metrics_a["transpiled_depth"], 0)
        self.assertGreater(metrics_a["transpiled_total_gates"], 0)

        qc_b, metrics_b = self.proc_b.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_b["is_compilation_successful"])
        self.assertGreater(metrics_b["transpiled_depth"], 0)
        self.assertGreater(metrics_b["transpiled_total_gates"], 0)

    def test_04_connectivity_compliance_zero_violations(self):
        """Verify that 100% of 2-qubit gates strictly adhere to the target coupling map."""
        _, metrics_a = self.proc_a.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_a["is_connectivity_compliant"])
        self.assertEqual(len(metrics_a["coupling_violations"]), 0)

        _, metrics_b = self.proc_b.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_b["is_connectivity_compliant"])
        self.assertEqual(len(metrics_b["coupling_violations"]), 0)

    def test_05_native_basis_compliance_zero_violations(self):
        """Verify that final circuits contain only native basis gates (plus measure/barrier)."""
        _, metrics_a = self.proc_a.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_a["is_basis_compliant"])
        self.assertEqual(len(metrics_a["non_basis_gates"]), 0)
        self.assertIn("cx", metrics_a["gate_breakdown"])

        _, metrics_b = self.proc_b.transpile_circuit(self.bound_qc, optimization_level=1)
        self.assertTrue(metrics_b["is_basis_compliant"])
        self.assertEqual(len(metrics_b["non_basis_gates"]), 0)
        self.assertIn("cx", metrics_b["gate_breakdown"])

    def test_06_physical_qubit_mapping_and_subsets(self):
        """Verify that logical circuit maps onto five physical qubits on A and a valid five-qubit subset of B."""
        _, metrics_a = self.proc_a.transpile_circuit(self.bound_qc, optimization_level=1)
        used_a = metrics_a["physical_qubits_used"]
        self.assertEqual(len(used_a), 5)
        self.assertEqual(metrics_a["physical_qubits_active_count"], 5)
        self.assertEqual(used_a, [0, 1, 2, 3, 4])
        # Verify initial layout maps 5 logical qubits to physical qubits
        layout_a = metrics_a["initial_layout"]
        self.assertEqual(len(layout_a), 5)
        self.assertTrue(all(p in used_a for p in layout_a.values()))

        _, metrics_b = self.proc_b.transpile_circuit(self.bound_qc, optimization_level=1)
        used_b = metrics_b["physical_qubits_used"]
        self.assertEqual(len(used_b), 5)
        self.assertEqual(metrics_b["physical_qubits_active_count"], 5)
        self.assertTrue(all(p in range(9) for p in used_b))
        # Verify initial layout maps 5 logical qubits to physical qubits
        layout_b = metrics_b["initial_layout"]
        self.assertEqual(len(layout_b), 5)
        self.assertTrue(all(p in used_b for p in layout_b.values()))

    def test_07_metrics_derived_from_artifacts_and_decomposition(self):
        """Verify that every metric is mathematically consistent with circuit decomposition."""
        qc_a, metrics_a = self.proc_a.transpile_circuit(self.bound_qc, optimization_level=1)
        # 6 logical RZZ x 2 CX = 12 basis synthesis 2Q gates
        self.assertEqual(metrics_a["basis_synthesis_2q_gates"], 12)
        # 3 routing SWAPs x 3 CX/SWAP = 9 routing 2Q gates
        self.assertEqual(metrics_a["routing_swap_count"], 3)
        self.assertEqual(metrics_a["routing_2q_gates"], 9)
        # Total native 2Q gates = 12 + 9 = 21
        self.assertEqual(metrics_a["transpiled_2q_gates"], 21)
        self.assertEqual(metrics_a["cx_gate_count"], 21)

        qc_b, metrics_b = self.proc_b.transpile_circuit(self.bound_qc, optimization_level=1)
        # 6 logical RZZ x 2 CX = 12 basis synthesis 2Q gates
        self.assertEqual(metrics_b["basis_synthesis_2q_gates"], 12)
        # 2 routing SWAPs x 3 CX/SWAP = 6 routing 2Q gates
        self.assertEqual(metrics_b["routing_swap_count"], 2)
        self.assertEqual(metrics_b["routing_2q_gates"], 6)
        # Total native 2Q gates = 12 + 6 = 18
        self.assertEqual(metrics_b["transpiled_2q_gates"], 18)
        self.assertEqual(metrics_b["cx_gate_count"], 18)

    def test_08_exported_artifacts_and_schema_integrity(self):
        """Verify that exported circuit files and serialized metrics match circuit definitions."""
        circuits_dir = self.repo_root / "results" / "circuits"
        data_dir = self.repo_root / "results" / "data"

        # Check circuit files exist
        logical_qasm = circuits_dir / "reduced_5q_logical_qaoa_p1.qasm"
        proc_a_qasm = circuits_dir / "reduced_5q_processor_a_opt1_transpiled.qasm"
        proc_b_qasm = circuits_dir / "reduced_5q_processor_b_opt1_transpiled.qasm"
        proc_a_canon = circuits_dir / "reduced_5q_processor_a_transpiled.qasm"
        proc_b_canon = circuits_dir / "reduced_5q_processor_b_transpiled.qasm"

        self.assertTrue(logical_qasm.exists())
        self.assertTrue(proc_a_qasm.exists())
        self.assertTrue(proc_b_qasm.exists())
        self.assertTrue(proc_a_canon.exists())
        self.assertTrue(proc_b_canon.exists())

        # Check metrics files exist and have required schema
        metrics_a_path = data_dir / "reduced_5q_processor_a_circuit_metrics.json"
        metrics_b_path = data_dir / "reduced_5q_processor_b_circuit_metrics.json"

        self.assertTrue(metrics_a_path.exists())
        self.assertTrue(metrics_b_path.exists())

        with open(metrics_a_path, "r", encoding="utf-8") as f:
            data_a = json.load(f)
        self.assertEqual(data_a["backend_name"], "Processor_A")
        self.assertEqual(data_a["native_2q_gate"], "ecr")
        self.assertEqual(data_a["physical_qubits_active_count"], 5)
        self.assertEqual(data_a["transpiled_2q_gates"], 18)
        self.assertTrue(data_a["is_connectivity_compliant"])

        with open(metrics_b_path, "r", encoding="utf-8") as f:
            data_b = json.load(f)
        self.assertEqual(data_b["backend_name"], "Processor_B")
        self.assertEqual(data_b["native_2q_gate"], "cz")
        self.assertEqual(data_b["physical_qubits_active_count"], 5)
        self.assertEqual(data_b["transpiled_2q_gates"], 18)
        self.assertTrue(data_b["is_connectivity_compliant"])


if __name__ == "__main__":
    unittest.main()
