"""Unit and integration tests for Processor A/B circuit resource analysis layer."""

import json
from pathlib import Path
import tempfile
import unittest

from qiskit.circuit import QuantumCircuit

from maxcut.domain.model import MaxCutGraph
from maxcut.io import load_challenge_instance
from maxcut.metrics.circuit_metrics import (
    CircuitResourceMetrics,
    extract_circuit_metrics,
)
from maxcut.metrics.comparator import ProcessorCircuitComparator
from maxcut.qaoa.ansatz import build_qaoa_circuit
from processors.processor_a import ProcessorABackend
from processors.processor_b import ProcessorBBackend


class TestCircuitMetricsAnalysis(unittest.TestCase):
    """Test suite for circuit resource extraction and Processor A/B comparator."""

    def setUp(self):
        # 4-node cycle graph (0-1-2-3-0)
        self.c4_graph = MaxCutGraph(
            name="C4_Test",
            nodes=[0, 1, 2, 3],
            edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        )
        self.logical_qc = build_qaoa_circuit(self.c4_graph, p=1, include_measurements=True).bind([0.5], [1.0])

        self.proc_a = ProcessorABackend(seed_transpiler=42)
        self.proc_b = ProcessorBBackend(seed_transpiler=42)

    def test_circuit_resource_metrics_extraction(self):
        """Verify extraction of all logical, transpiled, gate, and routing overhead metrics."""
        transpiled_qc = self.proc_a.run_circuit(self.logical_qc, shots=100, optimization_level=1).transpiled_circuit
        metrics = extract_circuit_metrics(
            logical_circuit=self.logical_qc,
            transpiled_circuit=transpiled_qc,
            backend_runner=self.proc_a,
            optimization_level=1,
            seed_transpiler=42,
        )

        self.assertIsInstance(metrics, CircuitResourceMetrics)
        self.assertEqual(metrics.backend_name, "Processor_A")
        self.assertEqual(metrics.logical_qubit_count, 4)
        self.assertEqual(metrics.physical_qubit_count, 5)
        self.assertGreater(metrics.logical_circuit_depth, 0)
        self.assertGreater(metrics.transpiled_circuit_depth, 0)
        self.assertGreater(metrics.transpiled_2q_gates, 0)
        self.assertGreaterEqual(metrics.swap_count, 0)
        self.assertGreaterEqual(metrics.two_qubit_gate_overhead, 0)
        self.assertGreaterEqual(metrics.routing_overhead_ratio, 1.0)
        self.assertGreaterEqual(metrics.depth_expansion_ratio, 1.0)
        self.assertIn("cx", metrics.gate_breakdown)
        self.assertEqual(metrics.native_2q_gate, "cx")

    def test_comparator_symmetric_compilation(self):
        """Verify that comparator evaluates Processor A and B under strictly symmetric settings."""
        comparator = ProcessorCircuitComparator(
            backend_a=self.proc_a,
            backend_b=self.proc_b,
            seed_transpiler=42,
        )

        results = comparator.compare(self.logical_qc, optimization_levels=[1, 2])
        self.assertEqual(len(results), 4)  # 2 backends * 2 levels

        # Check optimization level pairing
        levels = [r.optimization_level for r in results]
        self.assertEqual(levels, [1, 1, 2, 2])
        backend_names = [r.backend_name for r in results]
        self.assertEqual(backend_names, ["Processor_A", "Processor_B", "Processor_A", "Processor_B"])

        # Check that both used seed 42
        for r in results:
            self.assertEqual(r.seed_transpiler, 42)

    def test_machine_readable_exports(self):
        """Verify export to DataFrame, JSON, CSV, and Markdown table formats."""
        comparator = ProcessorCircuitComparator(
            backend_a=self.proc_a,
            backend_b=self.proc_b,
            seed_transpiler=42,
        )
        comparator.compare(self.logical_qc, optimization_levels=[1])

        # 1. DataFrame export
        df = comparator.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn("Transpiled Depth", df.columns)
        self.assertIn("Routing Ratio", df.columns)
        self.assertIn("SWAP Count", df.columns)

        # 2. Markdown export
        md_table = comparator.to_markdown_table()
        self.assertIn("| Backend |", md_table)
        self.assertIn("Processor_A", md_table)
        self.assertIn("Processor_B", md_table)

        # 3. JSON and CSV file export
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f_json:
            json_path = f_json.name
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f_csv:
            csv_path = f_csv.name

        try:
            comparator.to_json(json_path)
            comparator.to_csv(csv_path)

            with open(json_path, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            self.assertEqual(len(loaded_json), 2)
            self.assertEqual(loaded_json[0]["backend_name"], "Processor_A")
            self.assertEqual(loaded_json[1]["backend_name"], "Processor_B")

            with open(csv_path, "r", encoding="utf-8") as f:
                csv_content = f.read()
            self.assertIn("Processor_A", csv_content)
            self.assertIn("Processor_B", csv_content)
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(csv_path).unlink(missing_ok=True)

    def test_circuit_requiring_connectivity_aware_routing(self):
        """Regression test AUD-03: Verify routing metrics separation on a circuit requiring routing.

        Uses the official challenge graph with 5-qubit symmetry reduction on Processor A and Processor B.
        Tests that:
        - explicit_swap_count == 0 (all SWAPs decomposed to target basis)
        - routing_swap_count == 2
        - routing_2q_gates == 6 (2 SWAPs * 3 native 2Q gates)
        - basis_synthesis_2q_gates == 12 (6 logical RZZ gates * 2 native 2Q gates)
        - transpiled_2q_gates == 18 (12 basis synthesis + 6 routing)
        - routing_overhead_ratio == 18 / 12 == 1.5
        """
        from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
        challenge_graph = MaxCutGraph.from_challenge_instance(load_challenge_instance())
        logical_qc = build_reduced_qaoa_circuit(challenge_graph, p=1, fixed_node=0, include_measurements=True).bind([0.5], [1.0])

        for backend in [self.proc_a, self.proc_b]:
            result = backend.run_circuit(logical_qc, shots=100, optimization_level=1)
            metrics = extract_circuit_metrics(
                logical_circuit=logical_qc,
                transpiled_circuit=result.transpiled_circuit,
                backend_runner=backend,
                optimization_level=1,
                seed_transpiler=42,
            )

            expected_swaps = 3 if backend.backend_name == "Processor_A" else 2
            expected_routing_2q = expected_swaps * 3
            expected_total_2q = 12 + expected_routing_2q

            # Verification of 5 separate metrics:
            self.assertEqual(metrics.explicit_swap_count, 0)
            self.assertEqual(metrics.routing_swap_count, expected_swaps)
            self.assertEqual(metrics.routing_2q_gates, expected_routing_2q)
            self.assertEqual(metrics.basis_synthesis_2q_gates, 12)
            self.assertEqual(metrics.transpiled_2q_gates, expected_total_2q)
            self.assertEqual(
                metrics.transpiled_2q_gates,
                metrics.basis_synthesis_2q_gates + metrics.routing_2q_gates,
            )
            self.assertEqual(metrics.two_qubit_gate_overhead, expected_routing_2q)
            self.assertAlmostEqual(metrics.routing_overhead_ratio, expected_total_2q / 12, places=4)

            # Verification of adapter result fields
            self.assertEqual(result.explicit_swap_count, 0)
            self.assertEqual(result.routing_swap_count, expected_swaps)
            self.assertEqual(result.routing_2q_gates, expected_routing_2q)
            self.assertEqual(result.basis_synthesis_2q_gates, 12)
            self.assertEqual(result.transpiled_2q_gates, expected_total_2q)

    def test_circuit_not_requiring_routing(self):
        """Regression test AUD-03: Verify routing metrics on a circuit requiring NO routing.

        Uses a 2-node graph with a direct hardware coupling edge (0, 1).
        Tests that:
        - explicit_swap_count == 0
        - routing_swap_count == 0
        - routing_2q_gates == 0
        - basis_synthesis_2q_gates == 2 (1 RZZ * 2 native 2Q gates)
        - transpiled_2q_gates == 2
        - two_qubit_gate_overhead == 0
        - routing_overhead_ratio == 1.0
        """
        direct_graph = MaxCutGraph(
            name="Direct_Edge_2Node",
            nodes=[0, 1],
            edges=[(0, 1, 1.0)],
        )
        logical_qc = build_qaoa_circuit(direct_graph, p=1, include_measurements=True).bind([0.5], [1.0])

        for backend in [self.proc_a, self.proc_b]:
            result = backend.run_circuit(logical_qc, shots=100, optimization_level=1)
            metrics = extract_circuit_metrics(
                logical_circuit=logical_qc,
                transpiled_circuit=result.transpiled_circuit,
                backend_runner=backend,
                optimization_level=1,
                seed_transpiler=42,
            )

            # Verification of zero routing overhead:
            self.assertEqual(metrics.explicit_swap_count, 0)
            self.assertEqual(metrics.routing_swap_count, 0)
            self.assertEqual(metrics.routing_2q_gates, 0)
            self.assertEqual(metrics.basis_synthesis_2q_gates, 2)
            self.assertEqual(metrics.transpiled_2q_gates, 2)
            self.assertEqual(metrics.two_qubit_gate_overhead, 0)
            self.assertEqual(metrics.routing_overhead_ratio, 1.0)

            # Verification of adapter result fields
            self.assertEqual(result.explicit_swap_count, 0)
            self.assertEqual(result.routing_swap_count, 0)
            self.assertEqual(result.routing_2q_gates, 0)
            self.assertEqual(result.basis_synthesis_2q_gates, 2)
            self.assertEqual(result.transpiled_2q_gates, 2)

    def test_reconciliation_of_reported_totals_with_component_counts(self):
        """Regression test AUD-04: Ensure reported totals reconcile with component counts.

        Verifies that for all backends (Processor A, Processor B) across optimization levels (1, 2, 3):
        1. Final native 2Q gates = post-basis-synthesis 2Q gates + routing overhead (2Q gates).
        2. Post-basis-synthesis 2Q gates = logical 2Q interactions * 2 (for RZZ decomposition).
        3. Routing overhead = routing SWAP-equivalent count * 3 (for SWAP decomposition).
        4. Explicit final-circuit SWAP count = 0 (all SWAPs decomposed to basis).
        5. Routing overhead ratio = final native 2Q gates / post-basis-synthesis 2Q gates.
        6. Transpiled total gates = 1Q gates + 2Q gates + non-unitary gates.
        """
        from maxcut.qaoa.ansatz import build_reduced_qaoa_circuit
        challenge_graph = MaxCutGraph.from_challenge_instance(load_challenge_instance())
        logical_qc = build_reduced_qaoa_circuit(challenge_graph, p=1, fixed_node=0, include_measurements=True).bind([0.5], [1.0])

        comparator = ProcessorCircuitComparator(
            backend_a=self.proc_a,
            backend_b=self.proc_b,
            seed_transpiler=42,
        )
        results = comparator.compare(logical_qc, optimization_levels=[1, 2, 3])
        self.assertEqual(len(results), 6)

        for r in results:
            # 1. Exact reconciliation of two-qubit total
            self.assertEqual(
                r.transpiled_2q_gates,
                r.basis_synthesis_2q_gates + r.routing_2q_gates,
                f"Mismatch in total 2Q gates for {r.backend_name} level {r.optimization_level}",
            )
            # 2. Reconcile post-basis-synthesis 2Q gates with logical interactions
            self.assertEqual(r.logical_2q_gates, 6)
            self.assertEqual(r.basis_synthesis_2q_gates, 12)
            self.assertEqual(
                r.basis_synthesis_2q_gates,
                r.logical_2q_gates * 2,
            )
            # 3. Reconcile routing overhead with routing SWAP count
            expected_swaps = 3 if r.backend_name == "Processor_A" else 2
            expected_total_2q = 12 + expected_swaps * 3
            self.assertEqual(r.routing_swap_count, expected_swaps)
            self.assertEqual(r.routing_2q_gates, expected_swaps * 3)
            self.assertEqual(
                r.routing_2q_gates,
                r.routing_swap_count * 3,
            )
            # 4. Final circuit contains 0 explicit SWAPs
            self.assertEqual(r.explicit_swap_count, 0)
            # 5. Final native 2Q count matches expected
            self.assertEqual(r.transpiled_2q_gates, expected_total_2q)
            # 6. Reconcile routing overhead ratio with correct denominator
            expected_ratio = r.transpiled_2q_gates / r.basis_synthesis_2q_gates
            self.assertAlmostEqual(r.routing_overhead_ratio, expected_ratio, places=4)
            self.assertAlmostEqual(r.routing_overhead_ratio, expected_total_2q / 12, places=4)
            # 7. Total gate count reconciliation
            gb = r.gate_breakdown
            non_unitary = gb.get("measure", 0) + gb.get("barrier", 0)
            self.assertEqual(
                r.transpiled_total_gates,
                r.transpiled_1q_gates + r.transpiled_2q_gates,
            )
            self.assertEqual(
                sum(gb.values()),
                r.transpiled_total_gates + non_unitary,
            )


if __name__ == "__main__":
    unittest.main()
