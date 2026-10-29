"""Unit tests for the exact exhaustive classical reference solver."""

import json
from pathlib import Path
import tempfile
import unittest

from maxcut.classical import ClassicalMaxCutResult, ExactExhaustiveSolver
from maxcut.domain import MaxCutGraph


class TestClassicalReferenceSolver(unittest.TestCase):
    """Test suite for exhaustive classical search, timing, validation, and serialization."""

    def setUp(self):
        self.solver = ExactExhaustiveSolver(record_distribution=True)

    def test_synthetic_unweighted_triangle(self):
        """Verify exact ground truth for a 3-node triangle graph (K_3)."""
        # Triangle graph: 3 nodes, 3 edges (weight 1.0 each)
        # Optimal cut is 2.0 (any partition separating 1 node from the other 2)
        # Total assignments: 2^3 = 8.
        # Monochromatic: 000 (cut 0), 111 (cut 0) -> 2 assignments
        # Remaining: 6 assignments each have cut 2.0
        graph = MaxCutGraph.from_edges([(0, 1), (1, 2), (2, 0)], name="Synthetic_K3")

        result = self.solver.solve(graph)

        self.assertIsInstance(result, ClassicalMaxCutResult)
        self.assertEqual(result.graph_name, "Synthetic_K3")
        self.assertEqual(result.num_nodes, 3)
        self.assertEqual(result.num_edges, 3)
        self.assertEqual(result.total_weight, 3.0)
        self.assertEqual(result.optimal_cut_value, 2.0)
        self.assertEqual(result.total_assignments_evaluated, 8)
        self.assertEqual(len(result.optimal_bitstrings), 6)
        self.assertEqual(result.num_cut_edges, 2)
        self.assertAlmostEqual(result.cut_ratio, 2.0 / 3.0)

        # Validate solution self-consistency
        self.assertTrue(result.validate_solution(graph))
        self.assertGreater(result.execution_time_seconds, 0.0)

    def test_synthetic_complete_bipartite_graph(self):
        """Verify ground truth on bipartite graph K_2,2 (all edges cut)."""
        # Bipartite with parts {0, 1} and {2, 3}
        edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
        graph = MaxCutGraph.from_edges(edges, name="Synthetic_K2_2")

        result = self.solver.solve(graph)

        # Max-Cut must cut all 4 edges
        self.assertEqual(result.optimal_cut_value, 4.0)
        self.assertEqual(result.cut_ratio, 1.0)
        self.assertEqual(result.num_cut_edges, 4)

        # Bipartite cut partitions: {0,1} vs {2,3} -> bitstrings "1100" and "0011"
        self.assertIn("1100", result.optimal_bitstrings)
        self.assertIn("0011", result.optimal_bitstrings)

        self.assertTrue(result.validate_solution(graph))

    def test_synthetic_weighted_diamond_graph(self):
        """Verify exact solution on a weighted 4-node graph."""
        edges = [
            (0, 1, 3.0),
            (0, 2, 2.0),
            (1, 2, 1.0),
            (1, 3, 4.0),
            (2, 3, 5.0),
        ]
        graph = MaxCutGraph.from_edges(edges, name="Synthetic_Weighted_Diamond")

        result = self.solver.solve(graph)

        self.assertTrue(graph.is_weighted)
        self.assertEqual(result.total_weight, 15.0)

        # Check cut consistency: optimal cut value must be valid
        self.assertTrue(result.validate_solution(graph))
        self.assertGreater(result.optimal_cut_value, 0.0)
        self.assertLessEqual(result.optimal_cut_value, 15.0)

        # Distribution should sum to 2^4 = 16
        total_in_dist = sum(result.cut_value_distribution.values())
        self.assertEqual(total_in_dist, 16)

    def test_complementary_solutions_preservation(self):
        """Verify that complementary solutions (bitwise inversions) are preserved."""
        edges = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (3, 0, 4.0)]
        graph = MaxCutGraph.from_edges(edges, name="Synthetic_Cycle_4")

        result = self.solver.solve(graph)

        # For every optimal bitstring, its bitwise inverted partner must also be optimal
        for bs in result.optimal_bitstrings:
            inverted = "".join("1" if b == "0" else "0" for b in bs)
            self.assertIn(
                inverted,
                result.optimal_bitstrings,
                f"Complementary bitstring '{inverted}' missing for '{bs}'",
            )

        summary = result.summary()
        self.assertIn("Z2 spin-flip symmetry", summary)

    def test_result_serialization_and_validation(self):
        """Verify JSON export and re-import of classical reference results."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = MaxCutGraph.from_edges(edges, name="SerializationTest")

        result = self.solver.solve(graph)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            json_str = result.to_json(temp_path)
            self.assertTrue(Path(temp_path).exists())

            with open(temp_path, "r", encoding="utf-8") as f:
                loaded_dict = json.load(f)

            self.assertEqual(loaded_dict["graph_name"], "SerializationTest")
            self.assertEqual(loaded_dict["optimal_cut_value"], 4.0)
            self.assertEqual(loaded_dict["num_nodes"], 4)
            self.assertEqual(loaded_dict["num_edges"], 4)
            self.assertIn("cut_edges", loaded_dict)
            self.assertIn("cut_value_distribution", loaded_dict)
            self.assertGreater(loaded_dict["execution_time_seconds"], 0.0)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_single_edge_exact_solver(self):
        """Verify exact classical solver on a single weighted edge."""
        graph = MaxCutGraph.from_edges([(0, 1, 7.5)], name="SingleEdgeClassical")
        result = self.solver.solve(graph)

        self.assertEqual(result.optimal_cut_value, 7.5)
        self.assertEqual(result.total_assignments_evaluated, 4)
        self.assertEqual(len(result.optimal_bitstrings), 2)
        self.assertIn("01", result.optimal_bitstrings)
        self.assertIn("10", result.optimal_bitstrings)
        self.assertTrue(result.validate_solution(graph))

    def test_disconnected_graph_exact_solver(self):
        """Verify exact classical solver on disconnected components."""
        graph = MaxCutGraph(nodes=[0, 1, 2, 3], edges=[(0, 1, 3.0), (2, 3, 5.0)], name="DisconnectedClassical")
        result = self.solver.solve(graph)

        self.assertEqual(result.optimal_cut_value, 8.0)
        self.assertEqual(result.total_assignments_evaluated, 16)
        self.assertTrue(result.validate_solution(graph))
        # 4 optimal bitstrings: (01 or 10 on first pair) x (01 or 10 on second pair)
        self.assertEqual(len(result.optimal_bitstrings), 4)
        for bs in ["0101", "0110", "1001", "1010"]:
            self.assertIn(bs, result.optimal_bitstrings)


if __name__ == "__main__":
    unittest.main()
