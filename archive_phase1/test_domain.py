"""Comprehensive unit tests for the Max-Cut domain model."""

from pathlib import Path
import sys
import tempfile
import unittest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from maxcut.domain import (
    CutResult,
    Edge,
    GraphValidationError,
    InvalidAssignmentError,
    MaxCutGraph,
    Partition,
)
from maxcut.visualization import plot_maxcut_graph


class TestMaxCutDomain(unittest.TestCase):
    """Test suite for graph modeling, cut calculation, partitions, and validation."""

    def test_simple_unweighted_graph(self):
        """Verify unweighted 4-cycle graph creation and cut calculation."""
        # 4-cycle C_4: (0-1-2-3-0)
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Cycle_4")

        self.assertEqual(graph.num_nodes, 4)
        self.assertEqual(graph.num_edges, 4)
        self.assertFalse(graph.is_weighted)
        self.assertEqual(graph.total_weight, 4.0)

        # Alternating partition [1, 0, 1, 0] cuts all 4 edges
        result = graph.compute_cut("1010")
        self.assertEqual(result.cut_value, 4.0)
        self.assertEqual(result.num_cut_edges, 4)
        self.assertEqual(result.cut_ratio, 1.0)
        self.assertEqual(len(result.uncut_edges), 0)

        # Monochromatic partition [0, 0, 0, 0] cuts 0 edges
        zero_res = graph.compute_cut([0, 0, 0, 0])
        self.assertEqual(zero_res.cut_value, 0.0)
        self.assertEqual(zero_res.num_cut_edges, 0)
        self.assertEqual(len(zero_res.uncut_edges), 4)

    def test_weighted_graph(self):
        """Verify weighted triangle graph and cut evaluations."""
        # K_3 triangle with distinct edge weights
        edges = [(0, 1, 2.5), (1, 2, 1.5), (0, 2, 4.0)]
        graph = MaxCutGraph.from_edges(edges, name="Weighted_Triangle")

        self.assertTrue(graph.is_weighted)
        self.assertEqual(graph.total_weight, 8.0)

        # Partition separating node 0 from {1, 2}: assignment [1, 0, 0]
        # Cuts edges (0, 1) [weight 2.5] and (0, 2) [weight 4.0] = 6.5
        result = graph.compute_cut("100")
        self.assertEqual(result.cut_value, 6.5)
        self.assertEqual(result.num_cut_edges, 2)
        self.assertEqual(result.uncut_edges[0].endpoints(), (1, 2))

        # Partition separating node 1 from {0, 2}: assignment [0, 1, 0]
        # Cuts edges (0, 1) [2.5] and (1, 2) [1.5] = 4.0
        result_node1 = graph.compute_cut([0, 1, 0])
        self.assertEqual(result_node1.cut_value, 4.0)

    def test_known_partitions_and_summary(self):
        """Verify known path graph P_4 solutions and summary string generation."""
        # Path graph 0-1-2-3
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        graph = MaxCutGraph.from_edges(edges, name="Path_4")

        # Optimal cut is alternating [0, 1, 0, 1]
        res = graph.compute_cut("0101")
        self.assertEqual(res.cut_value, 3.0)
        self.assertEqual(res.cut_ratio, 1.0)

        summary = res.summary()
        self.assertIn("Cut Value: 3.0000", summary)
        self.assertIn("Cut Ratio: 100.00%", summary)
        self.assertIn("Partition S (size 2): [1, 3]", summary)
        self.assertIn("Partition ~S (size 2): [0, 2]", summary)

    def test_zero_and_one_based_indexing(self):
        """Verify seamless handling of 1-based vertex labels."""
        # Nodes 1, 2, 3, 4 with 1-based cycle
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        nodes = [1, 2, 3, 4]
        graph = MaxCutGraph(nodes=nodes, edges=edges, name="OneBasedCycle")

        self.assertEqual(graph.nodes, [1, 2, 3, 4])
        self.assertEqual(graph.canonical_nodes, [0, 1, 2, 3])

        # Dictionary assignment directly mapping 1-based nodes
        dict_assignment = {1: 1, 2: 0, 3: 1, 4: 0}
        res_dict = graph.compute_cut(dict_assignment)
        self.assertEqual(res_dict.cut_value, 4.0)
        self.assertEqual(res_dict.partition.set_s, frozenset([1, 3]))
        self.assertEqual(res_dict.partition.set_not_s, frozenset([2, 4]))

        # Bitstring assignment ordered by sorted nodes: "1010" -> node 1:1, node 2:0, node 3:1, node 4:0
        res_str = graph.compute_cut("1010")
        self.assertEqual(res_str.cut_value, 4.0)
        self.assertEqual(res_str.partition.set_s, frozenset([1, 3]))

    def test_complementary_partitions(self):
        """Verify that bitwise complementary assignments produce identical cuts."""
        edges = [(0, 1, 3.0), (1, 2, 2.0), (2, 3, 4.0), (3, 0, 1.0), (0, 2, 5.0)]
        graph = MaxCutGraph.from_edges(edges, name="Pentagon_Diagonal")

        assignment_a = "1010"
        assignment_b = "0101"

        res_a = graph.compute_cut(assignment_a)
        res_b = graph.compute_cut(assignment_b)

        # Cut values must be identical
        self.assertEqual(res_a.cut_value, res_b.cut_value)
        self.assertEqual(res_a.num_cut_edges, res_b.num_cut_edges)

        # Partitions must be complementary
        self.assertTrue(res_a.partition.is_complementary(res_b.partition))
        self.assertEqual(res_a.partition.set_s, res_b.partition.set_not_s)
        self.assertEqual(res_a.partition.set_not_s, res_b.partition.set_s)

    def test_invalid_assignments(self):
        """Verify strict error raising for malformed or incomplete assignments."""
        edges = [(0, 1), (1, 2), (2, 0)]
        graph = MaxCutGraph.from_edges(edges, name="Triangle")

        # Wrong length (2 instead of 3)
        with self.assertRaises(InvalidAssignmentError):
            graph.compute_cut("10")

        # Missing vertex in dictionary
        with self.assertRaises(InvalidAssignmentError):
            graph.compute_cut({0: 1, 1: 0})

        # Non-binary value in bitstring
        with self.assertRaises(InvalidAssignmentError):
            graph.compute_cut("102")

        # Non-binary value in dictionary
        with self.assertRaises(InvalidAssignmentError):
            graph.compute_cut({0: 1, 1: 0, 2: 5})

        # Unsupported type
        with self.assertRaises(InvalidAssignmentError):
            graph.compute_cut(123)  # type: ignore

    def test_graph_structural_validation(self):
        """Verify detection of self-loops, duplicate edges, and disconnected references."""
        # Self loop on node 1
        with self.assertRaises(GraphValidationError):
            MaxCutGraph(nodes=[0, 1], edges=[(0, 1), (1, 1)])

        # Too few nodes (< 2)
        with self.assertRaises(GraphValidationError):
            MaxCutGraph(nodes=[0], edges=[])

        # Edge referencing non-existent node
        with self.assertRaises(GraphValidationError):
            MaxCutGraph(nodes=[0, 1], edges=[(0, 2)])

        # Duplicate edge
        with self.assertRaises(GraphValidationError):
            MaxCutGraph(nodes=[0, 1], edges=[(0, 1), (1, 0)])

        # Empty edges
        with self.assertRaises(GraphValidationError):
            MaxCutGraph(nodes=[0, 1], edges=[])

    def test_visualization_helper(self):
        """Verify that plot_maxcut_graph generates and exports a valid figure."""
        edges = [(0, 1, 1.5), (1, 2, 2.0), (2, 3, 1.0), (3, 0, 3.0)]
        graph = MaxCutGraph.from_edges(edges, name="VisTestGraph")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name

        try:
            fig = plot_maxcut_graph(
                graph=graph,
                partition="1010",
                title="Test Cut Visualization",
                save_path=temp_path,
            )
            self.assertIsNotNone(fig)
            self.assertTrue(Path(temp_path).exists())
            self.assertGreater(Path(temp_path).stat().st_size, 0)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_single_edge_domain(self):
        """Verify single-edge graph with 2 nodes evaluates correctly."""
        graph = MaxCutGraph.from_edges([(0, 1, 2.5)], name="SingleEdgeDomain")
        self.assertEqual(graph.num_nodes, 2)
        self.assertEqual(graph.num_edges, 1)
        self.assertEqual(graph.total_weight, 2.5)

        self.assertEqual(graph.compute_cut("00").cut_value, 0.0)
        self.assertEqual(graph.compute_cut("01").cut_value, 2.5)
        self.assertEqual(graph.compute_cut("10").cut_value, 2.5)
        self.assertEqual(graph.compute_cut("11").cut_value, 0.0)

    def test_disconnected_components_domain(self):
        """Verify domain behavior on disconnected components and isolated nodes."""
        # Graph with 2 disconnected edges: (0, 1) w=1.0 and (2, 3) w=2.0
        graph = MaxCutGraph(nodes=[0, 1, 2, 3], edges=[(0, 1, 1.0), (2, 3, 2.0)], name="DisconnectedDomain")
        self.assertEqual(graph.total_weight, 3.0)

        # Both edges cut: 0101 -> cut = 3.0
        self.assertEqual(graph.compute_cut("0101").cut_value, 3.0)
        # Only first edge cut: 0100 -> cut = 1.0
        self.assertEqual(graph.compute_cut("0100").cut_value, 1.0)
        # Only second edge cut: 0001 -> cut = 2.0
        self.assertEqual(graph.compute_cut("0001").cut_value, 2.0)
        # Neither edge cut: 0000 -> cut = 0.0
        self.assertEqual(graph.compute_cut("0000").cut_value, 0.0)


if __name__ == "__main__":
    unittest.main()
