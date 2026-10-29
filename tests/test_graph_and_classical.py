"""
tests/test_graph_and_classical.py - Comprehensive mathematical, graph, and classical cut tests.
"""
import pytest
from src.formulation import (
    NUM_VERTICES,
    LOCKED_EDGES,
    TOTAL_WEIGHT,
    EXACT_OPTIMUM,
    OPTIMAL_BITSTRINGS,
    cut_weight,
    enumerate_all_assignments,
    exact_classical_optimum,
    random_baseline,
    validate_graph,
)
from qff_phase2 import counts_to_x


@pytest.mark.unit
def test_official_graph_instance_properties():
    """Verify official PS-O1 graph topology, edge weights, and total weight."""
    assert NUM_VERTICES == 6
    assert len(LOCKED_EDGES) == 9
    assert TOTAL_WEIGHT == 27
    
    expected_edges = [
        (0, 2, 1),
        (0, 3, 4),
        (0, 4, 4),
        (1, 4, 3),
        (1, 5, 3),
        (2, 3, 5),
        (2, 4, 1),
        (3, 4, 4),
        (3, 5, 2),
    ]
    assert LOCKED_EDGES == expected_edges
    
    # Check vertex range and strictly positive weights
    for u, v, w in LOCKED_EDGES:
        assert 0 <= u < NUM_VERTICES
        assert 0 <= v < NUM_VERTICES
        assert u < v  # Canonical representation
        assert 1 <= w <= 5
        
    validate_graph(NUM_VERTICES, LOCKED_EDGES)


@pytest.mark.unit
def test_cut_function_endpoint_logic():
    """Verify cut function only accumulates edge weight across partition boundaries."""
    # Monolithic partitions (all 0s or all 1s) cut zero edges
    assert cut_weight("000000") == 0
    assert cut_weight("111111") == 0
    assert cut_weight((0, 0, 0, 0, 0, 0)) == 0
    assert cut_weight((1, 1, 1, 1, 1, 1)) == 0
    
    # Test single-edge cut logic on small isolated pair
    test_edge = [(0, 1, 7)]
    assert cut_weight((0, 0), edges=test_edge) == 0
    assert cut_weight((1, 1), edges=test_edge) == 0
    assert cut_weight((0, 1), edges=test_edge) == 7
    assert cut_weight((1, 0), edges=test_edge) == 7


@pytest.mark.unit
def test_brute_force_enumeration_and_exact_optimum():
    """Verify 64-assignment enumeration, exact optimum 22, and both complementary solutions."""
    scores = enumerate_all_assignments()
    assert len(scores) == 64
    
    max_c, opt_strings = exact_classical_optimum()
    assert max_c == EXACT_OPTIMUM == 22
    assert set(opt_strings) == OPTIMAL_BITSTRINGS == {"001011", "110100"}
    
    # Verify exact cut value of optimal bitstrings
    assert cut_weight("001011") == 22
    assert cut_weight("110100") == 22
    
    # Verify partition sets: for "001011", x0=0, x1=0, x2=1, x3=0, x4=1, x5=1
    # S = {0, 1, 3}, S' = {2, 4, 5}
    # Cut edges should be:
    # (0,2)=1, (0,4)=4, (1,4)=3, (1,5)=3, (2,3)=5, (3,4)=4, (3,5)=2 -> sum = 1+4+3+3+5+4+2 = 22
    # Uncut edges: (0,3)=4 (both in S), (2,4)=1 (both in S') -> sum = 5. Total = 22 + 5 = 27
    assert cut_weight("001011") + 5 == 27


@pytest.mark.unit
def test_complement_symmetry_all_assignments():
    """Verify that complementing every bit preserves the cut value exactly."""
    scores = enumerate_all_assignments()
    for bitstring, val in scores.items():
        comp = "".join("1" if b == "0" else "0" for b in bitstring)
        assert scores[comp] == val, f"Complement symmetry violated: C({bitstring}) != C({comp})"


@pytest.mark.unit
def test_cut_weight_bounds_and_optimum():
    """Verify cut weights are non-negative and never exceed the theoretical/exact maximum."""
    scores = enumerate_all_assignments()
    for bitstring, val in scores.items():
        assert 0 <= val <= TOTAL_WEIGHT, f"Cut weight out of bounds [0, 27]: {val}"
        assert val <= EXACT_OPTIMUM, f"Cut weight {val} exceeded exact classical optimum 22"


@pytest.mark.unit
def test_qiskit_bitstring_ordering_endianness():
    """Test Qiskit bitstring reversal convention to prevent endianness bugs."""
    # In Qiskit, bitstring keys from measurement are little-endian:
    # qubit 0 is the rightmost character.
    # Suppose measurement returned "110100":
    # Rightmost is q0=0, q1=0, q2=1, q3=0, q4=1, q5=1 (logical "001011")
    qiskit_counts_key = "110100"
    logical_x = counts_to_x(qiskit_counts_key)
    assert logical_x == (0, 0, 1, 0, 1, 1)
    assert cut_weight(logical_x) == 22
    
    # If one incorrectly parsed without reversing (big-endian assumption):
    # naive_x = (1, 1, 0, 1, 0, 0)
    naive_x = tuple(int(b) for b in qiskit_counts_key)
    assert naive_x == (1, 1, 0, 1, 0, 0)
    # Because "110100" is also an optimal solution (by complement symmetry),
    # test a non-symmetric string to show that reversal matters:
    asym_qiskit_key = "000001"  # q0=1, q1..q5=0 -> logical (1, 0, 0, 0, 0, 0)
    logical_asym = counts_to_x(asym_qiskit_key)
    naive_asym = tuple(int(b) for b in asym_qiskit_key)
    assert logical_asym == (1, 0, 0, 0, 0, 0)
    assert naive_asym == (0, 0, 0, 0, 0, 1)
    # For (1, 0, 0, 0, 0, 0), vertex 0 cut edges are (0,2,1), (0,3,4), (0,4,4) -> cut = 9
    assert cut_weight(logical_asym) == 9
    # For (0, 0, 0, 0, 0, 1), vertex 5 cut edges are (1,5,3), (3,5,2) -> cut = 5
    assert cut_weight(naive_asym) == 5
    assert cut_weight(logical_asym) != cut_weight(naive_asym)


@pytest.mark.unit
def test_hand_verified_small_graphs():
    """Verify classical cut and brute-force solver on hand-calculated small graph instances."""
    # 1. Single edge
    edge_g = [(0, 1, 5)]
    opt_val, opt_sol = exact_classical_optimum(num_vertices=2, edges=edge_g)
    assert opt_val == 5
    assert set(opt_sol) == {"01", "10"}
    
    # 2. Triangle K_3 (weights 1, 1, 1) -> max cut is 2 (cannot cut all 3 edges)
    triangle = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
    opt_t, opt_sol_t = exact_classical_optimum(num_vertices=3, edges=triangle)
    assert opt_t == 2
    assert len(opt_sol_t) == 6  # 6 non-monolithic assignments
    
    # 3. Path P_3: (0, 1, 2) and (1, 2, 3) -> max cut is 2+3 = 5
    path = [(0, 1, 2), (1, 2, 3)]
    opt_p, opt_sol_p = exact_classical_optimum(num_vertices=3, edges=path)
    assert opt_p == 5
    assert set(opt_sol_p) == {"010", "101"}
    
    # 4. Cycle C_4: bipartite graph, max cut cuts all 4 edges (weight 4)
    cycle = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (0, 3, 1)]
    opt_c, opt_sol_c = exact_classical_optimum(num_vertices=4, edges=cycle)
    assert opt_c == 4
    assert set(opt_sol_c) == {"0101", "1010"}


@pytest.mark.unit
def test_invalid_graph_inputs_rejected():
    """Verify that malformed or invalid graph inputs raise clear ValueError/TypeError."""
    # Out of range vertex
    with pytest.raises(ValueError, match="Vertex index out of range"):
        validate_graph(6, [(0, 6, 1)])
        
    with pytest.raises(ValueError, match="Vertex index out of range"):
        validate_graph(6, [(-1, 2, 1)])
        
    # Self-loops
    with pytest.raises(ValueError, match="Self-loops are not permitted"):
        validate_graph(6, [(2, 2, 1)])
        
    # Malformed edge tuple
    with pytest.raises(ValueError, match="Each edge must be a .* tuple of length 3"):
        validate_graph(6, [(0, 1)])  # type: ignore
        
    # Non-positive weight
    with pytest.raises(ValueError, match="Edge weights must be positive"):
        validate_graph(6, [(0, 1, 0)])
        
    with pytest.raises(ValueError, match="Edge weights must be positive"):
        validate_graph(6, [(0, 1, -3)])
        
    # Duplicate edges
    with pytest.raises(ValueError, match="Duplicate edge detected"):
        validate_graph(6, [(0, 1, 1), (1, 0, 2)])
        
    # Empty edges
    with pytest.raises(ValueError, match="edges list cannot be empty"):
        validate_graph(6, [])
