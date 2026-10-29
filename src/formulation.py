"""
src/formulation.py - Mathematical formulation, QUBO, and classical enumeration for PS-O1.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Tuple, Set
import numpy as np
from qiskit.quantum_info import SparsePauliOp

NUM_VERTICES: int = 6
LOCKED_EDGES: List[Tuple[int, int, int]] = [
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
TOTAL_WEIGHT: int = sum(w for _, _, w in LOCKED_EDGES)  # 27
EXACT_OPTIMUM: int = 22
OPTIMAL_BITSTRINGS: Set[str] = {"001011", "110100"}


def validate_graph(num_vertices: int, edges: List[Tuple[int, int, int]]) -> None:
    """Validate a graph definition against standard network partitioning constraints.
    
    Raises ValueError or TypeError for invalid graph specifications.
    """
    if not isinstance(num_vertices, int) or num_vertices <= 0:
        raise ValueError(f"num_vertices must be a positive integer, got {num_vertices}")
    if not isinstance(edges, list):
        raise TypeError(f"edges must be a list of tuples, got {type(edges)}")
    if len(edges) == 0:
        raise ValueError("edges list cannot be empty")
        
    seen_edges = set()
    for e in edges:
        if not (isinstance(e, (tuple, list)) and len(e) == 3):
            raise ValueError(f"Each edge must be a (u, v, weight) tuple of length 3, got {e}")
        u, v, w = e
        if not (isinstance(u, int) and isinstance(v, int)):
            raise TypeError(f"Vertex indices must be integers, got ({u}, {v})")
        if u < 0 or u >= num_vertices or v < 0 or v >= num_vertices:
            raise ValueError(f"Vertex index out of range [0, {num_vertices - 1}]: ({u}, {v})")
        if u == v:
            raise ValueError(f"Self-loops are not permitted in simple graph partitioning: ({u}, {v})")
        if not (isinstance(w, (int, float)) and w > 0):
            raise ValueError(f"Edge weights must be positive numbers, got {w}")
        canon = tuple(sorted((u, v)))
        if canon in seen_edges:
            raise ValueError(f"Duplicate edge detected between vertices {canon}")
        seen_edges.add(canon)


def cut_weight(
    x: Tuple[int, ...] | List[int] | str,
    edges: List[Tuple[int, int, int]] | None = None,
) -> int:
    """Evaluate Max-Cut objective C(x) for logical assignment x.
    
    C(x) = sum_{(i,j) in E} w_ij * (x_i + x_j - 2 * x_i * x_j)
    """
    if isinstance(x, str):
        x = tuple(int(b) for b in x)
    active_edges = LOCKED_EDGES if edges is None else edges
    val = 0
    for u, v, w in active_edges:
        if u >= len(x) or v >= len(x):
            raise IndexError(f"Assignment length {len(x)} insufficient for edge ({u}, {v})")
        val += int(w * (x[u] + x[v] - 2 * x[u] * x[v]))
    return val


def qubo_objective(
    x: Tuple[int, ...] | List[int] | str,
    edges: List[Tuple[int, int, int]] | None = None,
) -> int:
    """Evaluate QUBO minimization objective Q(x) = -C(x)."""
    return -cut_weight(x, edges=edges)


def qubo_matrix(
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Construct the symmetric QUBO matrix Q such that Q(x) = x^T Q x = -C(x)."""
    active_edges = LOCKED_EDGES if edges is None else edges
    Q = np.zeros((num_vertices, num_vertices), dtype=float)
    for u, v, w in active_edges:
        # Cut indicator: (x_u + x_v - 2 x_u x_v)
        # Q(x) = -C(x) = sum w * (2 x_u x_v - x_u - x_v)
        # Linear terms: diagonal entries -w
        Q[u, u] -= w
        Q[v, v] -= w
        # Quadratic terms: 2 w x_u x_v -> symmetric split w on (u, v) and (v, u)
        Q[u, v] += w
        Q[v, u] += w
    return Q


def enumerate_all_assignments(
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> Dict[str, int]:
    """Exhaustively enumerate all 2^N binary assignments."""
    scores: Dict[str, int] = {}
    for bits in itertools.product([0, 1], repeat=num_vertices):
        bit_str = "".join(str(b) for b in bits)
        scores[bit_str] = cut_weight(bits, edges=edges)
    return scores


def exact_classical_optimum(
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> Tuple[int, List[str]]:
    """Compute the exact maximum cut weight and optimal bitstrings by brute-force."""
    scores = enumerate_all_assignments(num_vertices=num_vertices, edges=edges)
    max_c = max(scores.values())
    opts = sorted([b for b, c in scores.items() if c == max_c])
    return max_c, opts


def random_baseline(
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> Dict[str, float]:
    """Exact analytical expectations under a uniform random distribution over all 2^N bitstrings."""
    active_edges = LOCKED_EDGES if edges is None else edges
    tot_weight = sum(w for _, _, w in active_edges)
    exp_cut = tot_weight / 2.0
    
    if edges is None:
        opt_val = EXACT_OPTIMUM
        num_opts = len(OPTIMAL_BITSTRINGS)
    else:
        opt_val, opts = exact_classical_optimum(num_vertices=num_vertices, edges=edges)
        num_opts = len(opts)
        
    approx_ratio = exp_cut / opt_val if opt_val > 0 else 0.0
    p_opt = num_opts / (2 ** num_vertices)
    return {
        "expected_cut": float(exp_cut),
        "approx_ratio": float(approx_ratio),
        "p_opt": float(p_opt),
    }


def cost_hamiltonian(
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> SparsePauliOp:
    """Construct the cost Hamiltonian H_C corresponding to Max-Cut.
    
    In qubit space, x_i = (I - Z_i)/2.
    x_i + x_j - 2 x_i x_j = (I - Z_i Z_j) / 2.
    H_C = sum_{(i,j) in E} (w_ij / 2) * (I - Z_i Z_j).
    """
    active_edges = LOCKED_EDGES if edges is None else edges
    tot_weight = sum(w for _, _, w in active_edges)
    
    pauli_list = []
    # Identity term
    pauli_list.append(("I" * num_vertices, tot_weight / 2.0))
    # - (w_ij / 2) * Z_i Z_j
    for u, v, w in active_edges:
        chars = ["I"] * num_vertices
        chars[num_vertices - 1 - u] = "Z"
        chars[num_vertices - 1 - v] = "Z"
        pauli_list.append(("".join(chars), -w / 2.0))
    return SparsePauliOp.from_list(pauli_list)

