"""
src/qaoa.py - QAOA circuit construction, statevector expectation, and multi-start optimization for PS-O1.
"""
from __future__ import annotations
import time
import itertools
from typing import List, Tuple, Dict, Any
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.formulation import LOCKED_EDGES, NUM_VERTICES, cut_weight, EXACT_OPTIMUM, OPTIMAL_BITSTRINGS


def build_qaoa_circuit(
    gamma: float | List[float] | Tuple[float, ...],
    beta: float | List[float] | Tuple[float, ...],
    p: int = 1,
    measure: bool = True,
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> QuantumCircuit:
    r"""Build a QAOA circuit for Max-Cut optimization.
    
    Default: p=1 depth on 6 logical qubits for the official PS-O1 instance.
    Optional: supports p >= 1 (e.g. p=2) when gammas and betas of length p are provided.
    
    1. Initial state: |+>^{\otimes n} via H on all qubits.
    2. Cost layer: e^{-i \gamma_l H_C} for each edge (u, v, w):
       CX(u, v) -> RZ(2 * gamma_l * w, v) -> CX(u, v).
    3. Mixer layer: e^{-i \beta_l H_M} as RX(2 * beta_l) on all qubits.
    4. Measurement: measure_all() if measure is True.
    """
    if not isinstance(p, int) or p < 1:
        raise ValueError(f"QAOA depth p must be a positive integer, got {p}")
        
    # Unpack gammas and betas
    if isinstance(gamma, (int, float)):
        gammas = [float(gamma)]
    else:
        gammas = [float(g) for g in gamma]
        
    if isinstance(beta, (int, float)):
        betas = [float(beta)]
    else:
        betas = [float(b) for b in beta]
        
    if len(gammas) != p or len(betas) != p:
        raise ValueError(f"Length of gamma ({len(gammas)}) and beta ({len(betas)}) must equal p={p}")
        
    for idx, (g, b) in enumerate(zip(gammas, betas)):
        if not (np.isfinite(g) and np.isfinite(b)):
            raise ValueError(f"QAOA angles at layer {idx} must be finite, got gamma={g}, beta={b}")

    active_edges = LOCKED_EDGES if edges is None else edges
    qc = QuantumCircuit(num_vertices, name=f"qaoa_p{p}")
    
    # Initial superposition
    qc.h(range(num_vertices))
    
    # Alternating Cost and Mixer layers
    for layer in range(p):
        g_l = gammas[layer]
        b_l = betas[layer]
        
        # Cost unitary
        for u, v, w in active_edges:
            qc.cx(u, v)
            qc.rz(2.0 * g_l * w, v)
            qc.cx(u, v)
            
        # Mixer unitary
        for q in range(num_vertices):
            qc.rx(2.0 * b_l, q)
            
    if measure:
        qc.measure_all()
        
    return qc


def evaluate_statevector_metrics(
    gamma: float | List[float] | Tuple[float, ...],
    beta: float | List[float] | Tuple[float, ...],
    p: int = 1,
    num_vertices: int = NUM_VERTICES,
    edges: List[Tuple[int, int, int]] | None = None,
) -> Dict[str, float]:
    """Evaluate exact ideal statevector expected cut, approximation ratio, and P_opt."""
    qc = build_qaoa_circuit(gamma, beta, p=p, measure=False, num_vertices=num_vertices, edges=edges)
    sv = Statevector(qc)
    probs = sv.probabilities_dict()
    
    active_edges = LOCKED_EDGES if edges is None else edges
    exact_opt = EXACT_OPTIMUM if edges is None else max(cut_weight(x, edges=active_edges) for x in itertools.product([0, 1], repeat=num_vertices))
    opt_strings = OPTIMAL_BITSTRINGS if edges is None else {
        "".join(str(b) for b in bits)
        for bits in itertools.product([0, 1], repeat=num_vertices)
        if cut_weight(bits, edges=active_edges) == exact_opt
    }
    
    exp_cut = 0.0
    p_opt = 0.0
    
    for bitstring, prob in probs.items():
        # In Qiskit Statevector bitstrings, qubit 0 is the rightmost character (little-endian)
        # To get logical x = (x0, ..., x_{n-1}):
        x = tuple(int(b) for b in reversed(bitstring))
        x_str = "".join(str(b) for b in x)
        
        c = cut_weight(x, edges=active_edges)
        exp_cut += prob * c
        if x_str in opt_strings:
            p_opt += prob
            
    return {
        "expected_cut": float(exp_cut),
        "approx_ratio": float(exp_cut / exact_opt) if exact_opt > 0 else 0.0,
        "p_opt": float(p_opt),
    }



def optimize_qaoa_seed(seed: int, maxiter: int = 60, method: str = "COBYLA") -> Dict[str, Any]:
    """Run classical optimization of (gamma, beta) for a single random initialization seed."""
    rng = np.random.default_rng(seed)
    gamma0 = float(rng.uniform(0.0, np.pi))
    beta0 = float(rng.uniform(0.0, np.pi))
    
    eval_count = 0
    t0 = time.time()
    
    def loss(params: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        g, b = float(params[0]), float(params[1])
        metrics = evaluate_statevector_metrics(g, b)
        # Minimize negative expected cut
        return -metrics["expected_cut"]
    
    res = minimize(
        loss,
        x0=np.array([gamma0, beta0]),
        method=method,
        options={"maxiter": maxiter, "tol": 1e-4}
    )
    runtime = time.time() - t0
    
    opt_gamma = float(res.x[0])
    opt_beta = float(res.x[1])
    final_metrics = evaluate_statevector_metrics(opt_gamma, opt_beta)
    init_metrics = evaluate_statevector_metrics(gamma0, beta0)
    
    return {
        "seed": seed,
        "optimizer": method,
        "maxiter": maxiter,
        "initial_gamma": gamma0,
        "initial_beta": beta0,
        "initial_cut": init_metrics["expected_cut"],
        "initial_r": init_metrics["approx_ratio"],
        "optimal_gamma": opt_gamma,
        "optimal_beta": opt_beta,
        "final_cut": final_metrics["expected_cut"],
        "final_r": final_metrics["approx_ratio"],
        "final_p_opt": final_metrics["p_opt"],
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "message": str(res.message),
        "runtime_sec": round(runtime, 4),
    }


def run_multi_start_qaoa(seeds: List[int] = [0, 1, 2, 3, 4], maxiter: int = 60) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run QAOA across the official 5 seeds [0, 1, 2, 3, 4] and select the best parameter set."""
    all_runs: List[Dict[str, Any]] = []
    for s in seeds:
        run_res = optimize_qaoa_seed(seed=s, maxiter=maxiter)
        all_runs.append(run_res)
        
    # Best selection under documented criterion: highest ideal expected cut
    best_run = max(all_runs, key=lambda r: r["final_cut"])
    return best_run, all_runs
