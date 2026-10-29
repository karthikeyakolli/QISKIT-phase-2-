"""
src/benchmarks.py - Hardware-constrained benchmarks, resource metrics, and CSV output for PS-O1.
"""
from __future__ import annotations
import csv
import os
from typing import Dict, List, Any, Tuple, Set
import numpy as np
from qiskit import QuantumCircuit

from qff_phase2 import (
    PROCESSOR_A,
    PROCESSOR_B,
    load_processor,
    circuit_metrics,
    transpile_for,
    simulator,
    counts_to_x,
    SHOTS,
)
from src.formulation import cut_weight, EXACT_OPTIMUM, OPTIMAL_BITSTRINGS, random_baseline
from src.qaoa import build_qaoa_circuit


def compute_sampled_metrics(
    counts: Dict[str, int],
    shots: int = SHOTS,
    exact_optimum: int = EXACT_OPTIMUM,
    optimal_bitstrings: Set[str] = OPTIMAL_BITSTRINGS,
) -> Dict[str, Any]:
    """Compute and validate statistical quality metrics from measurement counts.
    
    Validates input distribution and raises ValueError on malformed counts.
    """
    if not isinstance(counts, dict) or len(counts) == 0:
        raise ValueError("counts must be a non-empty dictionary of bitstrings to integer counts")
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    if not isinstance(exact_optimum, (int, float)) or exact_optimum <= 0:
        raise ValueError(f"exact_optimum must be positive, got {exact_optimum}")

    total_shots = 0
    total_cut = 0
    best_cut = 0
    opt_shots = 0

    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise TypeError(f"Bitstring key must be str, got {type(bitstring)}")
        clean_bits = bitstring.replace(" ", "")
        if len(clean_bits) != 6:
            raise ValueError(f"Bitstring '{bitstring}' length is {len(clean_bits)}, expected 6")
        if not all(c in ("0", "1") for c in clean_bits):
            raise ValueError(f"Bitstring '{bitstring}' contains non-binary characters")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Count for bitstring '{bitstring}' must be a non-negative integer, got {count}")
            
        total_shots += count
        x = counts_to_x(bitstring)
        x_str = "".join(str(b) for b in x)
        c = cut_weight(x)
        total_cut += c * count
        if c > best_cut:
            best_cut = c
        if x_str in optimal_bitstrings:
            opt_shots += count

    if total_shots != shots:
        raise ValueError(f"Sum of measurement counts ({total_shots}) does not match shots ({shots})")

    mean_cut = total_cut / float(shots)
    r = mean_cut / float(exact_optimum)
    r_best = best_cut / float(exact_optimum)
    p_opt = opt_shots / float(shots)

    # Sanity bounds check
    if not (0.0 <= r <= 27.0 / exact_optimum):
        raise ValueError(f"Calculated approximation ratio {r} out of valid bounds [0, {27.0 / exact_optimum}]")
    if not (0.0 <= p_opt <= 1.0):
        raise ValueError(f"Calculated p_opt {p_opt} out of probability bounds [0, 1]")

    return {
        "shots": shots,
        "mean_cut": mean_cut,
        "approx_ratio": r,
        "best_cut": best_cut,
        "best_approx_ratio": r_best,
        "p_opt": p_opt,
        "raw_counts": counts,
    }


def evaluate_circuit_on_processor(
    qc: QuantumCircuit,
    processor: dict,
    noisy: bool,
    shots: int = SHOTS,
) -> Dict[str, Any]:
    """Transpile and simulate a QAOA circuit on a processor under ideal or noisy conditions."""
    # Transpile using official settings
    tqc = transpile_for(qc, processor)
    sim = simulator(noisy=noisy, processor=processor)
    job = sim.run(tqc, shots=shots)
    counts = job.result().get_counts()
    
    return compute_sampled_metrics(counts, shots=shots, exact_optimum=EXACT_OPTIMUM, optimal_bitstrings=OPTIMAL_BITSTRINGS)



def run_all_benchmarks(
    best_gamma: float,
    best_beta: float,
    processor_d_path: str = "processors/processor_D.json",
    shots: int = SHOTS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run all official benchmarks on Processors A, B, and D and collect resource metrics."""
    qc = build_qaoa_circuit(best_gamma, best_beta, measure=True)
    D = load_processor(processor_d_path)
    
    rand = random_baseline()
    rand_r = rand["approx_ratio"]
    
    # 1. Resource Metrics via official circuit_metrics()
    resources = {
        "A": circuit_metrics(qc, PROCESSOR_A),
        "B": circuit_metrics(qc, PROCESSOR_B),
        "D": circuit_metrics(qc, D),
    }
    
    benchmark_rows: List[Dict[str, Any]] = []
    
    # 2. Processor A (Capacity limit)
    # Section 3 & 9: PS-O1 needs 6 qubits, Processor A has 5 -> not executable
    for cond in ["ideal", "noisy"]:
        benchmark_rows.append({
            "problem_id": "PS-O1",
            "processor": "A",
            "condition": cond,
            "executable": "false",
            "logical_qubits": 6,
            "physical_qubits": 5,
            "logical_cx": "",
            "cx": "",
            "depth": "",
            "two_q_depth": "",
            "swaps": "",
            "cx_overhead": "",
            "shots": "",
            "quality_metric": "approx_ratio",
            "quality_value": "",
            "p_opt": "",
            "classical_optimum": EXACT_OPTIMUM,
            "random_baseline": round(rand_r, 4),
            "reason": "not executable: needs 6 qubits, has 5",
        })
        
    # 3. Processor B (9 qubits)
    m_b = resources["B"]
    for noisy, cond in [(False, "ideal"), (True, "noisy")]:
        eval_res = evaluate_circuit_on_processor(qc, PROCESSOR_B, noisy=noisy, shots=shots)
        benchmark_rows.append({
            "problem_id": "PS-O1",
            "processor": "B",
            "condition": cond,
            "executable": "true",
            "logical_qubits": m_b["logical_qubits"],
            "physical_qubits": m_b["physical_qubits"],
            "logical_cx": m_b["logical_cx"],
            "cx": m_b["cx"],
            "depth": m_b["depth"],
            "two_q_depth": m_b["two_q_depth"],
            "swaps": m_b["swaps"],
            "cx_overhead": m_b["cx_overhead"],
            "shots": shots,
            "quality_metric": "approx_ratio",
            "quality_value": round(eval_res["approx_ratio"], 4),
            "p_opt": round(eval_res["p_opt"], 4),
            "classical_optimum": EXACT_OPTIMUM,
            "random_baseline": round(rand_r, 4),
            "best_cut": eval_res["best_cut"],
            "mean_cut": round(eval_res["mean_cut"], 4),
        })
        
    # 4. Processor D (Custom 6 qubits)
    m_d = resources["D"]
    for noisy, cond in [(False, "ideal"), (True, "noisy")]:
        eval_res = evaluate_circuit_on_processor(qc, D, noisy=noisy, shots=shots)
        benchmark_rows.append({
            "problem_id": "PS-O1",
            "processor": "D",
            "condition": cond,
            "executable": "true",
            "logical_qubits": m_d["logical_qubits"],
            "physical_qubits": m_d["physical_qubits"],
            "logical_cx": m_d["logical_cx"],
            "cx": m_d["cx"],
            "depth": m_d["depth"],
            "two_q_depth": m_d["two_q_depth"],
            "swaps": m_d["swaps"],
            "cx_overhead": m_d["cx_overhead"],
            "shots": shots,
            "quality_metric": "approx_ratio",
            "quality_value": round(eval_res["approx_ratio"], 4),
            "p_opt": round(eval_res["p_opt"], 4),
            "classical_optimum": EXACT_OPTIMUM,
            "random_baseline": round(rand_r, 4),
            "best_cut": eval_res["best_cut"],
            "mean_cut": round(eval_res["mean_cut"], 4),
        })
        
    return benchmark_rows, resources


def write_results_csv(
    benchmark_rows: List[Dict[str, Any]],
    output_path: str = "results/tables/results.csv"
) -> None:
    """Write benchmark rows in the exact schema specified by Section 9 of the Challenge Kit."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fieldnames = [
        "problem_id",
        "processor",
        "condition",
        "executable",
        "logical_qubits",
        "physical_qubits",
        "logical_cx",
        "cx",
        "depth",
        "two_q_depth",
        "swaps",
        "cx_overhead",
        "shots",
        "quality_metric",
        "quality_value",
        "p_opt",
        "classical_optimum",
        "random_baseline",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in benchmark_rows:
            writer.writerow(row)
