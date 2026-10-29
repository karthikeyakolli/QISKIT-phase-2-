"""
run_experiments.py - Complete automated execution pipeline for PS-O1.
"""
from __future__ import annotations
import json
import time
import numpy as np

from src.formulation import (
    TOTAL_WEIGHT,
    EXACT_OPTIMUM,
    OPTIMAL_BITSTRINGS,
    exact_classical_optimum,
    random_baseline,
)
from src.qaoa import run_multi_start_qaoa
from src.benchmarks import run_all_benchmarks, write_results_csv
from src.bell import run_bell_state_suite, write_bell_csv
from src.visualization import (
    plot_weighted_graph,
    plot_processor_d_topology,
    plot_quality_comparison,
    plot_resource_comparison,
)


def main():
    t_start = time.time()
    print("======================================================================")
    print("QISKIT FALL FEST 2026 - PHASE 2 (PS-O1) BENCHMARK PIPELINE")
    print("======================================================================")
    
    # 1. Classical Verification
    print("\n--- 1. Classical Formulation Verification ---")
    c_star, opt_strings = exact_classical_optimum()
    rand = random_baseline()
    print(f"Total Graph Weight: {TOTAL_WEIGHT}")
    print(f"Exact Classical Optimum C*: {c_star}")
    print(f"Optimal Bitstrings ({len(opt_strings)}): {opt_strings}")
    print(f"Uniform Random Expected Cut: {rand['expected_cut']:.3f}")
    print(f"Uniform Random Approx Ratio: {rand['approx_ratio']:.4f}")
    print(f"Uniform Random P_opt: {rand['p_opt']:.4f}")
    
    # 2. Multi-Start QAOA Optimization
    print("\n--- 2. Multi-Start QAOA Optimization (Seeds 0, 1, 2, 3, 4) ---")
    seeds = [0, 1, 2, 3, 4]
    best_run, all_runs = run_multi_start_qaoa(seeds=seeds, maxiter=60)
    
    r_values = [r["final_r"] for r in all_runs]
    mean_r = float(np.mean(r_values))
    std_r = float(np.std(r_values))
    
    for r in all_runs:
        print(f"Seed {r['seed']}: Init=({r['initial_gamma']:.3f}, {r['initial_beta']:.3f}) -> "
              f"Opt=({r['optimal_gamma']:.3f}, {r['optimal_beta']:.3f}) | "
              f"Cut={r['final_cut']:.3f}, r={r['final_r']:.3f}, P_opt={r['final_p_opt']:.4f} ({r['runtime_sec']:.2f}s)")
              
    print(f"\nConsistency across {len(seeds)} seeds: Mean r = {mean_r:.4f} +/- {std_r:.4f}")
    print(f"Selected Best Parameter Set (Seed {best_run['seed']}):")
    print(f"  gamma* = {best_run['optimal_gamma']:.5f}")
    print(f"  beta*  = {best_run['optimal_beta']:.5f}")
    print(f"  Ideal Statevector Expected Cut = {best_run['final_cut']:.3f} (r = {best_run['final_r']:.4f})")
    print(f"  Ideal Statevector P_opt = {best_run['final_p_opt']:.4f}")
    
    # 3. Hardware-Constrained Benchmarks
    print("\n--- 3. Running Hardware-Constrained Benchmarks (Processors A, B, D) ---")
    benchmark_rows, resources = run_all_benchmarks(
        best_gamma=best_run["optimal_gamma"],
        best_beta=best_run["optimal_beta"],
        processor_d_path="processors/processor_D.json",
        shots=4096,
    )
    
    write_results_csv(benchmark_rows, "results/tables/results.csv")
    print("Saved: results/tables/results.csv")
    
    print("\nCircuit Resource Metrics Summary:")
    for proc_name, res in resources.items():
        if res.get("executable"):
            print(f"  Processor {proc_name}: Logical CX={res['logical_cx']}, Mapped CX={res['cx']}, "
                  f"Depth={res['depth']}, 2Q Depth={res['two_q_depth']}, SWAPs={res['swaps']}, CX Overhead={res['cx_overhead']}")
        else:
            print(f"  Processor {proc_name}: NOT EXECUTABLE ({res.get('reason')})")
            
    print("\nSampled Quality Summary (4096 shots):")
    for r in benchmark_rows:
        if r.get("executable") == "true":
            print(f"  Processor {r['processor']} ({r['condition']}): Mean Cut={r['mean_cut']:.3f}, "
                  f"r={r['quality_value']:.4f}, Best Cut={r['best_cut']}, P_opt={r['p_opt']:.4f}")
        else:
            print(f"  Processor {r['processor']} ({r['condition']}): Capacity limit - {r.get('reason')}")
            
    # 4. Bell-State Experiment on Processor D
    print("\n--- 4. Running Bell-State Verification on Processor D ---")
    bell_rows = run_bell_state_suite("processors/processor_D.json", shots=4096)
    write_bell_csv(bell_rows, "results/tables/bell.csv")
    print("Saved: results/tables/bell.csv")
    
    for br in bell_rows:
        print(f"  Pair {br['pair']} (dist={br['pair_distance']}) [{br['condition']}]: "
              f"XX={br['xx']:.3f}+/-{br['xx_se']:.3f}, YY={br['yy']:.3f}+/-{br['yy_se']:.3f}, "
              f"ZZ={br['zz']:.3f}+/-{br['zz_se']:.3f} | Fidelity = {br['fidelity']:.4f} +/- {br['fidelity_se']:.4f}")
              
    # 5. Visualizations
    print("\n--- 5. Generating Publication Figures ---")
    plot_weighted_graph("results/figures/weighted_graph.png")
    print("Saved: results/figures/weighted_graph.png")
    
    plot_processor_d_topology("processors/processor_D.json", "results/figures/processor_D_topology.png")
    print("Saved: results/figures/processor_D_topology.png")
    
    plot_quality_comparison(benchmark_rows, "results/figures/quality_comparison.png")
    print("Saved: results/figures/quality_comparison.png")
    
    plot_resource_comparison(resources, "results/figures/resource_comparison.png")
    print("Saved: results/figures/resource_comparison.png")
    
    elapsed = time.time() - t_start
    print(f"\n======================================================================")
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS (< 30 MINUTES LIMIT)")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
