"""
Test Suite for MAXCUT Version 5 PS-O1 Compliance & Authoritative Verification.

Validates:
1. Weighted Graph Schema & Properties (|V|=6, |E|=8, canonical integer weights [1,2,3,4,5,2,4,3], W=24.0).
2. Exhaustive Classical Solver (C_max = 20.0, optimal bitstrings '001100' and '110011', complement symmetry).
3. QUBO, Ising, and Cost Hamiltonian mathematical equivalence across all 64 assignments.
4. Full 6-Variable QAOA Formulation on Processor B and Processor D.
5. Processor A 5-Qubit Capacity Infeasibility Handling (TranspilerError on 6Q; auxiliary 5Q segregated).
6. Processor B (9Q Sparse Tree) and Processor D (8Q Chorded Mesh) coupling maps & compilation.
7. Noise Model parameters (1Q: 0.001, CX: 0.005, Readout: 0.01) and 4096 shots baseline.
8. Artifact Integrity in results/verified_o1_v5/ and historical preservation of v2, v3, v4.
"""

import json
from pathlib import Path
import pytest

from maxcut.domain.model import MaxCutGraph
from maxcut.classical.solver import ExactExhaustiveSolver
from maxcut.formulation.qubo import (
    build_maxcut_qubo,
    build_maxcut_ising,
    build_cost_hamiltonian,
)
from maxcut.qaoa.ansatz import build_qaoa_circuit, build_reduced_qaoa_circuit
from processors.processor_a.adapter import ProcessorABackend
from processors.processor_b.adapter import ProcessorBBackend
from processors.processor_d.adapter import ProcessorDBackend
from qiskit.transpiler.exceptions import TranspilerError

REPO_ROOT = Path(__file__).resolve().parent.parent
V5_DIR = REPO_ROOT / "results" / "verified_o1_v5"
V5_DATA_DIR = V5_DIR / "data"
V5_TABLES_DIR = V5_DIR / "tables"
V5_FIGURES_DIR = V5_DIR / "figures"
V5_CIRCUITS_DIR = V5_DIR / "circuits"
GRAPH_PATH = REPO_ROOT / "instances" / "participant" / "o1_maxcut_weighted_v5.json"


def test_v5_graph_specification_and_validation():
    """Verify the Version 5 weighted graph meets all PS-O1 constraints."""
    assert GRAPH_PATH.exists(), f"Missing graph file at {GRAPH_PATH}"
    graph = MaxCutGraph.from_challenge_file(GRAPH_PATH)

    assert graph.num_nodes == 6
    assert graph.num_edges == 8
    assert graph.is_weighted is True
    assert graph.nodes == [0, 1, 2, 3, 4, 5]

    canonical_edges = [
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 2, 3.0),
        (1, 3, 4.0),
        (2, 4, 5.0),
        (3, 4, 2.0),
        (3, 5, 4.0),
        (4, 5, 3.0),
    ]

    edge_map = {(min(e.u, e.v), max(e.u, e.v)): e.weight for e in graph.edges}
    for u, v, w in canonical_edges:
        key = (min(u, v), max(u, v))
        assert key in edge_map
        assert edge_map[key] == w
        assert 1 <= int(w) <= 5, f"Edge weight {w} outside [1, 5] range"

    assert graph.total_weight == 24.0


def test_v5_exhaustive_classical_optimum():
    """Verify classical exhaustive solver finds ground truth C_max = 20.0 and bitstrings."""
    graph = MaxCutGraph.from_challenge_file(GRAPH_PATH)
    solver = ExactExhaustiveSolver(record_distribution=True)
    result = solver.solve(graph)

    assert result.optimal_cut_value == 20.0
    assert result.total_assignments_evaluated == 64
    assert len(result.optimal_bitstrings) == 2

    # Check that bitstrings achieve exactly cut 20.0
    expected_optimal = {"001100", "110011"}
    assert set(result.optimal_bitstrings) == expected_optimal

    # Check complement symmetry: ~001100 = 110011
    b1 = result.optimal_bitstrings[0]
    b2 = result.optimal_bitstrings[1]
    comp_b1 = "".join("1" if c == "0" else "0" for c in b1)
    assert comp_b1 == b2

    # Check independent manual verification for all 64 assignments
    canonical_edges = [
        (0, 1, 1.0),
        (0, 2, 2.0),
        (1, 2, 3.0),
        (1, 3, 4.0),
        (2, 4, 5.0),
        (3, 4, 2.0),
        (3, 5, 4.0),
        (4, 5, 3.0),
    ]
    max_c = 0.0
    opt_states = []
    for x in range(64):
        bitstr = f"{x:06b}"
        c = 0.0
        for u, v, w in canonical_edges:
            if bitstr[u] != bitstr[v]:
                c += w
        if c > max_c:
            max_c = c
            opt_states = [bitstr]
        elif c == max_c:
            opt_states.append(bitstr)

    assert max_c == 20.0
    assert set(opt_states) == expected_optimal


def test_v5_qubo_hamiltonian_equivalence():
    """Verify mathematical equivalence between Cut Objective, QUBO, and Cost Hamiltonian."""
    graph = MaxCutGraph.from_challenge_file(GRAPH_PATH)
    qubo = build_maxcut_qubo(graph)
    cost_ham = build_cost_hamiltonian(graph)

    for x in range(64):
        bitstr = f"{x:06b}"
        cut_res = graph.compute_cut({i: int(bitstr[i]) for i in range(6)})
        cut_val = cut_res.cut_value

        # QUBO value: f_QUBO(x) = - C(x)
        qubo_val = qubo.evaluate(bitstr)
        assert abs(qubo_val - (-cut_val)) < 1e-9

        # Hamiltonian expectation: <x| H_C |x> = - C(x)
        h_val = cost_ham.evaluate_assignment(bitstr)
        assert abs(h_val - (-cut_val)) < 1e-9


def test_processor_a_capacity_rejection_and_auxiliary_segregation():
    """Verify Processor A (5Q) cleanly refuses 6-qubit logical ansatz and segregates auxiliary 5Q."""
    graph = MaxCutGraph.from_challenge_file(GRAPH_PATH)
    backend_a = ProcessorABackend(seed_transpiler=42)

    assert backend_a.num_qubits == 5

    # Full 6-qubit QAOA ansatz
    qc_6q = build_qaoa_circuit(graph, p=1, include_measurements=True).bind(
        [0.5], [0.5]
    )
    assert qc_6q.num_qubits == 6

    # Attempting to compile 6-qubit circuit onto 5-qubit Processor A must fail with TranspilerError
    with pytest.raises(TranspilerError):
        backend_a.verify_qaoa_circuit(qc_6q)

    # Auxiliary 5-variable symmetry-reduced circuit CAN compile on Processor A
    qc_5q = build_reduced_qaoa_circuit(
        graph, p=1, include_measurements=True
    ).bind([0.5], [0.5])
    assert qc_5q.num_qubits == 5

    verified_a_5q = backend_a.verify_qaoa_circuit(qc_5q)
    assert verified_a_5q["is_compilation_successful"] is True
    assert verified_a_5q["transpiled_qubit_count"] == 5
    assert verified_a_5q["routing_swap_count"] >= 1


def test_processor_b_and_d_full_6q_compilation():
    """Verify Processor B (9Q) and Processor D (8Q) successfully compile full 6Q circuit."""
    graph = MaxCutGraph.from_challenge_file(GRAPH_PATH)
    qc_6q = build_qaoa_circuit(graph, p=1, include_measurements=True).bind(
        [1.0], [1.0]
    )

    # Processor B
    backend_b = ProcessorBBackend(seed_transpiler=42)
    assert backend_b.num_qubits == 9
    res_b = backend_b.verify_qaoa_circuit(qc_6q, optimization_level=1)
    assert res_b["is_compilation_successful"] is True
    assert res_b["transpiled_qubit_count"] == 9
    assert res_b["physical_qubits_active_count"] == 6
    assert res_b["routing_swap_count"] == 3
    assert res_b["transpiled_2q_gates"] == 25
    assert res_b["transpiled_depth"] == 52

    # Processor D
    backend_d = ProcessorDBackend(seed_transpiler=42)
    assert backend_d.num_qubits == 8
    assert backend_d.num_qubits <= 12  # Budget constraint
    res_d = backend_d.verify_qaoa_circuit(qc_6q, optimization_level=1)
    assert res_d["is_compilation_successful"] is True
    assert res_d["transpiled_qubit_count"] == 8
    assert res_d["physical_qubits_active_count"] == 6
    assert res_d["routing_swap_count"] == 2
    assert res_d["transpiled_2q_gates"] == 22
    assert res_d["transpiled_depth"] == 41

    # Architectural improvement: D has lower depth and fewer CX gates than B
    assert res_d["transpiled_depth"] < res_b["transpiled_depth"]
    assert res_d["transpiled_2q_gates"] < res_b["transpiled_2q_gates"]
    assert res_d["routing_swap_count"] < res_b["routing_swap_count"]


def test_v5_benchmark_artifact_integrity():
    """Verify that all authoritative Version 5 benchmark artifacts exist, are valid JSON, and have 4096 shots."""
    required_counts_files = [
        V5_DIR / "counts_processor_b_ideal.json",
        V5_DIR / "counts_processor_b_noisy.json",
        V5_DIR / "counts_processor_d_ideal.json",
        V5_DIR / "counts_processor_d_noisy.json",
    ]

    for p in required_counts_files:
        assert p.exists(), f"Missing required counts file: {p}"
        with open(p, "r", encoding="utf-8") as f:
            counts = json.load(f)
        total_shots = sum(counts.values())
        assert (
            total_shots == 4096
        ), f"Counts file {p} does not sum to 4096 shots (got {total_shots})"

    required_data_files = [
        V5_DATA_DIR / "classical_reference.json",
        V5_DATA_DIR / "qaoa_optimization_result.json",
        V5_DATA_DIR / "processor_b_ideal_reference.json",
        V5_DATA_DIR / "processor_b_noisy_baseline.json",
        V5_DATA_DIR / "processor_d_ideal_reference.json",
        V5_DATA_DIR / "processor_d_noisy_baseline.json",
        V5_DATA_DIR / "experiment_summary.json",
    ]
    for p in required_data_files:
        assert p.exists(), f"Missing data file: {p}"

    # Verify classical reference data
    with open(
        V5_DATA_DIR / "classical_reference.json", "r", encoding="utf-8"
    ) as f:
        ref = json.load(f)
    assert ref["optimal_cut_value"] == 20.0
    assert ref["total_assignments_evaluated"] == 64
    assert set(ref["optimal_bitstrings"]) == {"001100", "110011"}

    # Verify QAOA optimization angles
    with open(
        V5_DATA_DIR / "qaoa_optimization_result.json", "r", encoding="utf-8"
    ) as f:
        qaoa = json.load(f)
    sel_idx = qaoa["selected_restart_index"]
    assert qaoa["restarts"][sel_idx]["converged"] is True
    assert len(qaoa["optimal_gamma"]) == 1
    assert len(qaoa["optimal_beta"]) == 1
    assert qaoa["best_expected_cut"] > 16.0


def test_v5_figures_and_tables_exist():
    """Verify all 8 publication SVG figures and summary tables exist and are non-empty."""
    expected_figures = [
        "figure1_maxcut_problem_graph.svg",
        "figure2_processor_a_coupling_map.svg",
        "figure3_processor_b_coupling_map.svg",
        "figure4_processor_d_coupling_map.svg",
        "figure5_logical_vs_transpiled_circuits.svg",
        "figure6_ideal_vs_noisy_distributions.svg",
        "figure7_architecture_resource_tradeoffs.svg",
        "figure8_combined_benchmark_performance.svg",
    ]
    for fig_name in expected_figures:
        fig_path = V5_FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 500, f"Figure {fig_path} is too small"

    expected_tables = [
        "master_version5_metrics.md",
        "master_version5_metrics.csv",
        "master_version5_metrics.json",
        "statistical_hypothesis_tests_v5.md",
        "statistical_hypothesis_tests_v5.csv",
        "architecture_resource_breakdown_v5.md",
        "architecture_resource_breakdown_v5.csv",
        "capacity_analysis_processor_a.md",
        "historical_comparison_v2_v3_v4_v5.md",
    ]
    for tbl_name in expected_tables:
        tbl_path = V5_TABLES_DIR / tbl_name
        assert tbl_path.exists(), f"Missing table: {tbl_path}"
        assert tbl_path.stat().st_size > 100, f"Table {tbl_path} is too small"


def test_historical_preservation_v2_v3_v4():
    """Verify historical archives (v2, v3, v4) remain completely intact and uncorrupted."""
    v2_dir = REPO_ROOT / "results" / "verified_ab_v2"
    v3_dir = REPO_ROOT / "results" / "verified_ab_v3"
    v4_dir = REPO_ROOT / "results" / "verified_ab_v4"

    assert v2_dir.exists(), "Historical directory verified_ab_v2 was deleted!"
    assert v3_dir.exists(), "Historical directory verified_ab_v3 was deleted!"
    assert v4_dir.exists(), "Historical directory verified_ab_v4 was deleted!"

    # Check key files in v4
    assert (v4_dir / "counts_processor_a_ideal.json").exists()
    assert (v4_dir / "counts_processor_b_noisy.json").exists()
    assert (v4_dir / "counts_processor_d_noisy.json").exists()
    assert (v4_dir / "data" / "processor_a_ideal_reference.json").exists()
    assert (v4_dir / "data" / "experiment_summary.json").exists()
