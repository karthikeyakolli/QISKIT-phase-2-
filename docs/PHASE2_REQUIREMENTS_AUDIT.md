# Phase 2 Requirements Audit and Traceability Matrix
## PS-O1: Quantum Optimization of Network Partitioning Using QAOA

**Track:** Quantum Optimization Track  
**Submission Scope:** Phase 2 Submission  
**Audit Date:** 2026-10-29  
**Auditor:** Team Antigravity Quantum  

---

## 1. Authoritative Precedence & Standards

Requirements are derived strictly from:
1. `QFF2026_Optimization_Challenge_Kit.docx` (Governing document for all benchmark rules, processor constraints, and judging rubrics)
2. `QFF_2026_Phase2_Optimization_Judging_Rubric_2page.pdf` (100-point rubric, automatic deduction rules, score thresholds)
3. `OPTIMISATION_PS.docx` (Refined Phase 2 problem statement for PS-O1)
4. `QFF2026_Phase2_Optimization_StarterPack/qff_phase2.py` (Official measurement and transpilation utilities)
5. `QFF2026_Phase2_Optimization_StarterPack/instances.json` (Locked benchmark instances)

---

## 2. Requirements Traceability Matrix

| Req ID | Requirement Description | Source Document & Section | Implementation Location | Verification Method | Status | Evidence / Verification Output |
|:---:|:---|:---|:---|:---|:---:|:---|
| **REQ-01** | Locked 6-vertex weighted graph ($|V|=6, |E|=9, W=27$) | Challenge Kit §6; `instances.json` | `data/instance.json`, `src/formulation.py` | Automated test `test_instance_json_consistency`, `test_graph_properties` | **PASS** | `LOCKED_EDGES` matches exact weights (1 to 5), total weight = 27. |
| **REQ-02** | Exact classical optimum $C^*=22$ with complementary solutions | Challenge Kit §6; Rubric Obj 3 | `src/formulation.py:exact_classical_optimum()` | `test_exact_optimum_and_solutions` | **PASS** | Brute force over 64 bitstrings yields $C^*=22$, solutions `001011` and `110100`. |
| **REQ-03** | Max-Cut QUBO and cost Hamiltonian mathematical derivation | Challenge Kit §6; Rubric Obj 3 | `src/formulation.py:qubo_objective()`, `cost_hamiltonian()` | `test_qubo_cut_equality`, `test_cost_hamiltonian_expectation` | **PASS** | $Q(x) == -C(x)$ for all 64 bitstrings; $\langle x\|H_C\|x\rangle == C(x)$. |
| **REQ-04** | Bitstring ordering convention (`counts_to_x`) | Challenge Kit §3; `qff_phase2.py` | `src/formulation.py`, `qff_phase2.py` | `test_counts_to_x_convention` | **PASS** | Reverses Qiskit little-endian strings into logical $(x_0, \dots, x_5)$ tuples. |
| **REQ-05** | Exact uniform-random baseline | Challenge Kit §9; Rubric Obj 3 | `src/formulation.py:random_baseline()` | `test_random_baseline` | **PASS** | Analytical $E[C] = 13.5, r = 13.5/22 = 0.6136, P_{opt} = 2/64 = 0.03125$. |
| **REQ-06** | QAOA $p=1$ ansatz circuit implementation | Challenge Kit §6; Rubric Obj 3 | `src/qaoa.py:build_qaoa_circuit()` | `test_qaoa_circuit_structure` | **PASS** | 6 qubits, $H^{\otimes 6}$ init, 18 CX + 9 RZ cost layer, 6 RX mixer layer. |
| **REQ-07** | 5 Prescribed random initializations (Seeds 0, 1, 2, 3, 4) | Challenge Kit §6; Rubric Obj 3 | `src/qaoa.py:run_multi_start_qaoa()` | `test_qaoa_reproducible_seeds` | **PASS** | Uniform angle sampling in $[0, \pi]$; Seed 3 selected ($r=0.8051, P_{opt}=0.1377$). |
| **REQ-08** | Multi-start consistency reporting (mean and std of $r$) | Challenge Kit §6; Rubric Obj 3 | `src/qaoa.py`, `run_experiments.py` | Run pipeline execution log | **PASS** | Consistency over 5 seeds: Mean $r = 0.7250 \pm 0.0401$. |
| **REQ-09** | Reference Processor A capacity limitation rule | Challenge Kit §3, §9; Rubric Obj 3 | `processors/processor_A.json`, `src/benchmarks.py` | `test_processor_a_definition_and_capacity` | **PASS** | Handled via capacity rule: `executable=false`, `reason="needs 6 qubits > 5"`. |
| **REQ-10** | Reference Processor B benchmark (ideal & noisy) | Challenge Kit §3, §9; Rubric Obj 3 | `processors/processor_B.json`, `src/benchmarks.py` | `test_processor_b_definition`, `test_benchmarks_execution` | **PASS** | Mapped CX=21, Depth=37, SWAPs=1, CX Overhead=3; Noisy $r=0.7874, P_{opt}=0.1260$. |
| **REQ-11** | Processor D compliance ($N \le 12$, max deg $\le 4$, connected, gate set) | Challenge Kit §5; Rubric Obj 2 | `processors/processor_D.json` | `test_processor_d_validation_and_freeze_hash` | **PASS** | `validate_processor_d` returns `[]`; 6 qubits, connected, max degree 4. |
| **REQ-12** | Processor D frozen state and hash verification | Challenge Kit §5; Rubric Obj 2 | `processors/processor_D.json` | SHA-256 verification | **PASS** | SHA-256: `652e064ecfd10f405a4226bc274f941a36f45865326901030a898ba8d2f37c57`. |
| **REQ-13** | Processor D benchmark (ideal & noisy) | Challenge Kit §5, §9; Rubric Obj 2, 3 | `src/benchmarks.py` | `test_benchmarks_execution` | **PASS** | Mapped CX=18, Depth=29, SWAPs=0, CX Overhead=0; Noisy $r=0.7862, P_{opt}=0.1250$. |
| **REQ-14** | Circuit resource metrics via official helper | Challenge Kit §9; `qff_phase2.py` | `src/benchmarks.py:circuit_metrics()` | `test_benchmarks_execution` | **PASS** | Logical CX, CX, Depth, 2Q Depth, SWAPs, CX overhead captured before decomposition. |
| **REQ-15** | Objective 1 Bell-state verification on D (2 pairs, 3 bases, 2 conditions) | Challenge Kit §4; Rubric Obj 1 | `src/bell.py:run_bell_state_suite()` | `test_bell_experiment_suite`, `test_bell_pair_discovery` | **PASS** | 12 circuits evaluated; Dist 1 noisy $F=0.9633 \pm 0.0021$; Dist 2 noisy $F=0.9542 \pm 0.0023$. |
| **REQ-16** | Valid Bell state fidelity formula $F = (1+\langle XX\rangle - \langle YY\rangle + \langle ZZ\rangle)/4$ | Challenge Kit §4; Rubric Obj 1 | `src/bell.py`, `qff_phase2.py` | `test_bell_experiment_suite` | **PASS** | Tested and verified; avoids invalid $P(00)+P(11)$ formulation. |
| **REQ-17** | Results CSV in exact required format | Challenge Kit §9; Rubric Obj 4 | `results/tables/results.csv` | `test_results_csv_schema`, `test_results_csv_consistency` | **PASS** | 6 rows matching exact fieldnames and capacity rules. |
| **REQ-18** | Bell CSV in exact required format | Challenge Kit §4; Rubric Obj 4 | `results/tables/bell.csv` | `test_bell_csv_schema` | **PASS** | 4 rows with correlation values, standard errors, distances, and fidelities. |
| **REQ-19** | Four required figures generated | Challenge Kit §10; Rubric Obj 5 | `results/figures/` | `test_required_files_exist` | **PASS** | Generated 300 DPI publication plots for graph, topology, quality, and resources. |
| **REQ-20** | Reproducible top-to-bottom notebook | Challenge Kit §10; Rubric Obj 4 | `main.ipynb` | Headless nbconvert execution | **PASS** | Completes in < 2 minutes on standard CPU without manual intervention. |
| **REQ-21** | Technical report (4–8 pages) covering all required sections | Challenge Kit §10; Rubric Obj 5 | `report.md` | Verification of sections 1–14 | **PASS** | Full scientific documentation including trade-offs, bottlenecks, and limitations. |
| **REQ-22** | Judging eligibility gates G1–G4 satisfied | Rubric §1 | `README.md` | Gate verification checklist | **READY** | G2, G3, G4 fully verified; G1 public repo prepared for portal entry prior to Day 2 deadline. |
| **REQ-23** | AI tool usage disclosure | Challenge Kit §10 | `README.md` §5 | Content verification | **PASS** | Explicitly declared tool name, scope of assistance, and human responsibility. |
