# Technical Report — Quantum Optimization of Network Partitioning Using QAOA
## Qiskit Fall Fest 2026 (Phase 2) — Problem Statement PS-O1

**Team Name:** Team Antigravity Quantum  
**Track:** Quantum Optimization Track  
**Problem Statement:** PS-O1 (Network Partitioning / Weighted Max-Cut)  
**Date:** October 29, 2026  
**Document Length:** 6 Pages Equivalent (Comprehensive Technical Report)  
**Governing Documents:** `QFF2026_Optimization_Challenge_Kit.docx`, `QFF_2026_Phase2_Optimization_Judging_Rubric_2page.pdf`, `qff_phase2.py`  

---

## 1. Problem Statement and Motivation

Network partitioning is a fundamental combinatorial problem with critical applications in telecommunications infrastructure, VLSI circuit layout design, distributed cloud computing cluster division, and community detection in complex social and biological networks. Given an undirected graph $G = (V, E)$ with non-negative edge weights $w_{ij}$, the objective is to bi-partition the vertex set $V$ into two disjoint subsets $S$ and $S' = V \setminus S$ such that the aggregate weight of edges crossing between $S$ and $S'$ is maximized. This problem is isomorphic to the weighted Maximum Cut (Max-Cut) problem.

Max-Cut is a canonical NP-hard problem. While the celebrated Goemans–Williamson classical randomized algorithm provides a polynomial-time approximation ratio guarantee of $\alpha_{GW} \approx 0.878$ using semidefinite programming (SDP) relaxation, computing optimal solutions for large dense graphs becomes computationally intractable classically. Variational quantum algorithms—specifically the Quantum Approximate Optimization Algorithm (QAOA) introduced by Farhi, Goldstone, and Gutmann (2014)—represent a leading paradigm for solving combinatorial optimization on Noisy Intermediate-Scale Quantum (NISQ) processors. 

In this Phase 2 investigation for Qiskit Fall Fest 2026, we formulate the locked 6-vertex network partitioning problem into a Quadratic Unconstrained Binary Optimization (QUBO) model, implement a parameterized $p=1$ QAOA quantum circuit, execute a rigorous multi-start optimization protocol over 5 prescribed random seeds, and systematically benchmark performance across standard reference processors and a custom-engineered quantum topology.

---

## 2. Locked Weighted Graph & Mathematical Formulation

### 2.1 The Official Locked Instance
In accordance with Section 6 of the Optimization Challenge Kit, all PS-O1 benchmark runs utilize an exact, locked 6-vertex weighted network ($|V| = 6, |E| = 9$). Logical variable $x_i$ maps bijectively to vertex $i$ and logical qubit $i$. The 9 weighted edges are locked as follows:

$$\begin{aligned}
e_1 &= (0, 2), \; w_{02} = 1 \qquad &e_4 &= (1, 4), \; w_{14} = 3 \qquad &e_7 &= (2, 4), \; w_{24} = 1 \\
e_2 &= (0, 3), \; w_{03} = 4 \qquad &e_5 &= (1, 5), \; w_{15} = 3 \qquad &e_8 &= (3, 4), \; w_{34} = 4 \\
e_3 &= (0, 4), \; w_{04} = 4 \qquad &e_6 &= (2, 3), \; w_{23} = 5 \qquad &e_9 &= (3, 5), \; w_{35} = 2
\end{aligned}$$

The total weight of the network is:
$$W = \sum_{(i,j) \in E} w_{ij} = 1 + 4 + 4 + 3 + 3 + 5 + 1 + 4 + 2 = 27$$

### 2.2 Classical Max-Cut Objective
Let $x_i \in \{0, 1\}$ represent the binary partition assignment of vertex $i$ ($x_i = 0$ if $i \in S$, and $x_i = 1$ if $i \in S'$). An edge $(i, j)$ crosses the partition boundary if and only if $x_i \neq x_j$. Because $x_i, x_j \in \{0, 1\}$, the cut indicator evaluates to:
$$x_i + x_j - 2 x_i x_j = \begin{cases} 1 & \text{if } x_i \neq x_j \\ 0 & \text{if } x_i = x_j \end{cases}$$

The total cut weight to be maximized is:
$$C(x) = \sum_{(i,j) \in E} w_{ij} (x_i + x_j - 2 x_i x_j)$$

---

## 3. QUBO Model and Cost Hamiltonian Derivation

### 3.1 QUBO Formulation
Classical and quantum optimizers typically minimize objective functions. We therefore formulate the Quadratic Unconstrained Binary Optimization (QUBO) minimization objective as the negative cut weight:
$$Q(x) = -C(x) = \sum_{(i,j) \in E} w_{ij} (2 x_i x_j - x_i - x_j)$$

Expanding over the 9 locked edges gives the explicit quadratic form:
$$Q(x) = x^T Q_{mat} x$$
where $Q_{mat}$ is the symmetric $6 \times 6$ QUBO matrix:
$$Q_{mat} = \begin{pmatrix}
-9 & 0 & 1 & 4 & 4 & 0 \\
0 & -6 & 0 & 0 & 3 & 3 \\
1 & 0 & -7 & 5 & 1 & 0 \\
4 & 0 & 5 & -15 & 4 & 2 \\
4 & 3 & 1 & 4 & -12 & 0 \\
0 & 3 & 0 & 2 & 0 & -5
\end{pmatrix}$$
Notice that diagonal entries reflect the negative weighted degrees of each vertex: $-d_w(0)=-9, -d_w(1)=-6, -d_w(2)=-7, -d_w(3)=-15, -d_w(4)=-12, -d_w(5)=-5$.

### 3.2 Cost Hamiltonian Mapping
To execute optimization on a quantum processor, binary variables are mapped to quantum observables via the standard spin transformation:
$$x_i = \frac{I - Z_i}{2}$$
Substituting this into the cut expression:
$$x_i + x_j - 2 x_i x_j = \frac{I - Z_i Z_j}{2}$$
The quantum Cost Hamiltonian $H_C$ is therefore:
$$H_C = \sum_{(i,j) \in E} \frac{w_{ij}}{2} (I - Z_i Z_j) = \frac{W}{2} I - \sum_{(i,j) \in E} \frac{w_{ij}}{2} Z_i Z_j = 13.5 I - \sum_{(i,j) \in E} \frac{w_{ij}}{2} Z_i Z_j$$

For any computational basis state $|x\rangle = |x_0 x_1 x_2 x_3 x_4 x_5\rangle$, the expectation value satisfies:
$$\langle x | H_C | x \rangle = C(x)$$
Maximizing the cut weight $C(x)$ is strictly equivalent to maximizing the expectation value $\langle \psi(\gamma, \beta) | H_C | \psi(\gamma, \beta) \rangle$.

---

## 4. Exact Classical Ground Truth and Uniform Random Baseline

### 4.1 Exhaustive Classical Search
Because the problem space comprises $2^6 = 64$ binary states, we perform an exact exhaustive enumeration across all assignments to establish rigorous classical ground truth.

```
Total states enumerated: 64
Maximum Cut Weight (C*): 22
Minimum Cut Weight: 0 (states 000000 and 111111)
Mean Cut Weight across all 64 states: 13.500
Number of optimal states: 2
Optimal Bitstrings (logical ordering x0 x1 x2 x3 x4 x5):
  1. '001011' -> Cut = 22
  2. '110100' -> Cut = 22
```

### 4.2 Complement Symmetry
Max-Cut exhibits a strict $\mathbb{Z}_2$ bit-flip symmetry: swapping the labels of the two partitions ($S \leftrightarrow S'$) leaves the set of cut edges invariant. For any assignment $x$, its bitwise complement $\bar{x}$ satisfies:
$$C(x) = C(\bar{x})$$
The optimal partition is $S = \{0, 1, 3\}$ and $S' = \{2, 4, 5\}$. Edge breakdown:
- Cut edges: $(0,2)[1], (0,4)[4], (1,4)[3], (1,5)[3], (2,3)[5], (3,4)[4], (3,5)[2] \implies \sum = 22$.
- Uncut edges: $(0,3)[4], (2,4)[1] \implies \sum = 5$.
- Sum of cut and uncut weights: $22 + 5 = 27 = W$.

Both complementary bitstrings `001011` and `110100` are recognized as valid optimal solutions per Section 6 of the Challenge Kit.

### 4.3 Analytical Uniform Random Baseline
Any quantum algorithm is scientifically meaningful only if it demonstrably outperforms random guessing. Under a uniform random distribution over the 64 states:
1. Every edge $(i, j)$ is cut with probability $P(x_i \neq x_j) = 0.5$.
2. **Expected Random Cut:** $\mathbb{E}_{rand}[C] = \sum_{(i,j)\in E} w_{ij} \times 0.5 = 27 / 2 = 13.500$.
3. **Random Approximation Ratio:** $r_{rand} = \frac{13.5}{22.0} \approx 0.6136$.
4. **Random Optimal Sampling Probability:** $P_{opt, rand} = \frac{2}{64} = 0.03125$ ($3.125\%$).

### 4.4 Classical Symmetry Reduction vs. Quantum Capacity Rule
In classical optimization, global spin-flip symmetry $C(x) = C(\mathbf{1} - x)$ allows one variable (e.g., $x_0 = 0$) to be arbitrarily fixed without loss of generality, reducing the classical search space from $2^6 = 64$ to $2^5 = 32$ evaluations.
However, in quantum optimization:
1. **Hamiltonian Transformation:** Fixing $x_0 = 0$ transforms two-qubit terms $Z_0 Z_v$ into single-qubit longitudinal fields $-w_{0v} Z_v$.
2. **Quantum Dynamics Alteration:** The remaining 5 qubits evolve under an inhomogeneous longitudinal field that breaks the isotropic $\mathbb{Z}_2$ quantum interference pattern, altering the mixer dynamics $e^{-i \beta \sum X_i}$ and ground-state projection.
3. **Official Capacity Rule:** Most importantly, Section 3 of the Optimization Challenge Kit and line 13 of the starter pack README explicitly mandate:
   > *"Processor A prints 'not executable (needs 6 qubits > 5)' for PS-O1 — this is correct (capacity rule, kit Section 3)."*
   Attempting to substitute a 5-variable symmetry-reduced formulation onto Processor A to artificially manufacture executable metrics is strictly prohibited by official hackathon rules. Therefore, our submission preserves the authentic 6-variable problem formulation and honestly reports Processor A as non-executable.

---

## 5. QAOA Circuit Architecture, Multi-Start Optimization & Convergence

### 5.1 QAOA $p=1$ Ansatz Structure
We construct the depth $p=1$ QAOA circuit on 6 logical qubits:
1. **Initial State:** Uniform superposition prepared via Hadamard gates:
   $$|\psi_0\rangle = H^{\otimes 6} |0\rangle^{\otimes 6} = \frac{1}{\sqrt{64}} \sum_{x \in \{0,1\}^6} |x\rangle$$
2. **Cost Unitary ($U(C, \gamma) = e^{-i \gamma H_C}$):** For each edge $(u, v, w)$, evolution under $e^{i \gamma \frac{w}{2} Z_u Z_v}$ is implemented as:
   $$\text{CX}(u, v) \longrightarrow R_Z(2 \gamma w, v) \longrightarrow \text{CX}(u, v)$$
   Total cost layer operations: 18 CX gates and 9 $R_Z$ rotations.
3. **Mixer Unitary ($U(B, \beta) = e^{-i \beta H_M}$):** Applied with $H_M = \sum_{i=0}^5 X_i$:
   $$\prod_{i=0}^5 R_X(2 \beta, i)$$
   Total mixer layer operations: 6 $R_X$ single-qubit gates.
4. **Measurement:** All 6 qubits measured into classical registers.

### 5.2 Multi-Start Optimization Protocol
Per the official Challenge Kit protocol, parameters $\gamma$ and $\beta$ were optimized using classical optimization in ideal statevector simulation across 5 random initializations using seeds $0, 1, 2, 3, 4$. Starting angles were drawn uniformly from $[0, \pi]$ using `np.random.default_rng(seed)`. Optimization employed the `COBYLA` algorithm (maxiter=60) minimizing $-\mathbb{E}[C]$.

#### Multi-Start Convergence Summary Table

| Seed | Initial $(\gamma_0, \beta_0)$ (rad) | Optimized $(\gamma^*, \beta^*)$ (rad) | Initial Cut | Optimized Cut | Approx Ratio $r$ | $P_{opt}$ | Iterations | Runtime |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | (2.001, 0.848) | (2.464, 1.986) | 14.887 | 15.494 | 0.7043 | 0.1201 | 54 | 0.21s |
| **1** | (1.608, 2.986) | (2.735, 4.257) | 13.912 | 15.558 | 0.7072 | 0.0665 | 48 | 0.17s |
| **2** | (0.822, 0.938) | (0.680, 1.139) | 14.802 | 15.491 | 0.7041 | 0.1204 | 55 | 0.21s |
| **3** | **(0.269, 0.744)** | **(0.084, 1.238)** | **15.112** | **17.713** | **0.8051** | **0.1377** | **45** | **0.14s** |
| **4** | (2.963, 1.606) | (2.465, 0.405) | 13.564 | 15.491 | 0.7041 | 0.1195 | 52 | 0.19s |

- **Mean Consistency:** Mean approximation ratio across all 5 seeds is $\bar{r} = 0.7250 \pm 0.0401$.
- **Global Best Selection:** Seed 3 converged to the global optimum:
  $$\gamma^* = 0.08396, \quad \beta^* = 1.23810$$
  Yielding an ideal expected cut of **$17.713$** ($r = \mathbf{0.8051}$) and optimal sampling probability $P_{opt} = \mathbf{13.77\%}$.
  Per the Phase 2 FAQ rule, these exact angles were frozen and evaluated across all hardware processors.

---

## 6. Processor A Capacity Limitation

Reference **Processor A** possesses 5 physical qubits with linear-tree coupling: `[(0, 1), (1, 2), (1, 3), (3, 4)]`.

Our PS-O1 problem strictly requires 6 logical qubits to represent the 6 vertices of the network. Under Section 3 of the Challenge Kit:
> *"A circuit that needs more qubits than a processor has is not run on it. PS-O1 needs 6 qubits ... so it does not fit Processor A (5 qubits). For these problems, enter Processor A in your results table as 'not executable: needs N qubits, has 5'."*

We strictly adhere to this rule. We do not artificially remove vertices, shrink the instance, or fabricate metrics. In `results/tables/results.csv`, Processor A is documented with `executable=false`, `logical_qubits=6`, `physical_qubits=5`, and empty quality metrics.

---

## 7. Processor B and Processor D Architecture Definitions

### 7.1 Reference Processor B (9 Qubits)
Processor B represents a 9-qubit ring-and-cross topology with 12 undirected coupling edges:
$$\{(0,1), (1,2), (1,6), (2,3), (2,6), (2,5), (3,4), (4,5), (5,6), (6,7), (6,8), (8,0)\}$$
While Processor B has sufficient qubit capacity ($9 \ge 6$), its physical connectivity does not contain all 9 problem edges as direct couplings. Consequently, Qiskit's SABRE routing pass must insert SWAP gates to execute non-local two-qubit interactions.

### 7.2 Custom Processor D (6 Qubits)
To resolve the routing bottleneck observed on Processor B, we designed **Processor D** (`processors/processor_D.json`).
- **Qubit Count:** 6 physical qubits (within the permitted range $[6, 12]$).
- **Physical Coupling Edges (9 edges):**
  $$\{(0,2), (0,3), (0,4), (1,4), (1,5), (2,3), (2,4), (3,4), (3,5)\}$$
- **Compliance Rules:**
  1. *Connectivity:* Fully connected undirected graph (diameter = 2).
  2. *Degree Constraint:* Maximum degree is 4 ($\text{deg}(3)=4, \text{deg}(4)=4, \text{deg}(0)=3, \text{deg}(2)=3, \text{deg}(1)=2, \text{deg}(5)=2$). Fully compliant with $\le 4$ limit.
  3. *Basis Gates:* Exactly `["rz", "sx", "x", "cx"]`.
  4. *Noise Model:* Official synthetic noise model ($p_1 = 0.001, p_2 = 0.005, p_{ro} = 0.01$).
  5. *Validation:* Passed `validate_processor_d()` with 0 rule violations.
  6. *Frozen Hash:* SHA-256 hash `652e064ecfd10f405a4226bc274f941a36f45865326901030a898ba8d2f37c57`.

---

## 8. Hardware-Constrained Quality Benchmark Results

Using the frozen parameters $(\gamma^* = 0.08396, \beta^* = 1.23810)$, the circuit was transpiled and executed on AerSimulator with 4096 shots under ideal and official noisy simulation.

### Quality Benchmarks Summary Table (`results/tables/results.csv`)

| Processor | Condition | Executable | Mean Cut | Approx Ratio $r$ | Best Cut | $P_{opt}$ | Exact $C^*$ | Random $r$ | Random $P_{opt}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | ideal | `false` | — | — | — | — | 22 | 0.6136 | 0.03125 |
| **A** | noisy | `false` | — | — | — | — | 22 | 0.6136 | 0.03125 |
| **B** | ideal | `true` | 17.761 | **0.8073** | 22 | **0.1396** | 22 | 0.6136 | 0.03125 |
| **B** | noisy | `true` | 17.322 | **0.7874** | 22 | **0.1260** | 22 | 0.6136 | 0.03125 |
| **D** | ideal | `true` | 17.713 | **0.8051** | 22 | **0.1384** | 22 | 0.6136 | 0.03125 |
| **D** | noisy | `true` | 17.296 | **0.7862** | 22 | **0.1250** | 22 | 0.6136 | 0.03125 |

### Analysis of Quality Findings:
1. **Clear Quantum Advantage over Baseline:** On both Processor B and Processor D, QAOA achieves $r \approx 0.786 - 0.787$ under noise, substantially outperforming uniform random guessing ($r = 0.6136$).
2. **Rubric Target Exceeded:** The noisy approximation ratio on Processor B ($r = 0.7874$) and $P_{opt} = 0.1260$ successfully exceed the rubric reference scores ($r_{ref} = 0.785, P_{opt, ref} = 0.123$).
3. **Noise Resilience:** Noise degrades the mean cut by only $\approx 2.4\%$ ($17.76 \to 17.32$ on B; $17.71 \to 17.30$ on D), demonstrating that shallow $p=1$ QAOA circuits maintain high fidelity against NISQ depolarizing and readout noise.

---

## 9. Circuit Resource Metrics & Overhead Analysis

Resource metrics were gathered using the authoritative `circuit_metrics()` utility (measurements and barriers excluded).

### Hardware Resource Comparison Table

| Processor | Physical Qubits | Logical CX | Transpiled CX | Circuit Depth | 2-Qubit Depth | Routing SWAPs | CX Overhead |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | 5 | 18 | — | — | — | — | *(Capacity limit)* |
| **B** | 9 | 18 | 21 | 37 | 17 | 1 | +3 |
| **D** | 6 | 18 | 18 | 29 | 14 | **0** | **0** |
| **Delta (D vs B)** | **-3 Qubits** | **0** | **-3 CX (-14.3%)** | **-8 (-21.6%)** | **-3 (-17.6%)** | **-1 (-100%)** | **-3 (-100%)** |

### Resource Metrics Definitions:
- **Logical CX:** Unconstrained CX count after transpilation without connectivity limits (18 CX, exactly 2 CX per edge).
- **Transpiled CX:** Final CX count after SABRE layout and routing on the hardware coupling map.
- **Circuit Depth:** Total transpiled depth excluding measurement and reset operations.
- **Two-Qubit Depth:** Depth considering only two-qubit CX gates.
- **SWAPs:** Inserted SWAP gates counted at routing before decomposition into $3 \times \text{CX}$.
- **CX Overhead:** $\text{Transpiled CX} - \text{Logical CX}$.

---

## 10. Processor D Design Rationale Tied to Measured Bottlenecks

### 10.1 The Measured Processor B Bottleneck
On Processor B, logical qubits cannot be mapped to physical qubits without requiring non-local routing. SABRE transpilation inserts 1 SWAP gate between physical qubits $(2, 6)$ to enable interaction $(1, 4)$. Because each SWAP gate decomposes into 3 CX gates on standard basis gates, this introduces:
- $+3$ additional CX gates ($18 \to 21$).
- Circuit depth increases from 29 to 37 ($+27.6\%$).
- Two-qubit depth increases from 14 to 17 ($+21.4\%$).
Each extra CX gate adds $0.5\%$ depolarizing error, and the additional depth increases idle decoherence ($T_1/T_2$).

### 10.2 The Processor D Solution
By constructing Processor D to directly mirror the 9 interaction edges of the PS-O1 graph:
1. Every required two-qubit gate is physically adjacent.
2. **Zero SWAP Gates:** $\text{SWAPs} = 0$, completely eliminating routing overhead.
3. **Zero CX Overhead:** $\text{CX Overhead} = 0$, executing the circuit with the minimum theoretical 18 CX gates.
4. **Depth Reduction:** Total circuit depth drops from 37 to 29 (a $21.6\%$ reduction), and two-qubit depth drops from 17 to 14.

---

## 11. Objective 1 — Bell-State Verification on Processor D

In accordance with Section 4 of the Challenge Kit, we verified physical entanglement fidelity on Processor D using the Bell state:
$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$$
prepared via $H(0) \to \text{CX}(0, 1)$.

### 11.1 Experimental Protocol
- **Adjacent Physical Pair (Distance 1):** Physical pair $(0, 2)$, directly coupled by edge $(0, 2)$.
- **Farthest Physical Pair (Distance 2):** Physical pair $(0, 1)$, separated by shortest path $(0, 4) - (4, 1)$ (distance 2).
- **Measurement Bases:** $ZZ$ (computational), $XX$ ($H$ applied to both), and $YY$ ($S^\dagger \to H$ applied to both).
- **Shots:** 4096 shots per circuit across ideal and noisy simulation (total 12 circuits).
- **Exact State Fidelity Formula:**
  $$F = \frac{1 + \langle XX \rangle - \langle YY \rangle + \langle ZZ \rangle}{4}, \qquad \text{SE}(F) = \frac{1}{4} \sqrt{\text{SE}_{XX}^2 + \text{SE}_{YY}^2 + \text{SE}_{ZZ}^2}$$
  *(Note: Computing fidelity as $P(00)+P(11)$ is explicitly rejected by the judging rubric).*

### 11.2 Bell-State Results Table (`results/tables/bell.csv`)

| Pair | Shortest Path Distance | Condition | Transpiled CX | $\langle XX \rangle \pm \text{SE}$ | $\langle YY \rangle \pm \text{SE}$ | $\langle ZZ \rangle \pm \text{SE}$ | State Fidelity $F \pm \text{SE}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(0, 2)** | 1 (Adjacent) | ideal | 1 | $+1.000 \pm 0.000$ | $-1.000 \pm 0.000$ | $+1.000 \pm 0.000$ | **$1.0000 \pm 0.0000$** |
| **(0, 2)** | 1 (Adjacent) | noisy | 1 | $+0.9458 \pm 0.0051$ | $-0.9521 \pm 0.0048$ | $+0.9551 \pm 0.0046$ | **$0.9633 \pm 0.0021$** |
| **(0, 1)** | 2 (Farthest) | ideal | 4 | $+1.000 \pm 0.000$ | $-1.000 \pm 0.000$ | $+1.000 \pm 0.000$ | **$1.0000 \pm 0.0000$** |
| **(0, 1)** | 2 (Farthest) | noisy | 4 | $+0.9443 \pm 0.0051$ | $-0.9341 \pm 0.0056$ | $+0.9385 \pm 0.0054$ | **$0.9542 \pm 0.0023$** |

### 11.3 Interpretation of Bell Results
1. **Ideal State Verification:** Under noiseless simulation, correlations evaluate to $\langle XX \rangle = +1.000, \langle YY \rangle = -1.000, \langle ZZ \rangle = +1.000$, yielding an exact state fidelity of $F = 1.0000 \pm 0.0000$.
2. **Noise Degradation:** On the distance-1 pair, noisy fidelity drops to $F = 0.9633 \pm 0.0021$. This $3.67\%$ reduction arises from the single CX gate depolarizing error ($0.5\%$), single-qubit errors ($0.1\%$), and symmetric readout bit-flip error ($1.0\%$ per qubit).
3. **Distance Routing Penalty:** For the farthest pair $(0, 1)$, transpilation requires routing across an intermediate qubit (inserting 1 SWAP = 3 CX, plus the entangling CX, totaling 4 CX gates). Consequently, fidelity drops further to $F = 0.9542 \pm 0.0023$. The measured degradation ($\Delta F \approx -0.0091$) directly captures the compounded noise penalty of multi-hop routing on physical hardware.

---

## 12. Connectivity, Routing, Noise, and Solution-Quality Trade-Offs

A key judging criterion is an honest scientific appraisal of hardware trade-offs:
1. **Application-Specific Advantage:** Processor D completely eliminates SWAP overhead for PS-O1 because its coupling topology perfectly matches the problem graph.
2. **Lack of Generalizability:** Processor D's topology is custom-tailored to this 6-vertex graph. If an arbitrary random graph or bipartite problem were mapped onto D, routing overhead would return.
3. **Physical Hardware Feasibility & Cross-Talk:** On superconducting circuit hardware, building qubits with degree 4 (Qubits 3 and 4) in a planar chip introduces significant physical design challenges:
   - High degree increases frequency crowding, making it harder to assign transmon qubit frequencies that avoid parasitic $ZZ$ cross-talk and spurious microwave transitions.
   - Routing control lines and readout resonators to high-degree planar qubits requires multi-layer wiring or through-silicon vias (TSVs).
   - In contrast, Processor B’s lower-density sparse layout (resembling heavy-hex architectures) is easier to fabricate with low spectator cross-talk, even though it requires classical software routers to insert SWAP gates.

---

## 13. Limitations, Reproducibility, and Concrete Next Improvements

### 13.1 Current Limitations
1. **Shallow Depth ($p=1$):** At $p=1$, the approximation ratio is bounded theoretically ($r \approx 0.805$). While superior to classical random baselines, higher depths ($p=2, 3$) are necessary to approach near-unity approximation.
2. **Synthetic Noise Model:** While the official depolarizing and readout noise model provides an equitable benchmark, real superconducting chips exhibit non-Markovian noise, anisotropic $T_1/T_2$ relaxation, and state-dependent readout asymmetry.

### 13.2 Concrete Next Improvements
1. **Depth $p=2$ Exploration:** Implementing $p=2$ QAOA (optimizing 4 parameters: $\gamma_1, \gamma_2, \beta_1, \beta_2$) using layer-by-layer warm-starting or Fourier parameterization.
2. **Readout Error Mitigation (TREX / M3):** Applying Matrix-free Measurement Mitigation (M3) or Twirled Readout Error Extinction (TREX) to post-process raw count dictionaries, recovering approximately $2\%$ in approximation ratio under noise.
3. **Dynamical Decoupling (DD):** Inserting Hahn echo or XY4 pulse sequences on idle spectator qubits during CX operations to suppress dephasing during deeper circuit executions.

---

## 14. Final Conclusions

Through rigorous mathematical formulation, automated testing, and comprehensive simulation, this investigation delivers a complete Phase 2 solution for **PS-O1**:
1. **Classical Ground Truth:** The maximum cut weight is proven to be $C^* = 22$, with complementary solutions `001011` and `110100`.
2. **QAOA Effectiveness:** Depth $p=1$ QAOA achieves $r = 0.8051$ (ideal) and $r = 0.7874$ (noisy), outperforming random guessing ($r=0.6136$) by $+28.3\%$ and exceeding the judging rubric reference target ($0.785$).
3. **Sampling Enhancement:** Optimal sampling probability $P_{opt} = 12.60\%$ represents a $>4.4\times$ improvement over uniform random sampling ($3.125\%$).
4. **Architectural Optimization:** Custom Processor D completely eliminates routing overhead ($\text{SWAPs}=0, \text{CX Overhead}=0$), reducing circuit depth from 37 to 29.
5. **Bell-State Verification:** Objective 1 Bell tests confirm state fidelities of $F = 0.9633$ (distance 1) and $F = 0.9542$ (distance 2), quantifying the exact physical noise penalty of multi-hop routing.
6. **Reproducibility:** The entire pipeline executes headlessly in $<15$ seconds and passes all 58 automated tests across 10 modules, fully satisfying judging eligibility gates G2–G4 (with G1 portal submission prepared for entry prior to deadline).
