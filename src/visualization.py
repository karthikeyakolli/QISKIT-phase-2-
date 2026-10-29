"""
src/visualization.py - Publication-grade figures for PS-O1.
"""
from __future__ import annotations
import os
from typing import Dict, List, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from qff_phase2 import load_processor
from src.formulation import LOCKED_EDGES, NUM_VERTICES, EXACT_OPTIMUM, random_baseline


def plot_weighted_graph(output_path: str = "results/figures/weighted_graph.png") -> None:
    """Plot the locked 6-vertex graph with cut edges highlighted and partition colors."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    G = nx.Graph()
    for u, v, w in LOCKED_EDGES:
        G.add_edge(u, v, weight=w)
        
    # Optimal partition: Group 0 = {0, 1, 3}, Group 1 = {2, 4, 5}
    # Corresponding to bitstring '001011' (or '110100')
    group0 = {0, 1, 3}
    group1 = {2, 4, 5}
    
    pos = nx.spring_layout(G, seed=42, k=1.2)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Draw cut edges vs uncut edges
    cut_edges = [(u, v) for u, v, w in LOCKED_EDGES if (u in group0 and v in group1) or (u in group1 and v in group0)]
    uncut_edges = [(u, v) for u, v, w in LOCKED_EDGES if (u in group0 and v in group0) or (u in group1 and v in group1)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=list(group0), node_color="#2b5c8f", node_size=800, label="Partition S (Nodes 0, 1, 3)", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=list(group1), node_color="#d95f02", node_size=800, label="Partition S' (Nodes 2, 4, 5)", ax=ax)
    
    # Cut edges (dashed red/crimson, thicker)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, width=2.8, edge_color="#b2182b", style="solid", ax=ax, label="Cut Edges (Weight = 22)")
    # Uncut edges (light gray, dotted)
    nx.draw_networkx_edges(G, pos, edgelist=uncut_edges, width=1.5, edge_color="#999999", style="dashed", ax=ax, label="Uncut Edges (Weight = 5)")
    
    # Edge labels
    edge_labels = {(u, v): f"w={w}" for u, v, w in LOCKED_EDGES}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_weight="bold", ax=ax)
    
    # Node labels
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=12, ax=ax)
    
    ax.set_title("PS-O1: Locked 6-Vertex Network Partitioning (Max-Cut C* = 22)", fontsize=14, pad=15, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_processor_d_topology(
    processor_d_path: str = "processors/processor_D.json",
    output_path: str = "results/figures/processor_D_topology.png"
) -> None:
    """Plot the custom 6-qubit Processor D coupling graph."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    D = load_processor(processor_d_path)
    
    G = nx.Graph()
    G.add_nodes_from(range(D["num_qubits"]))
    for u, v in D["edges"]:
        G.add_edge(u, v)
        
    pos = nx.spring_layout(G, seed=123, k=1.0)
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    degrees = dict(G.degree())
    node_colors = ["#1b7837" if d == 4 else "#7fbf7b" for _, d in degrees.items()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.2, edge_color="#2166ac", ax=ax)
    
    labels = {q: f"Q{q}\n(deg {degrees[q]})" for q in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="black", font_weight="bold", font_size=9, ax=ax)
    
    ax.set_title("Processor D Custom Architecture (6 Qubits, 9 Coupling Edges, Max Deg <= 4)", fontsize=12, pad=15, fontweight="bold")
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_quality_comparison(
    results_rows: List[Dict[str, Any]],
    output_path: str = "results/figures/quality_comparison.png"
) -> None:
    """Plot solution quality comparison across processors and conditions vs baselines."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter executable rows on B and D
    valid_rows = [r for r in results_rows if r.get("executable") == "true" or r.get("executable") is True]
    
    labels = []
    approx_ratios = []
    p_opts = []
    
    for r in valid_rows:
        label = f"Proc {r['processor']}\n({r['condition']})"
        labels.append(label)
        approx_ratios.append(float(r["quality_value"]))
        p_opts.append(float(r["p_opt"]))
        
    rand = random_baseline()
    rand_r = rand["approx_ratio"]
    rand_p_opt = rand["p_opt"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=300)
    
    x = np.arange(len(labels))
    width = 0.55
    
    # Subplot 1: Approximation Ratio r
    bars1 = ax1.bar(x, approx_ratios, width, color=["#4575b4", "#74add1", "#313695", "#35978f"])
    ax1.axhline(rand_r, color="#d73027", linestyle="--", linewidth=1.8, label=f"Random Baseline (r = {rand_r:.3f})")
    ax1.axhline(1.0, color="#1a9850", linestyle=":", linewidth=1.5, label="Exact Optimum (r = 1.000)")
    ax1.axhline(0.785, color="#f46d43", linestyle="-.", linewidth=1.2, label="Rubric Ref (r = 0.785)")
    ax1.set_ylabel("Approximation Ratio r = E[C] / C*", fontweight="bold")
    ax1.set_title("Primary Solution Quality (Approximation Ratio)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="lower right", fontsize=8)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    # Subplot 2: Optimal Sampling Probability P_opt
    bars2 = ax2.bar(x, p_opts, width, color=["#4575b4", "#74add1", "#313695", "#35978f"])
    ax2.axhline(rand_p_opt, color="#d73027", linestyle="--", linewidth=1.8, label=f"Random Baseline (P_opt = {rand_p_opt:.4f})")
    ax2.axhline(0.123, color="#f46d43", linestyle="-.", linewidth=1.2, label="Rubric Ref (P_opt = 0.123)")
    ax2.set_ylabel("Optimal Solution Probability P_opt", fontweight="bold")
    ax2.set_title("Optimal-Solution Sampling Probability", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(p_opts) * 1.3)
    ax2.legend(loc="upper right", fontsize=8)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_resource_comparison(
    resources: Dict[str, Any],
    output_path: str = "results/figures/resource_comparison.png"
) -> None:
    """Plot resource metric comparison between Processor B and Processor D."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    b_res = resources["B"]
    d_res = resources["D"]
    
    categories = ["Transpiled CX", "Circuit Depth", "Two-Qubit Depth", "SWAPs", "CX Overhead"]
    b_vals = [b_res["cx"], b_res["depth"], b_res["two_q_depth"], b_res["swaps"], b_res["cx_overhead"]]
    d_vals = [d_res["cx"], d_res["depth"], d_res["two_q_depth"], d_res["swaps"], d_res["cx_overhead"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    rects1 = ax.bar(x - width/2, b_vals, width, label="Processor B (9 Qubits)", color="#d95f02")
    rects2 = ax.bar(x + width/2, d_vals, width, label="Processor D (Custom 6 Qubits)", color="#1b7837")
    
    ax.set_ylabel("Count / Depth", fontweight="bold")
    ax.set_title("Hardware Execution Resource Overhead Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.legend(frameon=True)
    
    for rects in [rects1, rects2]:
        for bar in rects:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.4, str(yval), ha="center", va="bottom", fontsize=9, fontweight="bold")
            
    ax.set_ylim(0, max(max(b_vals), max(d_vals)) * 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
