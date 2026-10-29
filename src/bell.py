"""
src/bell.py - Official Bell-state verification experiment on Processor D.
"""
from __future__ import annotations
import csv
import os
from typing import Dict, List, Tuple, Any
import numpy as np

from qff_phase2 import (
    load_processor,
    coupling_map_from_edges,
    bell_experiment,
    SHOTS,
)


def find_bell_pairs(processor: dict) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Identify one distance-1 (adjacent) physical pair and one maximum-distance physical pair in the processor graph."""
    cm = coupling_map_from_edges(processor["edges"], processor["num_qubits"])
    n = processor["num_qubits"]
    
    # 1. Adjacent pair (distance 1)
    adj_pair = None
    for a, b in processor["edges"]:
        if cm.distance(a, b) == 1:
            adj_pair = (a, b)
            break
            
    # 2. Maximum distance pair
    max_dist = -1
    farthest_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            d = cm.distance(i, j)
            if d > max_dist:
                max_dist = d
                farthest_pair = (i, j)
                
    if adj_pair is None or farthest_pair is None:
        raise ValueError(f"Could not find valid Bell pairs in {processor['name']}")
        
    return adj_pair, farthest_pair


def run_bell_state_suite(
    processor_d_path: str = "processors/processor_D.json",
    shots: int = SHOTS,
) -> List[Dict[str, Any]]:
    """Run the 12 required Bell-state circuits on Processor D:
    2 pairs (adjacent & farthest) x 2 conditions (ideal & noisy) x 3 bases (XX, YY, ZZ).
    """
    D = load_processor(processor_d_path)
    adj_pair, far_pair = find_bell_pairs(D)
    
    pairs_to_test = [adj_pair, far_pair]
    bell_rows: List[Dict[str, Any]] = []
    
    for pair in pairs_to_test:
        for noisy in (False, True):
            res = bell_experiment(D, physical_pair=pair, noisy=noisy, shots=shots)
            cond_str = "noisy" if noisy else "ideal"
            bell_rows.append({
                "pair": f"({pair[0]},{pair[1]})",
                "physical_qubit_0": pair[0],
                "physical_qubit_1": pair[1],
                "pair_distance": res["pair_distance"],
                "condition": cond_str,
                "transpiled_cx": res.get("cx", 1),
                "xx": round(res["XX"], 4),
                "xx_se": round(res["XX_se"], 4),
                "yy": round(res["YY"], 4),
                "yy_se": round(res["YY_se"], 4),
                "zz": round(res["ZZ"], 4),
                "zz_se": round(res["ZZ_se"], 4),
                "fidelity": round(res["fidelity"], 4),
                "fidelity_se": round(res["fidelity_se"], 4),
            })
            
    return bell_rows


def write_bell_csv(
    bell_rows: List[Dict[str, Any]],
    output_path: str = "results/tables/bell.csv",
) -> None:
    """Write the Bell results table in the required format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "pair",
        "pair_distance",
        "condition",
        "transpiled_cx",
        "xx",
        "xx_se",
        "yy",
        "yy_se",
        "zz",
        "zz_se",
        "fidelity",
        "fidelity_se",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in bell_rows:
            writer.writerow(row)
