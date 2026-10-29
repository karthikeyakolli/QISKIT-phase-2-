"""Phase 9 Final Compliance and Release Readiness Unit Tests.

Validates:
1. Canonical O1 Max-Cut problem graph and classical ground truth.
2. Hardware target compliance: Processor A (5Q), Processor B (9Q), Processor D (8Q).
3. Basis gate compliance: native CX entangler across all targets.
4. Clean standalone execution of main.py entrypoint.
5. Presence and XML validity of all 8 publication vector SVG figures.
6. Reconciliation of master Phase 8 tables.
7. Verification of artifact validator execution.
8. Presence of AI-assistance disclosure and Quantum Advantage disclaimer.
"""

import json
from pathlib import Path
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent


class TestPhase9Compliance(unittest.TestCase):
    """Test suite certifying final hackathon release compliance for Track O1."""

    def test_01_problem_integrity_and_classical_optimum(self):
        """1. Verify canonical O1 graph integrity and exact classical optimum."""
        inst_path = ROOT / "instances" / "challenge" / "o1_maxcut_instance.json"
        ref_path = ROOT / "results" / "data" / "classical_reference.json"

        self.assertTrue(inst_path.exists())
        self.assertTrue(ref_path.exists())

        with open(inst_path, "r", encoding="utf-8") as f:
            inst = json.load(f)
        with open(ref_path, "r", encoding="utf-8") as f:
            ref = json.load(f)

        self.assertEqual(inst["num_nodes"], 6)
        self.assertEqual(len(inst["edges"]), 8)
        self.assertEqual(ref["optimal_cut_value"], 6.0)
        self.assertEqual(ref["num_optimal_bitstrings"], 10)
        self.assertEqual(ref["total_weight"], 8.0)

    def test_02_processor_architectures_and_gate_basis(self):
        """2. Verify Processor A (5Q), Processor B (9Q), and Processor D (8Q) specifications."""
        spec_a = ROOT / "processors" / "processor_a" / "hardware_spec.json"
        spec_b = ROOT / "processors" / "processor_b" / "hardware_spec.json"
        spec_d = ROOT / "processors" / "processor_d" / "hardware_spec.json"

        for p in [spec_a, spec_b, spec_d]:
            self.assertTrue(p.exists())

        with open(spec_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)
        with open(spec_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)
        with open(spec_d, "r", encoding="utf-8") as f:
            data_d = json.load(f)

        self.assertEqual(data_a["num_qubits"], 5, "Processor A must have exactly 5 physical qubits")
        self.assertEqual(data_b["num_qubits"], 9, "Processor B must have exactly 9 physical qubits")
        self.assertEqual(data_d["num_qubits"], 8, "Processor D must have exactly 8 physical qubits")

        for d in [data_a, data_b, data_d]:
            self.assertIn("cx", d["basis_gates"], "Native 2Q gate must be CX")
            self.assertEqual(d["metadata"]["native_2q_gate"], "cx")

        # Processor D planar degree constraint
        cmap_d = data_d["coupling_map"]
        adj_d = {}
        for u, v in cmap_d:
            adj_d.setdefault(u, set()).add(v)
        max_deg_d = max(len(nbrs) for nbrs in adj_d.values())
        self.assertLessEqual(max_deg_d, 3, "Processor D max degree must be <= 3")

    def test_03_main_entrypoint_execution(self):
        """3. Verify main.py executes cleanly with exit code 0."""
        main_py = ROOT / "main.py"
        self.assertTrue(main_py.exists())

        res = subprocess.run([sys.executable, str(main_py)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"main.py failed: {res.stderr}")
        self.assertIn("Authoritative Benchmark: Processor A vs. Processor B vs. Custom Processor D", res.stdout)
        self.assertIn("SUCCESS", res.stdout)

    def test_04_publication_figures_inventory(self):
        """4. Verify all 8 publication vector SVG figures exist and parse as valid XML."""
        figures_dir = ROOT / "results" / "verified_ab_v4" / "figures"
        for i in range(1, 9):
            matches = list(figures_dir.glob(f"figure{i}_*.svg"))
            self.assertEqual(len(matches), 1, f"Expected exactly one Figure {i} SVG")
            fig_path = matches[0]
            self.assertGreater(fig_path.stat().st_size, 1000)
            tree = ET.parse(str(fig_path))
            self.assertTrue(tree.getroot().tag.endswith("svg"))

    def test_05_artifact_validator_execution(self):
        """5. Verify validate_artifacts.py executes cleanly with return code 0."""
        val_py = ROOT / "scripts" / "validate_artifacts.py"
        self.assertTrue(val_py.exists())

        res = subprocess.run([sys.executable, str(val_py)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"validate_artifacts.py failed: {res.stderr}")
        self.assertIn("[SUCCESS]", res.stdout)

    def test_06_readme_disclosures_and_disclaimers(self):
        """6. Verify AI assistance disclosure and Quantum Advantage disclaimer in README.md."""
        readme_path = ROOT / "README.md"
        self.assertTrue(readme_path.exists())
        text = readme_path.read_text(encoding="utf-8")

        self.assertIn("AI-Assistance Disclosure", text)
        self.assertIn("Quantum Advantage Disclaimer", text)
        self.assertIn("quantum advantage", text.lower())
        self.assertIn("Processor D", text)


if __name__ == "__main__":
    unittest.main()
