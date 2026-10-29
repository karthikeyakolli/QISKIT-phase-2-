"""Basic package import and foundation integrity tests.

Supports both pytest and standard unittest runner.
"""

import importlib
import json
from pathlib import Path
import sys
import unittest

# Ensure src/ is on sys.path for test discovery
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestFoundationImports(unittest.TestCase):
    """Integrity tests for maxcut package foundation and configuration."""

    def test_package_root_import(self):
        """Verify that root maxcut package is importable and has version."""
        import maxcut

        self.assertTrue(hasattr(maxcut, "__version__"))
        self.assertIsInstance(maxcut.__version__, str)
        self.assertGreaterEqual(len(maxcut.__all__), 9)
        self.assertIn("io", maxcut.__all__)

    def test_subpackages_import(self):
        """Verify that all architectural layer subpackages are importable."""
        layers = [
            "maxcut.io",
            "maxcut.domain",
            "maxcut.classical",
            "maxcut.formulation",
            "maxcut.qaoa",
            "maxcut.backends",
            "maxcut.experiments",
            "maxcut.metrics",
            "maxcut.visualization",
        ]
        for layer in layers:
            module = importlib.import_module(layer)
            self.assertIsNotNone(module, f"Failed to import {layer}")

    def test_core_dependencies_available(self):
        """Verify that core quantum and scientific dependencies are importable."""
        import qiskit
        import qiskit_aer
        import qiskit_ibm_runtime
        import networkx
        import numpy
        import scipy
        import pandas
        import matplotlib
        import nbformat
        import nbclient
        import ipykernel

        self.assertTrue(qiskit.__version__.startswith("2."))
        self.assertIsNotNone(qiskit_aer.__version__)
        self.assertIsNotNone(qiskit_ibm_runtime.__version__)
        self.assertIsNotNone(networkx.__version__)
        self.assertIsNotNone(nbformat.__version__)
        self.assertIsNotNone(nbclient.__version__)
        self.assertIsNotNone(ipykernel.__version__)

    def test_experiment_config_validity(self):
        """Verify that configs/experiment.json exists and contains required keys."""
        config_path = Path(__file__).resolve().parent.parent / "configs" / "experiment.json"
        self.assertTrue(config_path.exists(), "configs/experiment.json does not exist")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.assertIn("project", config)
        self.assertIn("track", config)
        self.assertIn("reproducibility", config)
        self.assertEqual(config["reproducibility"]["seed_global"], 42)
        self.assertIn("qaoa", config)
        self.assertIn("execution", config)
        self.assertIn("io", config)


if __name__ == "__main__":
    unittest.main()
