"""Unit and integration tests for the reproducible QAOA experiment framework."""

import json
from pathlib import Path
import tempfile
import unittest

from maxcut.backends import (
    AerBackendRunner,
    BaseBackendRunner,
    ProcessorBackendRunner,
    create_backend_runner,
)
from maxcut.domain.model import MaxCutGraph
from maxcut.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    load_experiment_result,
    save_experiment_result,
)
from maxcut.io import ProcessorHardwareSpec


class TestExperimentFramework(unittest.TestCase):
    """Test suite for ExperimentConfig, ExperimentResult schema, and ExperimentRunner."""

    def setUp(self):
        # 3-node unweighted triangle graph
        self.triangle_graph = MaxCutGraph(
            name="Triangle_Test",
            nodes=[0, 1, 2],
            edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        )

        # Realistic mock hardware spec for Processor A (linear topology: 0-1-2-3-4)
        self.proc_a_spec = ProcessorHardwareSpec(
            name="Processor_A",
            num_qubits=5,
            coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)],
            basis_gates=["rz", "sx", "x", "cx"],
            qubit_properties={"t1_us_mean": 200.0, "t2_us_mean": 120.0},
        )

        # Realistic mock hardware spec for Processor B (sparse tree topology)
        self.proc_b_spec = ProcessorHardwareSpec(
            name="Processor_B",
            num_qubits=9,
            coupling_map=[
                (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (2, 4), (4, 2),
                (4, 5), (5, 4), (5, 6), (6, 5), (4, 7), (7, 4), (7, 8), (8, 7)
            ],
            basis_gates=["rz", "sx", "x", "cx"],
            qubit_properties={"t1_us_mean": 250.0, "t2_us_mean": 150.0},
        )

    def test_experiment_config_roundtrip_serialization(self):
        """Verify that ExperimentConfig serializes and deserializes cleanly without data loss."""
        cfg = ExperimentConfig(
            graph_name="Triangle_Test",
            problem_instance_path="instances/challenge/o1_maxcut_instance.json",
            depth_p=2,
            shots=1024,
            optimizer="SLSQP",
            initial_parameters=[0.5, 1.0, 0.2, 0.4],
            optimization_settings={"max_iterations": 50, "tolerance": 1e-5},
            random_seeds={"seed_global": 123, "seed_simulator": 123, "seed_transpiler": 123},
            backend_name="aer_simulator",
            transpilation_level=2,
            output_location="results/data",
        )

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            cfg_path = f.name

        try:
            cfg.to_json(cfg_path)
            loaded_cfg = ExperimentConfig.from_json(cfg_path)

            self.assertEqual(loaded_cfg.graph_name, "Triangle_Test")
            self.assertEqual(loaded_cfg.depth_p, 2)
            self.assertEqual(loaded_cfg.shots, 1024)
            self.assertEqual(loaded_cfg.optimizer, "SLSQP")
            self.assertEqual(loaded_cfg.initial_parameters, [0.5, 1.0, 0.2, 0.4])
            self.assertEqual(loaded_cfg.random_seeds["seed_global"], 123)
            self.assertEqual(loaded_cfg.transpilation_level, 2)
        finally:
            Path(cfg_path).unlink(missing_ok=True)

    def test_schema_field_completeness(self):
        """Verify that ExperimentResult contains all required fields per specification."""
        cfg = ExperimentConfig(
            graph_name="Triangle_Test",
            depth_p=1,
            shots=500,
            optimizer="COBYLA",
            optimization_settings={"max_iterations": 20, "tolerance": 1e-4},
            random_seeds={"seed_global": 42, "seed_simulator": 42, "seed_transpiler": 42},
            backend_name="aer_simulator",
        )

        runner = ExperimentRunner()
        result = runner.run(config=cfg, graph=self.triangle_graph, save_result=False)

        # Check top-level schema fields
        self.assertIsInstance(result, ExperimentResult)
        self.assertIn("name", result.problem_metadata)
        self.assertIn("backend_name", result.backend_metadata)
        self.assertIn("optimal_parameters", result.algorithm_parameters)
        self.assertGreater(result.cut_value, 0.0)
        self.assertIsNotNone(result.best_measured_bitstring)
        self.assertIsNotNone(result.most_frequent_bitstring)
        self.assertGreater(result.expected_cut, 0.0)

        # Check probabilities, runtime, circuit metrics, and reproducibility
        self.assertIn("num_unique_bitstrings", result.probability_metrics)
        self.assertIn("total_pipeline_time_seconds", result.runtime)
        self.assertIn("transpiled_depth", result.circuit_metrics)
        self.assertIn("transpiled_2q_gates", result.circuit_metrics)
        self.assertIn("raw_counts", result.to_dict())
        self.assertIn("decoded_counts", result.to_dict())
        self.assertIn("start_time_iso", result.timestamps)
        self.assertIn("random_seeds", result.reproducibility_information)
        self.assertIn("qiskit_version", result.reproducibility_information)

    def test_result_roundtrip_serialization(self):
        """Verify saving and loading ExperimentResult produces exact replica."""
        cfg = ExperimentConfig(
            graph_name="Triangle_Test",
            depth_p=1,
            shots=400,
            optimization_settings={"max_iterations": 15},
        )
        runner = ExperimentRunner()
        result = runner.run(config=cfg, graph=self.triangle_graph, save_result=False)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            res_path = f.name

        try:
            save_experiment_result(result, res_path)
            loaded_result = load_experiment_result(res_path)

            self.assertEqual(loaded_result.problem_metadata["name"], result.problem_metadata["name"])
            self.assertEqual(loaded_result.cut_value, result.cut_value)
            self.assertEqual(loaded_result.best_measured_bitstring, result.best_measured_bitstring)
            self.assertEqual(loaded_result.raw_counts, result.raw_counts)
            self.assertEqual(loaded_result.decoded_counts, result.decoded_counts)
            self.assertEqual(
                loaded_result.algorithm_parameters["optimal_parameters"],
                result.algorithm_parameters["optimal_parameters"],
            )

            # Summary formatting test
            summary = loaded_result.summary()
            self.assertIn("Experiment Result: Triangle_Test", summary)
            self.assertIn("CIRCUIT COMPILATION METRICS", summary)
        finally:
            Path(res_path).unlink(missing_ok=True)

    def test_runner_reusability_across_backends(self):
        """Verify that the SAME ExperimentRunner executes across Simulator, Processor A, and Processor B."""
        runner = ExperimentRunner()
        cfg = ExperimentConfig(
            graph_name="Triangle_Test",
            depth_p=1,
            shots=400,
            optimization_settings={"max_iterations": 15},
            random_seeds={"seed_global": 42, "seed_simulator": 42, "seed_transpiler": 42},
        )

        # 1. Target: Aer Simulator
        aer_backend = AerBackendRunner(seed_simulator=42, seed_transpiler=42)
        res_aer = runner.run(config=cfg, graph=self.triangle_graph, backend=aer_backend, save_result=False)
        self.assertEqual(res_aer.backend_metadata["target_type"], "simulator")
        self.assertEqual(res_aer.backend_metadata["backend_name"], "aer_simulator")

        # 2. Target: Processor A (linear topology)
        proc_a_backend = ProcessorBackendRunner(spec=self.proc_a_spec, seed_simulator=42, seed_transpiler=42)
        res_proc_a = runner.run(config=cfg, graph=self.triangle_graph, backend=proc_a_backend, save_result=False)
        self.assertEqual(res_proc_a.backend_metadata["target_type"], "hardware_processor")
        self.assertEqual(res_proc_a.backend_metadata["backend_name"], "Processor_A")
        self.assertGreaterEqual(res_proc_a.circuit_metrics["transpiled_2q_gates"], 1)

        # 3. Target: Processor B (T-shape topology)
        proc_b_backend = ProcessorBackendRunner(spec=self.proc_b_spec, seed_simulator=42, seed_transpiler=42)
        res_proc_b = runner.run(config=cfg, graph=self.triangle_graph, backend=proc_b_backend, save_result=False)
        self.assertEqual(res_proc_b.backend_metadata["target_type"], "hardware_processor")
        self.assertEqual(res_proc_b.backend_metadata["backend_name"], "Processor_B")
        self.assertGreaterEqual(res_proc_b.circuit_metrics["transpiled_2q_gates"], 1)

        # Structural consistency check: all 3 results share identical schema keys
        self.assertEqual(set(res_aer.to_dict().keys()), set(res_proc_a.to_dict().keys()))
        self.assertEqual(set(res_proc_a.to_dict().keys()), set(res_proc_b.to_dict().keys()))

    def test_structural_consistency_between_runs(self):
        """Verify that identical configurations produce structurally consistent output JSON files."""
        cfg = ExperimentConfig(
            graph_name="Triangle_Test",
            depth_p=1,
            shots=500,
            optimization_settings={"max_iterations": 20},
            random_seeds={"seed_global": 99, "seed_simulator": 99, "seed_transpiler": 99},
        )
        runner = ExperimentRunner()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg.output_location = tmp_dir
            res1 = runner.run(config=cfg, graph=self.triangle_graph, save_result=True)
            res2 = runner.run(config=cfg, graph=self.triangle_graph, save_result=True)

            dict1 = res1.to_dict()
            dict2 = res2.to_dict()

            # Schema keys must match identically
            self.assertEqual(dict1.keys(), dict2.keys())
            self.assertEqual(dict1["problem_metadata"].keys(), dict2["problem_metadata"].keys())
            self.assertEqual(dict1["backend_metadata"].keys(), dict2["backend_metadata"].keys())
            self.assertEqual(dict1["algorithm_parameters"].keys(), dict2["algorithm_parameters"].keys())
            self.assertEqual(dict1["circuit_metrics"].keys(), dict2["circuit_metrics"].keys())

            # Seeded values must be identical
            self.assertEqual(
                dict1["algorithm_parameters"]["optimal_parameters"],
                dict2["algorithm_parameters"]["optimal_parameters"],
            )
            self.assertEqual(dict1["cut_value"], dict2["cut_value"])
            self.assertEqual(dict1["raw_counts"], dict2["raw_counts"])


if __name__ == "__main__":
    unittest.main()
