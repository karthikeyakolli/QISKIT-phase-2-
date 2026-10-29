"""Test suite validating that the 10-gate pre-hardware audit passes cleanly."""

import unittest

from scripts.run_validation_audit import run_full_audit


class TestValidationAuditGate(unittest.TestCase):
    """Test suite executing the 10 pre-hardware validation gates."""

    def test_full_10_gate_audit(self):
        """Verifies that all 10 gates pass without exception or discrepancy."""
        results, meta = run_full_audit()
        self.assertEqual(len(results), 10)
        self.assertTrue(meta["all_gates_passed"])

        for gate in results:
            with self.subTest(gate=gate.gate_id, name=gate.name):
                self.assertTrue(
                    gate.passed,
                    f"Gate {gate.gate_id} ({gate.name}) failed! Expected: {gate.expected}, Actual: {gate.actual}",
                )


if __name__ == "__main__":
    unittest.main()
