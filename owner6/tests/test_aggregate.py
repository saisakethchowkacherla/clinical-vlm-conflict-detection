from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from owner6_rigor.aggregate import summarize_metrics


class AggregateTests(unittest.TestCase):
    def test_summarize_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "rows_scored": 4,
                        "baseline": {
                            "baseline_valid": True,
                            "accuracy": {"rate": 1.0},
                        },
                        "negative_flip": {
                            "conflict_A_B": {"rate": 0.5},
                            "control_C_D": {"rate": 0.0},
                            "gap": 0.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_metrics([path])
            self.assertIn("50.0%", summary)
            self.assertIn("Baseline Valid", summary)


if __name__ == "__main__":
    unittest.main()
