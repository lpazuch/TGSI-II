from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTest(unittest.TestCase):
    def test_dry_run_creates_report(self) -> None:
        env = dict(os.environ)
        pythonpath = str(ROOT / "src")
        env["PYTHONPATH"] = pythonpath if not env.get("PYTHONPATH") else f"{pythonpath}:{env['PYTHONPATH']}"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tgsi_pipeline.cli",
                "--config",
                str(ROOT / "configs" / "pipeline.example.json"),
                "--dry-run",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report_path = ROOT / "data" / "run_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["dry_run"])


if __name__ == "__main__":
    unittest.main()
