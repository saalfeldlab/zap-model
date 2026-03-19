#!/usr/bin/env python
"""Setup check: verify that the environment is configured correctly.

Checks:
1. local_paths.py has been created by the user
2. local_paths_test.py passes
3. neuprint can be queried via fishfuncem
"""

import subprocess
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "zap_model"


class TestSetup(unittest.TestCase):
    def test_local_paths_exists(self):
        """local_paths.py has been created by the user."""
        local_paths_file = SRC_DIR / "local_paths.py"
        self.assertTrue(
            local_paths_file.is_file(),
            f"Copy local_paths.example.py to {local_paths_file} and fill in your paths.",
        )

    def test_local_paths_test_passes(self):
        """local_paths_test.py passes."""
        local_paths_file = SRC_DIR / "local_paths.py"
        self.assertTrue(local_paths_file.is_file(), "Skipped: local_paths.py missing")

        subprocess.check_output(
            [sys.executable, "-m", "unittest", str(SRC_DIR / "local_paths_test.py"), "-v"],
            stderr=subprocess.STDOUT,
        )

    def test_neuprint_query(self):
        """neuprint can be queried via fishfuncem."""
        from fishfuncem.em.NeuprintServer import NeuprintServer

        server = NeuprintServer()
        client = server.client
        neuron_data = client.fetch_custom(
            """
            MATCH (n:Neuron)
            RETURN n.bodyId AS bid, n.type AS type
            LIMIT 3
            """,
            format="pandas",
        )
        self.assertGreater(len(neuron_data), 0)


if __name__ == "__main__":
    unittest.main()
