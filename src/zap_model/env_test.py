"""Tests that configured environment paths are accessible.

All tests skip when the relevant env var is not set, so the suite passes
in CI and on machines without local data.
"""

import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_LOCAL = os.environ.get("ZAPBENCH_LOCAL_PATH")
_CEI = os.environ.get("ZAP_CELL_EPHYS_INDEX_PATH")
_NEUPRINT = os.environ.get("NEUPRINT_DOWNLOAD_DIR")
_TRAINING = os.environ.get("TRAINING_DIR")


@unittest.skipUnless(_LOCAL, "ZAPBENCH_LOCAL_PATH not set")
class TestZapbenchLocalPath(unittest.TestCase):
    """Readability checks for ZAPBENCH_LOCAL_PATH."""

    root = Path(_LOCAL) if _LOCAL else Path()

    def test_root_exists(self):
        self.assertTrue(self.root.is_dir(), f"{self.root} is not a directory")

    def test_traces_exists(self):
        p = self.root / "traces"
        self.assertTrue(p.exists(), f"{p} does not exist")

    def test_raw_ephys_exists(self):
        p = self.root / "stimuli_raw" / "stimuli_and_ephys.10chFlt"
        self.assertTrue(p.exists(), f"{p} does not exist")

    def test_segmentation_exists(self):
        p = self.root / "segmentation"
        self.assertTrue(p.exists(), f"{p} does not exist")


@unittest.skipUnless(_CEI, "ZAP_CELL_EPHYS_INDEX_PATH not set")
class TestCellEphysIndexPath(unittest.TestCase):
    """Writability check for ZAP_CELL_EPHYS_INDEX_PATH."""

    def test_parent_writable(self):
        parent = Path(_CEI).parent
        self.assertTrue(
            os.access(parent, os.W_OK),
            f"{parent} is not writable. Run: mkdir -p {parent}",
        )


@unittest.skipUnless(_NEUPRINT, "NEUPRINT_DOWNLOAD_DIR not set")
class TestNeuprintDownloadDir(unittest.TestCase):
    """Writability check for NEUPRINT_DOWNLOAD_DIR."""

    def test_writable(self):
        p = Path(_NEUPRINT)
        self.assertTrue(
            os.access(p, os.W_OK),
            f"{p} is not writable. Run: mkdir -p {p}",
        )


@unittest.skipUnless(_TRAINING, "TRAINING_DIR not set")
class TestTrainingDir(unittest.TestCase):
    """Writability check for TRAINING_DIR."""

    def test_writable(self):
        p = Path(_TRAINING)
        self.assertTrue(
            os.access(p, os.W_OK),
            f"{p} is not writable. Run: mkdir -p {p}",
        )


if __name__ == "__main__":
    unittest.main()
