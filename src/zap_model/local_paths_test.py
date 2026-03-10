"""Tests that configured local paths are accessible.

All tests skip when local_paths.py doesn't exist, so the suite passes
in CI and on machines without local data.
"""

import os
import unittest

try:
    from zap_model.local_paths import (
        NEUPRINT_DOWNLOAD_DIR,
        TRAINING_DIR,
        ZAP_CELL_EPHYS_INDEX_PATH,
        ZAPBENCH_LOCAL_PATH,
    )

    _HAS_LOCAL_PATHS = True
except ImportError:
    _HAS_LOCAL_PATHS = False


@unittest.skipUnless(_HAS_LOCAL_PATHS, "local_paths.py not found")
class TestZapbenchLocalPath(unittest.TestCase):
    """Readability checks for ZAPBENCH_LOCAL_PATH."""

    def test_root_exists(self):
        self.assertTrue(ZAPBENCH_LOCAL_PATH.is_dir(), f"{ZAPBENCH_LOCAL_PATH} is not a directory")

    def test_traces_exists(self):
        p = ZAPBENCH_LOCAL_PATH / "traces"
        self.assertTrue(p.exists(), f"{p} does not exist")

    def test_raw_ephys_exists(self):
        p = ZAPBENCH_LOCAL_PATH / "stimuli_raw" / "stimuli_and_ephys.10chFlt"
        self.assertTrue(p.exists(), f"{p} does not exist")

    def test_segmentation_exists(self):
        p = ZAPBENCH_LOCAL_PATH / "segmentation"
        self.assertTrue(p.exists(), f"{p} does not exist")


@unittest.skipUnless(_HAS_LOCAL_PATHS, "local_paths.py not found")
class TestCellEphysIndexPath(unittest.TestCase):
    """Writability check for ZAP_CELL_EPHYS_INDEX_PATH."""

    def test_parent_writable(self):
        parent = ZAP_CELL_EPHYS_INDEX_PATH.parent
        self.assertTrue(
            os.access(parent, os.W_OK),
            f"{parent} is not writable. Run: mkdir -p {parent}",
        )


@unittest.skipUnless(_HAS_LOCAL_PATHS, "local_paths.py not found")
class TestNeuprintDownloadDir(unittest.TestCase):
    """Writability check for NEUPRINT_DOWNLOAD_DIR."""

    def test_writable(self):
        self.assertTrue(
            os.access(NEUPRINT_DOWNLOAD_DIR, os.W_OK),
            f"{NEUPRINT_DOWNLOAD_DIR} is not writable. Run: mkdir -p {NEUPRINT_DOWNLOAD_DIR}",
        )


@unittest.skipUnless(_HAS_LOCAL_PATHS, "local_paths.py not found")
class TestTrainingDir(unittest.TestCase):
    """Writability check for TRAINING_DIR."""

    def test_writable(self):
        self.assertTrue(
            os.access(TRAINING_DIR, os.W_OK),
            f"{TRAINING_DIR} is not writable. Run: mkdir -p {TRAINING_DIR}",
        )


if __name__ == "__main__":
    unittest.main()
