"""Tests that configured local paths are accessible.

All tests skip when local_paths.py doesn't exist, so the suite passes
in CI and on machines without local data.
"""

import os
import unittest

try:
    from zap_model.local_paths import (
        OUTPUT_DATA_DIR,
        TRAINING_DIR,
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
class TestOutputDataDir(unittest.TestCase):
    """Writability check for OUTPUT_DATA_DIR."""

    def test_writable(self):
        self.assertTrue(
            os.access(OUTPUT_DATA_DIR, os.W_OK),
            f"{OUTPUT_DATA_DIR} is not writable. Run: mkdir -p {OUTPUT_DATA_DIR}",
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
