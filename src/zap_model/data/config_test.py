"""Config use-case tests."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from zap_model.data.conditions import (
    HOLDOUT_CONDITIONS,
    OFFSETS,
    PADDING,
    TEST_FRACTION,
    TRAIN_CONDITIONS,
    VAL_FRACTION,
    Condition,
)
from zap_model.data.config import (
    ActivityConfig,
    ConditionSplitConfig,
    DataConfig,
)


class TestDataConfig(unittest.TestCase):
    def test_minimal_config(self):
        """Minimal config: just traces path, everything else defaults."""
        cfg = DataConfig(activity=ActivityConfig(traces_path="/data/traces.zarr"))

        self.assertEqual(cfg.activity.traces_path, Path("/data/traces.zarr"))
        self.assertEqual(cfg.splits.train_conditions, TRAIN_CONDITIONS)
        self.assertEqual(cfg.splits.holdout_conditions, HOLDOUT_CONDITIONS)
        self.assertEqual(cfg.splits.val_fraction, VAL_FRACTION)
        self.assertEqual(cfg.splits.test_fraction, TEST_FRACTION)
        self.assertEqual(cfg.splits.padding, PADDING)
        self.assertIsNotNone(cfg.neuprint)
        self.assertIsNone(cfg.body_ids_path)

        # verify split contiguity and coverage for every condition
        ranges = cfg.splits.get_ranges()
        for cond in Condition:
            r = ranges[cond]
            raw_start, raw_end = OFFSETS[cond]
            usable_start = raw_start + PADDING
            usable_end = raw_end - PADDING

            self.assertEqual(r.train.start, usable_start)
            self.assertEqual(r.train.end, r.val.start)
            self.assertEqual(r.val.end, r.test.start)
            self.assertEqual(r.test.end, usable_end)

    def test_train_on_all_data(self):
        """Use all frames for training: val and test fractions set to 0."""
        cfg = DataConfig(
            activity=ActivityConfig(traces_path="/data/traces.zarr"),
            splits=ConditionSplitConfig(val_fraction=0.0, test_fraction=0.0),
        )
        ranges = cfg.splits.get_ranges()
        for cond in TRAIN_CONDITIONS:
            r = ranges[cond]
            raw_start, raw_end = OFFSETS[cond]
            self.assertEqual(len(r.train), raw_end - raw_start - 2 * PADDING)
            self.assertEqual(len(r.val), 0)
            self.assertEqual(len(r.test), 0)

    def test_subset_of_neurons(self):
        """Restrict to a subset of neurons via body IDs file."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"id": [1, 2, 3]}).write_parquet(f.name)
            cfg = DataConfig(
                activity=ActivityConfig(traces_path="/data/traces.zarr"),
                body_ids_path=f.name,
            )
            self.assertEqual(cfg.body_ids_path, Path(f.name))

    def test_body_ids_missing_column(self):
        """Reject body IDs parquet without an 'id' column."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"wrong": [1, 2]}).write_parquet(f.name)
            with self.assertRaises(ValueError, msg="must have an 'id' column"):
                DataConfig(
                    activity=ActivityConfig(traces_path="/data/traces.zarr"),
                    body_ids_path=f.name,
                )

    def test_body_ids_wrong_dtype(self):
        """Reject body IDs parquet with non-integer 'id' column."""
        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            pl.DataFrame({"id": ["a", "b"]}).write_parquet(f.name)
            with self.assertRaises(ValueError, msg="must be integer"):
                DataConfig(
                    activity=ActivityConfig(traces_path="/data/traces.zarr"),
                    body_ids_path=f.name,
                )


if __name__ == "__main__":
    unittest.main()
