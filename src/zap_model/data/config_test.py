"""Config use-case tests."""

import unittest
from pathlib import Path

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
        self.assertIsNone(cfg.neuprint)
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
        cfg = DataConfig(
            activity=ActivityConfig(traces_path="/data/traces.zarr"),
            body_ids_path="/data/region_body_ids.parquet",
        )
        self.assertEqual(
            cfg.body_ids_path,
            Path("/data/region_body_ids.parquet"),
        )


if __name__ == "__main__":
    unittest.main()
