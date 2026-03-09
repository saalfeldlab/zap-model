"""Data configuration schema."""

from __future__ import annotations

import math
from pathlib import Path  # noqa: TC003 — pydantic needs this at runtime

from pydantic import BaseModel, model_validator

from zap_model.data.conditions import (
    HOLDOUT_CONDITIONS,
    OFFSETS,
    PADDING,
    TEST_FRACTION,
    TRAIN_CONDITIONS,
    VAL_FRACTION,
    Condition,
)
from zap_model.data.functional import MIN_MAX_VALUES


class FrameRange(BaseModel, frozen=True, extra="forbid"):
    """Half-open frame range [start, end)."""

    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)


class ConditionRanges(BaseModel, frozen=True, extra="forbid"):
    """Train/val/test frame ranges for a single condition."""

    train: FrameRange
    val: FrameRange
    test: FrameRange


class ActivityConfig(BaseModel, extra="forbid"):
    """Configuration for activity trace data."""

    traces_path: Path
    min_value: float = MIN_MAX_VALUES[0]
    max_value: float = MIN_MAX_VALUES[1]


class NeuprintConfig(BaseModel, extra="forbid"):
    """Configuration for neuprint EM data."""

    data_dir: Path
    min_weight: int = 1
    # from fishfuncem NeuprintServer.tracing_status_filter
    status_filter: tuple[str, ...] = (
        "Sensory Anchor",
        "Cervical Anchor",
        "Soma Anchor",
        "Examined Soma Anchor",
        "Primary Anchor",
        "Leaves",
        "PRT Orphan",
        "Reviewed",
        "Prelim Roughly traced",
        "RT Hard to trace",
        "RT Orphan",
        "Roughly traced",
        "Traced in ROI",
        "Traced",
        "Finalized",
    )


class ConditionSplitConfig(BaseModel, extra="forbid"):
    """Configuration for condition-based train/val/test splitting.

    For train conditions, frames are split as [train | val | test].
    For holdout conditions, all frames go to test.
    """

    train_conditions: tuple[Condition, ...] = TRAIN_CONDITIONS
    holdout_conditions: tuple[Condition, ...] = HOLDOUT_CONDITIONS
    val_fraction: float = VAL_FRACTION
    test_fraction: float = TEST_FRACTION
    padding: int = PADDING

    @model_validator(mode="after")
    def _check_no_overlap(self) -> ConditionSplitConfig:
        overlap = set(self.train_conditions) & set(self.holdout_conditions)
        if overlap:
            msg = f"conditions appear in both train and holdout: {overlap}"
            raise ValueError(msg)
        return self

    def get_ranges(self) -> dict[Condition, ConditionRanges]:
        """Compute train/val/test FrameRanges for each condition."""
        result: dict[Condition, ConditionRanges] = {}

        for cond in (*self.train_conditions, *self.holdout_conditions):
            raw_start, raw_end = OFFSETS[cond]
            start = raw_start + self.padding
            end = raw_end - self.padding
            usable = end - start

            if cond in self.holdout_conditions:
                result[cond] = ConditionRanges(
                    train=FrameRange(start=start, end=start),
                    val=FrameRange(start=start, end=start),
                    test=FrameRange(start=start, end=end),
                )
            else:
                n_test = math.floor(usable * self.test_fraction)
                n_val = math.floor(usable * self.val_fraction)
                n_train = usable - n_val - n_test

                train_end = start + n_train
                val_end = train_end + n_val
                result[cond] = ConditionRanges(
                    train=FrameRange(start=start, end=train_end),
                    val=FrameRange(start=train_end, end=val_end),
                    test=FrameRange(start=val_end, end=end),
                )

        return result


class IdMapping(BaseModel, extra="forbid"):
    """Mapping between EM body IDs and functional cell indices.

    Default uses neuprint's zapbenchId field (SomaCol.ZB_ID).
    Set path to override with a custom mapping file.
    """

    path: Path | None = None


class DataConfig(BaseModel, extra="forbid"):
    """Top-level data configuration.

    Examples::

        # minimal — just activity + default splits/id_mapping
        # this would be running a zapbench-style benchmarking model
        # all traces that have an EM soma would be used
        DataConfig(
            activity=ActivityConfig(traces_path=Path("/data/traces.zarr")),
        )

        # everything specified
        # id_mapping can be used to specify a subset of zapbench IDs
        # from a region of interest
        DataConfig(
            activity=ActivityConfig(
                traces_path=Path("/data/traces.zarr"),
                min_value=-0.3,
                max_value=2.0,
            ),
            splits=ConditionSplitConfig(
                holdout_conditions=(Condition.TAXIS, Condition.DARK),
                val_fraction=0.15,
            ),
            neuprint=NeuprintConfig(data_dir=Path("/data/neuprint_data/latest")),
            id_mapping=IdMapping(path=Path("/data/custom_mapping.parquet")),
        )
    """

    activity: ActivityConfig
    splits: ConditionSplitConfig = ConditionSplitConfig()
    neuprint: NeuprintConfig | None = None
    id_mapping: IdMapping = IdMapping()
