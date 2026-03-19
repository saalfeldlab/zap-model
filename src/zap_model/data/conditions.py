"""Condition metadata for zapbench functional data.

Note that condition data is specified in "frames" and not time steps.

Glossary
--------
Frame : Nominal timestep index (0..T-1) — one row of the traces matrix.
        All cells share the same Frame index, but were not acquired simultaneously.
AcqIndex : Index into the 6 kHz ephys acquisition stream. Represents the actual
           wall-clock time at which a specific cell was imaged. Different cells in
           the same Frame have different AcqIndex values because the microscope
           scans z-planes sequentially.

"""

from enum import IntEnum
from typing import NewType


class Condition(IntEnum):
    """Different stimulus conditions."""

    GAIN = 0
    DOTS = 1
    FLASH = 2
    TAXIS = 3
    TURNING = 4
    POSITION = 5
    OPEN_LOOP = 6
    ROTATION = 7
    DARK = 8


# Imaging geometry
# Adding a new type will let the IDE know that these integers correspond
# to frame indices and NOT bin indices.
Frame = NewType("Frame", int)

# number frames = number of nominal time steps = number of acquired z stacks
NUM_FRAMES = Frame(7879)


# (start, end) offsets along the Frame dimension, indexed by Condition
# e.g., OFFSETS[Condition.GAIN]  # (0, 649)
OFFSETS = (
    (Frame(0), Frame(649)),  # GAIN
    (Frame(649), Frame(2422)),  # DOTS
    (Frame(2422), Frame(3078)),  # FLASH
    (Frame(3078), Frame(3735)),  # TAXIS
    (Frame(3735), Frame(5047)),  # TURNING
    (Frame(5047), Frame(5638)),  # POSITION
    (Frame(5638), Frame(6623)),  # OPEN_LOOP
    (Frame(6623), Frame(7279)),  # ROTATION
    (Frame(7279), Frame(7879)),  # DARK
)

# frames excluded at the beginning and end of each condition
PADDING = Frame(1)

TRAIN_CONDITIONS = (
    Condition.GAIN,
    Condition.DOTS,
    Condition.FLASH,
    Condition.TURNING,
    Condition.POSITION,
    Condition.OPEN_LOOP,
    Condition.ROTATION,
    Condition.DARK,
)
HOLDOUT_CONDITIONS = (Condition.TAXIS,)

# fractions of timesteps per condition used for validation and test
VAL_FRACTION = 0.1
TEST_FRACTION = 0.2


# # per-condition stimulus parameter metadata
# CONDITION_STIM_METADATA: dict[Condition, dict] = {
#     Condition.GAIN: {
#         "stimParam3": {"description": "gain level", "values": {"1": "low", "2": "high"}},
#         "stimParam4": {"description": "unused"},
#     },
#     Condition.DOTS: {
#         "stimParam3": {"description": "orientation (degrees)"},
#         "stimParam4": {"description": "coherence"},
#     },
#     Condition.FLASH: {
#         "stimParam3": {"description": "luminance", "values": {"0": "dark", "1": "bright"}},
#         "stimParam4": {"description": "unused"},
#     },
#     Condition.TAXIS: {
#         "stimParam3": {"description": "left brightness", "values": {"0": "dark", "1": "bright"}},
#         "stimParam4": {"description": "right brightness", "values": {"0": "dark", "1": "bright"}},
#     },
#     Condition.TURNING: {
#         "stimParam3": {"description": "velocity"},
#         "stimParam4": {
#             "description": "orientation",
#             "values": {"1": "forward", "-1": "forward", "90": "sideways", "-90": "sideways"},
#         },
#     },
#     Condition.POSITION: {
#         "stimParam3": {
#             "description": "grating type",
#             "values": {"-1": "long forward", "0": "stationary", "1": "short pulse"},
#         },
#         "stimParam4": {"description": "delay between gratings"},
#     },
#     Condition.OPEN_LOOP: {
#         "stimParam3": {
#             "description": "mode",
#             "values": {"1": "closed loop", "2": "open loop"},
#         },
#         "stimParam4": {"description": "orientation"},
#     },
#     Condition.ROTATION: {
#         "stimParam3": {"description": "undocumented"},
#         "stimParam4": {"description": "undocumented"},
#     },
#     Condition.DARK: {
#         "stimParam3": {"description": "unused"},
#         "stimParam4": {"description": "unused"},
#     },
# }
