"""Condition metadata for zapbench functional data."""

from enum import IntEnum


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


# (start, end) offsets along the T dimension, indexed by Condition
# e.g., OFFSETS[Condition.GAIN]  # (0, 649)
OFFSETS = (
    (0, 649),  # GAIN
    (649, 2422),  # DOTS
    (2422, 3078),  # FLASH
    (3078, 3735),  # TAXIS
    (3735, 5047),  # TURNING
    (5047, 5638),  # POSITION
    (5638, 6623),  # OPEN_LOOP
    (6623, 7279),  # ROTATION
    (7279, 7879),  # DARK
)

# frames excluded at the beginning and end of each condition
PADDING = 1

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
