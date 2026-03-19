from pathlib import Path

# zapbench public release data
ZAPBENCH_GCS_URI = "gs://zapbench-release/volumes/20240930"
# read only, local copy of zarrs downloaded from URI above
ZAPBENCH_LOCAL_PATH = Path("/path/to/zapbench-release/volumes/20240930")

# writable root for generated data

# To place at repo root (gitignored path)
# OUTPUT_DATA_DIR = Path(__file__).parent.joinpath("../../data")
OUTPUT_DATA_DIR = Path("/path/to/output_data")

# To place at repo root (gitignored path)
# TRAINING_DIR = Path(__file__).parent.joinpath("../../runs")
TRAINING_DIR = Path("/path/to/training_runs")
