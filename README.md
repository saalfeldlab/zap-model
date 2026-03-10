# zap-model

Data processing and modeling for zapbench data.

## Environment setup

### Linux (CUDA)

```bash
conda env create -f envs/environment.linux.yaml
conda activate zap-model
pip install --no-deps git+https://github.com/ahrens-fish-lab/fishFuncEM.git@422bb2a
pip install -e .
pre-commit install
```

### macOS

```bash
conda env create -f envs/environment.mac.yaml
conda activate zap-model
pip install --no-deps git+https://github.com/ahrens-fish-lab/fishFuncEM.git@422bb2a
pip install -e .
pre-commit install
```

Note: conda env creation may take a while due to pip building some packages from source.

### Verify

```bash
python scripts/verify_fishfuncem.py
```

This queries neuprint and prints 5 rows. Requires `NEUPRINT_TOKEN` or
`NEUPRINT_APPLICATION_CREDENTIALS` in your environment (see fishfuncem docs).

### Data paths

Copy the example env file and fill in your local paths:

```bash
cp .env.example .env
```

| Variable | Purpose | Used by |
|---|---|---|
| `ZAPBENCH_LOCAL_PATH` | Local root for zapbench release data | configs, `build_cell_ephys_index.py` |
| `ZAPBENCH_GCS_URI` | GCS root for remote data | `build_cell_ephys_index.py` |
| `ZAP_CELL_EPHYS_INDEX_PATH` | Path to `cell_ephys_index.zarr` (computed output) | configs, `build_cell_ephys_index.py` |
| `NEUPRINT_DOWNLOAD_DIR` | Neuprint download root | configs, `download_neuprint.py` |
| `TRAINING_DIR` | Training run output directory | `TrainingConfig` |

The `.env` file is gitignored. When set, these variables provide defaults for
config fields (`ActivityConfig.traces_path`, `NeuprintConfig.data_dir`,
`TrainingConfig.run_dir`) and replace hardcoded paths in scripts.

### fishfuncem

fishfuncem is installed with `--no-deps` because its `pyproject.toml` pins
strict version ranges (e.g. `numpy<2`, `python<3.12`) that conflict with our
stack. Its actual runtime dependencies are listed explicitly in the conda env
files.

## Contributing

### Linting and formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Install the [ruff VS Code extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
(recommended in `.vscode/extensions.json`) and enable format-on-save:

```json
"[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true
}
```

Configuration lives in `pyproject.toml` (line length 100, strict rule set).

### Pre-commit hooks

Install the git hooks after setting up your environment:

```bash
pre-commit install
```

This runs ruff check + format on every commit and rejects files > 1 MB.

### Tests

Run all tests with:

```bash
make test
```

Tests have a 10-second timeout. Keep tests lightweight — no network calls, no
large data loads.

Place test files next to the module they test: `module.py` is tested by
`module_test.py` in the same directory. This keeps tests discoverable in the
file tree.

### Commits

Use [conventional commits](https://www.conventionalcommits.org/):

- `feat:` — new feature or capability
- `fix:` — bug fix
- `refactor:` — code restructuring without behavior change
- `build:` — dependency or build system changes
- `chore:` — maintenance (CI, configs, tooling)

Keep messages short and high-level. State the motivation, not a list of files changed.
