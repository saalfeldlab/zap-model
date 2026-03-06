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

### fishfuncem

fishfuncem is installed with `--no-deps` because its `pyproject.toml` pins
strict version ranges (e.g. `numpy<2`, `python<3.12`) that conflict with our
stack. Its actual runtime dependencies are listed explicitly in the conda env
files.
