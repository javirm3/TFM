`glmhmmt` is the installable GLM-HMM / GLM-HMMT package that lives inside this thesis repository.

Install from a clone with `uv`:

```bash
cd code/glmhmmt
uv sync
uv run glmhmmt-fit-glmhmmt --help
```

If you want the marimo notebooks too:

```bash
cd code/glmhmmt
uv sync --extra notebooks
uv run marimo edit notebooks/glmhmmt_analysis.py
```

The default runtime configuration lives in [config.toml](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt/config.toml). Path resolution precedence is:

1. CLI flags such as `--data-dir`, `--results-dir`, `--config-path`
2. Env vars `GLMHMMT_DATA_DIR`, `GLMHMMT_RESULTS_DIR`, `GLMHMMT_CONFIG_PATH`, `GLMHMMT_ALEXIS_DIR`
3. `[paths]` in `code/glmhmmt/config.toml`
4. Repo-aware fallbacks

The repo ships a mutable [tasks](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt/tasks) plugin folder with the bundled MCDR, 2AFC, and Nuo auditory adapters. In a clone, you can add or delete task modules there and `glmhmmt.tasks` will auto-discover them. Extra lab tasks can still be published as separate packages via the `glmhmmt.tasks` entry-point group.
