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
uv run marimo edit ../notebooks/glmhmmt_analysis.py
```

The default runtime configuration lives in [config.toml](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt/config.toml). Path resolution precedence is:

1. CLI flags such as `--data-dir`, `--results-dir`, `--config-path`
2. Env vars `GLMHMMT_DATA_DIR`, `GLMHMMT_RESULTS_DIR`, `GLMHMMT_CONFIG_PATH`, `GLMHMMT_ALEXIS_DIR`, `GLMHMMT_TASK_PATHS`
3. `[paths]` in `code/glmhmmt/config.toml`
4. Repo-aware fallbacks

The thesis repo keeps its editable task adapters in [code/tasks](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/tasks) and its notebooks in [code/notebooks](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/notebooks). `glmhmmt.tasks` discovers task modules from `GLMHMMT_TASK_PATHS`, from `[plugins].task_paths` in [config.toml](/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/glmhmmt/config.toml), and from `./tasks` in the current working directory.
