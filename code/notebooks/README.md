These notebooks live alongside the public `glmhmmt` package so the thesis repo keeps the analysis workspace and the installable library together.

Install notebook extras from `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code` with:

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv sync --extra notebooks
uv run marimo edit notebooks/glmhmmt_analysis.py
```
