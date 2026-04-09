These notebooks live alongside the public `glmhmmt` package so the thesis repo keeps the analysis workspace and the installable library together.

Install notebook extras from `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code` with:

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv sync --extra notebooks
uv run marimo edit notebooks/glmhmmt_analysis.py
```

For public `molab` / remote-fit usage:

- fill `/Users/javierrodriguezmartinez/Documents/MAMME/TFM/code/.env`
- publish the non-hash fit aliases to `R2` with:

```bash
cd /Users/javierrodriguezmartinez/Documents/MAMME/TFM/code
uv run --with boto3 python scripts/publish_public_fits.py
```

The notebooks first try local fits under `results/fits`. If `GLMHMMT_PUBLIC_FITS_BASE_URL`
or `GLMHMMT_PUBLIC_FITS_MANIFEST_URL` is configured, they switch to the published `R2`
manifest/files instead.
