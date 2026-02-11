from pathlib import Path

# Carpeta raíz del proyecto (dos niveles por encima de este archivo)
ROOT = Path(__file__).resolve().parents[1]

CODE_DIR  = ROOT / "code"
DATA_PATH = ROOT / "data"
FITTING_DIR = CODE_DIR / "fitting"
PARAMS_DIR = ROOT / "params"
EXPRESSIONS = CODE_DIR / "expressions"
PLOTS = ROOT / "plots"
CSV_PATH = DATA_PATH / "df_filtered.csv"
CSV_BAD_PATH = DATA_PATH / "df_filtered_bad.csv"

def show_paths():
    print(f"ROOT         = {ROOT}")
    print(f"CODE_DIR     = {CODE_DIR}")
    print(f"DATA_PATH    = {DATA_PATH}")
    print(f"FITTING_DIR  = {FITTING_DIR}")
    print(f"CSV_PATH     = {CSV_PATH}")