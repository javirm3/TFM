import polars as pl
import tomllib
try:
    df = pl.read_parquet("data/df_filtered.parquet")
    print("Loaded parquet")
except:
    try:
        df = pl.read_csv("data/df_filtered.csv")
        print("Loaded csv")
    except:
        print("Could not load data")
        exit()

# Check columns
cols = df.columns
print("Columns:", cols)

if "ttype_c" in cols:
    print("\nttype_c counts:")
    print(df["ttype_c"].value_counts())
else:
    print("ttype_c missing")

if "stimd_c" in cols:
    print("\nstimd_c counts:")
    print(df["stimd_c"].value_counts())
else:
    print("stimd_c missing")

# Check config mapping if needed
import paths
with paths.CONFIG.open("rb") as f:
    cfg = tomllib.load(f)

print("\nEncoding in Config:")
print(cfg["encoding"])

# Check subject breakdown
print("\nCounts by subject:")
print(df.group_by("subject").agg([
    pl.col("ttype_c").unique().alias("types"),
    pl.col("stimd_c").unique().alias("stims")
]).sort("subject"))
