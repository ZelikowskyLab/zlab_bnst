import os
import glob
import numpy as np
import pandas as pd
import tifffile
import caiman as cm

BASE_DIR = "/Volumes/zelikowsky-group2/Michael/Miniscope_Desktop_Backup"
CSV_PATH = os.path.join(BASE_DIR, "dbnst_data_info.csv.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mean_projections")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# File is UTF-16 tab-separated; header may be at end so use header=None
df = pd.read_csv(CSV_PATH, encoding="utf-16", sep="\t", header=None,
                 names=["Mouse", "Session", "TSeries_g"])

# Strip whitespace and drop the header row if it ended up as a data row
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df[df["Mouse"] != "Mouse"]
df = df.dropna(subset=["Mouse", "Session", "TSeries_g"])

print(f"Loaded {len(df)} datasets from CSV\n")

for _, row in df.iloc.iterrows():
    animal = row["Mouse"]
    session = row["Session"]
    tseries = row["TSeries_g"]

    reg_tif_dir = os.path.join(BASE_DIR, tseries, "suite2p", "plane0", "reg_tif")

    if not os.path.isdir(reg_tif_dir):
        print(f"[SKIP] {animal} {session}: reg_tif not found at {reg_tif_dir}")
        continue

    tif_files = sorted(glob.glob(os.path.join(reg_tif_dir, "*.tif")))
    if not tif_files:
        print(f"[SKIP] {animal} {session}: no .tif files in {reg_tif_dir}")
        continue

    print(f"[{animal} {session}] Loading {len(tif_files)} tif files from {tseries} ...")

    # Load all registered tifs as a single movie (T x H x W)
    movie = cm.load(tif_files)

    # Mean Z-projection across time
    mean_proj = np.array(movie).mean(axis=0)

    out_fname = f"{animal}_{session}_mean_proj.tif"
    out_path = os.path.join(OUTPUT_DIR, out_fname)
    tifffile.imwrite(out_path, mean_proj.astype(np.float32))
    print(f"  -> Saved: {out_path}  (shape {mean_proj.shape})\n")

print("Done.")
