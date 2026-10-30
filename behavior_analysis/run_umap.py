"""UMAP embedding of every frame's 33 kinematic features
(data/features_export/). Step 3 of the pipeline (see README.md).

Clustering is a separate step (watershed, explored interactively in
explore_watershed.ipynb), which reads the run folder this writes, so the
embedding (the slow part) is computed once and can be clustered with
different settings.

  - reads the cleaned per-trial feature CSVs (33 named columns)
  - keeps unannotated frames by default (--drop-unannotated to run on
    labeled behavior only)
  - z-scores each feature (--no-zscore to skip)

Usage:

    python3 run_umap.py                                   # defaults, full dataset (very long havent run)
    python3 run_umap.py --drop-unannotated 
    python3 run_umap.py --n-neighbors 30 --min-dist 0.05  # override parameters
    python3 run_umap.py --num-samples 50000               # quick pass on a subsample

    # from a notebook / debug console
    from run_umap import UMAPConfig, run
    result = run(UMAPConfig(n_neighbors=30, num_samples=50000))

Each run is saved to its own folder, never overwritten:

    data/umap_results/<date_time>_nn<n_neighbors>_md<min_dist>_<metric>_<n>d[_tag]/
        config.json           every parameter used
        point_metadata.csv    one row per embedded frame: group, trial, behavior,
                              intruder_behavior, frame, trial_time, umap_x, umap_y
                              (trial + frame locate the frame's row in data/features_export/)
        umap_by_group.png
        umap_by_behavior.png
"""
import argparse
import dataclasses
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import umap
import matplotlib
if __name__ == "__main__":
    # only force the headless backend for a direct script run (no display
    # available on a server) -- forcing it unconditionally at import time
    # would hijack the backend of anything that imports this module too,
    # including a Jupyter notebook's inline/interactive one
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from export_features_csv import FEATURE_NAMES, OUT_DIR as FEATURES_DIR
from pipeline_paths import GROUPS


@dataclass
class UMAPConfig:
    # --- data selection ---
    features_dir: Path = FEATURES_DIR
    drop_unannotated: bool = False   # False = all frames
    zscore: bool = True
    num_samples: Optional[int] = None  # None = use every row; set an int to subsample for a fast pass
    random_seed: int = 42            # used for subsampling (separate from UMAP's own random_state)

    # --- UMAP parameters (defaults match the original notebook) ---
    n_neighbors: int = 50
    min_dist: float = 0.001
    n_components: int = 2
    metric: str = "euclidean"
    umap_random_state: Optional[int] = 42

    # --- output ---
    tag: str = ""  # optional label appended to the results folder name

    def __post_init__(self):
        self.features_dir = Path(self.features_dir)


@dataclass
class UMAPResult:
    config: UMAPConfig
    meta: pd.DataFrame        # per embedded frame: group, trial, behavior, intruder_behavior, frame, trial_time, umap_x, umap_y
    data: np.ndarray          # [n, 33] features fed to UMAP (z-scored if enabled)
    embedding: np.ndarray     # [n, n_components] UMAP output
    out_dir: Path


def load_features(config: UMAPConfig) -> pd.DataFrame:
    """Reads every trial CSV under features_dir, tagging each row with its
    group and trial so downstream plots can break out by either."""
    frames = []
    for group_folder, group_label in GROUPS:
        for csv_path in sorted(glob.glob(str(config.features_dir / group_folder / "*.csv"))):
            df = pd.read_csv(csv_path, low_memory=False)
            df["group"] = group_label
            df["trial"] = Path(csv_path).stem
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No feature CSVs found under {config.features_dir}")

    combined = pd.concat(frames, ignore_index=True)

    if config.drop_unannotated:
        combined = combined[combined["behavior"].notna()].reset_index(drop=True)

    return combined


def prepare_data(df: pd.DataFrame, config: UMAPConfig):
    """Extracts the 33 feature columns, optionally z-scores and/or
    subsamples. Returns (data, meta) where meta is df's non-feature
    columns realigned to the same rows/order as data."""
    rng = np.random.default_rng(config.random_seed)

    if config.num_samples is not None and config.num_samples < len(df):
        idx = np.sort(rng.choice(len(df), size=config.num_samples, replace=False))
        df = df.iloc[idx].reset_index(drop=True)

    data = df[FEATURE_NAMES].to_numpy(dtype=float)

    if config.zscore:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std_safe = np.where(std > 1e-9, std, 1.0)  # guard any zero-variance column
        data = (data - mean) / std_safe

    meta = df[["group", "trial", "behavior", "intruder_behavior", "frame", "trial_time"]].reset_index(drop=True)
    return data, meta


def run_umap_embedding(data: np.ndarray, config: UMAPConfig) -> np.ndarray:
    print(f"running UMAP on {data.shape[0]} points x {data.shape[1]} features "
          f"(n_neighbors={config.n_neighbors}, min_dist={config.min_dist})...")
    t0 = time.time()
    model = umap.UMAP(
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        n_components=config.n_components,
        metric=config.metric,
        random_state=config.umap_random_state,)
    embedding = model.fit_transform(data)
    print(f"  done in {time.time() - t0:.1f}s")
    return embedding


# --- plots ---

def plot_umap_by_group(embedding, groups, out_path):
    unique_groups = sorted(pd.unique(groups))
    cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_groups)])
    codes = pd.Categorical(groups, categories=unique_groups).codes

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=codes, cmap=cmap, s=1.5, alpha=0.4)
    ax.set_xlabel("UMAP Dim 1"); ax.set_ylabel("UMAP Dim 2"); ax.set_title("UMAP by Group")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i), markersize=8)
               for i in range(len(unique_groups))]
    ax.legend(handles, unique_groups, loc="best")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap_by_behavior(embedding, behavior, out_path):
    behavior_filled = pd.Series(behavior).fillna("Unannotated")
    labels = sorted(behavior_filled.unique())
    codes = pd.Categorical(behavior_filled, categories=labels).codes
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors
    cmap = ListedColormap([base_colors[i % len(base_colors)] for i in range(len(labels))])

    fig, ax = plt.subplots(figsize=(12, 9))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=codes, cmap=cmap, s=2.5, alpha=0.4)
    ax.set_xlabel("UMAP Dim 1"); ax.set_ylabel("UMAP Dim 2"); ax.set_title("UMAP by Behavior")
    cbar = fig.colorbar(scatter, ticks=range(len(labels)))
    cbar.ax.set_yticklabels(labels, fontsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- main script ---

def run(config: UMAPConfig) -> UMAPResult:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # UMAP parameters in the folder name so runs are distinguishable at a
    # glance; the full config is in config.json
    umap_params = f"nn{config.n_neighbors}_md{config.min_dist:g}_{config.metric}_{config.n_components}d"
    run_name = "_".join(part for part in [timestamp, umap_params, config.tag] if part)
    out_dir = config.features_dir.parent / "umap_results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, default=str)

    print(f"loading features from {config.features_dir} ...")
    df = load_features(config)
    print(f"  {len(df)} frames across {df['trial'].nunique()} trials, {df['group'].nunique()} groups")

    data, meta = prepare_data(df, config)
    if data.shape[0] > 200_000:
        print(f"note: fitting UMAP on {data.shape[0]} points -- this can take a while. "
              f"Pass --num-samples for a faster pass while iterating on parameters.")

    embedding = run_umap_embedding(data, config)

    print(f"saving to {out_dir} ...")
    plot_umap_by_group(embedding, meta["group"].to_numpy(), out_dir / "umap_by_group.png")
    plot_umap_by_behavior(embedding, meta["behavior"].to_numpy(), out_dir / "umap_by_behavior.png")

    meta = meta.copy()
    meta["umap_x"] = embedding[:, 0]
    meta["umap_y"] = embedding[:, 1]
    meta.to_csv(out_dir / "point_metadata.csv", index=False)

    print(f"done. results in {out_dir}")
    return UMAPResult(config=config, meta=meta, data=data, embedding=embedding, out_dir=out_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    # defaults come from UMAPConfig so they can't drift from it
    d = UMAPConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", type=Path, default=d.features_dir)
    p.add_argument("--drop-unannotated", action="store_true")
    p.add_argument("--no-zscore", dest="zscore", action="store_false")
    p.add_argument("--num-samples", type=int, default=d.num_samples)
    p.add_argument("--random-seed", type=int, default=d.random_seed)

    p.add_argument("--n-neighbors", type=int, default=d.n_neighbors)
    p.add_argument("--min-dist", type=float, default=d.min_dist)
    p.add_argument("--n-components", type=int, default=d.n_components)
    p.add_argument("--metric", type=str, default=d.metric)
    p.add_argument("--umap-random-state", type=int, default=d.umap_random_state)

    p.add_argument("--tag", type=str, default=d.tag)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run(UMAPConfig(**vars(args)))
