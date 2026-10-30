import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.decomposition import PCA
from PIL import Image

from pipeline_paths import DATA_ROOT

MOUSE_IMAGE = Path(__file__).resolve().parent / "white-mouse-grey-outline-md.png"

def plot_bout_path(path, ax=None, fps=30, limit=None, cmap="viridis", colorbar=True,
                   mouse_image_path=MOUSE_IMAGE, mouse_extent=(-2, 2, 3.4, -3.0)):
    """Draw one bout's approach path (resident nose in the intruder's frame, [n_frames, 2] in cm) as a
    continuous line colored by time before the attack, over the intruder outline. Returns the LineCollection."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.6))
    seconds = (np.arange(len(path)) - (len(path) - 1)) / fps        # last frame = the frame before the attack
    segments = np.stack([path[:-1], path[1:]], axis=1)               # [n_frames - 1, 2, 2]: one segment per step
    line = LineCollection(segments, cmap=cmap, linewidth=2.2, capstyle="round")
    line.set_array((seconds[:-1] + seconds[1:]) / 2)                 # each segment colored by its midpoint time
    line.set_clim(seconds[0], seconds[-1])
    ax.add_collection(line)
    ax.plot(path[-1, 0], path[-1, 1], "o", color=plt.get_cmap(cmap)(1.0), markeredgecolor="k", markersize=6)   # the frame before the attack

    ax.imshow(Image.open(mouse_image_path), extent=mouse_extent, zorder=0)
    if limit is None:
        limit = np.ceil(np.nanmax(np.abs(path)) / 5) * 5
    ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_aspect("equal")   # after imshow, which resets them
    ax.set_xlabel("left / right of intruder (cm)"); ax.set_ylabel("behind / ahead of intruder (cm)")
    ax.spines[["top", "right"]].set_visible(False)
    if colorbar:
        ax.figure.colorbar(line, ax=ax, label="time before attack (s)", shrink=0.8)
    return line

def main():
    # Set Smoothing and Load in Data
    BOUTS_DIR = DATA_ROOT / "attack_bouts"
    PATH_SMOOTH_WINDOW = 7     # light Savitzky-Golay smoothing over time, in frames (odd; 1 = none)
    PATH_MIRROR_LEFT_RIGHT = True  # treat left and right approaches the same by mirroring
    SAVEPATH = DATA_ROOT / "pca_results"
    bouts = pd.read_csv(BOUTS_DIR / "bouts.csv")
    paths = np.load(BOUTS_DIR / "bout_paths.npz")["paths"]      # [n_bouts, 60, 2]: resident nose in the intruder's frame
    kinematics = pd.read_csv(BOUTS_DIR / "bout_kinematics.csv")

    def per_frame(column):
            """[n_bouts, 60] array of one column of bout_kinematics.csv, in bouts.csv order."""
            return kinematics.pivot(index="bout_id", columns="frame_in_window", values=column).loc[bouts["bout_id"]].to_numpy()
    
    ## Construct Feature Dataframe 
    facing, relative_heading = np.radians(per_frame("resident_facing_angle")), np.radians(per_frame("relative_heading"))
    kinematic_series = {
        "path_x": paths[:, :, 0], 
        "path_y": paths[:, :, 1],
        "facing_cos": np.cos(facing),                         # 1 = facing the intruder, -1 = facing away
        "facing_sin": np.sin(facing),
        "resident_approach_speed": per_frame("resident_approach_speed"),
        "intruder_approach_speed": per_frame("intruder_approach_speed"),}

    # each feature is a set of series measured in the same units
    features = {
        "path": ["path_x", "path_y"],                                           # cm
        "facing": ["facing_cos", "facing_sin"],                                 # -1 to 1
        "approach_speed": ["resident_approach_speed", "intruder_approach_speed"],  # cm/s
    }
    # Normalize features 
    # z-score every column (each series at each time point) on its own: subtract its mean and divide
    # by its SD across bouts, so every column counts the same in the PCA. 
    columns = [f"{name}_t{t:02d}" for series_names in features.values() for name in series_names
               for t in range(paths.shape[1])]
    raw = np.hstack([kinematic_series[name] for series_names in features.values() for name in series_names])
    X = pd.DataFrame(zscore(raw, axis=0), index=bouts["bout_id"], columns=columns)
    print(f"{X.shape[0]} bouts x {X.shape[1]} columns; total variance per feature: "
          + ", ".join(f"{feature_name} {X.filter(regex='^(' + '|'.join(series_names) + ')_t').var(ddof=0).sum():.0f}"
                      for feature_name, series_names in features.items()))

    # FIT PCA 
    pca = PCA().fit(X.to_numpy())
    pc_scores = pca.transform(X.to_numpy())                     # [n_bouts, n_components]: each bout's position on every PC
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=X.columns,
                            columns=[f"PC{i + 1}" for i in range(len(explained))])
    loadings.index.name = "column"

    # SAVE RESULTS: one folder per run, so a new run never overwrites one a figure was made from
    run_dir = Path(SAVEPATH) / f"{datetime.now():%Y%m%d_%H%M%S}_kinematics_zscore"
    run_dir.mkdir(parents=True)
    pc_columns = [f"PC{i + 1}" for i in range(len(explained))]

    # one row per bout: what it is, and its score on every PC (row order = bouts.csv = bout_paths.npz)
    scores_out = bouts[["bout_id", "group", "group_folder", "trial_number"]].copy()
    scores_out["animal"] = bouts["group_folder"] + " / trial " + bouts["trial_number"].astype(str)
    scores_out = pd.concat([scores_out, pd.DataFrame(pc_scores, columns=pc_columns)], axis=1)
    scores_out.to_csv(run_dir / "pc_scores.csv", index=False)

    loadings.to_csv(run_dir / "loadings.csv")                   # one row per input column, one column per PC
    pd.DataFrame({"PC": pc_columns, "share": explained, "cumulative": np.cumsum(explained)}).to_csv(
        run_dir / "explained_variance.csv", index=False)

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "inputs": [str(BOUTS_DIR / name) for name in ("bouts.csv", "bout_paths.npz", "bout_kinematics.csv")],
        "features": features,                                    # feature -> the series in it
        "frames_per_series": int(paths.shape[1]),
        "normalization": "scipy.stats.zscore per column (ddof=0)",
        "path_smoothing": None,                                  # paths go in as recorded
        "path_mirrored_left_right": False,
        "n_bouts": int(X.shape[0]),
        "n_columns": int(X.shape[1]),
        "n_components": int(len(explained)),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved PCA results to {run_dir}")
    
    #Plot explained variance 
    n_show = min(20, len(explained))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(np.arange(1, n_show + 1), explained[:n_show], color="0.6", label="each PC")
    ax.plot(np.arange(1, n_show + 1), np.cumsum(explained)[:n_show], "o-", color="k", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("share of variance"); ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(1, n_show + 1))
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Variance explained (first {n_show} of {len(explained)} PCs)", fontsize=10)
    plt.tight_layout()
    plt.savefig(run_dir / "explained_variance.png")
    plt.show()

    # PLOT of PCs 
    # What each PC looks like as an approach: mean path of the bouts with the highest (red) and lowest
    # (blue) scores on that PC, with those bouts' own paths underneath.
    # Paths as they went into the PCA (not smoothed or mirrored).
    EXTREME_FRACTION = 0.10    # share of bouts taken from each end of a component
    N_PCS_SHOWN = 4
    mouse_image = Image.open(MOUSE_IMAGE)    
    MOUSE_EXTENT = [-2, 2, 3.4, -3.0]

    n_extreme = max(1, int(round(EXTREME_FRACTION * len(paths))))
    limit = np.ceil(np.nanpercentile(np.abs(paths), 99) / 5) * 5    # 99th percentile, so one stray path doesn't shrink the rest
    fig, axes = plt.subplots(1, N_PCS_SHOWN, figsize=(4.3 * N_PCS_SHOWN, 4.6), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        order = np.argsort(pc_scores[:, k])
        for rows, color, label in [(order[-n_extreme:], "tab:red", f"highest {n_extreme} bouts"),
                                   (order[:n_extreme], "tab:blue", f"lowest {n_extreme} bouts")]:
            for xy in paths[rows]:
                ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.5, alpha=0.2)
            mean_path = np.nanmean(paths[rows], axis=0)
            ax.plot(mean_path[:, 0], mean_path[:, 1], color=color, linewidth=2.2, label=label)
            ax.plot(mean_path[-1, 0], mean_path[-1, 1], "o", color=color, markeredgecolor="k", markersize=6)   # the frame before the attack
        ax.imshow(mouse_image, extent=MOUSE_EXTENT, zorder=0)   # zorder 0: under the paths (lines are drawn at 2)
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_aspect("equal")   # after imshow, which resets them
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"PC{k + 1} ({explained[k]:.0%})", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.supxlabel("left / right of intruder (cm)")
    fig.supylabel("behind / ahead of intruder (cm)")
    fig.suptitle(f"Mean path of the {EXTREME_FRACTION:.0%} of bouts at each end of every component "
                 "(dot = the frame before the attack)", fontsize=10)
    plt.tight_layout()
    plt.savefig(run_dir / "pc_highloadings_bout_paths.png")
    plt.show()

if __name__=='__main__':
    main()