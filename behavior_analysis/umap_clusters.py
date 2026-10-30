"""UMAP Watershed Clustering Functions 
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

GROUP_LABELS = {"Group A": "NS", "Group B": "CD-1", "Group C": "FS"}
GROUP_COLORS = {"Group A": "#FFFFFF", "Group B": "#B5DEB0", "Group C": "#F4A460"}  # NS, CD-1, FS


# --- watershed clustering ---

@dataclass
class WatershedConfig:
    """Watershed settings (the slider values in explore_watershed.ipynb).

    Attributes
    ----------
    bins : int
        Number of histogram bins per axis.
    gaussian_stddev : float
        Standard deviation of the Gaussian smoothing, in bins.
    num_clusters : int
        Maximum number of clusters: one per density peak, tallest first.
    compactness : float
        Watershed compactness. Higher values give more evenly sized,
        rounder regions.
    """
    bins: int = 200               # histogram bins per axis
    gaussian_stddev: float = 1.5  # smoothing, in bins
    num_clusters: int = 6        # maximum number of clusters: one per density peak, tallest first
    compactness: float = 0.05     # higher = more evenly sized, rounder regions


def run_watershed(embedding: np.ndarray, config: WatershedConfig):
    """Cluster a 2D embedding with a watershed on its smoothed density.

    The embedding is binned into a 2D histogram and smoothed with a
    Gaussian. Each cluster grows from one of the ``num_clusters`` tallest
    local peaks of the smoothed density, and the watershed floods the
    negated density, so clusters grow downhill from their peak and meet in
    the low-density valleys between peaks.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_points, 2)
        2D embedding, x in column 0 and y in column 1.
    config : WatershedConfig
        Bins, smoothing, maximum number of clusters and compactness.

    Returns
    -------
    clusters : np.ndarray of int, shape (bins, bins)
        Cluster label of each histogram bin, indexed ``[y_bin, x_bin]``
        (transposed from the histogram's ``[x_bin, y_bin]``). Labels start
        at 1; cluster 1 is the tallest peak, 2 the next, and so on.
    centers : np.ndarray, shape (num_clusters, 2)
        Density-weighted center of each cluster as ``(y_bin, x_bin)``.
        NaN for a label with no bins.
    histogram_x : np.ndarray, shape (bins + 1,)
        Bin edges along x.
    histogram_y : np.ndarray, shape (bins + 1,)
        Bin edges along y.

    Notes
    -----
    If the density has fewer local peaks than ``num_clusters``, fewer
    clusters are returned.
    """
    # bin and smooth 2d 
    histogram, histogram_x, histogram_y = np.histogram2d(
        embedding[:, 0], embedding[:, 1], bins=config.bins,)
    kernel = Gaussian2DKernel(x_stddev=config.gaussian_stddev,
                              y_stddev=config.gaussian_stddev)
    smoothed_histogram = convolve_fft(histogram, kernel)  # indexed [x_bin, y_bin]

    # one seed per density peak (tallest first), instead of an even grid of seeds
    peaks = peak_local_max(smoothed_histogram, num_peaks=config.num_clusters, exclude_border=False)
    markers = np.zeros(smoothed_histogram.shape, dtype=int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1) #peak markers as a 2d array 

    # only bins that contain data take part; empty bins stay label 0 (no cluster)
    occupied = histogram > 0                                       # [x_bin, y_bin], like the histogram
    clusters = watershed(-smoothed_histogram, markers, mask=occupied, compactness=config.compactness).T
    # a requested label index with zero pixels (fewer peaks than num_clusters)
    # makes center_of_mass divide by zero mass; NaN is the right answer for
    # "the center of an empty region" and callers skip NaN centers, so
    # silence numpy's warning. The density is transposed to match clusters.
    with np.errstate(invalid="ignore"):
        centers = np.array(center_of_mass(smoothed_histogram.T, clusters, range(1, config.num_clusters + 1)))

    return clusters, centers, histogram_x, histogram_y


def assign_point_clusters(embedding: np.ndarray, clusters: np.ndarray,
                          histogram_x: np.ndarray, histogram_y: np.ndarray, bins: int) -> np.ndarray:
    """Give each embedded point the cluster label of its histogram bin.

    Parameters
    ----------
    embedding : np.ndarray, shape (n_points, 2)
        2D embedding, x in column 0 and y in column 1.
    clusters : np.ndarray of int, shape (bins, bins)
        Cluster label image from `run_watershed`, indexed
        ``[y_bin, x_bin]``.
    histogram_x, histogram_y : np.ndarray, shape (bins + 1,)
        Bin edges from the same `run_watershed` call.
    bins : int
        Number of bins per axis used in that call.

    Returns
    -------
    np.ndarray of int, shape (n_points,)
        Cluster label of each point, in the same order as ``embedding``.
    """
    x_bins = np.clip(np.digitize(embedding[:, 0], histogram_x) - 1, 0, bins - 1)
    y_bins = np.clip(np.digitize(embedding[:, 1], histogram_y) - 1, 0, bins - 1)
    return clusters[y_bins, x_bins]


def cluster_occupancy(meta: pd.DataFrame, point_clusters: np.ndarray):
    """Compute each trial's proportion of frames in each cluster.

    Parameters
    ----------
    meta : pd.DataFrame
        One row per embedded frame, with ``trial`` and ``group`` columns.
    point_clusters : np.ndarray of int, shape (n_points,)
        Cluster label of each frame, in the same order as ``meta``.

    Returns
    -------
    proportions : pd.DataFrame
        Trials (rows) by clusters (columns). Each row sums to 1. Rows are
        sorted by group, then trial.
    long_df : pd.DataFrame
        The same values in long form, with columns ``trial``, ``cluster``,
        ``proportion`` and ``group``.
    cluster_ids : np.ndarray of int
        Ids of the non-empty clusters.
    groups : list of str
        Sorted group names.
    trial_group : pd.Series
        Group of each trial, indexed by trial.
    """
    # cluster ids actually present, not just range(1, num_clusters + 1) --
    # the requested and non-empty counts can differ, and an always-empty
    # cluster would otherwise show up as an all-zero column
    cluster_ids = np.unique(point_clusters[point_clusters > 0])

    meta_clustered = meta.copy()
    meta_clustered["cluster"] = point_clusters

    counts = meta_clustered.groupby(["trial", "cluster"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=cluster_ids, fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    groups = sorted(meta_clustered["group"].unique())
    trial_group = meta_clustered.drop_duplicates("trial").set_index("trial")["group"]
    trial_order = sorted(proportions.index, key=lambda t: (groups.index(trial_group[t]), t))
    proportions = proportions.loc[trial_order]

    long_df = proportions.reset_index().melt(id_vars="trial", var_name="cluster", value_name="proportion")
    long_df["group"] = long_df["trial"].map(trial_group)
    return proportions, long_df, cluster_ids, groups, trial_group


# --- behavior colors ---

# behavior -> family, from the resident-intruder ethogram. Spelling variants
# from older exports ("Self-Grooming", "Tail-rattling", ...) are listed too,
# so older UMAP runs still map cleanly.
BEHAVIOR_FAMILY = {
    # Prosocial
    "Anogenital Sniffing": "Prosocial",
    "Body Sniffing": "Prosocial",
    "Face Sniffing": "Prosocial",
    "Grooming": "Prosocial",
    "Tail Sniffing": "Prosocial",
    # Avoidance
    "Digging": "Avoidance",
    "Escape": "Avoidance",
    "Freezing (non-social)": "Avoidance",
    "Freezing (social)": "Avoidance",
    "Non Prosocial (Misc)": "Avoidance",
    "Social Avoidance (Misc)": "Avoidance",
    "Stretched Approach": "Avoidance",
    "Thigmotaxis": "Avoidance",
    "Thigmotaxis (without freezing)": "Avoidance",
    "Wall Jumping": "Avoidance",
    # Aggression
    "Aggression (Misc)": "Aggression",
    "Aggressive Grooming Barbing": "Aggression",
    "Aggressive Grooming/Barbing": "Aggression",
    "Attack": "Aggression",
    "Chasing": "Aggression",
    "Mounting": "Aggression",
    "Tail Rattling": "Aggression",
    "Tail-rattling": "Aggression",
    "White Mouse Aggression": "Aggression",
    # Asocial
    "Other Behavior of Interest": "Asocial",
    "Self Grooming": "Asocial",
    "Self-Grooming": "Asocial",
    "White Mouse Behavior of Interest": "Asocial",
}
FAMILY_ORDER = ["Prosocial", "Avoidance", "Aggression", "Asocial"]
FAMILY_CMAP_NAME = {"Prosocial": "Blues", "Avoidance": "Purples", "Aggression": "Reds", "Asocial": "Greys"}


def behavior_family_palette(behaviors_present):
    """Order behaviors by family and give each a shade of its family's color.

    Families follow `FAMILY_ORDER` (Prosocial, Avoidance, Aggression,
    Asocial). Each behavior gets a shade from its family's sequential
    colormap, so hue shows the family and shade the specific behavior.

    Parameters
    ----------
    behaviors_present : iterable of str
        Behavior labels to color.

    Returns
    -------
    ordered_behaviors : list of str
        The behaviors, sorted by family and then by name. Behaviors not in
        `BEHAVIOR_FAMILY` come last.
    colors : list of RGBA tuples
        One color per behavior, in the same order.

    Notes
    -----
    A behavior missing from `BEHAVIOR_FAMILY` is colored green and reported
    with a printed warning, instead of being placed in a family silently.
    """
    ordered_behaviors, colors = [], []
    for family in FAMILY_ORDER:
        members = sorted(b for b in behaviors_present if BEHAVIOR_FAMILY.get(b) == family)
        if not members:
            continue
        shades = plt.get_cmap(FAMILY_CMAP_NAME[family])(np.linspace(0.35, 0.9, len(members)))
        ordered_behaviors.extend(members)
        colors.extend(shades)

    unclassified = sorted(b for b in behaviors_present if b not in BEHAVIOR_FAMILY)
    if unclassified:
        print(f"warning: {len(unclassified)} behavior label(s) not in BEHAVIOR_FAMILY, "
              f"shown in green: {unclassified}")
        shades = plt.get_cmap("Greens")(np.linspace(0.35, 0.9, len(unclassified)))
        ordered_behaviors.extend(unclassified)
        colors.extend(shades)

    return ordered_behaviors, colors


# --- plotting helpers ---

def label_cluster_centroids(ax, embedding, point_clusters, text_color, outline_color,
                            fontsize=8, stroke_width=2):
    """Write each cluster's number at the mean position of its points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    embedding : np.ndarray, shape (n_points, 2)
        2D embedding of all points.
    point_clusters : np.ndarray of int, shape (n_points,)
        Cluster label of each point.
    text_color : color
        Color of the numbers.
    outline_color : color
        Color of the outline drawn around each number.
    fontsize : float, default 8
        Font size of the numbers, in points.
    stroke_width : float, default 2
        Width of the outline, in points.
    """
    for c in np.unique(point_clusters[point_clusters > 0]):
        cx, cy = embedding[point_clusters == c].mean(axis=0)
        ax.text(cx, cy, str(c), fontsize=fontsize, fontweight="bold", ha="center", va="center",
                color=text_color, path_effects=[pe.withStroke(linewidth=stroke_width, foreground=outline_color)])


def draw_cluster_outlines(ax, clusters, histogram_x, histogram_y, color, linewidth=0.6):
    """Draw the boundaries between clusters as contour lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    clusters : np.ndarray of int, shape (bins, bins)
        Cluster label image from `run_watershed`, indexed
        ``[y_bin, x_bin]``.
    histogram_x, histogram_y : np.ndarray, shape (bins + 1,)
        Bin edges from the same `run_watershed` call. They place the
        outlines in embedding coordinates.
    color : color
        Line color. Choose one that contrasts with the panel background.
    linewidth : float, default 0.6
        Line width, in points.
    """
    for label_id in np.unique(clusters):
        mask = (clusters == label_id).astype(float)
        ax.contour(mask, levels=[0.5], extent=[histogram_x[0], histogram_x[-1], histogram_y[0], histogram_y[-1]],
                   colors=color, linewidths=linewidth, alpha=0.7)


def per_point_density(x, y, bins, stddev, xlim, ylim):
    """Compute the smoothed local density at each point.

    The points are binned into a 2D histogram and smoothed with a Gaussian
    (the same technique as the watershed). The map is scaled to a maximum
    of 1 within this set of points, so groups of different sizes can be
    compared by shape, not raw count.

    Parameters
    ----------
    x, y : np.ndarray, shape (n_points,)
        Point coordinates (e.g. one group's embedding).
    bins : int
        Number of histogram bins per axis.
    stddev : float
        Standard deviation of the Gaussian smoothing, in bins.
    xlim, ylim : tuple of float
        (min, max) range of the histogram along each axis. Use the same
        limits for every group so their maps line up.

    Returns
    -------
    point_density : np.ndarray, shape (n_points,)
        Density at each point's bin, between 0 and 1.
    smoothed : np.ndarray, shape (bins, bins)
        The normalized density map, indexed ``[x_bin, y_bin]``.
    """
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    smoothed = convolve_fft(hist, Gaussian2DKernel(x_stddev=stddev, y_stddev=stddev))
    smoothed = smoothed / smoothed.max()

    x_bins = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    y_bins = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return smoothed[x_bins, y_bins], smoothed
