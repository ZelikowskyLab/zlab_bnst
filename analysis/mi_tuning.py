"""
mi_tuning.py

Computes per-cell mutual information (MI) between activity and a categorical
behavior label, bias-corrected against a circular-shift null.

Metric: MI(activity, label) in bits, where label is either the fine-grained
'Behavior' column or the coarser 'behavior_type' column (selectable via
label_col). Decomposes additively into each state's non-negative contribution:
    MI = sum_state p(state) * KL( p(activity | state) || p(activity) )

Null: circular shift of the full trace (same mechanism as zscore_tuning.py),
re-binned and re-scored at the same frame labels. Used both to bias-correct
the plug-in MI estimate (a known finite-sample bias, worse for low-occupancy
states) and to get a z-score / p-value per cell.

Note the p-value here is one-sided (P(null >= observed)): MI and its
per-state contributions are non-negative by construction, so unlike the
zscore_tuning two-sided mean-shift test, only "more informative than chance"
is meaningful.

Outputs saved to data/mi_bits_*.csv, data/mi_z_*.csv, data/mi_p_*.csv,
one set per (animal, session, label_col).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.extend(['/Users/amonast/Documents/GitHub/zlab_miniscope'])
from load_traces import load_aligned_act
from zscore_tuning import get_intervals


def assign_frame_labels(act_times: np.ndarray, intervals: pd.DataFrame,
                         label_col: str = "behavior_type",
                         baseline_cutoff: float = 120) -> np.ndarray:
    """
    Per-frame categorical label ('Behavior' or 'behavior_type') from the
    resolved, non-overlapping interval partition returned by get_intervals().
    Frames not covered by any scored bout, plus frames before
    baseline_cutoff, are labeled "Other".
    """
    scored = intervals[intervals["behavior_type"] != "Baseline"]
    labels = np.full(len(act_times), "Other", dtype=object)
    for _, bout in scored.iterrows():
        mask = (act_times >= bout["start"]) & (act_times < bout["end"])
        labels[mask] = bout[label_col]
    labels[act_times < baseline_cutoff] = "Other"
    return labels


def discretize(act_vals: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Per-cell quantile bins computed once from the true (unshifted) trace.
    Circularly shifting the trace and rebinning is equivalent to rolling
    these bin indices directly (digitize is applied pointwise), so shuffles
    reuse this array instead of rebinning each iteration.

    Flat/near-constant traces collapse duplicate quantile edges via
    np.unique, yielding fewer effective bins for that cell (MI ~ 0 either
    way).
    """
    T, n_cells = act_vals.shape
    bin_idx = np.zeros((T, n_cells), dtype=int)
    for c in range(n_cells):
        edges = np.unique(np.quantile(act_vals[:, c], np.linspace(0, 1, n_bins + 1)))
        if len(edges) <= 2:
            continue
        bin_idx[:, c] = np.clip(np.digitize(act_vals[:, c], edges[1:-1]), 0, len(edges) - 2)
    return bin_idx


def _mi_batch(bin_idx: np.ndarray, y_idx: np.ndarray,
              n_bins: int, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized plug-in MI (bits) and per-state contributions for every cell
    at once, from a single scatter-add contingency table.

    bin_idx : (T, N_cells) int array, values in [0, n_bins)
    y_idx   : (T,) int array, values in [0, n_states)

    Returns
    -------
    mi_total : (N_cells,) total MI in bits
    contrib  : (n_states, N_cells) each state's non-negative contribution
               to that cell's total MI (contrib.sum(axis=0) == mi_total)
    """
    T, n_cells = bin_idx.shape
    combined = bin_idx * n_states + y_idx[:, None]
    cell_idx = np.broadcast_to(np.arange(n_cells), combined.shape)

    counts = np.zeros((n_bins * n_states, n_cells))
    np.add.at(counts, (combined.ravel(), cell_idx.ravel()), 1.0)
    counts = counts.reshape(n_bins, n_states, n_cells)

    pxy = counts / T
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(pxy > 0, pxy * np.log2(pxy / (px * py)), 0.0)

    contrib = terms.sum(axis=0)
    mi_total = contrib.sum(axis=0)
    return mi_total, contrib


def mi_against_shuffle(act_vals: np.ndarray,
                        act_times: np.ndarray,
                        cell_cols: list,
                        intervals: pd.DataFrame,
                        animal: str,
                        session: str,
                        label_col: str = "behavior_type",
                        n_bins: int = 8,
                        n_shuffles: int = 1000,
                        seed: int = 42,
                        baseline_cutoff: float = 120) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Shuffle-bias-corrected mutual information between each cell's activity
    and a categorical label ('Behavior' or 'behavior_type').

    Returns three DataFrames indexed by (animal, session, state), columns
    are cell_cols:
      mi_bits  — bias-corrected MI (observed - null mean). state="Total" is
                 the omnibus MI across all states of label_col; other rows
                 are that state's contribution to the total.
      z_scores — (observed - null_mean) / null_std
      p_values — one-sided: P(null >= observed)
    """
    rng = np.random.default_rng(seed)
    T, n_cells = act_vals.shape

    labels = assign_frame_labels(act_times, intervals, label_col=label_col,
                                  baseline_cutoff=baseline_cutoff)
    states = sorted(pd.unique(labels).tolist())
    state_to_idx = {s: i for i, s in enumerate(states)}
    y_idx = np.array([state_to_idx[s] for s in labels])
    n_states = len(states)

    bin_idx = discretize(act_vals, n_bins=n_bins)

    obs_total, obs_contrib = _mi_batch(bin_idx, y_idx, n_bins, n_states)

    null_total = np.zeros((n_shuffles, n_cells))
    null_contrib = np.zeros((n_shuffles, n_states, n_cells))
    for i in range(n_shuffles):
        shift = rng.integers(low=1, high=T)
        shifted_bins = np.roll(bin_idx, shift, axis=0)
        null_total[i], null_contrib[i] = _mi_batch(shifted_bins, y_idx, n_bins, n_states)

    row_labels = ["Total"] + states
    index = pd.MultiIndex.from_tuples(
        [(animal, session, r) for r in row_labels],
        names=["animal", "session", label_col],
    )
    mi_bits = pd.DataFrame(index=index, columns=cell_cols, dtype=float)
    z_scores = pd.DataFrame(index=index, columns=cell_cols, dtype=float)
    p_values = pd.DataFrame(index=index, columns=cell_cols, dtype=float)

    def _fill(row_label, obs, null):
        mu = null.mean(axis=0)
        sd = null.std(axis=0)
        sd = np.where(sd == 0, np.nan, sd)
        mi_bits.loc[(animal, session, row_label)] = obs - mu
        z_scores.loc[(animal, session, row_label)] = (obs - mu) / sd
        p_values.loc[(animal, session, row_label)] = (null >= obs).mean(axis=0)

    _fill("Total", obs_total, null_total)
    for s in states:
        si = state_to_idx[s]
        _fill(s, obs_contrib[si], null_contrib[:, si, :])

    return mi_bits, z_scores, p_values


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Config ─────────────────────────────────────────────────────────────────
    BEHAVIOR_CSV = Path("/Users/amonast/Desktop/miniscope/data/intervals_allmice.csv")
    TRACES_DIR = Path("/Users/amonast/Desktop/miniscope/Traces")
    INFO_CSV = Path("/Users/amonast/Desktop/miniscope/dbnst_data_info.csv")
    OUT_DIR = Path("/Users/amonast/Desktop/miniscope/data")
    FPS = 30
    BASELINE_CUTOFF = 120
    N_BINS = 8
    N_SHUFFLES = 1000
    SEED = 42
    LABEL_COLS = ["behavior_type", "Behavior"]

    behavior_df = pd.read_csv(BEHAVIOR_CSV, index_col=0)
    behavior_df["time"] = pd.to_timedelta(behavior_df["time"])

    animal_sessions = [
        ('C1F2', 'R1'), ('C1F2', 'R2'),
        ('C1M1', 'R1'), ('C1M1', 'R2'),
        ('C1M2', 'R1'), ('C1M2', 'R2'),
        ('C2F1', 'R1'), ('C2F1', 'R2'),
        ('C2M4', 'R1'), ('C2M4', 'R2'),
        ('C3F3', 'R1'), ('C3F3', 'R2'),
        ('C3F4', 'R1'), ('C3F4', 'R2'),
        ('C3M2', 'R1'), ('C3M2', 'R2'),
        ('C3M3', 'R1'), ('C3M3', 'R2'),
        ('C3M4', 'R1'), ('C3M4', 'R2'),
        ('C0F2C', 'R2')]

    for animal, session in animal_sessions:
        print(f"  {animal} {session} ...", end=" ", flush=True)
        try:
            act_df = load_aligned_act(animal, session,
                                       traces_dir=TRACES_DIR,
                                       behavior_csv=BEHAVIOR_CSV,
                                       info_csv=INFO_CSV,
                                       baseline_cutoff=BASELINE_CUTOFF,
                                       fps=FPS)
            cell_cols = [c for c in act_df.columns
                         if str(c).isdigit() or isinstance(c, (int, np.integer))]
            act_times = act_df.index.get_level_values("timestamp").total_seconds().to_numpy()
            act_vals = act_df[cell_cols].values.astype(float)
        except StopIteration:
            print("no trace file, skipping")
            continue

        intervals = get_intervals(behavior_df, animal, session)

        for label_col in LABEL_COLS:
            mi, z, p = mi_against_shuffle(
                act_vals, act_times, cell_cols, intervals,
                animal, session, label_col=label_col,
                n_bins=N_BINS, n_shuffles=N_SHUFFLES, seed=SEED,
                baseline_cutoff=BASELINE_CUTOFF,
            )
            suffix = "type" if label_col == "behavior_type" else "behavior"
            mi.to_csv(OUT_DIR / f"mi_bits_{suffix}_{animal}_{session}.csv")
            z.to_csv(OUT_DIR / f"mi_z_{suffix}_{animal}_{session}.csv")
            p.to_csv(OUT_DIR / f"mi_p_{suffix}_{animal}_{session}.csv")

        print(f"done ({len(cell_cols)} cells)")


if __name__ == "__main__":
    main()
