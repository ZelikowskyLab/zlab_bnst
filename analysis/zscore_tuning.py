"""
zscore_tuning.py

Computes per-cell behavior-tuning z-scores against a circular-shift null.

Metric: for each behavior type, pool all labeled frames across every bout
and compute a single mean: a time-weighted average of activity during that behavior.

Null: circular shift of the full trace, re-extracted at the same bout intervals.

Outputs saved to data/z_scores_all.csv and data/p_values_all.csv.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import sys
sys.path.extend(['/Users/amonast/Documents/GitHub/zlab_miniscope'])
from load_traces import load_aligned_act

def _resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by start; merge same-behavior overlaps, clip different-behavior overlaps."""
    df = df.sort_values("start").reset_index(drop=True)
    result = []
    i = 0
    while i < len(df):
        cur = df.iloc[i].copy()
        while i + 1 < len(df):
            nxt = df.iloc[i + 1]
            if nxt["start"] >= cur["end"]:
                break
            if nxt["Behavior"] == cur["Behavior"]:
                cur["end"] = max(cur["end"], nxt["end"])
                i += 1
            else:
                cur["end"] = nxt["start"]
                break
        if cur["end"] > cur["start"]:
            result.append(cur)
        i += 1
    return pd.DataFrame(result).reset_index(drop=True)


def get_intervals(behavior_df: pd.DataFrame, animal: str, session: str) -> pd.DataFrame:
    ani_beh = behavior_df[
        (behavior_df["animal"]  == animal) &
        (behavior_df["session"] == session)
    ]
    records = []
    for (behavior, btype), grp in ani_beh.groupby(["Behavior", "behavior_type"]):
        starts = grp[grp["Event_Type"] == "State start"]["time"].dt.total_seconds().to_numpy()
        stops  = grp[grp["Event_Type"] == "State stop" ]["time"].dt.total_seconds().to_numpy()
        for s, e in zip(starts, stops):
            records.append(dict(Behavior=behavior, behavior_type=btype, start=s, end=e))
    intervals = _resolve_overlaps(pd.DataFrame(records))
    intervals["duration"] = intervals["end"] - intervals["start"]
    return intervals


def _time_mean(act_vals: np.ndarray, act_times: np.ndarray,
               intervals_bt: pd.DataFrame, extra_mask: np.ndarray = None) -> np.ndarray:
    """
    Mean activity across all frames belonging to any bout (time-weighted).
    Returns (N_cells,) array; NaN if no bouts overlap the trace.
    """
    mask = np.zeros(len(act_times), dtype=bool)
    for _, bout in intervals_bt.iterrows():
        mask |= (act_times >= bout["start"]) & (act_times < bout["end"])
    if extra_mask is not None:
        mask |= extra_mask
    if mask.sum() == 0:
        return np.full(act_vals.shape[1], np.nan)
    return act_vals[mask].mean(axis=0)


def zscore_against_shuffle(act_vals:  np.ndarray,
                            act_times: np.ndarray,
                            cell_cols: list,
                            intervals: pd.DataFrame,
                            animal:    str,
                            session:   str,
                            OUT_DIR = str,
                            n_shuffles: int=1000,
                            seed:       int=42,
                            exclude_early_shuffle: bool=True,
                            baseline_cutoff: float=120,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each behavior_type, compute a per-cell z-score and permutation p-value
    against a circular-shift null.

    The observed metric is the mean across all labeled frames (time-weighted).
    The null is built by circularly shifting the full trace and re-computing
    the same metric at the original bout intervals

    The baseline period (frames before `baseline_cutoff`) is folded into the
    "Other" category rather than scored as its own separate behavior_type.

    Returns z_scores and p_values as DataFrames indexed by
    (animal, session, behavior_type).
    """
    rng = np.random.default_rng(seed)
    T   = len(act_vals)

    scored = intervals[~intervals["behavior_type"].isin(["Baseline"])]
    btypes = scored["behavior_type"].unique().tolist()
    if "Other" not in btypes:
        btypes.append("Other")

    baseline_mask = act_times < baseline_cutoff

    # ── Observed: mean of per-bout means ──────────────────────────────────────
    observed = {}
    for bt in btypes:
        bt_ivs = scored[scored["behavior_type"] == bt]
        observed[bt] = _time_mean(act_vals, act_times, bt_ivs,
                                   extra_mask=baseline_mask if bt == "Other" else None)

    # ── Null via circular shift ────────────────────────────────────────────────
    scored_null = scored[scored["start"] >= 200] if exclude_early_shuffle else scored

    shuffle_dist  = {bt: np.zeros((n_shuffles, len(cell_cols))) for bt in btypes}
    rolled_traces = np.zeros((n_shuffles, T, len(cell_cols)), dtype=act_vals.dtype)

    for i in range(n_shuffles):
        shift        = rng.integers(low=1, high=T)
        shifted_vals = np.roll(act_vals, shift, axis=0)
        rolled_traces[i] = shifted_vals

        for bt in btypes:
            bt_ivs               = scored_null[scored_null["behavior_type"] == bt]
            shuffle_dist[bt][i]  = _time_mean(shifted_vals, act_times, bt_ivs,
                                               extra_mask=baseline_mask if bt == "Other" else None)

    # # ── Save full rolled traces (n_shuffles, n_timepoints, n_cells) ───────────
    # shuf_path = OUT_DIR / f"{animal}_{session}_shuffles.npz"
    # np.savez_compressed(shuf_path, rolled_traces=rolled_traces,
    #                     act_times=act_times, cell_cols=np.array(cell_cols))

    # ── Z-score and two-sided permutation p-value ─────────────────────────────
    index = pd.MultiIndex.from_tuples(
        [(animal, session, bt) for bt in btypes],
        names=["animal", "session", "behavior_type"],
    )
    z_scores = pd.DataFrame(index=index, columns=cell_cols, dtype=float)
    p_values = pd.DataFrame(index=index, columns=cell_cols, dtype=float)

    for bt in btypes:
        obs  = observed[bt]
        shuf = shuffle_dist[bt]
        mu   = shuf.mean(axis=0)
        sd   = shuf.std(axis=0)
        sd   = np.where(sd == 0, np.nan, sd)

        z_scores.loc[(animal, session, bt)] = (obs - mu) / sd
        p_values.loc[(animal, session, bt)] = (
            (np.abs(shuf - mu) >= np.abs(obs - mu)).mean(axis=0)
        )

    return z_scores, p_values


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # ── Config ─────────────────────────────────────────────────────────────────────
    BEHAVIOR_CSV    = Path("/Users/amonast/Desktop/miniscope/data/tuning/intervals_allmice.csv")
    TRACES_DIR      = Path("/Users/amonast/Desktop/miniscope/Traces")
    INFO_CSV        = Path("/Users/amonast/Desktop/miniscope/dbnst_data_info.csv")
    OUT_DIR         = Path("/Users/amonast/Desktop/miniscope/data")
    FPS             = 30
    BASELINE_CUTOFF = 120   # seconds; frames before this are labelled Baseline
    N_SHUFFLES      = 1000
    SEED            = 42

    behavior_df = pd.read_csv(BEHAVIOR_CSV, index_col=0)
    behavior_df["time"] = pd.to_timedelta(behavior_df["time"])

    animal_sessions = [
        # ('C1F2', 'R1'), ('C1F2', 'R2'),
        # ('C1M1', 'R1'), ('C1M1', 'R2'),
        # ('C1M2', 'R1'), ('C1M2', 'R2'),
        # ('C2F1', 'R1'), ('C2F1', 'R2'),
        # ('C2M4', 'R1'), ('C2M4', 'R2'),
        # ('C3F3', 'R1'), ('C3F3', 'R2'),
        # ('C3F4', 'R1'), ('C3F4', 'R2'),
        # ('C3M2', 'R1'), ('C3M2', 'R2'),
        # ('C3M3', 'R1'), ('C3M3', 'R2'),
        # ('C3M4', 'R1'), ('C3M4', 'R2'),
        ('C0F2C','R2')]

    all_z, all_p = [], []

    for animal,session in animal_sessions:
            print(f"  {animal} {session} ...", end=" ", flush=True)
            try:
                act_df    = load_aligned_act(animal, session,
                                             traces_dir=TRACES_DIR,
                                             behavior_csv=BEHAVIOR_CSV,
                                             info_csv=INFO_CSV,
                                             baseline_cutoff=BASELINE_CUTOFF,
                                             fps=FPS)
                cell_cols = [c for c in act_df.columns
                             if str(c).isdigit() or isinstance(c, (int, np.integer))]
                act_times = act_df.index.get_level_values("timestamp").total_seconds().to_numpy()
                act_vals  = act_df[cell_cols].values.astype(float)
            except StopIteration:
                print("no trace file, skipping")
                continue

            intervals = get_intervals(behavior_df, animal, session)

            z, p = zscore_against_shuffle(
                            act_vals, act_times, cell_cols,
                            intervals, animal, session,
                            OUT_DIR=OUT_DIR,n_shuffles=N_SHUFFLES,
                            baseline_cutoff=BASELINE_CUTOFF,)
            
            z.to_csv(OUT_DIR / f"z_scores_{animal}_{session}.csv")
            p.to_csv(OUT_DIR / f"p_values_{animal}_{session}.csv")

            print(f"done ({z.shape[1]} cells)")



if __name__ == "__main__":
    main()
