"""
zscore_tuning_v2.py

"""

import pandas as pd
from pathlib import Path
import numpy as np
from load_traces import load_aligned_act
from zscore_tuning import get_intervals, zscore_against_shuffle


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    BEHAVIOR_CSV    = Path("/Users/amonast/Desktop/miniscope/data/intervals_allmice.csv")
    TRACES_DIR      = Path("/Users/amonast/Desktop/miniscope/Traces")
    INFO_CSV        = Path("/Users/amonast/Desktop/miniscope/dbnst_data_info.csv")
    OUT_DIR         = Path("/Users/amonast/Desktop/miniscope/data")
    FPS             = 30
    BASELINE_CUTOFF = 120
    N_SHUFFLES      = 1000
    SEED            = 42

    UNCLASSIFIED_CSV = Path("/Users/amonast/Desktop/miniscope/data/intervals_unclassified.csv")

    behavior_df = pd.read_csv(BEHAVIOR_CSV, index_col=0)
    behavior_df["time"] = pd.to_timedelta(behavior_df["time"])

    unclassified_df = pd.read_csv(UNCLASSIFIED_CSV, index_col=0)
    unclassified_df["time"] = pd.to_timedelta(unclassified_df["time"])
    behavior_df = pd.concat([behavior_df, unclassified_df], ignore_index=True)

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
    ]

    all_z, all_p = [], []

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
            act_vals  = act_df[cell_cols].values.astype(float)
        except StopIteration:
            print("no trace file, skipping")
            continue

        intervals = get_intervals(behavior_df, animal, session)

        z, p = zscore_against_shuffle(
            act_vals, act_times, cell_cols,
            intervals, animal, session,
            OUT_DIR=OUT_DIR, n_shuffles=N_SHUFFLES, seed=SEED,
        )

        all_z.append(z)
        all_p.append(p)
        print(f"done ({z.shape[1]} cells)")

    if not all_z:
        print("No sessions processed.")
        return

    # ── Combine into long format ───────────────────────────────────────────────
    z_all = pd.concat(all_z)
    p_all = pd.concat(all_p)

    # Stack cells: index (animal, session, behavior_type, cell)
    z_long = (z_all.stack()
                   .rename("z_score")
                   .reset_index()
                   .rename(columns={"level_3": "cell"}))
    p_long = (p_all.stack()
                   .rename("p_value")
                   .reset_index()
                   .rename(columns={"level_3": "cell"}))

    long_df = z_long.merge(p_long, on=["animal", "session", "behavior_type", "cell"])

    # ── Split by tuning direction ──────────────────────────────────────────────
    pos_tuned = long_df[long_df["z_score"] > 0].reset_index(drop=True)
    neg_tuned = long_df[long_df["z_score"] < 0].reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    long_df.to_csv(OUT_DIR / "tuned_all.csv", index=False)
    pos_tuned.to_csv(OUT_DIR / "zscore_positive.csv", index=False)
    neg_tuned.to_csv(OUT_DIR / "zscore_negative.csv", index=False)

    print(f"\nAll sessions combined: {len(long_df):,} rows")
    print(f"  Positively tuned (z>0): {len(pos_tuned):,} rows -> zscore_positive.csv")
    print(f"  Negatively tuned (z<0): {len(neg_tuned):,} rows -> zscore_negative.csv")


if __name__ == "__main__":
    main()
