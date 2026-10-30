import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys 
import argparse
from itertools import product 

_ROOT = Path('/Users/amonast/Desktop/miniscope')
_DEFAULT_TRACES_DIR  = _ROOT / "Traces"
_DEFAULT_BEHAVIOR_CSV = _ROOT / "data" /"tuning"/ "intervals_allmice.csv"
_DEFAULT_INFO_CSV     = _ROOT / "dbnst_data_info.csv"


def _resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:
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


def _build_timestamp_index(traces_dir: Path) -> dict:
    """Return {normalized_datetime_str: Path} for every timestamps_*.csv in traces_dir."""
    index = {}
    for p in traces_dir.glob("timestamps_*.csv"):
        raw = p.stem[len("timestamps_"):]   # strip prefix
        normalized = raw.replace("_", "-")  # unify separators
        index[normalized] = p
    return index


def _extract_tseries_suffix(tseries_g: str) -> str:
    """
    Pull the trailing datetime token out of a TSeries_g string and normalize
    underscores → hyphens.

    Handles:
      ...day1_2024-08-13-12-31-27   → '2024-08-13-12-31-27'
      ...v5_2026_02-02-11-43-00     → '2026-02-02-11-43-00'  (mixed sep)
      ...c2_07-17-10-36-34          → '07-17-10-36-34'       (no year)
    """
    s = tseries_g.strip()
    # Full match: YYYY[-_]MM-DD-HH-MM-SS
    m = re.search(r"(\d{4}[-_]\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", s)
    if m:
        return m.group(1).replace("_", "-")
    # Short match: MM-DD-HH-MM-SS (no year)
    m = re.search(r"(\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", s)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract datetime suffix from TSeries_g: {tseries_g!r}")


def _find_timestamp_path(tseries_g: str, trace_fpath: Path, traces_dir: Path) -> Path:
    """Resolve a TSeries_g value to its timestamps CSV file."""
    ts_index = _build_timestamp_index(traces_dir)
    suffix = _extract_tseries_suffix(tseries_g)

    # If suffix has no year (MM-DD-HH-MM-SS), prepend year from trace filename
    if re.fullmatch(r"\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", suffix):
        m = re.search(r"_(\d{4})(\d{2})(\d{2})_", trace_fpath.name)
        if m:
            suffix = f"{m.group(1)}-{suffix}"

    if suffix in ts_index:
        return ts_index[suffix]

    raise FileNotFoundError(
        f"No timestamp file matching '{suffix}' found in {traces_dir}.\n"
        f"Available keys: {sorted(ts_index)}"
    )

def build_intervals(animal: str, session: str, behavior_csv: Path | None = None) -> pd.DataFrame:
    """
    Build a start/end interval table for one animal/session from the raw behavior CSV.

    Returns a DataFrame with columns: animal, session, Behavior, behavior_type, start, end.
    """
    behavior_csv = Path(behavior_csv) if behavior_csv else _DEFAULT_BEHAVIOR_CSV
    behavior_df = pd.read_csv(behavior_csv, index_col=0)
    behavior_df["time"] = pd.to_timedelta(behavior_df["time"])

    ani_beh = behavior_df[(behavior_df["animal"]  == animal) &
                          (behavior_df["session"] == session)]

    records = []
    for (behavior, btype), grp in ani_beh.groupby(["Behavior", "behavior_type"]):
        starts = grp[grp["Event_Type"] == "State start"]["time"].dt.total_seconds().to_numpy()
        stops  = grp[grp["Event_Type"] == "State stop" ]["time"].dt.total_seconds().to_numpy()
        for s, e in zip(starts, stops):
            records.append(dict(animal=animal, session=session,
                                Behavior=behavior, behavior_type=btype, start=s, end=e))

    if not records:
        raise ValueError(f"No behavior intervals for {animal} {session} in {behavior_csv}")
    return _resolve_overlaps(pd.DataFrame(records))


def load_aligned_act(animal: str,
                        session: str,
                        traces_dir: Path,
                        behavior_csv: Path,
                        info_csv: Path,
                        baseline_cutoff: float = 120,
                        fps: float = 30,
                        zscore: bool = True,):
    """
    Load dF/F traces for one animal/session, label each frame with behavior,
    and attach real-world timestamps from the matching timestamps_*.csv file.

    Parameters
    ----------
    animal, session : str
        Animal ID (e.g. 'C3M2') and session label (e.g. 'R1').
    baseline_cutoff : float
        Frames before this time (seconds) are labelled 'Baseline'.
    fps : float
        Recording frame rate used to convert frame index → seconds.
    zscore : bool
        Z-score each cell column across the full trace before returning.
    traces_dir : Path, optional
        Directory containing both *_dff.csv and timestamps_*.csv files.
        Defaults to <repo_root>/Traces.
    behavior_csv : Path, optional
        Path to intervals_allmice.csv.  Defaults to <repo_root>/data/intervals_allmice.csv.
    info_csv : Path, optional
        Path to dbnst_data_info.csv.  Defaults to <repo_root>/dbnst_data_info.csv.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (animal, session, time), with columns for each cell,
        Behavior, behavior_type, and timestamp (Unix epoch seconds).
    """
    traces_dir   = Path(traces_dir)   if traces_dir   else _DEFAULT_TRACES_DIR
    behavior_csv = Path(behavior_csv) if behavior_csv else _DEFAULT_BEHAVIOR_CSV
    info_csv     = Path(info_csv)     if info_csv     else _DEFAULT_INFO_CSV

    # ── Behavior intervals ────────────────────────────────────────────────────
    intervals = build_intervals(animal, session, behavior_csv)

    # ── Activity trace ────────────────────────────────────────────────────────
    fpath = next(traces_dir.glob(f"{animal}_*_{session}_dff.csv"))
    act            = pd.read_csv(fpath, index_col=0)

    if zscore:
        cell_cols      = [c for c in act.columns if str(c).isdigit() or isinstance(c, (int, np.integer))]
        act[cell_cols] = (act[cell_cols] - act[cell_cols].mean()) / act[cell_cols].std()

    # ── Timestamps ───────────────────────────────────────────────────────────
    info_df = pd.read_csv(info_csv, index_col=None)
    info_df.columns = info_df.columns.str.strip()
    for col in info_df.columns:
        info_df[col] = info_df[col].astype(str).str.strip()

    row = info_df[(info_df["Mouse"]   == animal) &
                    (info_df["Session"] == session)]
    if row.empty:
        raise KeyError(f"No entry for ({animal!r}, {session!r}) in {info_csv}")

    tseries_g = row["TSeries_g"].iloc[0]
    ts_path   = _find_timestamp_path(tseries_g, fpath, traces_dir)

    ts_df = pd.read_csv(ts_path)
    ts_clean = ts_df[['Local Frame Number', 'Frame Timestamp (s)']].copy()
    ts_clean['timestamp'] = pd.to_datetime(ts_clean['Frame Timestamp (s)'], unit='s', utc=True)
    ts_clean['timestamp'] = ts_clean['timestamp'] - ts_clean['timestamp'].iloc[0]
    # Align by frame number: both trace and timestamp CSVs start at frame 0
    n = min(len(act), len(ts_df))
    act = act.iloc[:n].copy()
    act["timestamp"] = ts_clean['timestamp']
    act = act.set_index('timestamp')
    # ── Label frames with behavior ────────────────────────────────────────────
    iv_idx  = pd.IntervalIndex.from_arrays(intervals["start"], intervals["end"], closed="both")
    locs, _ = iv_idx.get_indexer_non_unique(act.index.total_seconds().to_numpy())
    safe    = locs.clip(0)

    act["Behavior"]      = np.where(locs >= 0, intervals["Behavior"].to_numpy()[safe],      "Unlabeled")
    act["behavior_type"] = np.where(locs >= 0, intervals["behavior_type"].to_numpy()[safe], "Other")
    act.loc[act.index.total_seconds() < baseline_cutoff, "Behavior"] = "Baseline"
    act.loc[act.index.total_seconds() < baseline_cutoff, "behavior_type"] = "Other"   
    act["animal"]  = animal
    act["session"] = session

    return act.reset_index().set_index(["animal", "session", "timestamp"])

## Data
def get_tuned_df(animals, data_dir, sessions=['R1', 'R2'], filter_pos=False, z_thr=2):
    """
    filter_pos : bool
        If True, also require zscore > z_thr (not just p<=0.05) for a cell to
        be included in tuned_cells. If False (default), tuned_cells is
        significance-only and can include negatively (inhibited) cells.
    z_thr : float
        Threshold used when filter_pos=True.
    """
    index_cols = ['animal', 'session', 'behavior_type']
    rows = []
    for ani, session in product(animals, sessions):
        pf = f"{data_dir}/tuning/p_values_{ani}_{session}.csv"
        zf = f"{data_dir}/tuning/z_scores_{ani}_{session}.csv"
        if not (Path(pf).exists() and Path(zf).exists()):
            continue
        df_p = pd.read_csv(pf).set_index(index_cols)
        df_z = pd.read_csv(zf).set_index(index_cols)

        chunk = pd.concat({'pval': df_p, 'zscore': df_z}, names=['metric'])
        chunk = chunk.reorder_levels(['animal', 'session', 'behavior_type', 'metric'])
        rows.append(chunk)
    df = pd.concat(rows).sort_index()

    sig = df.xs('pval', level='metric').le(0.05)
    mask = sig
    if filter_pos:
        pos = df.xs('zscore', level='metric').gt(z_thr)
        mask = sig & pos
    tuned = (mask.apply(lambda row: row.index[row].tolist(), axis=1)
                .rename('tuned_cells')
                .reset_index())

    ## exlcude these animals cells as aggessive tuned b/c one was the intruder mouse and one only had one tiny bout
    tuned = tuned[~((tuned["animal"].isin(["C2M4", "C2F1"])) & (tuned["behavior_type"] == "Aggressive"))]
    return tuned,df

def get_co_tuned_cells(df, behavior_a, behavior_b, exclude_animals=None):
    """Cells with p<=0.05 and z>0 for BOTH behavior_a and behavior_b, per (animal, session)."""
    sig = df.xs('pval', level='metric').le(0.05)
    pos = df.xs('zscore', level='metric').gt(0)
    sig_pos = sig & pos

    a = sig_pos.xs(behavior_a, level='behavior_type')
    b = sig_pos.xs(behavior_b, level='behavior_type')
    if exclude_animals:
        a = a.drop(index=exclude_animals, level='animal', errors='ignore')
        b = b.drop(index=exclude_animals, level='animal', errors='ignore')

    co_tuned_mask = a & b.reindex_like(a)
    return (co_tuned_mask.apply(lambda row: row.index[row].tolist(), axis=1)
                          .rename('co_tuned_cells')
                          .reset_index())



def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('animal', type=str, help='animal id')
    parser.add_argument('session', type=str, help='session descriptor, R1 or R2')
    args = parser.parse_args(argv)

    act = load_aligned_act(args.animal, args.session,
                           traces_dir=_DEFAULT_TRACES_DIR,
                           behavior_csv=_DEFAULT_BEHAVIOR_CSV,
                           info_csv=_DEFAULT_INFO_CSV)
    print(act)

if __name__ == '__main__':
    main(['C0F2C', 'R2'])