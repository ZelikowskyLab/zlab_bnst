"""
fill_handscored_rows.py

Fills the timestamp columns for rows that were hand-added to an Observer
export (only Behavior / Event_Type / Time_Relative_sf / Duration_sf entered).

Every Observer-generated row satisfies
    Date_Time_Absolute = session_start + Time_Relative_sf
so session_start is estimated from the complete rows and the missing columns
are rebuilt from Time_Relative_sf.

A bout of another behavior that fully encloses a hand-scored bout is split
around it (e.g. Digging 405.5-410.4 s around Aggression 407.8-409.4 s becomes
Digging 405.5-407.8 s and 409.4-410.4 s), since get_intervals would otherwise
clip the enclosing bout at the hand-scored start and drop the remainder.

Default is a dry run (prints the filled rows). Pass --write to save; the
original file is first copied to <cohort>_raw_backup/ (outside the glob the
loader uses, so it isn't picked up twice).

    python fill_handscored_rows.py "<path to xlsx>"
    python fill_handscored_rows.py "<path to xlsx>" --write
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLS = ["Date_Time_Absolute_dmy_hmsf", "Date_dmy", "Time_Absolute_hms",
             "Time_Absolute_f", "Time_Relative_hmsf", "Time_Relative_hms",
             "Time_Relative_f"]


def _bouts(df: pd.DataFrame) -> list[tuple]:
    """(behavior, start_idx, stop_idx), pairing starts/stops in row order as get_intervals does."""
    out = []
    for beh, g in df[df["Event_Type"].isin(["State start", "State stop"])].groupby("Behavior"):
        starts = g.index[g["Event_Type"] == "State start"]
        stops  = g.index[g["Event_Type"] == "State stop"]
        out += [(beh, s, e) for s, e in zip(starts, stops)]
    return out


def _find_enclosing(df: pd.DataFrame):
    t = df["Time_Relative_sf"]
    bouts = _bouts(df)
    for hb, hs, he in bouts:
        if not df.loc[hs, "_hand"]:
            continue
        for ob, os_, oe in bouts:
            if ob != hb and t[os_] < t[hs] and t[he] < t[oe]:
                return (hs, he), (os_, oe)
    return None


def split_enclosing_bouts(df: pd.DataFrame) -> pd.DataFrame:
    """Split other-behavior bouts that fully contain a hand-scored bout (one per pass)."""
    blank = TIME_COLS + ["Observation", "Event_Log", "Comment"]
    while (hit := _find_enclosing(df)) is not None:
        (hs, he), (os_, oe) = hit
        t = df["Time_Relative_sf"]

        stop = df.loc[oe].copy()
        stop[blank] = np.nan
        stop[["Time_Relative_sf", "Duration_sf", "_added"]] = [t[hs], 0.0, True]

        start = df.loc[os_].copy()
        start[blank] = np.nan
        start[["Time_Relative_sf", "Duration_sf", "_added"]] = [t[he], t[oe] - t[he], True]

        df.loc[os_, "Duration_sf"] = t[hs] - t[os_]
        # stop goes just before the hand-scored start, start just after its stop
        new = pd.DataFrame([stop, start], index=[hs - 0.5, he + 0.5])
        df = pd.concat([df, new]).sort_index().reset_index(drop=True)
    return df


def fill_missing_times(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    df = df.copy()
    # concat with blank rows leaves these as object dtype
    for col in ["Date_Time_Absolute_dmy_hmsf", "Date_dmy", "Time_Absolute_hms"]:
        df[col] = pd.to_datetime(df[col])
    missing = df["Date_Time_Absolute_dmy_hmsf"].isna() & df["Time_Relative_sf"].notna()
    ok = ~missing & df["Date_Time_Absolute_dmy_hmsf"].notna()

    offsets = (df.loc[ok, "Date_Time_Absolute_dmy_hmsf"]
               - pd.to_timedelta(df.loc[ok, "Time_Relative_sf"], unit="s"))
    spread = (offsets.max() - offsets.min()).total_seconds()
    assert spread < 0.01, f"session start not constant (spread {spread:.4f}s)"
    session_start = offsets.median()

    rel = pd.to_timedelta(df.loc[missing, "Time_Relative_sf"], unit="s").dt.round("ms")
    absolute = session_start.round("ms") + rel

    df.loc[missing, "Date_Time_Absolute_dmy_hmsf"] = absolute
    df.loc[missing, "Date_dmy"]                    = absolute
    df.loc[missing, "Time_Absolute_hms"]           = absolute.dt.floor("s")
    df.loc[missing, "Time_Absolute_f"]             = absolute.dt.microsecond // 1000
    df.loc[missing, "Time_Relative_hmsf"]          = [(pd.Timestamp(0) + r).time() for r in rel]
    df.loc[missing, "Time_Relative_hms"]           = [(pd.Timestamp(0) + r.floor("s")).time() for r in rel]
    df.loc[missing, "Time_Relative_f"]             = rel.dt.microseconds // 1000

    # Observation / Event_Log are constant within a single-observation export
    for col in ["Observation", "Event_Log"]:
        vals = df.loc[ok, col].dropna().unique()
        if len(vals) == 1:
            df.loc[missing, col] = vals[0]

    return df, df.index[missing]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    raw = pd.read_excel(args.xlsx)
    df = raw.loc[:, ~raw.columns.str.startswith("Unnamed")].copy()
    df["_hand"]  = df["Date_Time_Absolute_dmy_hmsf"].isna() & df["Time_Relative_sf"].notna()
    df["_added"] = False
    df = split_enclosing_bouts(df)
    df, filled = fill_missing_times(df)
    added = df.index[df["_added"]]
    df = df.drop(columns=["_hand", "_added"])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    show = TIME_COLS + ["Time_Relative_sf", "Duration_sf", "Behavior", "Event_Type"]
    print(f"{len(filled)} rows filled ({len(added)} of them added by splitting): {list(filled)}\n")
    for i in filled:
        print(df.loc[max(0, i - 1):i + 1, show].to_string(), "\n")

    if args.write:
        backup_dir = args.xlsx.parent.parent / f"{args.xlsx.parent.name}_raw_backup"
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(args.xlsx, backup_dir / args.xlsx.name)
        df.to_excel(args.xlsx, index=False)
        print(f"wrote {args.xlsx}\nbackup in {backup_dir}")


if __name__ == "__main__":
    main()
